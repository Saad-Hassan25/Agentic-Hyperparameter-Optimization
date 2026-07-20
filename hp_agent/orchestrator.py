"""The deterministic controller: trial loop, budgets, manifest, public API (§3, §6.8).

Two tracks run per call. The baseline sampler track (Optuna TPE, else random
search) always runs first, under `cfg.trial_budget` -- it is both the control
group and the full `--no-llm` fallback. When an LLM is configured and
reachable, a second, agent-guided track runs under its own, identically-sized
`cfg.trial_budget`: a Latin-Hypercube seed batch, then an LLM-proposal loop
gated by dedup/validation/repair (`propose.py`), a noise-floor- and
overfit-aware selection rule, and a patience-/noise-floor-based convergence
check (`select.py`). The LLM is called at exactly two points -- propose and
report; everything else here is deterministic so a run is reproducible from
its own `manifest.json` and `trials.jsonl`.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import baseline, design, propose, report, search_space, select
from .adapters import validate_family_for_task
from .config import TuningConfig
from .evaluate import TrialOutcome, evaluate_trial, resolve_metric
from .llm import BudgetExceeded, LLMClient
from .report import ReportContext, TuningReport
from .select import ConvergenceDecision

_SELECTION_RULE = (
    "within noise floor of top score (fold-to-fold metric_std), "
    "minimum overfit_gap, fastest fit_time_s as tie-break"
)

# Maps a propose.py rejection `reason` onto the TrialOutcome `status` it is
# persisted as in trials.jsonl (§7's audit trail requires every rejection
# kind, not just duplicates -- e.g. "num_leaves=310 rejected: exceeds
# 2**max_depth" is a constraint_violation). "llm_error" is deliberately not
# mapped: it carries no proposed config to log (the call failed before any
# values existed), so it is not a "the agent proposed X and it was rejected"
# audit-trail row the way the other three are.
_REJECTION_STATUS = {
    "invalid_schema": "invalid_schema_rejected",
    "constraint_violation": "constraint_violation_rejected",
    "duplicate": "duplicate_rejected",
}


@dataclass
class HPAgentResult:
    """Everything a caller of `HPAgent.run()` gets back. See §5 of the design doc."""

    best_config: dict
    best_metric: float
    baseline_sampler_metric: float
    baseline_sampler_name: str
    lift_over_baseline: float
    report_path: str
    run_dir: str
    agent_trials: list[TrialOutcome] = field(default_factory=list)
    baseline_trials: list[TrialOutcome] = field(default_factory=list)
    convergence: ConvergenceDecision | None = None
    report: TuningReport | None = None
    cost_usd: float = 0.0
    manifest: dict = field(default_factory=dict)


class HPAgent:
    """Public API -- one call. See §5."""

    def __init__(self, config: TuningConfig | None = None):
        self.config = config or TuningConfig.from_env()

    # ------------------------------------------------------------------ #
    def run(self, X, y, task: str, group=None) -> HPAgentResult:
        cfg = self.config
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        validate_family_for_task(cfg.model_family, task)  # raises ValueError, uncaught, on purpose

        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not isinstance(y, (pd.Series, np.ndarray)):
            y = np.asarray(y)

        group_values = group
        if group_values is None and cfg.group_column and cfg.group_column in X.columns:
            group_values = X[cfg.group_column].to_numpy()
        if cfg.group_column and cfg.group_column in X.columns:
            X = X.drop(columns=[cfg.group_column])

        if len(X) < cfg.min_rows_warn_threshold:
            self._log(
                f"Only {len(X)} rows (< min_rows_warn_threshold={cfg.min_rows_warn_threshold}) "
                "-- the CV noise floor will be wide; selection/convergence may be indecisive (§12)."
            )

        t0 = time.time()
        space = search_space.get_search_space(cfg.model_family)
        seed_k = max(5, round(cfg.seed_fraction * cfg.trial_budget))

        llm = LLMClient(cfg)
        llm_ok, why = llm.available()
        if cfg.model is not None and not llm_ok:
            self._log(f"LLM unavailable ({why}) -- proceeding in --no-llm mode.")
        active_llm = llm if llm_ok else None

        # -- Baseline track: always runs first, unconditionally (§6.8). ------ #
        self._log(f"Running baseline track ({cfg.trial_budget} trials)...")
        baseline_trials = baseline.run_baseline_track(space, cfg.model_family, task, X, y, cfg, group_values)
        baseline_best = select.select_best(baseline_trials)
        if baseline_best is None:
            raise RuntimeError(
                "Every baseline trial failed or timed out -- nothing to report. "
                "Check trial_timeout_s and the model family's fit cost on this dataset."
            )
        baseline_sampler_name = baseline.resolve_baseline_sampler_name(cfg)
        self._log(
            f"Baseline track done: best {resolve_metric(task, cfg.metric)}="
            f"{baseline_best.primary_metric:.4f} ({baseline_sampler_name})."
        )

        agent_trials: list[TrialOutcome] = []
        rejected_trials: list[TrialOutcome] = []

        if active_llm is None:
            # §6.8: --no-llm mode skips the agent-guided track entirely. The
            # baseline track's own result stands in for both best_config/
            # best_metric AND baseline_sampler_metric -- lift is 0.0 by
            # definition, not a fabricated comparison against nothing.
            self._log("No LLM configured -- skipping the agent-guided track (--no-llm mode).")
            agent_best = baseline_best
            lift_over_baseline = 0.0
            convergence = ConvergenceDecision(converged=False, reason="not_converged", trials_used=0)
        else:
            self._log(f"Running agent-guided track ({cfg.trial_budget} trials)...")
            seen_signatures: set = set()

            # -- Stage 0: seed (LHS warm start), budget-checked per point. --- #
            seed_budget_exhausted = False
            for seed_cfg in design.generate_seed_designs(space, seed_k, cfg.random_state):
                if llm.budget.would_exceed() or (time.time() - t0) > cfg.max_wall_time_s:
                    seed_budget_exhausted = True
                    break
                outcome = evaluate_trial(
                    config=seed_cfg, space=space, model_family=cfg.model_family, task=task,
                    X=X, y=y, cfg=cfg, group=group_values,
                    iteration=len(agent_trials) + 1, source="seed_design",
                )
                agent_trials.append(outcome)
                seen_signatures.add(search_space.config_signature(seed_cfg, space))

            if seed_budget_exhausted:
                convergence = ConvergenceDecision(
                    converged=True, reason="budget_exhausted", trials_used=len(agent_trials),
                )
            else:
                # -- Stages 1-2: propose -> validate/dedup/repair -> evaluate -- #
                convergence = None
                while len(agent_trials) < cfg.trial_budget:
                    if llm.budget.would_exceed() or (time.time() - t0) > cfg.max_wall_time_s:
                        convergence = ConvergenceDecision(
                            converged=True, reason="budget_exhausted", trials_used=len(agent_trials),
                        )
                        break

                    rejections: list[dict] = []
                    current_best = select.select_best(agent_trials)
                    try:
                        proposed_config, source, _rationale = propose.propose_next(
                            active_llm, space, cfg.model_family, agent_trials, cfg,
                            current_best, seen_signatures, rejections,
                        )
                    except BudgetExceeded:
                        # §6.1/§12: the cost ceiling was spent mid-attempt inside
                        # propose_next's own LLM call, not caught at the top-of-
                        # loop check above -- still a clean, logged stop, never a
                        # crash out of HPAgent.run().
                        convergence = ConvergenceDecision(
                            converged=True, reason="budget_exhausted", trials_used=len(agent_trials),
                        )
                        break
                    for rej in rejections:
                        status = _REJECTION_STATUS.get(rej.get("reason"))
                        if status is None:
                            continue
                        rejected_trials.append(TrialOutcome(
                            iteration=len(agent_trials) + 1,
                            source="llm_proposal",
                            config=rej["config"] if rej.get("config") is not None else {},
                            status=status,
                            error=rej.get("detail"),
                        ))

                    outcome = evaluate_trial(
                        config=proposed_config, space=space, model_family=cfg.model_family, task=task,
                        X=X, y=y, cfg=cfg, group=group_values,
                        iteration=len(agent_trials) + 1, source=source,
                    )
                    agent_trials.append(outcome)
                    seen_signatures.add(search_space.config_signature(proposed_config, space))

                    convergence = select.check_convergence(agent_trials, cfg, seed_k)
                    if convergence.converged:
                        break

                # Loop exited by reaching trial_budget without budget-exhaustion
                # or check_convergence ever firing -> still a budget, just a
                # trial-count one (§6.7's third bullet).
                if convergence is None or not convergence.converged:
                    convergence = ConvergenceDecision(
                        converged=True, reason="budget_exhausted", trials_used=len(agent_trials),
                    )

            agent_best = select.select_best(agent_trials)
            if agent_best is None:
                raise RuntimeError(
                    "Every agent-guided trial failed or timed out -- nothing to report. "
                    "Check trial_timeout_s and the model family's fit cost on this dataset."
                )
            lift_over_baseline = agent_best.primary_metric - baseline_best.primary_metric
            self._log(
                f"Agent track done ({convergence.reason}, {len(agent_trials)} trials): "
                f"best {resolve_metric(task, cfg.metric)}={agent_best.primary_metric:.4f}, "
                f"lift over baseline={lift_over_baseline:+.4f}."
            )

        # -- Reporting: deterministic stats in, LLM narrative out (§6.9). ---- #
        metric_name = resolve_metric(task, cfg.metric)
        influence = report.compute_hyperparameter_influence(agent_trials, space)
        agent_trials_to_best = report.compute_trials_to_best(agent_trials, agent_best.primary_metric)
        baseline_trials_to_best = report.compute_trials_to_best(baseline_trials, baseline_best.primary_metric)

        ctx = ReportContext(
            task=task,
            metric_name=metric_name,
            model_family=cfg.model_family,
            best_config=agent_best.config,
            best_metric=agent_best.primary_metric,
            selection_rule=_SELECTION_RULE,
            agent_trials_to_best=agent_trials_to_best,
            baseline_sampler_name=baseline_sampler_name,
            baseline_sampler_metric=baseline_best.primary_metric,
            baseline_trials_to_best=baseline_trials_to_best,
            lift_over_baseline=lift_over_baseline,
            hyperparameter_influence=influence,
            convergence=convergence,
            n_agent_trials=len(agent_trials),
            n_baseline_trials=len(baseline_trials),
        )
        tuning_report, report_source = report.llm_report(active_llm, ctx, cfg)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        report_md = report.render_report_markdown(tuning_report, ctx, report_source, generated_at)

        # -- Write the run directory (§7). ----------------------------------- #
        run_id = cfg.run_id or f"{cfg.model_family}-{task}-{int(t0)}"
        run_dir = Path(cfg.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._manifest(
            run_id=run_id, X=X, cfg=cfg, task=task, llm=llm, active_llm=active_llm,
            agent_best=agent_best, baseline_best=baseline_best,
            baseline_sampler_name=baseline_sampler_name, lift_over_baseline=lift_over_baseline,
            convergence=convergence, report_source=report_source, t0=t0,
        )
        (run_dir / "manifest.json").write_text(_json(manifest), encoding="utf-8")

        all_trials = [*agent_trials, *rejected_trials, *baseline_trials]
        (run_dir / "trials.jsonl").write_text(
            "\n".join(t.model_dump_json() for t in all_trials) + ("\n" if all_trials else ""),
            encoding="utf-8",
        )

        best_config_payload = {
            "best_config": agent_best.config,
            "best_metric": agent_best.primary_metric,
            "selection_rule": _SELECTION_RULE,
        }
        (run_dir / "best_config.json").write_text(_json(best_config_payload), encoding="utf-8")

        baseline_comparison = {
            "baseline_sampler_name": baseline_sampler_name,
            "baseline_sampler_metric": baseline_best.primary_metric,
            "agent_metric": agent_best.primary_metric,
            "lift_over_baseline": lift_over_baseline,
            "trials_to_best": {"agent": agent_trials_to_best, "baseline": baseline_trials_to_best},
        }
        (run_dir / "baseline_comparison.json").write_text(_json(baseline_comparison), encoding="utf-8")

        report_path = run_dir / "report.md"
        report_path.write_text(report_md, encoding="utf-8")

        self._log(
            f"Done. best {metric_name}={agent_best.primary_metric:.4f} | "
            f"lift {lift_over_baseline:+.4f} | cost ${llm.budget.cost_usd:.4f} | {run_dir}"
        )

        return HPAgentResult(
            best_config=agent_best.config,
            best_metric=agent_best.primary_metric,
            baseline_sampler_metric=baseline_best.primary_metric,
            baseline_sampler_name=baseline_sampler_name,
            lift_over_baseline=lift_over_baseline,
            report_path=str(report_path),
            run_dir=str(run_dir),
            agent_trials=agent_trials,
            baseline_trials=baseline_trials,
            convergence=convergence,
            report=tuning_report,
            cost_usd=llm.budget.cost_usd,
            manifest=manifest,
        )

    # ------------------------------------------------------------------ #
    def _manifest(
        self, *, run_id: str, X: pd.DataFrame, cfg: TuningConfig, task: str,
        llm: LLMClient, active_llm: LLMClient | None,
        agent_best: TrialOutcome, baseline_best: TrialOutcome,
        baseline_sampler_name: str, lift_over_baseline: float,
        convergence: ConvergenceDecision, report_source: str, t0: float,
    ) -> dict:
        return {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "hash": self._dataset_hash(X),
                "n_rows": int(len(X)),
                "n_cols": int(X.shape[1]),
            },
            "config": cfg.to_dict(),
            "model": {"id": cfg.model, "report_source": report_source},
            "seeds": {"random_state": cfg.random_state},
            "task": task,
            "model_family": cfg.model_family,
            "llm_mode": "llm_guided" if active_llm is not None else "no_llm",
            "agent_track": (
                "ran under its own trial_budget, identical in size to the baseline track."
                if active_llm is not None else
                "skipped -- no LLM configured or reachable; best_config/best_metric are the "
                "baseline track's own result, and lift_over_baseline is 0.0 by definition "
                "(there is no separate agent track to compare against, not a fabricated tie)."
            ),
            "result": {
                "best_metric": round(agent_best.primary_metric, 6),
                "baseline_sampler_metric": round(baseline_best.primary_metric, 6),
                "lift_over_baseline": round(lift_over_baseline, 6),
                "baseline_sampler_name": baseline_sampler_name,
            },
            "convergence": {
                "converged": convergence.converged,
                "reason": convergence.reason,
                "trials_used": convergence.trials_used,
            },
            "llm_calls": [c.__dict__ for c in llm.call_log],
            "cost_usd": round(llm.budget.cost_usd, 6),
            "tokens": {"prompt": llm.budget.prompt_tokens, "completion": llm.budget.completion_tokens},
            "wall_seconds": round(time.time() - t0, 2),
            "package_versions": _versions(),
        }

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[hp-agent] {msg}", file=sys.stderr)

    @staticmethod
    def _dataset_hash(X: pd.DataFrame) -> str:
        """Short sha256 over shape/columns/a sample of values (mirrors `feature_agent`)."""
        h = hashlib.sha256()
        h.update(str(X.shape).encode())
        h.update("|".join(map(str, X.columns)).encode())
        try:
            h.update(pd.util.hash_pandas_object(X.head(1000), index=True).to_numpy().tobytes())
        except Exception:
            pass
        return h.hexdigest()[:16]


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "optuna", "pydantic"):
        mod_name = pkg.replace("scikit-learn", "sklearn")
        try:
            out[pkg] = importlib.import_module(mod_name).__version__
        except Exception:
            out[pkg] = "not installed"
    return out


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
