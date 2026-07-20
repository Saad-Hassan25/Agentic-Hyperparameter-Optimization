"""Selection + convergence — never a raw metric-max (§6.6, §6.7).

`select_best` is the same spirit as the one-standard-error rule in `glmnet`:
prefer the simplest, least-overfit config that isn't measurably worse than the
best one found, rather than chasing a metric difference smaller than the CV
protocol's own noise. `check_convergence` replaces the prototype's "span of
the last 3 scores < 0.005" with three independently-loggable conditions,
gated behind a minimum-trials floor.

Division of responsibility (do not mistake this for a missing feature):
`check_convergence` only ever returns `reason` in
`{"not_converged", "patience_exhausted", "noise_floor_plateau"}`. It never
returns `"budget_exhausted"` — the orchestrator (a later stage) owns
trial-count/wall-time/cost budget checks and constructs its own
`ConvergenceDecision(converged=True, reason="budget_exhausted", ...)` when a
budget fires first, so a run that merely ran out of budget is never logged as
having actually plateaued.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .config import TuningConfig
from .evaluate import TrialOutcome


class ConvergenceDecision(BaseModel):
    converged: bool
    reason: Literal["patience_exhausted", "noise_floor_plateau", "budget_exhausted", "not_converged"]
    trials_used: int


def _ok_trials(trials: list[TrialOutcome]) -> list[TrialOutcome]:
    return [t for t in trials if t.status == "ok" and t.primary_metric is not None]


def compute_noise_floor(trials: list[TrialOutcome]) -> float:
    """The single best trial's fold-to-fold `metric_std` — a proxy for how much
    of an apparent improvement could be CV noise rather than signal (§6.6 step 1)."""
    ok = _ok_trials(trials)
    if not ok:
        return 0.0
    best = max(ok, key=lambda t: t.primary_metric)
    return best.metric_std if best.metric_std is not None else 0.0


def select_best(trials: list[TrialOutcome]) -> TrialOutcome | None:
    """§6.6's three steps: noise floor -> within-noise-floor candidate set ->
    least-overfit, fastest-tiebreak pick. `None` if no trial ever succeeded."""
    ok = _ok_trials(trials)
    if not ok:
        return None
    noise_floor = compute_noise_floor(ok)
    best_metric = max(t.primary_metric for t in ok)
    candidates = [t for t in ok if (best_metric - t.primary_metric) <= noise_floor]

    def sort_key(t: TrialOutcome) -> tuple[float, float]:
        gap = t.overfit_gap if t.overfit_gap is not None else float("inf")
        fit_time = t.fit_time_s if t.fit_time_s is not None else float("inf")
        return (gap, fit_time)

    return min(candidates, key=sort_key)


def check_convergence(trials: list[TrialOutcome], cfg: TuningConfig, seed_k: int) -> ConvergenceDecision:
    """§6.7. `trials` is the agent-guided track only (seed_design + llm_proposal +
    repair_perturbation), in iteration order — never `baseline_sampler` trials,
    which live on a fully separate track and are never passed here."""
    trials_used = len(trials)
    min_trials_floor = max(cfg.min_trials_before_convergence, seed_k)
    if trials_used < min_trials_floor:
        return ConvergenceDecision(converged=False, reason="not_converged", trials_used=trials_used)

    ok_sorted = sorted(_ok_trials(trials), key=lambda t: t.iteration)
    if not ok_sorted:
        return ConvergenceDecision(converged=False, reason="not_converged", trials_used=trials_used)

    # 1. patience_exhausted: replay select_best over the growing prefix of ok
    # trials, recording the iteration whenever the running-best actually changes.
    last_improvement_iteration = ok_sorted[0].iteration
    prev_best_iteration: int | None = None
    for i, t in enumerate(ok_sorted):
        current_best = select_best(ok_sorted[: i + 1])
        if current_best is not None and current_best.iteration != prev_best_iteration:
            last_improvement_iteration = t.iteration
            prev_best_iteration = current_best.iteration
    max_iteration = ok_sorted[-1].iteration
    if max_iteration - last_improvement_iteration >= cfg.patience:
        return ConvergenceDecision(converged=True, reason="patience_exhausted", trials_used=trials_used)

    # 2. noise_floor_plateau: spread of the last `convergence_window` ok trials,
    # against a tolerance never tighter than the CV protocol's own noise floor.
    if len(ok_sorted) >= cfg.convergence_window:
        window = ok_sorted[-cfg.convergence_window :]
        metrics = [t.primary_metric for t in window]
        spread = max(metrics) - min(metrics)
        if spread < max(cfg.convergence_tol, compute_noise_floor(trials)):
            return ConvergenceDecision(converged=True, reason="noise_floor_plateau", trials_used=trials_used)

    return ConvergenceDecision(converged=False, reason="not_converged", trials_used=trials_used)
