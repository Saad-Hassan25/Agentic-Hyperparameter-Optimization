"""Baseline sampler track — the control group and the `--no-llm` fallback (§6.8).

Every run also executes a classical sampler (Optuna's TPE, or uniform random
search if Optuna isn't installed) under the SAME `SearchSpace`, the SAME CV
protocol, and the SAME `cfg.trial_budget` as the agent-guided track — not
sharing the agent's budget, a fully separate track of its own, every trial
recorded with `source="baseline_sampler"`. This is deliberately one mechanism
doing two jobs: it is both the `--no-llm` fallback (the orchestrator returns
this track alone when `cfg.model is None`) and the control group the final
report's `lift_over_baseline` compares the agent against. Both jobs only mean
anything if the protocol really is identical, so this module never
re-implements CV scoring — every trial, in either sampler branch, is scored by
`evaluate.evaluate_trial()` directly, giving the baseline track the exact same
per-trial timeout and metric orientation as the agent track.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .config import TuningConfig
from .design import decode_unit_point
from .evaluate import TrialOutcome, evaluate_trial
from .search_space import SearchSpace, repair_config


def resolve_baseline_sampler_name(cfg: TuningConfig) -> Literal["optuna_tpe", "random_search"]:
    """Which sampler `run_baseline_track` will use, per §8's `baseline_sampler` knob.

    `"optuna_tpe"` only when explicitly requested, or requested via `"auto"`
    and optuna is importable. Every other case — explicit `"random_search"`,
    `"auto"` with optuna missing, or any unrecognized string — falls back to
    `"random_search"` rather than raising: a typo'd config value degrades
    gracefully instead of crashing the run (§12's risk table).
    """
    if cfg.baseline_sampler == "optuna_tpe":
        return "optuna_tpe"
    if cfg.baseline_sampler == "auto":
        try:
            import optuna  # noqa: F401
        except ImportError:
            return "random_search"
        return "optuna_tpe"
    return "random_search"


def run_baseline_track(
    space: SearchSpace,
    model_family: str,
    task: str,
    X,
    y,
    cfg: TuningConfig,
    group=None,
) -> list[TrialOutcome]:
    """Run `cfg.trial_budget` baseline trials, iterations 1..trial_budget.

    Dispatches on `resolve_baseline_sampler_name`. The optuna branch is itself
    guarded by an `ImportError` fallback to random search — belt-and-braces
    against the rare case where `resolve_baseline_sampler_name` picked
    `"optuna_tpe"` (e.g. via `"auto"`) but optuna turns out not to be
    importable by the time this actually runs.
    """
    if resolve_baseline_sampler_name(cfg) == "optuna_tpe":
        try:
            return _run_optuna_tpe(space, model_family, task, X, y, cfg, group)
        except ImportError:
            pass
    return _run_random_search(space, model_family, task, X, y, cfg, group)


def _run_optuna_tpe(
    space: SearchSpace, model_family: str, task: str, X, y, cfg: TuningConfig, group
) -> list[TrialOutcome]:
    """Optuna TPE baseline. Imported lazily so the rest of the package never
    depends on optuna being installed."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep run logs clean

    outcomes: list[TrialOutcome] = []
    next_iteration = [0]  # boxed so the objective closure can mutate it

    def objective(trial: "optuna.Trial") -> float:
        proposed = {}
        for p in space.params:
            if p.kind == "int":
                proposed[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log_scale)
            else:
                proposed[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log_scale)
        if not all(c(proposed) for c in space.constraints):
            proposed = repair_config(proposed, space)

        next_iteration[0] += 1
        outcome = evaluate_trial(
            config=proposed,
            space=space,
            model_family=model_family,
            task=task,
            X=X,
            y=y,
            cfg=cfg,
            group=group,
            iteration=next_iteration[0],
            source="baseline_sampler",
        )
        outcomes.append(outcome)
        # Optuna's TPE still needs a scalar signal for failing/timed-out trials
        # to steer away from that region, without the study itself crashing.
        return outcome.primary_metric if outcome.status == "ok" else float("-inf")

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.trial_budget, catch=())

    # The hand-built `outcomes` list, not optuna's own trial objects, is the
    # source of truth returned here — optuna only drove where to search next.
    return outcomes


def _run_random_search(
    space: SearchSpace, model_family: str, task: str, X, y, cfg: TuningConfig, group
) -> list[TrialOutcome]:
    """Uniform random-search baseline: the same unit-cube decode `design.py`
    uses for the warm-start design, one fresh random point per trial."""
    outcomes: list[TrialOutcome] = []
    for iteration in range(1, cfg.trial_budget + 1):
        rng = np.random.default_rng(cfg.random_state + iteration)
        unit_row = rng.uniform(size=len(space.params))
        proposed = decode_unit_point(unit_row, space)
        if not all(c(proposed) for c in space.constraints):
            proposed = repair_config(proposed, space)

        outcome = evaluate_trial(
            config=proposed,
            space=space,
            model_family=model_family,
            task=task,
            X=X,
            y=y,
            cfg=cfg,
            group=group,
            iteration=iteration,
            source="baseline_sampler",
        )
        outcomes.append(outcome)
    return outcomes
