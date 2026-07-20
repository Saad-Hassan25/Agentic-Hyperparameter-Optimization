"""CV harness — model-family-aware evaluation, never the LLM computing numbers (§6.5).

Fold strategy is chosen from the task and whether a `group` was supplied:
`StratifiedKFold` for classification, plain `KFold` for regression, and
`GroupKFold` whenever a group is given for either task — a repeated entity must
never straddle folds, the same leakage lesson `feature_agent` applies to
feature evaluation, applied here to model evaluation. `make_fold_splitter` is
exported (not private) so `baseline.py` can build the identical splitter and
guarantee the agent track and the baseline track are scored under the exact
same CV protocol (§6.8).

Metric orientation convention (READ THIS BEFORE TOUCHING A METRIC COMPUTATION):
`primary_metric` and `train_metric` are ALWAYS reported such that higher is
better, even for regression. `average_precision`/`roc_auc` are natively
higher-is-better; for regression we report NEGATIVE rmse/mae, mirroring
sklearn's own `neg_root_mean_squared_error`/`neg_mean_absolute_error` scorer
convention. This is why `resolve_metric` still returns the user-facing name
("rmse"/"mae", per §8's config enum) for logging, while the value actually
stored on `TrialOutcome.primary_metric` is its negation. Every downstream
comparison of two `TrialOutcome.primary_metric` values — `select.py`'s
selection rule, `report.py`'s Spearman influence, the convergence spread — can
then just do `a > b` and mean "a improved on b", across every family and every
metric. Do not flip the sign without updating every caller.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Literal

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .adapters import get_adapter
from .config import TuningConfig
from .search_space import SearchSpace


class TrialOutcome(BaseModel):
    iteration: int
    source: Literal["seed_design", "llm_proposal", "repair_perturbation", "baseline_sampler"]
    config: dict
    status: Literal[
        "ok", "failed", "duplicate_rejected", "timeout",
        # The two rejection kinds §7's own worked example requires in the
        # trials.jsonl audit trail alongside duplicates (e.g. "num_leaves=310
        # rejected: exceeds 2**max_depth" is a constraint_violation, not a
        # duplicate) -- doc §4's literal only listed "duplicate_rejected";
        # extended here rather than dropping the other two rejection kinds on
        # the floor, per §7's explicit requirement that they be persisted too.
        "invalid_schema_rejected", "constraint_violation_rejected",
    ]
    primary_metric: float | None = None
    metric_std: float | None = None
    train_metric: float | None = None
    overfit_gap: float | None = None       # train_metric - primary_metric; both already oriented
                                            # higher-is-better, so normally non-negative
    fit_time_s: float | None = None
    error: str | None = None


def resolve_metric(task: str, metric_cfg: str) -> str:
    """User-facing metric name (§6.5), NOT yet oriented — see module docstring.

    `metric_cfg == "auto"` resolves to the family default in both branches;
    an out-of-family value (e.g. "rmse" requested for classification) also
    falls back to the family default rather than raising, since §8 documents
    "auto" as the only cross-task-safe value.
    """
    if task == "regression":
        return metric_cfg if metric_cfg in ("rmse", "mae") else "rmse"
    return metric_cfg if metric_cfg in ("average_precision", "roc_auc") else "average_precision"


def _oriented_score(metric: str, y_true, pred_or_score) -> float:
    """Score `metric`, oriented so a larger value is always a better trial."""
    if metric == "average_precision":
        return float(average_precision_score(y_true, pred_or_score))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, pred_or_score))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y_true, pred_or_score)))
    if metric == "mae":
        return -float(mean_absolute_error(y_true, pred_or_score))
    raise ValueError(f"unknown metric {metric!r}")


def _score_input(estimator, metric: str, X):
    """The prediction shape each metric needs: proba/decision score for the
    ranking metrics, point predictions for the regression metrics."""
    if metric in ("average_precision", "roc_auc"):
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
        return estimator.decision_function(X)
    return estimator.predict(X)


def _select_rows(data, idx):
    """Row-index `data` (DataFrame/Series or ndarray) without assuming either."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def make_fold_splitter(task: str, cfg: TuningConfig, group):
    """Build the fold splitter (§6.5). Exported so `baseline.py` can reuse the
    identical CV protocol the agent track uses (§6.8's apples-to-apples requirement).

    `n_splits` is clipped to at most the number of distinct groups when
    grouped, and to at least 2 always.
    """
    if group is not None:
        n_splits = max(2, min(cfg.n_folds, len(np.unique(np.asarray(group)))))
        return GroupKFold(n_splits=n_splits)
    n_splits = max(2, cfg.n_folds)
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)


def _run_folds(
    config: dict, model_family: str, task: str, X, y, cfg: TuningConfig, group, metric: str
) -> tuple[list[float], list[float]]:
    """The actual fit+score work, run inside the timeout-guarded thread (below)."""
    splitter = make_fold_splitter(task, cfg, group)
    val_scores: list[float] = []
    train_scores: list[float] = []
    for tr_idx, va_idx in splitter.split(X, y, group):
        # Fresh estimator every fold — never refit/mutate one instance across folds.
        estimator = get_adapter(model_family).build_estimator(config, task, cfg.random_state)
        X_tr, X_va = _select_rows(X, tr_idx), _select_rows(X, va_idx)
        y_tr, y_va = _select_rows(y, tr_idx), _select_rows(y, va_idx)
        estimator.fit(X_tr, y_tr)
        val_scores.append(_oriented_score(metric, y_va, _score_input(estimator, metric, X_va)))
        train_scores.append(_oriented_score(metric, y_tr, _score_input(estimator, metric, X_tr)))
    return val_scores, train_scores


def evaluate_trial(
    config: dict,
    space: SearchSpace,  # noqa: ARG001 — accepted for call-signature symmetry with the
                          # propose/design stages, which are space-driven; the CV protocol
                          # itself needs only the already-validated `config` dict.
    model_family: str,
    task: str,
    X,
    y,
    cfg: TuningConfig,
    group,
    iteration: int,
    source: str,
) -> TrialOutcome:
    """Train+score `config` under CV, guarded by a soft wall-clock timeout.

    Runs the whole per-fold fit+score loop inside a `ThreadPoolExecutor` with
    `max_workers=1`, bounded by `cfg.trial_timeout_s`. Thread-based, not
    process-based, on purpose: sklearn/numpy fits release the GIL during the
    BLAS-heavy inner loop, so a thread-based timeout genuinely keeps the
    orchestrator's main loop from stalling past `trial_timeout_s` — the actual
    requirement (§6.5: "not allowed to stall the whole run") — even though it
    cannot forcibly kill an already-running fit the way a subprocess could.
    This trades perfect resource reclamation for portability (no
    multiprocessing/pickling complexity on Windows) and simplicity, and is a
    deliberate tradeoff, not a placeholder.

    Never raises: every path — success, timeout, or any other exception during
    fit/score — returns a `TrialOutcome`.
    """
    metric = resolve_metric(task, cfg.metric)
    executor = ThreadPoolExecutor(max_workers=1)
    t0 = time.perf_counter()
    future = executor.submit(_run_folds, config, model_family, task, X, y, cfg, group, metric)
    try:
        val_scores, train_scores = future.result(timeout=cfg.trial_timeout_s)
    except FutureTimeoutError:
        # Do NOT wait=True here: shutdown(wait=True) would block on the very fit
        # we're trying to stop stalling on. wait=False returns immediately; the
        # thread is abandoned to finish (or not) in the background.
        executor.shutdown(wait=False)
        return TrialOutcome(
            iteration=iteration,
            source=source,
            config=config,
            status="timeout",
            error=f"trial exceeded trial_timeout_s={cfg.trial_timeout_s}s",
        )
    except Exception as exc:  # noqa: BLE001 — any fit/score failure is a per-trial "failed", not a crash
        executor.shutdown(wait=False)
        return TrialOutcome(iteration=iteration, source=source, config=config, status="failed", error=str(exc))
    executor.shutdown(wait=False)

    fit_time_s = time.perf_counter() - t0
    primary_metric = float(np.mean(val_scores))
    train_metric = float(np.mean(train_scores))
    return TrialOutcome(
        iteration=iteration,
        source=source,
        config=config,
        status="ok",
        primary_metric=primary_metric,
        metric_std=float(np.std(val_scores, ddof=0)),
        train_metric=train_metric,
        overfit_gap=train_metric - primary_metric,
        fit_time_s=fit_time_s,
    )
