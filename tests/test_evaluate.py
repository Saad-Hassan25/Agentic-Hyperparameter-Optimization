"""CV harness (§6.5): sane per-trial metrics across all four families, honest
GroupKFold isolation, and genuine per-trial timeout enforcement."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from hp_agent import evaluate
from hp_agent.config import TuningConfig
from hp_agent.search_space import get_search_space

# Small, fast-to-fit configs per family -- deliberately modest (not midpoints,
# which can pick e.g. n_estimators=800 and slow the suite down for no reason).
_CLASSIFICATION_CONFIGS = {
    "random_forest": {"n_estimators": 60, "max_depth": 6, "min_samples_split": 4, "max_features": 0.7},
    "lightgbm": {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.1, "n_estimators": 60,
                 "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 0.1},
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.5},
}
_REGRESSION_CONFIGS = {
    "random_forest": {"n_estimators": 60, "max_depth": 6, "min_samples_split": 4, "max_features": 0.7},
    "lightgbm": {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.1, "n_estimators": 60,
                 "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 0.1},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5},
}


# --------------------------------------------------------------------------- #
# sane metrics, all four families, appropriately matched to task
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", list(_CLASSIFICATION_CONFIGS))
def test_evaluate_trial_ok_on_classification(family, make_classification_df):
    X, y = make_classification_df(n=300, seed=0)
    cfg = TuningConfig(model=None, model_family=family, n_folds=4, trial_timeout_s=60)
    space = get_search_space(family)
    outcome = evaluate.evaluate_trial(
        config=_CLASSIFICATION_CONFIGS[family], space=space, model_family=family,
        task="classification", X=X, y=y, cfg=cfg, group=None, iteration=1, source="seed_design",
    )
    assert outcome.status == "ok", outcome.error
    assert 0.0 <= outcome.primary_metric <= 1.0          # average_precision, oriented higher-is-better
    assert outcome.metric_std >= 0.0
    assert 0.0 <= outcome.train_metric <= 1.0
    assert math.isclose(outcome.overfit_gap, outcome.train_metric - outcome.primary_metric, abs_tol=1e-9)
    assert outcome.fit_time_s is not None and outcome.fit_time_s >= 0.0


@pytest.mark.parametrize("family", list(_REGRESSION_CONFIGS))
def test_evaluate_trial_ok_on_regression(family, make_regression_df):
    X, y = make_regression_df(n=300, seed=0)
    cfg = TuningConfig(model=None, model_family=family, n_folds=4, trial_timeout_s=60)
    space = get_search_space(family)
    outcome = evaluate.evaluate_trial(
        config=_REGRESSION_CONFIGS[family], space=space, model_family=family,
        task="regression", X=X, y=y, cfg=cfg, group=None, iteration=1, source="seed_design",
    )
    assert outcome.status == "ok", outcome.error
    # regression metrics are stored as NEGATIVE rmse/mae (module docstring's
    # higher-is-better convention), so both are always <= 0.
    assert outcome.primary_metric <= 0.0
    assert outcome.metric_std >= 0.0
    assert outcome.train_metric <= 0.0
    assert math.isclose(outcome.overfit_gap, outcome.train_metric - outcome.primary_metric, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# GroupKFold: actually used, and no group straddles a fold's train/val split
# --------------------------------------------------------------------------- #
def test_group_kfold_used_and_no_group_straddles_folds(make_grouped_classification_df):
    X, y, group = make_grouped_classification_df(n_groups=12, rows_per_group=5, seed=0)
    cfg = TuningConfig(model=None, n_folds=4)

    splitter = evaluate.make_fold_splitter("classification", cfg, group)
    assert isinstance(splitter, GroupKFold), "a group column must select GroupKFold, not be silently ignored"

    for tr_idx, va_idx in splitter.split(X, y, group):
        train_groups = set(np.asarray(group)[tr_idx])
        val_groups = set(np.asarray(group)[va_idx])
        assert train_groups.isdisjoint(val_groups), "a group's rows straddled train and validation"


def test_evaluate_trial_with_group_runs_ok_end_to_end(make_grouped_classification_df):
    X, y, group = make_grouped_classification_df(n_groups=10, rows_per_group=6, seed=1)
    cfg = TuningConfig(model=None, model_family="random_forest", n_folds=3, trial_timeout_s=60)
    space = get_search_space("random_forest")
    outcome = evaluate.evaluate_trial(
        config=_CLASSIFICATION_CONFIGS["random_forest"], space=space, model_family="random_forest",
        task="classification", X=X, y=y, cfg=cfg, group=group, iteration=1, source="seed_design",
    )
    assert outcome.status == "ok", outcome.error


# --------------------------------------------------------------------------- #
# per-trial timeout: genuinely fires, never stalls the caller
# --------------------------------------------------------------------------- #
class _SlowEstimator:
    """Stand-in estimator whose `.fit()` deliberately outlives `trial_timeout_s`."""

    def __init__(self, delay: float):
        self._delay = delay

    def fit(self, X, y):
        time.sleep(self._delay)
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        p = np.zeros((len(X), 2))
        p[:, 1] = 0.5
        p[:, 0] = 0.5
        return p


class _SlowAdapter:
    def __init__(self, delay: float):
        self._delay = delay

    def build_estimator(self, config, task, random_state):
        return _SlowEstimator(self._delay)


def test_trial_timeout_fires_and_records_status_timeout(monkeypatch, make_classification_df):
    """Direct check of evaluate_trial's ThreadPoolExecutor timeout guard (§6.5):
    a deliberately slow synthetic workload with a tiny trial_timeout_s must come
    back as status="timeout" promptly, not hang the run."""
    monkeypatch.setattr(evaluate, "get_adapter", lambda family: _SlowAdapter(delay=1.5))
    X, y = make_classification_df(n=60, seed=0)
    cfg = TuningConfig(model=None, model_family="random_forest", n_folds=2, trial_timeout_s=0.1)
    space = get_search_space("random_forest")

    t0 = time.perf_counter()
    outcome = evaluate.evaluate_trial(
        config={"n_estimators": 10, "max_depth": 3, "min_samples_split": 2, "max_features": 0.5},
        space=space, model_family="random_forest", task="classification",
        X=X, y=y, cfg=cfg, group=None, iteration=1, source="seed_design",
    )
    elapsed = time.perf_counter() - t0

    assert outcome.status == "timeout"
    assert outcome.error is not None and "trial_timeout_s" in outcome.error
    assert outcome.primary_metric is None
    assert elapsed < 1.0, "evaluate_trial waited for the slow fit instead of honoring the timeout"
