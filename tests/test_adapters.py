"""Model-family adapters (§6.2): config -> unfitted sklearn-compatible estimator,
task/family validation, and a portable LightGBM/HistGB backend check."""

from __future__ import annotations

import numpy as np
import pytest

from hp_agent.adapters import get_adapter, validate_family_for_task
from hp_agent.search_space import get_search_space, repair_config


def _midpoint_config(space):
    cfg = {}
    for p in space.params:
        mid = (p.low + p.high) / 2
        cfg[p.name] = int(round(mid)) if p.kind == "int" else float(mid)
    return cfg


def test_random_forest_estimator_has_right_params_and_fits():
    space = get_search_space("random_forest")
    cfg = _midpoint_config(space)
    est = get_adapter("random_forest").build_estimator(cfg, "classification", random_state=42)
    assert est.n_estimators == cfg["n_estimators"]
    assert est.max_depth == cfg["max_depth"]
    assert est.min_samples_split == cfg["min_samples_split"]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] > 0).astype(int)
    est.fit(X, y)
    preds = est.predict(X)
    assert len(preds) == 50


def test_lightgbm_adapter_backend_is_portable():
    adapter = get_adapter("lightgbm")
    assert adapter.backend in ("lightgbm", "histgb")


def test_lightgbm_estimator_has_right_params_and_fits():
    space = get_search_space("lightgbm")
    cfg = repair_config(_midpoint_config(space), space)
    adapter = get_adapter("lightgbm")
    est = adapter.build_estimator(cfg, "classification", random_state=42)

    assert est.learning_rate == pytest.approx(cfg["learning_rate"])
    if adapter.backend == "lightgbm":
        assert est.num_leaves == cfg["num_leaves"]
    else:
        assert est.max_leaf_nodes == cfg["num_leaves"]

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    est.fit(X, y)
    preds = est.predict(X)
    assert len(preds) == 60


def test_elasticnet_estimator_has_right_params_and_fits():
    space = get_search_space("elasticnet")
    cfg = _midpoint_config(space)
    est = get_adapter("elasticnet").build_estimator(cfg, "regression", random_state=42)
    assert est.alpha == pytest.approx(cfg["alpha"])
    assert est.l1_ratio == pytest.approx(cfg["l1_ratio"])

    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=50)
    est.fit(X, y)
    preds = est.predict(X)
    assert len(preds) == 50


def test_logistic_regression_estimator_has_right_params_and_fits():
    space = get_search_space("logistic_regression")
    cfg = _midpoint_config(space)
    est = get_adapter("logistic_regression").build_estimator(cfg, "classification", random_state=42)
    assert est.C == pytest.approx(cfg["C"])
    assert est.l1_ratio == pytest.approx(cfg["l1_ratio"])

    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    est.fit(X, y)
    preds = est.predict(X)
    assert len(preds) == 60


@pytest.mark.parametrize("family,task", [
    ("elasticnet", "classification"),
    ("logistic_regression", "regression"),
])
def test_family_task_mismatch_raises(family, task):
    with pytest.raises(ValueError):
        validate_family_for_task(family, task)


@pytest.mark.parametrize("family,task", [
    ("random_forest", "classification"),
    ("random_forest", "regression"),
    ("lightgbm", "classification"),
    ("lightgbm", "regression"),
    ("elasticnet", "regression"),
    ("logistic_regression", "classification"),
])
def test_family_task_match_does_not_raise(family, task):
    validate_family_for_task(family, task)  # must not raise
