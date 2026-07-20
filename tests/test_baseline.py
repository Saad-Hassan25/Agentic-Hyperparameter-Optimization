"""Baseline sampler resolution + run_baseline_track under a tiny trial_budget (§6.8, §10)."""

from __future__ import annotations

import pytest

from hp_agent.baseline import resolve_baseline_sampler_name, run_baseline_track
from hp_agent.config import TuningConfig
from hp_agent.search_space import get_search_space

# This suite assumes optuna is installed in this environment (it is, per the
# repo's requirements.txt); "auto" resolving to "optuna_tpe" specifically
# depends on that. If optuna is ever removed here, this whole module should be
# skipped rather than fail on an environment mismatch it isn't testing for.
pytest.importorskip("optuna")


@pytest.mark.parametrize("requested, expected", [
    ("auto", "optuna_tpe"),
    ("optuna_tpe", "optuna_tpe"),
    ("random_search", "random_search"),
])
def test_resolve_baseline_sampler_name(requested, expected):
    cfg = TuningConfig(model=None, baseline_sampler=requested)
    assert resolve_baseline_sampler_name(cfg) == expected


def test_unrecognized_sampler_name_degrades_to_random_search():
    """A typo'd config value degrades gracefully instead of crashing the run (§12)."""
    cfg = TuningConfig(model=None, baseline_sampler="not_a_real_sampler")
    assert resolve_baseline_sampler_name(cfg) == "random_search"


@pytest.mark.parametrize("sampler_name", ["optuna_tpe", "random_search"])
def test_run_baseline_track_returns_trial_budget_outcomes(sampler_name, make_classification_df):
    X, y = make_classification_df(n=250, seed=0)
    cfg = TuningConfig(
        model=None, model_family="logistic_regression", baseline_sampler=sampler_name,
        trial_budget=6, n_folds=3, trial_timeout_s=60,
    )
    space = get_search_space("logistic_regression")

    outcomes = run_baseline_track(space, "logistic_regression", "classification", X, y, cfg)

    assert len(outcomes) == cfg.trial_budget
    assert all(o.source == "baseline_sampler" for o in outcomes)
    assert all(o.status in ("ok", "failed", "timeout") for o in outcomes)
    assert sum(o.status == "ok" for o in outcomes) >= 1
    assert [o.iteration for o in outcomes] == list(range(1, cfg.trial_budget + 1))
