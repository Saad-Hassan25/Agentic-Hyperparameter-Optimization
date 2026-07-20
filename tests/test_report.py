"""Reporting stats and the deterministic report fallback (§6.9, §14, §10)."""

from __future__ import annotations

import pytest

from hp_agent.config import TuningConfig
from hp_agent.evaluate import TrialOutcome
from hp_agent.report import (
    ReportContext,
    TuningReport,
    compute_hyperparameter_influence,
    compute_trials_to_best,
    llm_report,
)
from hp_agent.search_space import get_search_space
from hp_agent.select import ConvergenceDecision


def _ok_trial(iteration, config, metric):
    return TrialOutcome(
        iteration=iteration, source="seed_design", config=config, status="ok",
        primary_metric=metric, metric_std=0.01, train_metric=metric + 0.01,
        overfit_gap=0.01, fit_time_s=1.0,
    )


# --------------------------------------------------------------------------- #
# compute_hyperparameter_influence: distinguishes correlated from uncorrelated
# --------------------------------------------------------------------------- #
def test_influence_reflects_correlated_param_vs_zero_variance_param():
    space = get_search_space("elasticnet")
    trials = [
        _ok_trial(i + 1, {"alpha": 1.0, "l1_ratio": 0.1 + 0.1 * i}, metric=0.50 + 0.05 * i)
        for i in range(8)
    ]
    influence = compute_hyperparameter_influence(trials, space)
    assert influence["l1_ratio"] == pytest.approx(1.0, abs=1e-6)  # rigged to correlate near-perfectly
    assert influence["alpha"] == 0.0                              # rigged constant -> zero variance -> 0.0


def test_influence_distinguishes_strong_correlation_from_unrelated_noise():
    space = get_search_space("elasticnet")
    n = 12
    metrics = [0.9 - 0.03 * i for i in range(n)]                 # monotonically decreasing
    alphas = [0.01 * (i + 1) for i in range(n)]                  # rigged: perfectly anti-correlated
    l1_pattern = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05, 0.95, 0.45]  # rigged: unrelated
    trials = [
        _ok_trial(i + 1, {"alpha": alphas[i], "l1_ratio": l1_pattern[i]}, metric=metrics[i])
        for i in range(n)
    ]
    influence = compute_hyperparameter_influence(trials, space)
    assert influence["alpha"] == pytest.approx(-1.0, abs=1e-6)
    assert abs(influence["l1_ratio"]) < 0.3


def test_influence_is_zero_with_fewer_than_three_ok_trials():
    space = get_search_space("elasticnet")
    trials = [_ok_trial(1, {"alpha": 0.5, "l1_ratio": 0.5}, metric=0.7)]
    influence = compute_hyperparameter_influence(trials, space)
    assert influence == {p.name: 0.0 for p in space.params}


# --------------------------------------------------------------------------- #
# compute_trials_to_best: earliest iteration whose own score already reached
# the eventually-selected best (§14's "doesn't reach that level until trial 24")
# --------------------------------------------------------------------------- #
def test_compute_trials_to_best_matches_doc_semantics():
    metrics = [0.70, 0.75, 0.78, 0.782, 0.795, 0.783, 0.790]
    trials = [_ok_trial(i + 1, {"x": i}, m) for i, m in enumerate(metrics)]
    best_metric = 0.783  # the overfit-aware selection's chosen metric, not necessarily the raw max
    assert compute_trials_to_best(trials, best_metric) == 5  # trial 5 (0.795) first reached 0.783


def test_compute_trials_to_best_empty_history_is_zero():
    assert compute_trials_to_best([], 0.5) == 0


# --------------------------------------------------------------------------- #
# llm_report(None, ...): the deterministic --no-llm fallback
# --------------------------------------------------------------------------- #
def test_llm_report_none_returns_valid_deterministic_report():
    space = get_search_space("elasticnet")
    trials = [
        _ok_trial(i + 1, {"alpha": 1.0, "l1_ratio": 0.1 + 0.1 * i}, metric=0.50 + 0.05 * i)
        for i in range(8)
    ]
    influence = compute_hyperparameter_influence(trials, space)
    ctx = ReportContext(
        task="regression", metric_name="rmse", model_family="elasticnet",
        best_config={"alpha": 0.1, "l1_ratio": 0.5}, best_metric=-1.23,
        selection_rule="within noise floor of top score, minimum overfit_gap, fastest fit_time_s",
        agent_trials_to_best=3, baseline_sampler_name="random_search",
        baseline_sampler_metric=-1.50, baseline_trials_to_best=4,
        lift_over_baseline=0.27, hyperparameter_influence=influence,
        convergence=ConvergenceDecision(converged=True, reason="patience_exhausted", trials_used=10),
        n_agent_trials=10, n_baseline_trials=6,
    )
    cfg = TuningConfig(model=None)

    report, source = llm_report(None, ctx, cfg)

    assert source == "deterministic"
    assert isinstance(report, TuningReport)
    # the report validates against its own schema (round-trip through model_dump)
    TuningReport.model_validate(report.model_dump())
    assert report.best_metric == pytest.approx(-1.23)
    assert report.baseline_sampler_name == "random_search"
    assert report.lift_over_baseline == pytest.approx(0.27)
    assert report.trials_to_best == 3
    assert report.hyperparameter_influence == influence
    assert isinstance(report.narrative, str) and report.narrative.strip()


# --------------------------------------------------------------------------- #
# llm_report(<mock returning bogus numeric fields>, ...): the LLM must never
# be able to overwrite a code-computed number (§6.9, §7 M4 acceptance)
# --------------------------------------------------------------------------- #
class _BogusNumbersLLM:
    """A misbehaving LLM that ignores the 'narrative only' instruction and
    tries to smuggle back fabricated numeric fields. `.structured()` is only
    ever asked for `_NarrativeOnly`'s schema (one string field) -- this fake
    proves that even if a model tries to return extra keys, pydantic's
    `_NarrativeOnly.model_validate` only ever surfaces `narrative`, so there is
    no field for the bogus numbers to land in."""

    def structured(self, *, stage, system, user, schema, temperature, model=None, max_retries=2):
        payload = {
            "narrative": "This narrative is legitimate LLM prose.",
            # every one of these must NOT end up on the returned TuningReport
            "best_config": {"alpha": 999.0, "l1_ratio": 999.0},
            "best_metric": -9999.0,
            "hyperparameter_influence": {"alpha": 12.34, "l1_ratio": -56.78},
            "lift_over_baseline": 9999.0,
            "baseline_sampler_metric": -9999.0,
            "selection_rule": "the LLM made this up",
            "trials_to_best": 999,
        }
        return schema.model_validate(payload)


def test_llm_report_cannot_overwrite_code_computed_numeric_fields():
    space = get_search_space("elasticnet")
    trials = [
        _ok_trial(i + 1, {"alpha": 1.0, "l1_ratio": 0.1 + 0.1 * i}, metric=0.50 + 0.05 * i)
        for i in range(8)
    ]
    influence = compute_hyperparameter_influence(trials, space)
    ctx = ReportContext(
        task="regression", metric_name="rmse", model_family="elasticnet",
        best_config={"alpha": 0.1, "l1_ratio": 0.5}, best_metric=-1.23,
        selection_rule="within noise floor of top score, minimum overfit_gap, fastest fit_time_s",
        agent_trials_to_best=3, baseline_sampler_name="random_search",
        baseline_sampler_metric=-1.50, baseline_trials_to_best=4,
        lift_over_baseline=0.27, hyperparameter_influence=influence,
        convergence=ConvergenceDecision(converged=True, reason="patience_exhausted", trials_used=10),
        n_agent_trials=10, n_baseline_trials=6,
    )
    cfg = TuningConfig(model="fake-model")

    report, source = llm_report(_BogusNumbersLLM(), ctx, cfg)

    assert source == "llm"
    # the narrative is genuinely the (legitimate part of the) LLM's text
    assert report.narrative == "This narrative is legitimate LLM prose."
    # but every numeric/structural field is exactly what ctx said, never the
    # bogus values the fake LLM tried to smuggle back
    assert report.best_config == ctx.best_config
    assert report.best_metric == pytest.approx(ctx.best_metric)
    assert report.hyperparameter_influence == influence
    assert report.lift_over_baseline == pytest.approx(ctx.lift_over_baseline)
    assert report.baseline_sampler_metric == pytest.approx(ctx.baseline_sampler_metric)
    assert report.selection_rule == ctx.selection_rule
    assert report.trials_to_best == ctx.agent_trials_to_best
