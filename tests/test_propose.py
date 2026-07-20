"""Adversarial propose-loop suite (§6.4, §10): out-of-range value, missing key,
wrong type, a constraint violation, and an exact duplicate -- each must be
retried/rejected/perturbed as designed, and `propose_next` must NEVER silently
skip a trial slot: it always returns a usable config, never raises for these
cases, and never returns None. The LLM is fully mocked; nothing here touches
the network.
"""

from __future__ import annotations

import pytest

from hp_agent.config import TuningConfig
from hp_agent.evaluate import TrialOutcome
from hp_agent.propose import HyperparamProposal, propose_next
from hp_agent.search_space import config_signature, get_search_space


class _QueueLLM:
    """Fake LLMClient: returns a scripted sequence of `HyperparamProposal`s from
    `.structured()`, one per call -- lets a test script exactly what "the LLM
    said" on each propose attempt without ever touching the network."""

    def __init__(self, responses: list[HyperparamProposal]):
        self._responses = list(responses)
        self.calls = 0

    def structured(self, *, stage, system, user, schema, temperature, model=None, max_retries=2):
        assert schema is HyperparamProposal
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _lgbm_config(**overrides) -> dict:
    cfg = {
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "n_estimators": 200,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 0.1,
    }
    cfg.update(overrides)
    return cfg


def _seed_trial(iteration: int, config: dict, metric: float) -> TrialOutcome:
    return TrialOutcome(
        iteration=iteration, source="seed_design", config=config, status="ok",
        primary_metric=metric, metric_std=0.01, train_metric=metric + 0.02, overfit_gap=0.02,
        fit_time_s=1.0,
    )


def _assert_usable(result) -> None:
    """The standing invariant across every adversarial fixture: propose_next
    always returns a usable config, never None."""
    config, source, rationale = result
    assert config is not None and isinstance(config, dict)
    assert source in ("llm_proposal", "repair_perturbation")
    assert isinstance(rationale, str) and rationale


# --------------------------------------------------------------------------- #
# out-of-range value: retried, then succeeds on the corrected attempt
# --------------------------------------------------------------------------- #
def test_out_of_range_value_is_rejected_then_retry_succeeds():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    bad_values = {**valid_cfg, "num_leaves": 10_000}          # far above the 255 upper bound
    good_values = {**valid_cfg, "learning_rate": 0.09}         # distinct, still valid

    llm = _QueueLLM([
        HyperparamProposal(values=bad_values, reasoning="too many leaves"),
        HyperparamProposal(values=good_values, reasoning="scaled back after the rejection"),
    ])
    cfg = TuningConfig(model=None)
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=[], cfg=cfg, current_best=None,
        seen_signatures=set(), rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "llm_proposal"
    assert config["learning_rate"] == pytest.approx(0.09)
    assert llm.calls == 2
    assert any(r["reason"] == "invalid_schema" for r in rejections)


def test_out_of_range_value_falls_back_to_repair_perturbation_after_two_failures():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    bad_values = {**valid_cfg, "learning_rate": 999.0}  # far above the 0.3 upper bound

    llm = _QueueLLM([
        HyperparamProposal(values=bad_values, reasoning="oops1"),
        HyperparamProposal(values=bad_values, reasoning="oops2"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, valid_cfg, metric=0.8)]
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures={config_signature(valid_cfg, space)}, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "repair_perturbation"
    assert set(config) == {p.name for p in space.params}
    assert all(c(config) for c in space.constraints)
    for p in space.params:
        assert p.low <= config[p.name] <= p.high
    assert llm.calls == 2
    assert all(r["reason"] == "invalid_schema" for r in rejections)


# --------------------------------------------------------------------------- #
# missing key: a required dimension absent from `values`
# --------------------------------------------------------------------------- #
def test_missing_key_falls_back_to_repair_perturbation_after_two_failures():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    missing = dict(valid_cfg)
    del missing["max_depth"]

    llm = _QueueLLM([
        HyperparamProposal(values=missing, reasoning="oops1"),
        HyperparamProposal(values=missing, reasoning="oops2"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, valid_cfg, metric=0.8)]
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures={config_signature(valid_cfg, space)}, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "repair_perturbation"
    assert set(config) == {p.name for p in space.params}
    assert all(c(config) for c in space.constraints)
    assert llm.calls == 2
    assert all(r["reason"] == "invalid_schema" for r in rejections)


# --------------------------------------------------------------------------- #
# wrong type: a value that fails the per-family schema (non-integral float for
# a strictly-int dimension) though it is a legal `float | int` at the outer
# HyperparamProposal schema layer
# --------------------------------------------------------------------------- #
def test_wrong_type_value_falls_back_to_repair_perturbation_after_two_failures():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    wrong_type = {**valid_cfg, "num_leaves": 31.7}  # fractional value for a strictly-int dimension

    llm = _QueueLLM([
        HyperparamProposal(values=wrong_type, reasoning="oops1"),
        HyperparamProposal(values=wrong_type, reasoning="oops2"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, valid_cfg, metric=0.8)]
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures={config_signature(valid_cfg, space)}, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "repair_perturbation"
    assert all(c(config) for c in space.constraints)
    assert llm.calls == 2
    assert all(r["reason"] == "invalid_schema" for r in rejections)


# --------------------------------------------------------------------------- #
# constraint violation: individually in-bounds, but num_leaves >= 2**max_depth
# --------------------------------------------------------------------------- #
def test_constraint_violation_is_rejected_then_retry_succeeds():
    space = get_search_space("lightgbm")
    violating = _lgbm_config(num_leaves=200, max_depth=3)  # 200 >= 2**3 == 8
    assert not all(c(violating) for c in space.constraints), "fixture must actually violate the constraint"
    fixed = _lgbm_config(num_leaves=6, max_depth=3)         # 6 < 8, respects the constraint

    llm = _QueueLLM([
        HyperparamProposal(values=violating, reasoning="too many leaves for the depth"),
        HyperparamProposal(values=fixed, reasoning="reduced num_leaves to respect the constraint"),
    ])
    cfg = TuningConfig(model=None)
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=[], cfg=cfg, current_best=None,
        seen_signatures=set(), rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "llm_proposal"
    assert config["num_leaves"] == 6
    assert llm.calls == 2
    assert any(r["reason"] == "constraint_violation" for r in rejections)


def test_constraint_violation_falls_back_to_repair_perturbation_after_two_failures():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    violating = _lgbm_config(num_leaves=200, max_depth=3)

    llm = _QueueLLM([
        HyperparamProposal(values=violating, reasoning="oops1"),
        HyperparamProposal(values=violating, reasoning="oops2"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, valid_cfg, metric=0.8)]
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures={config_signature(valid_cfg, space)}, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "repair_perturbation"
    assert all(c(config) for c in space.constraints)
    assert llm.calls == 2
    assert all(r["reason"] == "constraint_violation" for r in rejections)


# --------------------------------------------------------------------------- #
# exact duplicate of a prior trial
# --------------------------------------------------------------------------- #
def test_exact_duplicate_is_rejected_then_retry_succeeds():
    space = get_search_space("lightgbm")
    prior_cfg = _lgbm_config()
    duplicate_values = dict(prior_cfg)             # exact duplicate of trial 1's config
    new_values = _lgbm_config(learning_rate=0.2)   # distinct

    llm = _QueueLLM([
        HyperparamProposal(values=duplicate_values, reasoning="reproposing the same point"),
        HyperparamProposal(values=new_values, reasoning="moved to unexplored territory"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, prior_cfg, metric=0.881)]
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures={config_signature(prior_cfg, space)}, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "llm_proposal"
    assert config["learning_rate"] == pytest.approx(0.2)
    assert llm.calls == 2
    dup_rejections = [r for r in rejections if r["reason"] == "duplicate"]
    assert len(dup_rejections) == 1
    assert "trial 1" in dup_rejections[0]["detail"]  # feedback names the trial it duplicated (§6.4)


def test_exact_duplicate_falls_back_to_repair_perturbation_after_two_failures():
    space = get_search_space("lightgbm")
    prior_cfg = _lgbm_config()
    duplicate_values = dict(prior_cfg)

    llm = _QueueLLM([
        HyperparamProposal(values=duplicate_values, reasoning="oops1"),
        HyperparamProposal(values=duplicate_values, reasoning="oops2"),
    ])
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, prior_cfg, metric=0.881)]
    seen = {config_signature(prior_cfg, space)}
    rejections: list[dict] = []

    result = propose_next(
        llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
        seen_signatures=seen, rejections=rejections,
    )
    _assert_usable(result)
    config, source, _ = result

    assert source == "repair_perturbation"
    assert all(c(config) for c in space.constraints)
    assert llm.calls == 2
    assert all(r["reason"] == "duplicate" for r in rejections)


# --------------------------------------------------------------------------- #
# never silently skipped: sweep every adversarial fixture and confirm each one
# comes back with a real, in-bounds, constraint-satisfying config
# --------------------------------------------------------------------------- #
def test_propose_next_never_raises_or_returns_none_across_adversarial_fixtures():
    space = get_search_space("lightgbm")
    valid_cfg = _lgbm_config()
    fixtures = {
        "out_of_range": {**valid_cfg, "num_leaves": 10_000},
        "missing_key": {k: v for k, v in valid_cfg.items() if k != "max_depth"},
        "wrong_type": {**valid_cfg, "num_leaves": 31.7},
        "constraint_violation": _lgbm_config(num_leaves=200, max_depth=3),
        "duplicate": dict(valid_cfg),
    }
    cfg = TuningConfig(model=None)
    history = [_seed_trial(1, valid_cfg, metric=0.8)]
    seen = {config_signature(valid_cfg, space)}

    for name, bad_values in fixtures.items():
        llm = _QueueLLM([
            HyperparamProposal(values=bad_values, reasoning=f"{name} attempt 1"),
            HyperparamProposal(values=bad_values, reasoning=f"{name} attempt 2"),
        ])
        result = propose_next(
            llm, space, "lightgbm", history=history, cfg=cfg, current_best=history[0],
            seen_signatures=seen, rejections=[],
        )
        config, source, rationale = result
        assert config is not None, f"{name}: propose_next silently returned no config"
        assert source in ("llm_proposal", "repair_perturbation"), f"{name}: unexpected source {source}"
        assert all(c(config) for c in space.constraints), f"{name}: fallback config violates a constraint"
        for p in space.params:
            assert p.low <= config[p.name] <= p.high, f"{name}: {p.name} out of bounds in fallback"
