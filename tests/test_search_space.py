"""SearchSpace round-trip, signature stability, and constraint repair (§4, §6.2, §6.4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hp_agent.search_space import (
    config_signature,
    get_search_space,
    list_families,
    repair_config,
)


def _midpoint_config(space):
    """A config at the midpoint of every dimension -- always in-bounds by construction."""
    cfg = {}
    for p in space.params:
        mid = (p.low + p.high) / 2
        cfg[p.name] = int(round(mid)) if p.kind == "int" else float(mid)
    return cfg


# --------------------------------------------------------------------------- #
# to_pydantic_model(): valid/invalid round-trip, all four families
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", list_families())
def test_in_bounds_config_validates_for_every_family(family):
    space = get_search_space(family)
    cfg = _midpoint_config(space)
    if not all(c(cfg) for c in space.constraints):
        cfg = repair_config(cfg, space)  # lightgbm's arithmetic midpoint can violate its constraint
    model = space.to_pydantic_model()
    validated = model.model_validate(cfg)
    for p in space.params:
        assert getattr(validated, p.name) == pytest.approx(cfg[p.name])


@pytest.mark.parametrize("family", list_families())
def test_value_below_low_bound_raises_validation_error(family):
    space = get_search_space(family)
    cfg = _midpoint_config(space)
    target = space.params[0]
    cfg[target.name] = target.low - (abs(target.low) + 1) * 2  # comfortably below the bound
    model = space.to_pydantic_model()
    with pytest.raises(ValidationError):
        model.model_validate(cfg)


@pytest.mark.parametrize("family", list_families())
def test_value_above_high_bound_raises_validation_error(family):
    space = get_search_space(family)
    cfg = _midpoint_config(space)
    target = space.params[0]
    cfg[target.name] = target.high + (abs(target.high) + 1) * 2  # comfortably above the bound
    model = space.to_pydantic_model()
    with pytest.raises(ValidationError):
        model.model_validate(cfg)


@pytest.mark.parametrize("family", list_families())
def test_missing_required_dimension_raises_validation_error(family):
    space = get_search_space(family)
    cfg = _midpoint_config(space)
    del cfg[space.params[0].name]
    model = space.to_pydantic_model()
    with pytest.raises(ValidationError):
        model.model_validate(cfg)


# --------------------------------------------------------------------------- #
# config_signature(): stable, order-independent, rounded
# --------------------------------------------------------------------------- #
def test_config_signature_is_independent_of_input_dict_key_order():
    space = get_search_space("lightgbm")
    cfg = {
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.1234567, "n_estimators": 100,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0000004,
    }
    reordered = dict(reversed(list(cfg.items())))
    assert config_signature(cfg, space) == config_signature(reordered, space)


def test_config_signature_rounds_floats_to_declared_decimals():
    space = get_search_space("elasticnet")
    cfg_a = {"alpha": 0.1000001, "l1_ratio": 0.5}
    cfg_b = {"alpha": 0.1000002, "l1_ratio": 0.5}  # same at 6 decimals -> equal signature
    cfg_c = {"alpha": 0.1001, "l1_ratio": 0.5}      # differs at 6 decimals -> distinct signature
    assert config_signature(cfg_a, space) == config_signature(cfg_b, space)
    assert config_signature(cfg_a, space) != config_signature(cfg_c, space)


def test_config_signature_keeps_int_params_exact():
    space = get_search_space("random_forest")
    cfg = {"n_estimators": 100, "max_depth": 8, "min_samples_split": 4, "max_features": 0.5}
    sig = dict(config_signature(cfg, space))
    assert sig["n_estimators"] == 100
    assert isinstance(sig["n_estimators"], int)


# --------------------------------------------------------------------------- #
# repair_config(): actually satisfies constraints within max_iters
# --------------------------------------------------------------------------- #
def test_repair_config_satisfies_lightgbm_num_leaves_constraint():
    space = get_search_space("lightgbm")
    cfg = {
        "num_leaves": 250,          # violates num_leaves < 2**max_depth at max_depth=4 (2**4=16)
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
    assert not all(c(cfg) for c in space.constraints), "fixture must actually violate the constraint"

    repaired = repair_config(cfg, space, max_iters=100)

    assert all(c(repaired) for c in space.constraints), "repair_config did not satisfy every constraint"
    for p in space.params:
        assert p.low <= repaired[p.name] <= p.high, f"{p.name} out of bounds after repair"


def test_repair_config_is_a_noop_when_already_valid():
    space = get_search_space("lightgbm")
    cfg = {
        "num_leaves": 15, "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    }
    assert all(c(cfg) for c in space.constraints)
    repaired = repair_config(cfg, space)
    assert repaired == cfg
