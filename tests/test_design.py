"""M1 acceptance criterion (§9, §10): the Latin-Hypercube warm-start design
produces `k` distinct, in-bounds, constraint-satisfying configs over every
registered search space, across at least 10 seeds per family."""

from __future__ import annotations

import pytest

from hp_agent.design import generate_seed_designs
from hp_agent.search_space import config_signature, get_search_space, list_families

SEEDS = list(range(10))
K = 8


@pytest.mark.parametrize("family", list_families())
@pytest.mark.parametrize("seed", SEEDS)
def test_lhs_design_distinct_in_bounds_and_constraint_satisfying(family, seed):
    space = get_search_space(family)
    configs = generate_seed_designs(space, k=K, random_state=seed)

    assert len(configs) == K, f"{family} seed={seed}: expected {K} configs, got {len(configs)}"

    signatures = {config_signature(cfg, space) for cfg in configs}
    assert len(signatures) == K, f"{family} seed={seed}: configs are not all distinct"

    for cfg in configs:
        for p in space.params:
            assert p.low <= cfg[p.name] <= p.high, (
                f"{family} seed={seed}: {p.name}={cfg[p.name]} out of bounds [{p.low}, {p.high}]"
            )
        for constraint in space.constraints:
            assert constraint(cfg), f"{family} seed={seed}: constraint violated by {cfg}"


@pytest.mark.parametrize("family", list_families())
def test_lhs_design_configs_have_every_dimension(family):
    space = get_search_space(family)
    configs = generate_seed_designs(space, k=K, random_state=0)
    expected_names = {p.name for p in space.params}
    for cfg in configs:
        assert set(cfg) == expected_names
