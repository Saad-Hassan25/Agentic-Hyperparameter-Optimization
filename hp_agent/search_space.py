"""Declarative search spaces — the single source of truth per model family.

A `SearchSpace` is the one place a family's tunable dimensions and their bounds
are declared. Everything downstream reads it instead of re-declaring ranges by
hand: `to_pydantic_model()` drives validation, `render_prompt_block()` drives
the LLM prompt, and `design.py`'s warm-start sampler draws from the same
`params` list. This is what makes the prototype's bug (ranges drifting between
a hand-written prompt block and a hand-written schema) structurally impossible.

This module also owns two family-agnostic utilities every downstream stage
needs — a hashable config fingerprint for duplicate detection (§6.4) and a
generic constraint-repair step (§6.4's repair-perturbation fallback groundwork)
— plus the registry of the four model families this agent tunes (§6.2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, create_model


@dataclass(frozen=True)
class ParamSpec:
    """One tunable dimension. The one place its bounds are declared."""
    name: str
    kind: Literal["int", "float"]
    low: float
    high: float
    log_scale: bool = False          # sample/report on log scale (e.g. learning_rate)
    description: str = ""


@dataclass(frozen=True)
class SearchSpace:
    """Declarative search space for one model family. Drives validation,
    the warm-start design, and the prompt — never duplicated by hand."""
    model_family: str
    params: list[ParamSpec]
    constraints: list[Callable[[dict], bool]] = field(default_factory=list)
    # e.g. lambda cfg: cfg["num_leaves"] < 2 ** cfg["max_depth"]

    def to_pydantic_model(self) -> type[BaseModel]:
        fields = {
            p.name: (
                (int if p.kind == "int" else float),
                Field(..., ge=p.low, le=p.high, description=p.description),
            )
            for p in self.params
        }
        return create_model(f"{self.model_family}_Config", **fields)

    def render_prompt_block(self) -> str:
        lines = [
            f"- {p.name} ({p.kind}, {'log-scale' if p.log_scale else 'linear'}): "
            f"{p.low}-{p.high} — {p.description}"
            for p in self.params
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Duplicate detection (§6.4): a hashable fingerprint of a config
# --------------------------------------------------------------------------- #
def config_signature(cfg: dict, space: SearchSpace, decimals: int = 6) -> tuple:
    """Round `cfg` to a fixed precision per `ParamSpec` and return a hashable key.

    Float params are rounded (on the raw value, not the log of it) to `decimals`
    places; int params are kept as exact ints. The result is a tuple of
    `(name, rounded_value)` pairs in `space.params` order, usable directly as a
    dict/set key for hashing one trial's config against all prior trials.
    """
    sig = []
    for p in space.params:
        value = cfg[p.name]
        if p.kind == "int":
            sig.append((p.name, int(round(value))))
        else:
            sig.append((p.name, round(float(value), decimals)))
    return tuple(sig)


# --------------------------------------------------------------------------- #
# Generic constraint repair (§6.4)
# --------------------------------------------------------------------------- #
def repair_config(cfg: dict, space: SearchSpace, max_iters: int = 100) -> dict:
    """Shrink an out-of-constraint config toward its lower bounds until it fits.

    Family-agnostic on purpose: it does not know which param relates to which
    constraint, so it works unchanged for every current and future
    `SearchSpace`. If `cfg` already satisfies every constraint, a shallow copy
    is returned untouched. Otherwise, each iteration shrinks every param 10%
    of the way from its current value toward its `ParamSpec.low` bound
    (`new_value = value - 0.1 * (value - p.low)`), clips back into
    `[p.low, p.high]`, rounds int params, and re-checks all constraints —
    returning as soon as they all pass.

    Contract: if `max_iters` is exhausted without satisfying every constraint,
    the best-effort (fully clipped/rounded) config is returned anyway — this
    function never raises on an unsatisfiable repair. Callers (`design.py`,
    `baseline.py`) are responsible for checking the constraints themselves and
    rejecting/resampling instead of accepting a config that still fails.
    """
    cfg = dict(cfg)
    if all(c(cfg) for c in space.constraints):
        return cfg

    working = dict(cfg)
    for _ in range(max_iters):
        for p in space.params:
            value = working[p.name]
            shrunk = value - 0.1 * (value - p.low)
            clipped = min(max(shrunk, p.low), p.high)
            working[p.name] = int(round(clipped)) if p.kind == "int" else clipped
        if all(c(working) for c in space.constraints):
            return working
    return working


# --------------------------------------------------------------------------- #
# Registry (§6.2): the four model families this agent tunes
# --------------------------------------------------------------------------- #
SEARCH_SPACES: dict[str, SearchSpace] = {
    "random_forest": SearchSpace(
        model_family="random_forest",
        params=[
            ParamSpec("n_estimators", "int", 50, 800, log_scale=True, description="Number of trees."),
            ParamSpec("max_depth", "int", 2, 32, description="Max tree depth."),
            ParamSpec("min_samples_split", "int", 2, 50, description="Min samples required to split a node."),
            ParamSpec("max_features", "float", 0.1, 1.0, description="Fraction of features considered per split."),
        ],
        constraints=[],
    ),
    "lightgbm": SearchSpace(
        model_family="lightgbm",
        params=[
            ParamSpec("num_leaves", "int", 4, 255, description="Max leaves per tree."),
            ParamSpec("max_depth", "int", 3, 16, description="Max tree depth."),
            ParamSpec("learning_rate", "float", 0.005, 0.3, log_scale=True, description="Boosting learning rate."),
            ParamSpec("n_estimators", "int", 50, 2000, log_scale=True, description="Number of boosting rounds."),
            ParamSpec("min_child_samples", "int", 5, 100, description="Min samples per leaf."),
            ParamSpec("subsample", "float", 0.5, 1.0, description="Row subsampling fraction per iteration."),
            ParamSpec("colsample_bytree", "float", 0.5, 1.0, description="Column subsampling fraction per tree."),
            ParamSpec("reg_lambda", "float", 0.001, 10.0, log_scale=True, description="L2 regularization."),
        ],
        constraints=[lambda cfg: cfg["num_leaves"] < 2 ** cfg["max_depth"]],
    ),
    "elasticnet": SearchSpace(
        model_family="elasticnet",
        params=[
            ParamSpec("alpha", "float", 0.0001, 10.0, log_scale=True, description="Regularization strength."),
            ParamSpec("l1_ratio", "float", 0.0, 1.0, description="Mix between L1 and L2 penalty."),
        ],
        constraints=[],
    ),
    "logistic_regression": SearchSpace(
        model_family="logistic_regression",
        params=[
            ParamSpec("C", "float", 0.001, 100.0, log_scale=True, description="Inverse regularization strength."),
            ParamSpec("l1_ratio", "float", 0.0, 1.0, description="Mix between L1 and L2 (elasticnet penalty)."),
        ],
        constraints=[],
    ),
}


def get_search_space(name: str) -> SearchSpace:
    """Look up a registered `SearchSpace` by model family name."""
    try:
        return SEARCH_SPACES[name]
    except KeyError:
        raise ValueError(
            f"Unknown model_family {name!r}. Valid options: {list_families()}"
        ) from None


def list_families() -> list[str]:
    """Registered model family names, in registration order."""
    return list(SEARCH_SPACES.keys())
