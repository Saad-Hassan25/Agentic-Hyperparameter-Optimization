"""Space-filling warm-start design (§6.3).

The prototype started every run from one hand-picked seed config, then handed
everything else to free-form LLM guessing — the reasoning loop had no spread
of evidence to reason *from*, and the search could anchor near that one point
and never sample a distant good region (§2 defect #3). Before the agent
proposes anything, this module draws a Latin-Hypercube sample over the active
`SearchSpace` (log-scale dimensions decoded in log space) so the first round
of trend-reasoning happens over real, diverse evidence instead of noise
around one point.

`decode_unit_point` is deliberately factored out of the LHS sampler: it is a
pure decode from a `[0, 1)` unit cube row to a config dict, with no opinion
about where the row came from. `baseline.py`'s random-search fallback reuses
it unchanged — a single random unit row is just
`np.random.default_rng(seed).uniform(size=d)`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import qmc

from .search_space import SearchSpace, config_signature, repair_config

_OVERSAMPLE_FACTOR = 6     # tight constraints (e.g. lightgbm's num_leaves < 2**max_depth)
                           # reject a meaningful fraction of raw draws; oversample for it
_MAX_BATCHES = 50          # hard cap on LHS batches drawn before giving up early


def decode_unit_point(unit_row: Sequence[float], space: SearchSpace) -> dict:
    """Decode one `[0, 1)` unit-cube row into a config dict, in `space.params` order.

    Log-scale dimensions are interpolated in log space then exponentiated;
    linear dimensions are interpolated directly. This is a pure decode with no
    knowledge of `space.constraints` — callers repair or reject separately.
    """
    point: dict = {}
    for value, p in zip(unit_row, space.params):
        if p.log_scale:
            lo, hi = math.log(p.low), math.log(p.high)
            x = math.exp(lo + value * (hi - lo))
        else:
            x = p.low + value * (p.high - p.low)
        if p.kind == "int":
            x = int(round(x))
            x = min(max(x, int(p.low)), int(p.high))
        else:
            x = float(min(max(x, p.low), p.high))
        point[p.name] = x
    return point


def generate_seed_designs(space: SearchSpace, k: int, random_state: int) -> list[dict]:
    """Draw `k` distinct, constraint-satisfying configs via Latin Hypercube sampling.

    Oversamples (`k * _OVERSAMPLE_FACTOR` rows per batch) since a tight
    constraint can reject a meaningful fraction of raw draws. A rejected draw
    is first repaired via `search_space.repair_config` and kept only if the
    repair now satisfies every constraint; otherwise it is discarded silently
    (expected to be rare given the registered bounds). Duplicates are dropped
    via `config_signature`. If one oversampled batch isn't enough, further
    batches are drawn from the same (state-advancing) sampler, up to
    `_MAX_BATCHES` total, and whatever was found by then is returned.
    """
    sampler = qmc.LatinHypercube(d=len(space.params), seed=random_state)
    batch_size = max(k * _OVERSAMPLE_FACTOR, 1)

    kept: list[dict] = []
    seen: set[tuple] = set()
    for _ in range(_MAX_BATCHES):
        if len(kept) >= k:
            break
        for row in sampler.random(n=batch_size):
            if len(kept) >= k:
                break
            cfg = decode_unit_point(row, space)
            if not all(c(cfg) for c in space.constraints):
                cfg = repair_config(cfg, space)
                if not all(c(cfg) for c in space.constraints):
                    continue
            sig = config_signature(cfg, space)
            if sig in seen:
                continue
            seen.add(sig)
            kept.append(cfg)
    return kept
