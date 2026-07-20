"""LLM proposal + validation/dedup/repair loop (§3 stage 1-2, §6.4).

The LLM reasons over the full trial history and returns one structured
`HyperparamProposal`. `llm.structured()` already retries (≤2 total attempts)
on malformed/non-schema-conforming JSON; the loop here layers a second,
domain-specific check on top of that — not instead of it — because a proposal
can be syntactically valid JSON and still be out-of-range, constraint-
violating, or an exact duplicate of a prior trial. On the same ≤2-attempt
budget, a rejection is fed back to the model as prompt text (the doc's
"you already proposed this at trial 4" / validation-error style). If both
attempts still fail, the deterministic repair-perturbation fallback (jitter
the current best ±15%, clip, repair constraints) guarantees the trial slot
still makes forward progress — a validation/parsing failure must never
propagate out of `propose_next`.

Genuine LLM-layer failures (§6.1, §12: "hard-fail only that trial slot, never
the run") are handled the same way, not left to escape uncaught: `LLMError`
(the model's own JSON-schema retries inside `llm.structured()` were already
exhausted, or the API call itself failed) is caught around each attempt and
counts as one of the ≤2 local attempts, so a persistently-broken LLM still
degrades to the repair-perturbation fallback like any other rejection.
`BudgetExceeded` is the one exception intentionally left to propagate: it
means the per-run cost ceiling is already spent, which is not "this proposal
was bad" but "the run itself must stop now" — the orchestrator catches it at
the call site and converts it into a clean `budget_exhausted` stop rather than
a crash.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .config import TuningConfig
from .evaluate import TrialOutcome
from .llm import BudgetExceeded, LLMClient, LLMError
from .search_space import SearchSpace, config_signature, repair_config

_MAX_HISTORY_ROWS = 30   # cap the rendered history so the prompt stays bounded;
                         # recent trials matter most for trend-reasoning anyway,
                         # so the cap drops the oldest rows, not the newest
_MAX_REPAIR_ATTEMPTS = 5  # nudges before giving up and returning a possibly-
                          # still-colliding config anyway (rare edge case, §6.4)

_SYSTEM = (
    "You are a rigorous, practical principal data scientist tuning hyperparameters "
    "by reasoning over trial history. You never train models or compute metrics "
    "yourself; you propose one configuration and one sentence of reasoning."
)


class HyperparamProposal(BaseModel):
    """What the LLM returns at the propose stage."""
    values: dict[str, float | int]
    reasoning: str = Field(..., description="One sentence: what trend drove this proposal")


# --------------------------------------------------------------------------- #
# prompt template loading (mirrors feature_agent's prompts_util.py pattern;
# inlined here since hp_agent's file tree has no shared prompts_util module)
# --------------------------------------------------------------------------- #
def _template(name: str) -> str:
    return resources.files("hp_agent.prompts").joinpath(name).read_text(encoding="utf-8")


def _fill(name: str, subs: dict[str, str]) -> str:
    text = _template(name)
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


# --------------------------------------------------------------------------- #
# prompt rendering
# --------------------------------------------------------------------------- #
def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def render_trial_history(trials: list[TrialOutcome]) -> str:
    """Compact, LLM-readable table of prior trials, ok only, most recent last.

    Capped to the most recent `_MAX_HISTORY_ROWS` trials for very long
    histories: this keeps the prompt bounded without silently hiding the cap
    from the reasoning (it's noted here, not just applied) — recent trials are
    what trend-reasoning over an evolving history actually needs anyway.
    """
    ok = sorted(
        (t for t in trials if t.status == "ok" and t.primary_metric is not None),
        key=lambda t: t.iteration,
    )
    if not ok:
        return "(no trials yet)"
    if len(ok) > _MAX_HISTORY_ROWS:
        ok = ok[-_MAX_HISTORY_ROWS:]
    lines = ["iteration | source | config | primary_metric | metric_std | overfit_gap | fit_time_s"]
    for t in ok:
        lines.append(
            f"{t.iteration} | {t.source} | {t.config} | {_fmt(t.primary_metric)} | "
            f"{_fmt(t.metric_std)} | {_fmt(t.overfit_gap)} | {_fmt(t.fit_time_s)}"
        )
    return "\n".join(lines)


def _duplicate_feedback(sig: tuple, history: list[TrialOutcome], space: SearchSpace) -> str:
    match = next(
        (t for t in history if t.status == "ok" and config_signature(t.config, space) == sig),
        None,
    )
    if match is not None:
        return (
            f"You already proposed this config at trial {match.iteration}, "
            f"val_metric={_fmt(match.primary_metric)} — propose something meaningfully different."
        )
    return "This configuration duplicates one already tried — propose something meaningfully different."


# --------------------------------------------------------------------------- #
# repair-perturbation fallback (§6.4)
# --------------------------------------------------------------------------- #
def _base_config_for_repair(
    current_best: TrialOutcome | None, history: list[TrialOutcome], space: SearchSpace
) -> dict:
    if current_best is not None:
        return dict(current_best.config)
    ok = [t for t in history if t.status == "ok"]
    if ok:
        return dict(max(ok, key=lambda t: t.iteration).config)
    if history:
        # No ok trial exists yet (e.g. every seed trial failed/timed out) --
        # fall back to the very first trial's config deterministically. In
        # practice propose_next only runs once seed trials exist, so this is
        # the first seed-design point.
        return dict(min(history, key=lambda t: t.iteration).config)
    # Extreme edge case: no history at all. Deterministic midpoint of every
    # dimension, so the fallback never depends on randomness it can't seed.
    return {
        p.name: (p.low + p.high) / 2 if p.kind == "float" else int(round((p.low + p.high) / 2))
        for p in space.params
    }


def _jitter(config: dict, space: SearchSpace, rng: np.random.Generator) -> dict:
    """±15% jitter per dimension, clipped to bounds, ints rounded."""
    jittered = {}
    for p in space.params:
        value = config[p.name]
        span = p.high - p.low
        if value == 0:
            # Multiplicative jitter can't move a value already at zero (a
            # valid low bound for e.g. l1_ratio) -- jitter relative to the
            # dimension's own range instead so it isn't permanently stuck.
            delta = rng.uniform(-0.15, 0.15) * span
        else:
            delta = rng.uniform(-0.15, 0.15) * value
        new_value = min(max(value + delta, p.low), p.high)
        jittered[p.name] = int(round(new_value)) if p.kind == "int" else float(new_value)
    return jittered


def _repair_perturbation(
    space: SearchSpace,
    history: list[TrialOutcome],
    cfg: TuningConfig,
    current_best: TrialOutcome | None,
    seen_signatures: set,
) -> tuple[dict, str, str]:
    """Deterministic fallback: jitter the current best ±15%, repair constraint
    violations, and retry (fresh jitter draw) up to `_MAX_REPAIR_ATTEMPTS` times
    if the result still collides with a prior trial's signature. This is a rare
    edge case (only matters when the best config's neighborhood is already
    densely sampled) -- forward progress on the trial budget matters more than
    a guaranteed-novel repair, so the last attempt is returned regardless.
    """
    base = _base_config_for_repair(current_best, history, space)
    candidate = base
    for attempt in range(_MAX_REPAIR_ATTEMPTS):
        # Seed combines the run's random_state, a per-call differentiator
        # (history length grows on every call, so consecutive propose_next
        # calls don't collide), and the local attempt count.
        rng = np.random.default_rng(cfg.random_state + len(history) * 10 + attempt)
        candidate = repair_config(_jitter(base, space, rng), space)
        if config_signature(candidate, space) not in seen_signatures:
            break
    rationale = (
        "LLM proposal invalid or duplicate after 2 attempts; deterministic "
        "±15% perturbation of the current best config."
    )
    return candidate, "repair_perturbation", rationale


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def propose_next(
    llm: LLMClient,
    space: SearchSpace,
    model_family: str,
    history: list[TrialOutcome],
    cfg: TuningConfig,
    current_best: TrialOutcome | None,
    seen_signatures: set,
    rejections: list[dict] | None = None,
) -> tuple[dict, str, str]:
    """Propose the next config: up to 2 LLM attempts, then a deterministic
    repair-perturbation fallback (§6.4). Returns (config, source, rationale).

    `BudgetExceeded` propagates uncaught -- the orchestrator owns stopping the
    run cleanly on that (a spent cost ceiling is a run-level stop, not a
    per-proposal rejection). Every other failure mode never escapes this
    function: a parsing/validation/constraint/dedup failure is the scenario
    the repair fallback exists for, and `LLMError` (the model's own JSON-retry
    budget inside `llm.structured()` already exhausted, or the API call itself
    failed) is caught around the call and treated the same way -- it consumes
    one of the two local attempts and falls through to repair-perturbation if
    both attempts end up failing, exactly like an invalid/duplicate proposal
    would (§6.1, §12: "hard-fail only that trial slot, never the run").

    `rejections`, if given, is appended to in place with one dict per
    rejected attempt -- `{"config": ..., "reason": "invalid_schema" |
    "constraint_violation" | "duplicate" | "llm_error", "detail": <feedback
    text>}` -- so a caller (the orchestrator) can surface rejected proposals in
    the run's audit trail (doc §7) without this function's own retry/fallback
    contract changing at all for callers that don't pass it.
    """
    feedback = "(first proposal for this trial slot -- no prior rejection.)"
    for _attempt in range(2):
        prompt = _fill("propose.md", {
            "<<MODEL_FAMILY>>": model_family,
            "<<SEARCH_SPACE>>": space.render_prompt_block(),
            "<<TRIAL_HISTORY>>": render_trial_history(history),
            "<<FEEDBACK>>": feedback,
        })
        try:
            proposal = llm.structured(
                stage="propose", system=_SYSTEM, user=prompt,
                schema=HyperparamProposal, temperature=cfg.proposal_temperature,
            )
        except BudgetExceeded:
            raise  # run-level stop -- the orchestrator converts this to a clean budget_exhausted halt
        except LLMError as exc:
            feedback = (
                f"Your previous response could not be used ({exc}). "
                "Return a corrected JSON proposal matching the schema exactly."
            )
            if rejections is not None:
                rejections.append({"config": None, "reason": "llm_error", "detail": str(exc)})
            continue
        try:
            coerced = space.to_pydantic_model().model_validate(proposal.values).model_dump()
        except ValidationError as exc:
            feedback = str(exc)
            if rejections is not None:
                rejections.append({"config": proposal.values, "reason": "invalid_schema", "detail": feedback})
            continue
        if not all(c(coerced) for c in space.constraints):
            feedback = f"Proposed values {coerced} violate a search-space constraint for {model_family}."
            if rejections is not None:
                rejections.append({"config": coerced, "reason": "constraint_violation", "detail": feedback})
            continue
        sig = config_signature(coerced, space)
        if sig in seen_signatures:
            feedback = _duplicate_feedback(sig, history, space)
            if rejections is not None:
                rejections.append({"config": coerced, "reason": "duplicate", "detail": feedback})
            continue
        return coerced, "llm_proposal", proposal.reasoning

    # Both attempts failed validation/constraints/dedup -> deterministic fallback.
    return _repair_perturbation(space, history, cfg, current_best, seen_signatures)
