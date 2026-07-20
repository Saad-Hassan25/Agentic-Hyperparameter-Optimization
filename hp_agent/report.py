"""Reporting — deterministic stats in, narrative out (§3 stage 5, §6.9).

`compute_hyperparameter_influence` and `compute_trials_to_best` are the only
places the "hyperparameter influence" and "trials-to-best" numbers in
`TuningReport` are computed — in code, from `TrialOutcome` history, via
`scipy.stats.spearmanr`. The LLM is handed those numbers, the selection
rationale, and the baseline comparison, and writes one grounded narrative
paragraph; it never computes or asserts a correlation itself. Crucially, the
LLM is only ever asked to return `{"narrative": str}` (`_NarrativeOnly`) — it
is never handed `TuningReport`'s own schema, so a model that ignores its
instructions and tries to re-assert `best_metric`, `hyperparameter_influence`,
or any other numeric field has no field to put it in; `llm_report` always
constructs the final `TuningReport` in code from `ReportContext`, splicing in
only the narrative string. `llm_report` mirrors `feature_agent.report`'s
(narrative, source) pattern: a deterministic f-string fallback stands in
whenever `llm=None` (the --no-llm path, and the only path this module's tests
exercise without a network) or the LLM call raises, so a run always produces
a valid `TuningReport`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from pydantic import BaseModel, Field
from scipy.stats import spearmanr

from .config import TuningConfig
from .evaluate import TrialOutcome
from .llm import BudgetExceeded, LLMClient, LLMError
from .search_space import SearchSpace
from .select import ConvergenceDecision

_REPORT_SYSTEM = (
    "You are a principal data scientist writing up a hyperparameter-tuning run. "
    "Every number given to you was computed deterministically; interpret them "
    "faithfully and never invent or recompute a figure."
)


class TuningReport(BaseModel):
    best_config: dict
    best_metric: float
    selection_rule: str
    baseline_sampler_metric: float
    baseline_sampler_name: Literal["optuna_tpe", "random_search"]
    lift_over_baseline: float
    trials_to_best: int
    hyperparameter_influence: dict[str, float]
    narrative: str


class _NarrativeOnly(BaseModel):
    """The narrow schema actually requested from the LLM at the report stage
    (§6.9). Every field of `TuningReport` besides `narrative` is code-computed
    and spliced in by `llm_report` — the LLM is structurally unable to re-assert
    or overwrite a numeric field because it is never shown `TuningReport`'s
    own schema, only this one."""

    narrative: str = Field(..., description="One paragraph, grounded only in the numbers given below")


# --------------------------------------------------------------------------- #
# prompt template loading (mirrors feature_agent's prompts_util.py pattern;
# inlined here since hp_agent's file tree has no shared prompts_util module —
# same convention propose.py already established)
# --------------------------------------------------------------------------- #
def _template(name: str) -> str:
    return resources.files("hp_agent.prompts").joinpath(name).read_text(encoding="utf-8")


def _fill(name: str, subs: dict[str, str]) -> str:
    text = _template(name)
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


# --------------------------------------------------------------------------- #
# deterministic stats (§6.9) — the LLM never computes these
# --------------------------------------------------------------------------- #
def _ok_trials(trials: list[TrialOutcome]) -> list[TrialOutcome]:
    return [t for t in trials if t.status == "ok" and t.primary_metric is not None]


def compute_hyperparameter_influence(trials: list[TrialOutcome], space: SearchSpace) -> dict[str, float]:
    """Spearman(param value, primary_metric) across every ok trial, per param.

    Handles both the modern `scipy.stats.spearmanr` return (a result object
    with a `.statistic` attribute) and the older `(correlation, pvalue)` tuple,
    since which one a given scipy install returns depends on its version.
    Fewer than 3 ok trials, or a param with zero variance across the history
    (or a metric with zero variance, which also makes the correlation
    undefined), yields 0.0 for that param rather than a NaN.
    """
    ok = _ok_trials(trials)
    influence: dict[str, float] = {}
    for p in space.params:
        if len(ok) < 3:
            influence[p.name] = 0.0
            continue
        xs = [float(t.config[p.name]) for t in ok]
        ys = [float(t.primary_metric) for t in ok]
        if len(set(xs)) <= 1 or len(set(ys)) <= 1:
            influence[p.name] = 0.0
            continue
        result = spearmanr(xs, ys)
        rho = getattr(result, "statistic", None)
        if rho is None:
            rho = getattr(result, "correlation", None)
        if rho is None:
            rho = result[0]  # oldest-style plain tuple
        rho = float(rho)
        influence[p.name] = 0.0 if math.isnan(rho) else round(rho, 4)
    return influence


def compute_trials_to_best(trials: list[TrialOutcome], best_metric: float) -> int:
    """Earliest iteration (among ok trials, any source) whose own raw score
    already reached the eventually-selected best (§14: "doesn't reach that
    level until trial 24"). 0 if `trials` is empty."""
    if not trials:
        return 0
    ok_sorted = sorted(_ok_trials(trials), key=lambda t: t.iteration)
    for t in ok_sorted:
        if t.primary_metric >= best_metric - 1e-9:
            return t.iteration
    # No ok trial reached the bar (shouldn't happen if best_metric was itself
    # drawn from this history) — fall back to the last ok iteration, else 0.
    return ok_sorted[-1].iteration if ok_sorted else 0


# --------------------------------------------------------------------------- #
# narrative
# --------------------------------------------------------------------------- #
@dataclass
class ReportContext:
    """Everything the narrative/markdown renderer needs, bundled once so
    `llm_report` and `render_report_markdown` share a single source of truth."""

    task: str
    metric_name: str
    model_family: str
    best_config: dict
    best_metric: float
    selection_rule: str
    agent_trials_to_best: int
    baseline_sampler_name: Literal["optuna_tpe", "random_search"]
    baseline_sampler_metric: float
    baseline_trials_to_best: int
    lift_over_baseline: float
    hyperparameter_influence: dict[str, float]
    convergence: ConvergenceDecision
    n_agent_trials: int
    n_baseline_trials: int


def _influence_table(influence: dict[str, float]) -> str:
    if not influence:
        return "(no hyperparameter influence could be computed — fewer than 3 ok trials)"
    lines = ["| param | spearman rho |", "|---|---|"]
    for name, rho in influence.items():
        lines.append(f"| `{name}` | {rho:+.4f} |")
    return "\n".join(lines)


def _ranked_influence(influence: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(influence.items(), key=lambda kv: abs(kv[1]), reverse=True)


def deterministic_narrative(ctx: ReportContext) -> str:
    """Plain f-string narrative, stated numbers only — no LLM prose. This is
    what --no-llm mode (and any LLM-call failure) produces."""
    ranked = _ranked_influence(ctx.hyperparameter_influence)
    if ranked:
        strongest_name, strongest_rho = ranked[0]
        weakest_name, weakest_rho = ranked[-1]
        influence_sentence = (
            f"`{strongest_name}` showed the strongest rank correlation with {ctx.metric_name} "
            f"(Spearman {strongest_rho:+.4f}), while `{weakest_name}` showed the weakest "
            f"(Spearman {weakest_rho:+.4f})."
        )
    else:
        influence_sentence = "Hyperparameter influence could not be computed from this trial history."
    if ctx.lift_over_baseline > 0:
        lift_clause = f"beat the {ctx.baseline_sampler_name} baseline by {ctx.lift_over_baseline:+.4f}"
    elif ctx.lift_over_baseline < 0:
        lift_clause = f"trailed the {ctx.baseline_sampler_name} baseline by {-ctx.lift_over_baseline:.4f}"
    else:
        lift_clause = f"matched the {ctx.baseline_sampler_name} baseline"
    return (
        f"Tuning {ctx.model_family} for {ctx.task} selected a configuration scoring "
        f"{ctx.best_metric:.4f} {ctx.metric_name} ({ctx.selection_rule}). {influence_sentence} "
        f"The agent {lift_clause} ({ctx.baseline_sampler_metric:.4f}), reaching its best in "
        f"{ctx.agent_trials_to_best} trials versus {ctx.baseline_trials_to_best} for the baseline "
        f"over {ctx.n_agent_trials} agent trials and {ctx.n_baseline_trials} baseline trials."
    )


def _report_from_ctx(ctx: ReportContext, narrative: str) -> TuningReport:
    """Build the final `TuningReport` entirely from code-computed `ctx` fields,
    splicing in only `narrative` — the one field either the LLM or
    `deterministic_narrative` produced. No field but `narrative` ever comes
    from anywhere but `ReportContext`, so this is the single place, whether
    the narrative source is the LLM or the deterministic fallback, that a
    `TuningReport` gets constructed."""
    return TuningReport(
        best_config=ctx.best_config,
        best_metric=ctx.best_metric,
        selection_rule=ctx.selection_rule,
        baseline_sampler_metric=ctx.baseline_sampler_metric,
        baseline_sampler_name=ctx.baseline_sampler_name,
        lift_over_baseline=ctx.lift_over_baseline,
        trials_to_best=ctx.agent_trials_to_best,
        hyperparameter_influence=ctx.hyperparameter_influence,
        narrative=narrative,
    )


def deterministic_report(ctx: ReportContext) -> TuningReport:
    return _report_from_ctx(ctx, deterministic_narrative(ctx))


def llm_report(llm: LLMClient | None, ctx: ReportContext, cfg: TuningConfig) -> tuple[TuningReport, str]:
    """LLM-written narrative with a deterministic fallback.

    `llm=None` means the caller already resolved --no-llm mode (or no
    `LLMClient` was ever constructed) — this branch never imports or touches
    `llm.py`'s OpenAI client machinery, it just builds the fallback report
    from `ctx`. Temperature is hardcoded to 0.0 here (§6.1 fixes the report
    stage to 0.0 for determinism; it is intentionally not a `TuningConfig`
    field). Returns (report, "llm") or (report, "deterministic").

    The LLM is only ever shown `_NarrativeOnly`'s schema (one string field),
    never `TuningReport`'s own schema — so even a model that ignores the
    "copy through exactly" system prompt and tries to restate `best_metric`,
    `hyperparameter_influence`, `lift_over_baseline`, etc. has nowhere to put
    them; `_report_from_ctx` builds every other field from `ctx` regardless of
    what the LLM returned.
    """
    if llm is None:
        return deterministic_report(ctx), "deterministic"

    prompt = _fill("report.md", {
        "<<TASK>>": ctx.task,
        "<<MODEL_FAMILY>>": ctx.model_family,
        "<<METRIC_NAME>>": ctx.metric_name,
        "<<SELECTION_RULE>>": ctx.selection_rule,
        "<<BEST_CONFIG>>": str(ctx.best_config),
        "<<BEST_METRIC>>": f"{ctx.best_metric:.4f}",
        "<<BASELINE_SAMPLER_NAME>>": ctx.baseline_sampler_name,
        "<<BASELINE_METRIC>>": f"{ctx.baseline_sampler_metric:.4f}",
        "<<LIFT>>": f"{ctx.lift_over_baseline:+.4f}",
        "<<AGENT_TRIALS_TO_BEST>>": str(ctx.agent_trials_to_best),
        "<<BASELINE_TRIALS_TO_BEST>>": str(ctx.baseline_trials_to_best),
        "<<HYPERPARAMETER_INFLUENCE_TABLE>>": _influence_table(ctx.hyperparameter_influence),
    })
    try:
        narrative_only = llm.structured(
            stage="report", system=_REPORT_SYSTEM, user=prompt,
            schema=_NarrativeOnly, temperature=0.0,
        )
        return _report_from_ctx(ctx, narrative_only.narrative), "llm"
    except (LLMError, BudgetExceeded):
        return deterministic_report(ctx), "deterministic"


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def render_report_markdown(report: TuningReport, ctx: ReportContext, source: str, generated_at: str) -> str:
    """Renders the `runs/<run_id>/report.md` artifact (§7)."""
    L: list[str] = []
    L += ["# Hyperparameter Tuning Report", ""]
    L.append(
        f"**Model family:** `{ctx.model_family}`  ·  **Task:** {ctx.task}  ·  "
        f"**Metric:** {ctx.metric_name}"
    )
    if generated_at:
        L.append(f"**Generated:** {generated_at}  ")
    L.append(f"**Narrative source:** {source}")
    L += ["", "## Narrative", "", report.narrative, ""]

    L += ["## Best configuration", "",
          f"```\n{report.best_config}\n```",
          f"- **{ctx.metric_name}:** {report.best_metric:.4f}",
          f"- **Selection rule:** {report.selection_rule}",
          ""]

    L += ["## Baseline comparison", "",
          f"- **Baseline sampler:** {report.baseline_sampler_name}",
          f"- **Baseline {ctx.metric_name}:** {report.baseline_sampler_metric:.4f}",
          f"- **Lift over baseline:** {report.lift_over_baseline:+.4f}",
          f"- **Trials to best — agent:** {report.trials_to_best}  ·  "
          f"**baseline:** {ctx.baseline_trials_to_best}",
          f"- **Trial counts — agent:** {ctx.n_agent_trials}  ·  "
          f"**baseline:** {ctx.n_baseline_trials}",
          ""]

    L += ["## Convergence", "",
          f"- **Converged:** {ctx.convergence.converged}",
          f"- **Reason:** {ctx.convergence.reason}",
          f"- **Trials used:** {ctx.convergence.trials_used}",
          ""]

    L += ["## Hyperparameter influence", "",
          "Spearman rank correlation between each dimension's value and the primary "
          "metric, across every `ok` trial (code-computed, never LLM-asserted).", "",
          _influence_table(report.hyperparameter_influence), ""]

    L += ["---", f"*Narrative source: {source}. All statistics above are computed "
          "deterministically from `trials.jsonl`; the narrative paragraph is the only "
          "LLM-authored content and is grounded in those numbers.*"]
    return "\n".join(L)
