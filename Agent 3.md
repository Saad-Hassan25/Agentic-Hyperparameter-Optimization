# Agent 3 — Agentic Hyperparameter Optimization

**Status:** Design approved for implementation · **LLM provider:** OpenRouter (all model calls) · **Companion:** tunes any sklearn-compatible estimator over any `(X, y)`; typically runs downstream of `feature_agent`'s exported pipeline, but has no hard dependency on it

---

## 1. Problem Statement

**What it replaces.** Grid search (exhaustive but wasteful), random search (efficient but blind to trend), and manual Bayesian optimization setup (powerful but boilerplate-heavy and opaque once running). All three treat hyperparameter tuning as a search problem over a metric surface. This agent treats it as a reasoning problem over an evolving trial history — the way a practitioner actually tunes: try something sensible, look at what moved the needle, form a hypothesis, adjust.

**What the agent does.** Given a dataset, a target, a task type, and a model family, the agent:

1. Selects (or is given) a **model family** and its declarative search space — RandomForest, LightGBM/HistGradientBoosting, or a linear baseline (ElasticNet/LogisticRegression).
2. Runs a small **space-filling warm-start design** (not one arbitrary seed point) to give the reasoning loop a spread of real evidence before it starts guessing.
3. Proposes the next configuration via an LLM that reasons over the **full trial history**, with structured, schema-validated output and a repair loop for invalid or duplicate proposals.
4. Evaluates each candidate with a **model-family-aware cross-validation harness** — correct stratification, optional grouping, per-trial timeout, configurable metric — never the LLM computing numbers itself.
5. Applies a **noise-floor- and overfit-aware selection rule**: the winner is never just `max(metric)`.
6. Detects convergence with a **patience- and noise-floor-based rule**, not a fixed span-of-last-3 heuristic.
7. Runs an **Optuna TPE (or random-search) baseline** under the identical budget and search space — this is both the `--no-llm` fallback and the control group the final report compares the agent against.
8. Emits reproducible artifacts: a full trial ledger, the best config, the baseline comparison, and a report explaining — with numbers computed in code, not asserted by the LLM — which hyperparameters actually drove the improvement.

**Non-goals (v1).** Neural-network / deep-learning hyperparameter search (different cost/time regime, needs early-stopping-aware schedulers like ASHA/Hyperband), multi-objective tuning (accuracy vs. latency vs. fairness simultaneously), distributed/parallel trial execution across machines, and AutoML-style model-family *selection* (this agent tunes a given family; picking the family is a future agent's job — see §13).

---

## 2. Lessons from the Prototype (what this plan fixes)

The prototype validated the core idea — an LLM reasoning over trial history beats blind search — but has eight defects that would sink it in production. Each drives a design decision below.

| # | Prototype defect | Consequence | Fix (section) |
|---|---|---|---|
| 1 | Direct `OpenAI()` client, hardcoded `gpt-4o-mini`, no schema-retry on malformed JSON | Wrong provider; brittle parsing; a bad response silently produces a garbage trial | OpenRouter client wrapper, model in config, validated-JSON retry loop (§6.1, §6.4) |
| 2 | One hardcoded model family (`RandomForestClassifier`); parameter ranges duplicated in Pydantic `Field` bounds **and** a separately hand-written prompt text block | Can't tune LightGBM or a linear model; the two range statements drift out of sync the first time either is edited | Declarative `SearchSpace` per model family — single source of truth for validation *and* the prompt (§6.2) |
| 3 | Search starts from one hardcoded seed config, then every subsequent trial is a free-form LLM guess | No diversity in early evidence; the search can anchor near the seed and never sample a distant good region | Space-filling warm-start design (Latin Hypercube) before agent-guided refinement (§6.3) |
| 4 | No duplicate-config detection; a malformed proposal is silently `continue`-skipped | Wastes a trial slot for zero information; the LLM never learns it repeated itself or broke the schema | Hash-based dedup + validation-error-in-the-loop repair, falling back to a deterministic perturbation (§6.4) |
| 5 | Single metric (`roc_auc`), undeclared stratification, no grouping option, classification-only | Misleading on imbalanced data or regression tasks; optimistic scores when rows from the same entity split across folds | Model-family-aware CV harness: configurable metric, explicit `StratifiedKFold`/`GroupKFold`, regression support, per-trial timeout (§6.5) |
| 6 | `overfit_gap` is computed and shown as prompt text but never used — the run just returns `max(results, key=val_auc)` | Can crown a badly overfit config that merely got a lucky validation fold | Noise-floor- and overfit-aware selection rule: never a raw metric-max (§6.6) |
| 7 | Convergence = span of the last 3 val-AUC scores `< 0.005`, no floor on minimum trials, no comparison to fold noise | Can "converge" after 3 trials that happen to score similarly by chance, or never converge when genuinely flat because `tol` is smaller than CV noise | Patience-based **and** noise-floor-based convergence, with a minimum-trials floor (§6.7) |
| 8 | No non-LLM path; no baseline proving the LLM step earns its cost/latency over a well-known sampler | Can't tell if the reasoning loop adds value; the tool is unusable and untestable without an API key | Optuna TPE / random-search baseline track, run under the identical budget — doubles as `--no-llm` fallback and control group (§6.8) |

---

## 3. Architecture

One agent, six stages, orchestrated by a deterministic controller, plus a baseline track that runs under the same budget for comparison. The LLM is called at exactly two points (propose, report); everything else — training, scoring, gating, convergence — is deterministic code.

```
                          ┌──────────────────────────────────────────────┐
                          │                ORCHESTRATOR                  │
                          │  budget: trials / wall-time / $ · manifest   │
                          └──────────────────────────────────────────────┘
                                              │
   ┌──────────┐    ┌────────────┐    ┌────────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────┐
   │ 0.       │    │ 1.         │    │ 2.             │    │ 3.         │    │ 4.          │    │ 5.       │
   │ SEED     │───▶│ PROPOSE    │───▶│ VALIDATE +     │───▶│ EVALUATE   │───▶│ SELECT /    │───▶│ REPORT   │
   │          │    │ (LLM)      │    │ DEDUP + REPAIR │    │ (CV)       │    │ CONVERGE    │    │ (LLM)    │
   │ space-   │    │ reasons    │    │ Pydantic       │    │ model-     │    │ noise-floor,│    │ hyper-   │
   │ filling  │    │ over full  │    │ schema, hash   │    │ family CV, │    │ overfit-    │    │ param    │
   │ design   │    │ trial      │    │ dedup, retry   │    │ per-trial  │    │ aware pick, │    │ influence│
   │ (LHS)    │    │ history    │    │ w/ error text  │    │ timeout    │    │ patience    │    │ + report │
   └──────────┘    └────────────┘    └────────────────┘    └────────────┘    └─────────────┘    └──────────┘
                         ▲                                                          │
                         └────────────── feedback: trial history, ─────────────────┘
                                         rejections, best-so-far

   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │  BASELINE TRACK (parallel run, identical budget/space/CV): Optuna TPE, else random search │
   │  → also the full --no-llm mode, and the control group §5's report compares against        │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Stage responsibilities**

0. **Seed** — A Latin-Hypercube (or Sobol, if `scipy.stats.qmc` version supports it) space-filling design over the active `SearchSpace`, size `k = max(5, round(0.2 × trial_budget))`. Gives the agent a spread of real, diverse evidence instead of one arbitrary starting point.
1. **Propose** — LLM reasons over the full trial history (config, metric, std, overfit gap, fit time per trial) and returns one structured proposal plus a one-sentence rationale.
2. **Validate + Dedup + Repair** — The proposal is checked against the dynamically-generated Pydantic model for the active search space. Out-of-range or malformed responses are retried (≤2×) with the validation error appended to the prompt. A config within floating-point tolerance of one already tried is rejected as a duplicate and fed back ("already tried, propose something meaningfully different"). If repair fails twice, the orchestrator falls back to a deterministic local perturbation of the current best config rather than burning the trial.
3. **Evaluate** — The model-family adapter trains under CV with the configured metric, fold strategy, and a per-trial wall-clock timeout; returns validation metric, std across folds, train metric, overfit gap, and fit time.
4. **Select / Converge** — Deterministic gate: compute the noise floor from fold std, choose the running-best under the overfit-aware rule (§6.6), and check the convergence conditions (§6.7).
5. **Report** — Deterministic code computes rank-correlation "hyperparameter influence" scores from the trial history and the lift over the baseline track; the LLM turns those numbers into a short narrative. It never computes the numbers itself.

---

## 4. Data Contracts

All LLM I/O and inter-stage handoffs are Pydantic models. The search space, not free prompt text, is the single source of truth for what the LLM is allowed to propose.

```python
from pydantic import BaseModel, Field, create_model
from typing import Literal
from dataclasses import dataclass, field
from collections.abc import Callable

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


class HyperparamProposal(BaseModel):
    """What the LLM returns at the propose stage."""
    values: dict[str, float | int]
    reasoning: str = Field(..., description="One sentence: what trend drove this proposal")


class TrialOutcome(BaseModel):
    iteration: int
    source: Literal["seed_design", "llm_proposal", "repair_perturbation", "baseline_sampler"]
    config: dict
    status: Literal["ok", "failed", "duplicate_rejected", "timeout"]
    primary_metric: float | None = None
    metric_std: float | None = None
    train_metric: float | None = None
    overfit_gap: float | None = None
    fit_time_s: float | None = None
    error: str | None = None


class ConvergenceDecision(BaseModel):
    converged: bool
    reason: Literal[
        "patience_exhausted", "noise_floor_plateau", "budget_exhausted", "not_converged"
    ]
    trials_used: int


class TuningReport(BaseModel):
    best_config: dict
    best_metric: float
    selection_rule: str                       # human-readable, e.g. "within noise floor of
                                               #   top score, minimum overfit_gap"
    baseline_sampler_metric: float
    baseline_sampler_name: Literal["optuna_tpe", "random_search"]
    lift_over_baseline: float
    trials_to_best: int
    hyperparameter_influence: dict[str, float]  # Spearman(param, metric) across history,
                                                 #   computed in code, not by the LLM
    narrative: str                              # LLM-written, grounded in the fields above
```

**Why a `SearchSpace` object instead of prompt text:** the prototype's bug (ranges silently drifting between the Pydantic schema and the prompt) is structurally impossible here — `to_pydantic_model()` and `render_prompt_block()` both read the same `list[ParamSpec]`. Adding a model family means writing one `SearchSpace`, not three places that must stay in sync.

---

## 5. Repository Layout

Mirrors the standalone-agent convention established by `eda_agent` and `feature_agent`.

```
hp_agent/
├── __init__.py
├── config.py            # TuningConfig dataclass; YAML-loadable
├── llm.py                # OpenRouter client wrapper: retries, JSON-schema
│                        #   enforcement, cost accounting (shared pattern w/ feature_agent)
├── search_space.py      # ParamSpec, SearchSpace, per-family registry
├── adapters.py           # model-family adapters: RandomForest, LightGBM/
│                        #   HistGradientBoosting fallback, ElasticNet/LogisticRegression
├── design.py             # Latin-Hypercube / Sobol warm-start sampler
├── propose.py            # LLM proposal + Pydantic validation + dedup + repair loop
├── evaluate.py           # CV harness: stratified/grouped folds, configurable metric,
│                        #   per-trial timeout, overfit gap
├── select.py             # noise-floor + overfit-aware best-config rule, convergence
│                        #   decision (patience + noise floor + min-trials floor)
├── baseline.py           # Optuna TPE sampler, else random-search fallback; also the
│                        #   full --no-llm path
├── report.py             # hyperparameter-influence stats (Spearman, code-computed),
│                        #   LLM narrative writer
├── orchestrator.py       # trial loop, budgets, manifest, public API
├── prompts/
│   ├── propose.md         # next-config proposal prompt template
│   └── report.md          # final narrative prompt template
└── tests/                # see §10
```

Public API kept to one call:

```python
from hp_agent import HPAgent, TuningConfig

result = HPAgent(TuningConfig(
    model="<openrouter-model-id>",
    model_family="lightgbm",
    trial_budget=25,
)).run(X, y, task="classification")

result.best_config          # dict
result.best_metric           # float
result.baseline_sampler_metric
result.report_path           # report.md
```

---

## 6. Key Design Decisions

### 6.1 LLM access — OpenRouter only

Same wrapper contract as `feature_agent.llm`, so both agents can eventually share one `llm.py` module across the suite:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

- **Model from config**, never hardcoded; default is a non-OpenAI OpenRouter model, overridable via `OPENROUTER_MODEL`.
- **Structured output enforcement**: request JSON, parse into `HyperparamProposal`, retry ≤2× with the validation error appended on failure. Hard-fail only that trial slot (falls to repair-perturbation, §6.4), never the run.
- **Determinism posture**: `temperature=0.4` for proposals (exploration is useful, this is a search), `0.0` for the final report. Prompt/response hashes recorded in the run manifest.
- **Budget accounting**: token counts and cost per call accumulated; orchestrator enforces a per-run `max_cost_usd` ceiling exactly like `feature_agent`.

### 6.2 Search space as the single source of truth

Each model family registers one `SearchSpace` (§4). `search_space.py` ships three to start:

| Family | Key dimensions | Notable constraint |
|---|---|---|
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` | none |
| `lightgbm` (falls back to `HistGradientBoostingClassifier/Regressor` if `lightgbm` isn't installed — same fallback already used in `feature_agent`) | `num_leaves`, `max_depth`, `learning_rate` (log), `n_estimators` (log), `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda` (log) | `num_leaves < 2**max_depth` |
| `elasticnet` / `logistic_regression` | `alpha`/`C` (log), `l1_ratio` | none |

Adding a fourth family (e.g. XGBoost) is a one-file addition to `search_space.py` plus one adapter in `adapters.py` — nothing else changes, because the Pydantic schema, the prompt block, and the warm-start sampler are all derived from the same `SearchSpace`.

### 6.3 Warm start — space-filling design before agent-guided refinement

The prototype started from a single hand-picked config, then handed everything else to free-form LLM guessing — the reasoning loop had no spread of evidence to reason *from*. Before the LLM proposes anything, `design.py` draws `k = max(5, round(0.2 × trial_budget))` points from a Latin-Hypercube sample over the normalized search space (log-scale dimensions sampled in log space, then exponentiated), evaluated in stage 3 exactly like any other trial. Only after the seed batch completes does control pass to the propose stage. This mirrors how real Bayesian optimizers warm-start, and it means the agent's first real trend-reasoning happens over genuine diversity, not noise around one point.

### 6.4 Structured proposal + validation/dedup/repair loop

- **Validation.** The proposal's `values` dict is parsed against the search space's dynamically-generated Pydantic model (§4). Constraint violations (e.g. `num_leaves >= 2**max_depth`) are checked separately and reported the same way. On failure, the validation error is appended verbatim to the next prompt and the LLM gets one retry (≤2 total attempts).
- **Deduplication.** Every accepted config is hashed (rounded to a fixed precision per `ParamSpec`) against all prior trials. A duplicate is rejected without spending a training run, and the rejection ("you already proposed `{config}` at trial 4, val_metric=0.881 — propose something meaningfully different") is fed back into the next prompt.
- **Repair fallback (graceful degradation within a run).** If two consecutive proposals fail validation or come back duplicate, the orchestrator stops asking the LLM for that slot and generates a deterministic local perturbation of the current best config (±15% jitter per dimension, clipped to bounds) instead — the run always makes forward progress on its trial budget even if the LLM is misbehaving.

### 6.5 Evaluation protocol — model-family-aware, budgeted, honest about grouping

- **Fold strategy.** `StratifiedKFold` for classification, plain `KFold` for regression, `GroupKFold` when the caller supplies a `group_column` (repeated entities must not straddle folds — the same leakage lesson `feature_agent` applies to feature evaluation applies here to model evaluation).
- **Metric.** Configurable, task-appropriate default: `average_precision` for classification (informative under imbalance — a plain `roc_auc` default, as the prototype hardcoded, overstates quality on skewed targets), `neg_root_mean_squared_error` for regression. Always paired with `metric_std` across folds, never reported alone.
- **Per-trial timeout and run budget.** Each trial is bounded by a wall-clock timeout (config, default 120 s); a config that would run away (e.g. `n_estimators=2000`, `max_depth=50` on LightGBM) is killed and recorded as `status="timeout"`, not allowed to stall the whole run. The orchestrator additionally enforces a total wall-time and `max_cost_usd` ceiling across the run, not just an iteration count.
- **Overfit gap** (`train_metric - primary_metric`) is computed every trial and, unlike the prototype, is actually consumed downstream (§6.6) rather than only shown as prompt text.

### 6.6 Selection rule — never a raw metric-max

The prototype's `max(results, key=lambda r: r.val_auc)` can crown a config that merely got a lucky validation fold or is badly overfit. The rule instead:

1. Compute the **noise floor**: the top trial's `metric_std` (fold-to-fold spread), a proxy for how much of an apparent improvement could be CV noise rather than signal.
2. Form the **candidate set**: all trials whose metric is within the noise floor of the single best metric.
3. **Select** the candidate with the smallest `overfit_gap`; tie-break by fastest `fit_time_s`.

This is the same spirit as the one-standard-error rule in `glmnet`: prefer the simplest, least-overfit model that isn't measurably worse than the best one found, rather than chasing a metric difference smaller than the evaluation's own noise.

### 6.7 Convergence — patience, noise floor, and a minimum-trials floor

Replaces the prototype's "span of the last 3 scores `< 0.005`" with three explicit, independently-loggable conditions, evaluated only after a **minimum-trials floor** (default: seed batch size + 5) has been reached:

- **Patience exhausted:** no new best-selected config (§6.6) in the last `patience` trials (default 8).
- **Noise-floor plateau:** the spread of the last `window` trials' metrics (default 5) is below `max(tol, noise_floor)` — i.e., the tolerance is never allowed to be tighter than what the CV protocol itself can resolve.
- **Budget exhausted:** trial/wall-time/cost ceiling reached — always a valid, logged stop reason, distinct from genuine convergence.

The `ConvergenceDecision.reason` is recorded in the manifest so a run that stopped on budget is never mistaken for one that actually plateaued.

### 6.8 Baseline comparison and graceful degradation — one mechanism, two jobs

Every run also executes a classical sampler — **Optuna's TPE sampler** if installed, else uniform random search — under the **identical** search space, CV protocol, and trial budget, recorded as `source="baseline_sampler"` trials. This single mechanism does two jobs the prototype had no answer for:

- **It is the `--no-llm` fallback** required across this agent suite: with no `OPENROUTER_API_KEY` set, the orchestrator skips stages 1–2 entirely and returns the baseline-sampler result, still fully useful and fully testable offline.
- **It is the control group.** When the LLM path *does* run, the final report states `lift_over_baseline` (best-selected metric, agent vs. sampler, same budget) and `trials_to_best` for each — so the reasoning loop has to demonstrate it earns its added cost and latency, every run, not just in a one-off benchmark.

### 6.9 Reporting — deterministic stats in, narrative out

`report.py` computes `hyperparameter_influence` as the Spearman rank correlation between each search-space dimension's value and the trial's primary metric, across the full trial history (seed + proposals + repairs), **in code**. The LLM is given these numbers, the selection rationale, and the baseline comparison, and writes the one-paragraph narrative — it never computes or asserts a correlation itself, consistent with this suite's standing rule that the LLM reasons and code does arithmetic.

---

## 7. Outputs & Reproducibility

```
runs/<run_id>/
├── manifest.json          # dataset hash, config, model id, prompt hashes,
│                          #   seeds, package versions, cost, wall time,
│                          #   convergence reason
├── trials.jsonl           # every TrialOutcome — seed, proposals, repairs,
│                          #   rejections (duplicate/invalid), and the baseline
│                          #   sampler's trials, one line each
├── best_config.json        # winning config + why it was selected (§6.6)
├── baseline_comparison.json  # sampler's best, agent's best, lift, trials-to-best
└── report.md               # LLM narrative grounded in the above
```

`trials.jsonl` includes rejected and repaired proposals, not just accepted ones — "the agent proposed `num_leaves=310` at trial 6, rejected: exceeds `2**max_depth`" is exactly the audit trail a manual tuning session never produces.

---

## 8. Configuration

```python
@dataclass
class TuningConfig:
    # LLM (OpenRouter)
    model: str | None = None          # None → --no-llm mode, baseline sampler only
    proposal_temperature: float = 0.4
    max_cost_usd: float = 1.00

    # Search
    model_family: str = "lightgbm"    # random_forest | lightgbm | elasticnet | logistic_regression
    trial_budget: int = 25
    seed_fraction: float = 0.2        # fraction of trial_budget spent on warm-start LHS design

    # Evaluation
    n_folds: int = 5
    metric: str = "auto"              # auto | average_precision | roc_auc | rmse | mae
    group_column: str | None = None
    trial_timeout_s: float = 120.0
    max_wall_time_s: float = 900.0
    random_state: int = 42

    # Convergence
    patience: int = 8
    convergence_window: int = 5
    convergence_tol: float = 0.005
    min_trials_before_convergence: int = 10

    # Baseline
    baseline_sampler: str = "auto"    # auto → optuna_tpe if installed, else random_search
```

---

## 9. Implementation Plan

Five milestones, each independently testable with a demoable acceptance criterion. Estimated total: **~2 engineer-weeks.**

| Milestone | Scope | Acceptance criterion | Est. |
|---|---|---|---|
| **M0 — Scaffolding** | Package skeleton, `config.py`, `llm.py` (OpenRouter wrapper reused/adapted from `feature_agent`), `search_space.py` with the three registered families, run-manifest writer | `SearchSpace.to_pydantic_model()` round-trips valid/invalid configs correctly for all three families; wrapper returns a validated `HyperparamProposal` from a live OpenRouter call, and from a mocked malformed-JSON fixture triggers exactly one retry | 2 d |
| **M1 — Adapters + Design** | `adapters.py` (RandomForest, LightGBM w/ HistGB fallback, ElasticNet/LogisticRegression), `design.py` LHS sampler | LHS design over each search space produces `k` distinct, in-bounds, constraint-satisfying configs across 10 seeds | 2 d |
| **M2 — Evaluate + Select** | `evaluate.py` (stratified/grouped CV, configurable metric, per-trial timeout), `select.py` (noise-floor+overfit selection, convergence decision) | On a synthetic dataset with a known best region, the selection rule picks a config within the noise floor of the true best and never the single highest-variance lucky fold; a deliberately runaway config (huge `n_estimators`) is correctly killed at `trial_timeout_s` | 3 d |
| **M3 — Propose + Baseline** | `propose.py` (LLM loop: validate/dedup/repair), `baseline.py` (Optuna TPE, random-search fallback, `--no-llm` path) | Adversarial fixture suite: out-of-range proposal triggers repair retry then perturbation fallback; duplicate proposal is rejected and logged; `--no-llm` run completes end-to-end using only `baseline.py` | 3 d |
| **M4 — Orchestrator + Reporting + Hardening** | `orchestrator.py` (trial loop, budgets, manifest), `report.py` (Spearman influence stats + LLM narrative), full test suite (§10), README | 25-trial run on the Census Income benchmark (§14) stays under `max_cost_usd`; benchmark suite (§11) meets targets; `report.md` narrative references only numbers present in `manifest.json`/`trials.jsonl` | 3 d |

Dependency order is strict M0 → M1 → M2 → M3 → M4, but M1 and the first half of M3 (`baseline.py`, which needs no LLM) can proceed in parallel once M0's search-space contracts are frozen.

---

## 10. Testing Strategy

- **Unit, no network.** All LLM interactions mocked with recorded fixtures. `search_space.py`, `evaluate.py`, `select.py`, and `design.py` are pure functions of data — property-test them (e.g., every LHS-generated config must satisfy the space's constraints by construction).
- **Adversarial propose-loop suite.** Fixtures covering: out-of-range value, missing key, wrong type, constraint violation (`num_leaves >= 2**max_depth`), and an exact duplicate of a prior trial — each must retry/reject/perturb exactly as designed, never silently skip a trial slot.
- **Selection-rule benchmark.** Synthetic objective with a known noisy plateau near the optimum: assert the selection rule (§6.6) picks a low-overfit-gap config within the noise floor of the best, across 20 seeds, and never the single highest-variance outlier trial.
- **Convergence benchmark.** Synthetic flat objective (no true signal past a point): assert the agent stops within a bounded number of trials past `min_trials_before_convergence`, and that a genuinely still-improving objective is never falsely declared converged.
- **Timeout/budget enforcement.** A deliberately slow adapter config must be killed at `trial_timeout_s` and recorded as `status="timeout"`, and a run must halt at `max_wall_time_s`/`max_cost_usd` even mid-trial-budget.
- **`--no-llm` path.** Full run with `model=None` completes using only `baseline.py`, produces every artifact in §7, and never imports `llm.py`'s client.
- **Golden E2E.** One live-LLM smoke test (marked, excluded from CI default) on the Census Income scenario, asserting artifact completeness and that `lift_over_baseline` is computed, not asserting an exact metric value.

---

## 11. Success Metrics

Measured on a fixed benchmark suite (Census Income, a synthetic imbalanced-classification set, and a regression set, each across all three model families):

- **Lift over baseline sampler:** agent's selected metric ≥ Optuna TPE's selected metric on ≥ 4/6 benchmark runs, under an identical trial budget.
- **Sample efficiency:** when the agent wins, it reaches within 1% of its own final best metric in fewer trials than the baseline sampler needs to reach the same bar, on average.
- **Selection soundness:** on the selection-rule benchmark (§10), the chosen config's overfit gap is never in the top decile of all evaluated trials.
- **Robustness:** zero silently-skipped trial slots across the adversarial suite — every malformed/duplicate proposal is retried, perturbed, or explicitly logged as rejected.
- **Cost & latency:** ≤ $1 and ≤ 15 min per 25-trial run on a 50k-row dataset (LLM path); `--no-llm` path has no cost ceiling to observe beyond compute time.

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM repeatedly proposes invalid/duplicate configs, stalling progress | Medium | Medium | Repair-perturbation fallback (§6.4) guarantees forward progress regardless of LLM behavior |
| CV noise on small datasets makes the selection rule indecisive (wide noise floor) | Medium | Medium | Noise floor is explicit and logged; below a row-count threshold (config), warn that the tuning run's resolution is limited |
| Optuna not installed in a given environment | Low | Low | `baseline.py` falls back to uniform random search — same interface, same trials.jsonl schema |
| A model-family adapter has a slow or unstable configuration region (e.g. LightGBM `num_leaves` near the constraint boundary) | Medium | Low | Per-trial timeout (§6.5) plus the constraint check in `SearchSpace` rejects the region before training starts |
| Cost blowup from a long trial budget | Low | Low | Hard `max_cost_usd` and `max_wall_time_s` ceilings, enforced by the orchestrator independent of trial count |
| OpenRouter model deprecation / rate limits | Low | Medium | Model id in config; wrapper retries with backoff; `--no-llm` path always available as a hard fallback |

---

## 13. Extensions (explicitly out of v1)

1. **Multi-objective tuning** — accuracy vs. inference latency vs. model size, requiring a Pareto-front report instead of a single winner.
2. **Model-family selection** — a preceding agent that picks *which* family (tree ensemble vs. linear vs. other) is worth tuning at all, handing this agent its search space.
3. **Early-stopping-aware schedulers** — ASHA/Hyperband-style trial pruning for expensive model families (deep nets, huge ensembles), where a trial can be killed early based on a learning curve rather than only a wall-clock timeout.
4. **Cross-agent orchestration** — `feature_agent`'s exported pipeline → this agent's tuned estimator, sharing the run-manifest convention already used across the suite.

---

## 14. Reference Scenario (acceptance narrative)

Census Income classification (UCI, 48,842 rows, ~24% positive class — imbalanced enough that `average_precision` is the configured metric, not `roc_auc`). `model_family="lightgbm"`, `trial_budget=25`, `seed_fraction=0.2` (5 LHS seed trials), `max_cost_usd=1.00`.

**Seed stage:** 5 Latin-Hypercube trials spread across the LightGBM search space; best seed average-precision 0.782, worst 0.741 — already showing `max_depth` and `learning_rate` moving the metric more than `subsample`/`colsample_bytree`.

**Propose stage:** trial 9's proposal is rejected (`num_leaves=310` violates `num_leaves < 2**max_depth` at `max_depth=8`); the repaired retry proposes `num_leaves=200` instead, with reasoning: *"reducing num_leaves to respect the depth constraint while keeping learning_rate low, since trials 3 and 6 showed lower learning_rate with more estimators outperforming high learning_rate with few."* Trial 14 is rejected as a near-duplicate of trial 7 and the agent is told so; its next proposal moves to unexplored territory instead of resubmitting.

**Convergence:** patience (8 trials with no new best) fires at trial 22 — logged as `reason="patience_exhausted"`, not a budget cutoff.

**Selection:** the raw highest-average-precision trial (23) scores 0.798 but has `overfit_gap=0.041`, well above the noise floor; the selection rule instead picks trial 19 (average-precision 0.795, within the noise floor of trial 23, `overfit_gap=0.018`) as `best_config`.

**Baseline comparison:** Optuna TPE, same 25-trial budget and search space, reaches its own best of 0.783 and doesn't reach that level until trial 24. `lift_over_baseline = 0.795 - 0.783 = +0.012`, `trials_to_best`: agent 19, sampler 24 — the report states plainly that the reasoning loop both reached a better point *and* reached it in fewer trials, on this run.

**Report excerpt (LLM-written, grounded in code-computed numbers):** *"learning_rate showed the strongest rank correlation with average precision (Spearman −0.71 — lower is better, within the tested range), followed by max_depth (+0.44); subsample and colsample_bytree showed negligible correlation (|ρ| < 0.1) and are not worth tuning further for this dataset. The selected configuration trades 0.003 average precision for roughly half the overfit gap of the single best-scoring trial."* Every number in that paragraph traces back to `trials.jsonl`, not to the model's own arithmetic.
