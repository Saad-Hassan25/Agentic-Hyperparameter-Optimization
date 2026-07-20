# Hyperparameter Optimization Agent

An **agentic** hyperparameter tuner that treats tuning as a reasoning problem over
an evolving trial history — the way a practitioner actually tunes: try something
sensible, look at what moved the needle, form a hypothesis, adjust — rather than
as a blind search problem over a metric surface.

It replaces three things at once: **grid search** (exhaustive but wasteful),
**random search** (efficient but blind to trend), and **manual Bayesian-optimization
setup** (powerful but boilerplate-heavy and opaque once running). Given a dataset,
a target, a task type, and a model family, the agent runs a space-filling
warm-start design, then an LLM proposes each next configuration by reasoning over
the *full* trial history (config, metric, std, overfit gap, fit time), while
every number — training, scoring, the selection rule, convergence — is computed
by deterministic code, never asserted by the LLM. It is powered by **OpenRouter**
(any JSON-capable model) and built on one core principle:

> **The LLM proposes; deterministic statistics decide.**
> An LLM reasons over trial history and proposes the next config, schema-validated
> and deduplicated against every prior trial. A model-family-aware cross-validation
> harness scores it. A noise-floor- and overfit-aware rule — never a raw
> `max(metric)` — picks the winner, and a patience- and noise-floor-based rule
> decides when to stop.

## Why this over grid/random search or plain Optuna

| Approach | What it misses |
|---|---|
| Grid search | Exhaustive but wasteful; no notion of which region is worth resolving finer |
| Random search | Efficient coverage but blind to trend — never reasons about *why* one region outperformed another |
| Plain Optuna (or any blind sampler) | A strong statistical prior, but no narrative: no hypothesis about which hyperparameter actually drove an improvement, no accounting for overfit gap or CV noise when picking a winner |
| This agent | Runs an identical-budget Optuna TPE / random-search track as the **baseline and control group**, then has the LLM reason over the same trial history a practitioner would read, while every accept/reject/select/converge decision is still deterministic code — the reasoning loop has to *earn* its cost every run by beating that baseline, not just once in a benchmark |

`OPENROUTER_API_KEY` is **optional**. Without it (or with `model=None`), the
agent runs in `--no-llm` mode: only the baseline sampler (Optuna TPE if
installed, else uniform random search) executes, under the identical search
space and trial budget, and the run still produces every artifact. This is the
same mechanism that serves as the control group when the LLM path *does* run —
one code path, two jobs.

## Quick start

```python
from hp_agent import HPAgent, TuningConfig

result = HPAgent(TuningConfig(
    model="anthropic/claude-sonnet-4",   # omit / None -> --no-llm mode
    model_family="lightgbm",
    trial_budget=25,
)).run(X, y, task="classification")

result.best_config              # dict
result.best_metric               # float
result.baseline_sampler_metric
result.report_path               # report.md
```

Loading config from the environment (`.env` + `OPENROUTER_*` vars) instead of
in-code:

```python
from hp_agent import HPAgent, TuningConfig

result = HPAgent(TuningConfig.from_env(model_family="random_forest")).run(
    X, y, task="regression",
)
```

## How it works

One agent, six deterministic-controller stages, plus a baseline track run under
the same budget:

```
SEED (LHS warm start) -> PROPOSE (LLM) -> VALIDATE + DEDUP + REPAIR
    -> EVALUATE (CV) -> SELECT / CONVERGE -> REPORT (LLM)
```

The LLM is called at exactly two points — propose and report. Training,
scoring, gating, and convergence are always deterministic code. A parallel
baseline track (Optuna TPE, else random search) runs under the identical
search space, CV protocol, and trial budget — it is both the `--no-llm`
fallback and the control group the final report compares the agent against.

1. **Seed** — a Latin-Hypercube space-filling design over the active
   `SearchSpace`, sized `k = max(5, round(0.2 × trial_budget))`, so the
   reasoning loop starts from a spread of real evidence instead of one
   arbitrary point.
2. **Propose** — the LLM reasons over the full trial history and returns one
   structured proposal plus a one-sentence rationale.
3. **Validate + dedup + repair** — checked against the search space's
   dynamically-generated Pydantic model; out-of-range/malformed responses
   retry once with the validation error appended; near-duplicates of a prior
   trial are rejected and fed back; two consecutive failures fall back to a
   deterministic perturbation of the current best config so the run always
   makes forward progress.
4. **Evaluate** — a model-family-aware CV harness (stratified/grouped folds,
   configurable metric, per-trial timeout) trains and scores the candidate.
5. **Select / converge** — a noise-floor- and overfit-aware selection rule
   (never a raw metric-max) and a patience + noise-floor convergence check.
6. **Report** — deterministic code computes rank-correlation hyperparameter
   influence and the lift over the baseline track; the LLM turns those
   numbers into a short narrative — it never computes them itself.

See `Agent 3.md` in this repo for the full design doc.

## Install

```bash
git clone <this-repo>
cd hp-agent

pip install -e .                          # or: pip install -e ".[lightgbm,optuna,dev]"
# LightGBM is the preferred booster for model_family="lightgbm" (falls back to
# scikit-learn's HistGradientBoosting if absent). Optuna is the preferred
# baseline sampler (falls back to uniform random search if absent).

cp .env.example .env                      # optional: add OPENROUTER_API_KEY
```

## Configuration

Everything is on `TuningConfig` (`hp_agent/config.py`). Common knobs:

| Field / env | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` / — | — | optional; unset -> `--no-llm` mode |
| `OPENROUTER_MODEL` / `model` | `None` | any JSON-capable OpenRouter model id |
| `model_family` | `lightgbm` | `random_forest` \| `lightgbm` \| `elasticnet` \| `logistic_regression` |
| `trial_budget` | `25` | total trials across seed + propose stages |
| `seed_fraction` | `0.2` | fraction of `trial_budget` spent on the warm-start LHS design |
| `n_folds` | `5` | CV folds |
| `metric` | `auto` | `average_precision` / `roc_auc` / `rmse` / `mae` |
| `group_column` | — | entity id -> `GroupKFold` (no entity straddles folds) |
| `trial_timeout_s` / `max_wall_time_s` | `120.0` / `900.0` | per-trial and per-run wall-clock ceilings |
| `max_cost_usd` | `1.00` | hard per-run LLM cost ceiling |
| `patience` / `convergence_window` / `convergence_tol` | `8` / `5` / `0.005` | convergence rule (§6.7 of the design doc) |
| `baseline_sampler` | `auto` | `optuna_tpe` if Optuna is installed, else `random_search` |

## Outputs & reproducibility

```
runs/<run_id>/
├── manifest.json            # dataset hash, config, model id, prompt hashes,
│                             #   seeds, package versions, cost, wall time,
│                             #   convergence reason
├── trials.jsonl             # every trial — seed, proposals, repairs,
│                             #   rejections (duplicate/invalid), baseline trials
├── best_config.json          # winning config + why it was selected
├── baseline_comparison.json  # sampler's best, agent's best, lift, trials-to-best
└── report.md                 # LLM narrative grounded in the above
```

`trials.jsonl` records rejected and repaired proposals too, not just accepted
ones — the audit trail a manual tuning session never produces.

## Tests

```bash
python -m pytest tests/
```

The suite runs fully offline (the LLM is stubbed/mocked): search-space property
tests, an adversarial propose-loop suite, the selection-rule and convergence
benchmarks, timeout/budget enforcement, and a `--no-llm` end-to-end run.

## Project layout

```
hp-agent/
├── hp_agent/
│   ├── config.py           # TuningConfig dataclass; YAML-loadable
│   ├── llm.py               # OpenRouter client wrapper: retries, JSON-schema
│   │                        #   enforcement, cost accounting
│   ├── search_space.py      # ParamSpec, SearchSpace, per-family registry
│   ├── adapters.py           # model-family adapters: RandomForest, LightGBM/
│   │                        #   HistGradientBoosting fallback, ElasticNet/LogisticRegression
│   ├── design.py             # Latin-Hypercube / Sobol warm-start sampler
│   ├── propose.py            # LLM proposal + Pydantic validation + dedup + repair loop
│   ├── evaluate.py           # CV harness: stratified/grouped folds, configurable metric,
│   │                        #   per-trial timeout, overfit gap
│   ├── select.py             # noise-floor + overfit-aware best-config rule, convergence
│   │                        #   decision (patience + noise floor + min-trials floor)
│   ├── baseline.py           # Optuna TPE sampler, else random-search fallback; also the
│   │                        #   full --no-llm path
│   ├── report.py             # hyperparameter-influence stats (Spearman, code-computed),
│   │                        #   LLM narrative writer
│   ├── orchestrator.py       # trial loop, budgets, manifest, public API
│   └── prompts/
│       ├── propose.md        # next-config proposal prompt template
│       └── report.md         # final narrative prompt template
├── tests/
├── Agent 3.md                # the design doc this implements
├── pyproject.toml
└── requirements.txt
```

All milestones in `Agent 3.md` §9 (M0-M4) are implemented: every module in the
layout above is real, `orchestrator.py` ties them together behind the single
`HPAgent.run()` call, and a `--no-llm` end-to-end run (baseline sampler only)
has been smoke-tested to produce every artifact in §7.

## What it deliberately does **not** do (v1)

Neural-network hyperparameter search, multi-objective tuning (accuracy vs.
latency vs. fairness simultaneously), distributed/parallel trial execution, and
AutoML-style model-family *selection* (this agent tunes a given family; picking
the family is a future agent's job). See §13 of `Agent 3.md`.
