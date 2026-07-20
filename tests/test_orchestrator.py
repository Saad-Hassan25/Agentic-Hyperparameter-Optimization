"""End-to-end orchestration (§10): the --no-llm full run, and one golden
live-LLM smoke test gated on OPENROUTER_API_KEY (skipped by default)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from hp_agent.config import TuningConfig
from hp_agent.llm import LLMClient
from hp_agent.orchestrator import HPAgent

_ARTIFACTS = ("manifest.json", "trials.jsonl", "best_config.json", "baseline_comparison.json", "report.md")


def test_no_llm_full_run_produces_every_artifact_and_never_touches_openai_client(
    monkeypatch, make_classification_df
):
    """§10's --no-llm test: a full run with model=None must complete using only
    baseline.py, produce every artifact in §7, and never even attempt to build
    the OpenAI client -- proven by making that construction path raise, then
    confirming the run still completes successfully."""

    def _must_not_be_called(self):
        raise AssertionError("LLMClient._ensure_client was called in --no-llm mode")

    monkeypatch.setattr(LLMClient, "_ensure_client", _must_not_be_called)

    X, y = make_classification_df(n=300, seed=0)
    run_dir = tempfile.mkdtemp(prefix="hp_agent_no_llm_")
    cfg = TuningConfig(
        model=None, model_family="random_forest", trial_budget=6, n_folds=3,
        trial_timeout_s=30, output_dir=run_dir, verbose=False,
    )

    result = HPAgent(cfg).run(X, y, task="classification")

    run_path = Path(result.run_dir)
    for fname in _ARTIFACTS:
        assert (run_path / fname).exists(), f"missing artifact {fname}"

    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_mode"] == "no_llm"
    assert manifest["model"]["report_source"] == "deterministic"

    assert isinstance(result.best_metric, float)
    assert isinstance(result.baseline_sampler_metric, float)
    assert result.lift_over_baseline == 0.0
    assert result.agent_trials == []  # --no-llm mode skips the agent-guided track entirely
    assert len(result.baseline_trials) == cfg.trial_budget

    trials_lines = (run_path / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trials_lines) == cfg.trial_budget
    for line in trials_lines:
        assert json.loads(line)["source"] == "baseline_sampler"

    best_config_payload = json.loads((run_path / "best_config.json").read_text(encoding="utf-8"))
    assert best_config_payload["best_config"] == result.best_config

    baseline_comparison = json.loads((run_path / "baseline_comparison.json").read_text(encoding="utf-8"))
    assert baseline_comparison["lift_over_baseline"] == 0.0


def test_rejected_proposals_are_persisted_to_trials_jsonl_not_just_duplicates(
    monkeypatch, make_classification_df
):
    """§7's own worked example ("num_leaves=310 ... rejected: exceeds
    2**max_depth" -- a constraint_violation, not a duplicate) requires every
    rejection kind in the audit trail, not only duplicates. Offline (no
    network): the fake LLM's first propose-stage response is out-of-range
    (invalid_schema), its second is valid -- and the orchestrator must write
    an `invalid_schema_rejected` row to trials.jsonl for the first, distinct
    from any accepted `ok` trial."""
    from hp_agent.propose import HyperparamProposal
    from hp_agent.report import _NarrativeOnly

    valid_cfg = {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "max_features": 0.5}
    invalid_cfg = {**valid_cfg, "n_estimators": 100_000}  # far above the 800 upper bound

    proposal_queue = [
        HyperparamProposal(values=invalid_cfg, reasoning="oops, out of range"),
        HyperparamProposal(values=valid_cfg, reasoning="corrected"),
    ]

    def _fake_structured(self, *, stage, system, user, schema, temperature, model=None, max_retries=2):
        if schema is HyperparamProposal:
            return proposal_queue.pop(0)
        assert schema is _NarrativeOnly
        return _NarrativeOnly(narrative="offline fake narrative")

    monkeypatch.setattr(LLMClient, "available", lambda self: (True, ""))
    monkeypatch.setattr(LLMClient, "structured", _fake_structured)

    X, y = make_classification_df(n=200, seed=3)
    run_dir = tempfile.mkdtemp(prefix="hp_agent_rejections_")
    cfg = TuningConfig(
        model="offline-test-model", model_family="random_forest", trial_budget=6,
        seed_fraction=0.2, n_folds=3, trial_timeout_s=30, output_dir=run_dir, verbose=False,
    )

    result = HPAgent(cfg).run(X, y, task="classification")

    trials_lines = (Path(result.run_dir) / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in trials_lines]
    rejected = [r for r in rows if r["status"] == "invalid_schema_rejected"]
    assert len(rejected) == 1, "the out-of-range proposal must be logged, not silently dropped"
    assert rejected[0]["config"]["n_estimators"] == 100_000
    assert rejected[0]["error"]  # the validation-error feedback text is preserved
    # and the corrected, accepted proposal still made it into the run as an ok trial
    assert any(r["status"] == "ok" and r["source"] == "llm_proposal" for r in rows)


def test_budget_exhausted_halts_agent_track_before_full_trial_budget(monkeypatch, make_classification_df):
    """§10 'Timeout/budget enforcement': "a run must halt at ... max_cost_usd
    even mid-trial-budget". Fully offline (no network): `LLMClient.available`
    and `.structured()` are faked so the agent-guided track actually starts,
    but a tiny `max_cost_usd` is spent by the very first propose-stage call,
    so the orchestrator's own budget check must halt the agent track — logged
    as `convergence.reason == "budget_exhausted"` — well short of `trial_budget`,
    never crashing and never silently running the full budget anyway."""
    from hp_agent.propose import HyperparamProposal
    from hp_agent.report import _NarrativeOnly
    from hp_agent.search_space import get_search_space

    space = get_search_space("random_forest")
    call_count = {"n": 0}

    def _fake_structured(self, *, stage, system, user, schema, temperature, model=None, max_retries=2):
        call_count["n"] += 1
        # Burn the entire cost ceiling on the very first call, regardless of
        # stage -- proves the *next* budget check halts the run, not that this
        # call itself was ever going to be rejected as invalid/duplicate.
        self.budget.add(prompt_tokens=50, completion_tokens=50, cost_usd=self.config.max_cost_usd)
        if schema is HyperparamProposal:
            frac = (call_count["n"] % 97) / 97.0  # deterministic spread; avoids dedup collisions
            values = {}
            for p in space.params:
                v = p.low + frac * (p.high - p.low)
                values[p.name] = int(round(v)) if p.kind == "int" else float(v)
            return HyperparamProposal(values=values, reasoning="offline fake proposal")
        assert schema is _NarrativeOnly
        return _NarrativeOnly(narrative="offline fake narrative")

    monkeypatch.setattr(LLMClient, "available", lambda self: (True, ""))
    monkeypatch.setattr(LLMClient, "structured", _fake_structured)

    X, y = make_classification_df(n=200, seed=2)
    run_dir = tempfile.mkdtemp(prefix="hp_agent_budget_")
    cfg = TuningConfig(
        model="offline-test-model", model_family="random_forest", trial_budget=20,
        seed_fraction=0.2, n_folds=3, trial_timeout_s=30, max_cost_usd=0.01,
        output_dir=run_dir, verbose=False,
    )

    result = HPAgent(cfg).run(X, y, task="classification")

    assert result.convergence is not None
    assert result.convergence.reason == "budget_exhausted"
    assert len(result.agent_trials) < cfg.trial_budget
    # the seed batch (space-filling design) still ran in full before the
    # budget-driven halt -- the agent track made forward progress, it just
    # didn't reach the end of its trial_budget.
    seed_k = max(5, round(cfg.seed_fraction * cfg.trial_budget))
    assert len(result.agent_trials) >= seed_k

    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["convergence"]["reason"] == "budget_exhausted"


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="golden live-LLM smoke test requires OPENROUTER_API_KEY; excluded from CI default (§10)",
)
def test_golden_live_llm_smoke_end_to_end(make_classification_df):
    """One live-LLM smoke test on a tiny budget: asserts artifact completeness
    and that lift_over_baseline is computed, never an exact metric value."""
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    X, y = make_classification_df(n=300, seed=1)
    run_dir = tempfile.mkdtemp(prefix="hp_agent_golden_")
    cfg = TuningConfig(
        model=model, model_family="lightgbm", trial_budget=6, n_folds=3,
        trial_timeout_s=60, max_cost_usd=0.25, output_dir=run_dir, verbose=False,
    )

    result = HPAgent(cfg).run(X, y, task="classification")

    run_path = Path(result.run_dir)
    for fname in _ARTIFACTS:
        assert (run_path / fname).exists(), f"missing artifact {fname}"
    assert isinstance(result.lift_over_baseline, float)
    assert isinstance(result.best_metric, float)
    assert result.report is not None
