"""Central configuration for the Hyperparameter Optimization agent.

Design principle (shared with `feature_agent` and `eda_agent`): every knob that
changes what the agent *does* — the search budget, the CV protocol, the LLM
connection, the convergence rule — lives here, documented and tunable, instead of
being scattered as magic numbers. A principal data scientist should be able to
read this file and know exactly how trials are proposed, evaluated, and selected.

The config is a plain dataclass so it is trivially constructable in code, loadable
from YAML, and serializable into the run manifest. `model=None` is the load-bearing
default: it means --no-llm mode, where the orchestrator never even attempts a
network call and the baseline sampler (Optuna TPE or random search) runs alone.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The four supported model families. Hardcoded here (rather than imported from
# search_space.py) so config.py has no dependency on search_space.py and the two
# modules can be imported in either order without a circular import.
_MODEL_FAMILIES = ("random_forest", "lightgbm", "elasticnet", "logistic_regression")


# --------------------------------------------------------------------------- #
# Minimal .env loader (no hard dependency on python-dotenv)
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class TuningConfig:
    """All settings for one run of the hyperparameter-tuning agent.

    Fields mirror the design doc (§8) and add the operational plumbing needed to
    actually run: the OpenRouter connection, budgets, and the evaluation/
    convergence rules for the trial loop.
    """

    # --- LLM (OpenRouter, OpenAI-compatible) --------------------------------- #
    model: str | None = None                      # None -> --no-llm mode: baseline sampler only
    proposal_temperature: float = 0.4              # diversity helps proposal exploration
    max_cost_usd: float = 1.00                     # hard per-run cost ceiling
    # Price estimates (USD per 1M tokens) used for the cost ceiling when the API
    # does not report an exact cost. Override per model.
    input_price_per_mtok: float = 3.0
    output_price_per_mtok: float = 15.0
    max_output_tokens: int = 4096
    request_timeout: float = 120.0
    # OpenRouter connection / attribution (usually read from env, see from_env).
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = ""
    app_name: str = "hp-agent"

    # --- Search --------------------------------------------------------------- #
    model_family: str = "lightgbm"                 # random_forest | lightgbm | elasticnet | logistic_regression
    trial_budget: int = 25
    seed_fraction: float = 0.2                      # fraction of trial_budget spent on the warm-start LHS design

    # --- Evaluation ------------------------------------------------------------ #
    n_folds: int = 5
    metric: str = "auto"                            # auto | average_precision | roc_auc | rmse | mae
    group_column: str | None = None
    trial_timeout_s: float = 120.0
    max_wall_time_s: float = 900.0
    random_state: int = 42
    max_eval_rows: int = 200_000                     # stratified subsample above this, mirrors feature_agent
    min_rows_warn_threshold: int = 500               # below this row count, log that the CV noise floor is wide

    # --- Convergence ------------------------------------------------------------ #
    patience: int = 8
    convergence_window: int = 5
    convergence_tol: float = 0.005
    min_trials_before_convergence: int = 10

    # --- Baseline --------------------------------------------------------------- #
    baseline_sampler: str = "auto"                   # auto -> optuna_tpe if optuna installed, else random_search

    # --- Output ------------------------------------------------------------------ #
    output_dir: str = "runs"                         # runs/<run_id> is created underneath
    run_id: str | None = None                        # explicit id (else derived from data hash + time)
    verbose: bool = True

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2 for cross-validation.")
        if not (0 < self.seed_fraction <= 1):
            raise ValueError("seed_fraction must be in (0, 1].")
        if self.trial_budget < 1:
            raise ValueError("trial_budget must be >= 1.")
        if self.model_family not in _MODEL_FAMILIES:
            raise ValueError(
                f"model_family must be one of {_MODEL_FAMILIES}, got {self.model_family!r}."
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **overrides: Any) -> "TuningConfig":
        """Build a config from environment / .env, then apply keyword overrides.

        Precedence: explicit overrides > environment > dataclass defaults.
        `None` overrides are ignored so callers can pass every field through blindly.
        """
        load_dotenv()
        cfg = cls(
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            site_url=os.getenv("OPENROUTER_SITE_URL", ""),
            app_name=os.getenv("OPENROUTER_APP_NAME", "hp-agent"),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown setting: {key}")
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str], **overrides: Any) -> "TuningConfig":
        """Load a config from a YAML file (requires PyYAML), env for secrets."""
        try:
            import yaml  # optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError("from_yaml needs PyYAML: pip install pyyaml") from exc
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_env(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serializable view for the run manifest (never emits the API key)."""
        d = asdict(self)
        d.pop("openrouter_api_key", None)
        return d
