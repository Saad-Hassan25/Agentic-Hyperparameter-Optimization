"""Model-family adapters — the only module that knows sklearn constructor kwargs.

`search_space.py` declares *what* is tunable per family; this module is the sole
place that maps a validated `{param_name: value}` config onto a real, unfitted
estimator. Keeping that mapping in one file means adding a fifth family later
(§6.2's XGBoost example) is one new `SearchSpace` plus one new adapter here —
nothing else in the package needs to change.

LightGBM is optional: if it isn't installed we fall back to scikit-learn's
HistGradientBoosting, the same fallback `feature_agent.evaluate` uses so both
agents behave identically in a LightGBM-less environment.
"""

from __future__ import annotations

import abc

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False


class ModelAdapter(abc.ABC):
    """Turns a flat config dict + task type into an unfitted estimator."""

    model_family: str
    supported_tasks: tuple[str, ...]

    @property
    def backend(self) -> str:
        """Informational label for the manifest, e.g. 'lightgbm' | 'histgb' | 'sklearn'."""
        return "sklearn"

    @abc.abstractmethod
    def build_estimator(self, config: dict, task: str, random_state: int):
        """Build an unfitted sklearn-compatible estimator from a validated config."""


class RandomForestAdapter(ModelAdapter):
    model_family = "random_forest"
    supported_tasks = ("classification", "regression")

    def build_estimator(self, config: dict, task: str, random_state: int):
        kwargs = dict(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            max_features=config["max_features"],
            random_state=random_state,
            n_jobs=1,  # single-threaded: keeps the per-trial timeout and fit_time comparable across trials
        )
        if task == "classification":
            return RandomForestClassifier(**kwargs)
        return RandomForestRegressor(**kwargs)


class LightGBMAdapter(ModelAdapter):
    model_family = "lightgbm"
    supported_tasks = ("classification", "regression")

    @property
    def backend(self) -> str:
        return "lightgbm" if _HAS_LGBM else "histgb"

    def build_estimator(self, config: dict, task: str, random_state: int):
        if _HAS_LGBM:
            kwargs = dict(
                num_leaves=config["num_leaves"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                n_estimators=config["n_estimators"],
                min_child_samples=config["min_child_samples"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_lambda=config["reg_lambda"],
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
                deterministic=True,
                force_row_wise=True,
            )
            if task == "classification":
                return LGBMClassifier(**kwargs)
            return LGBMRegressor(**kwargs)

        # HistGradientBoosting fallback: subsample/colsample_bytree have no
        # equivalent knob here and are deliberately dropped in this path only —
        # the search space still defines and tunes them whenever the real
        # LightGBM backend is active.
        kwargs = dict(
            max_leaf_nodes=config["num_leaves"],
            max_depth=config["max_depth"] if config["max_depth"] > 0 else None,
            learning_rate=config["learning_rate"],
            max_iter=config["n_estimators"],
            min_samples_leaf=config["min_child_samples"],
            l2_regularization=config["reg_lambda"],
            random_state=random_state,
        )
        if task == "classification":
            return HistGradientBoostingClassifier(**kwargs)
        return HistGradientBoostingRegressor(**kwargs)


class ElasticNetAdapter(ModelAdapter):
    model_family = "elasticnet"
    supported_tasks = ("regression",)

    def build_estimator(self, config: dict, task: str, random_state: int):
        return ElasticNet(
            alpha=config["alpha"],
            l1_ratio=config["l1_ratio"],
            random_state=random_state,
        )


class LogisticRegressionAdapter(ModelAdapter):
    model_family = "logistic_regression"
    supported_tasks = ("classification",)

    def build_estimator(self, config: dict, task: str, random_state: int):
        return LogisticRegression(
            C=config["C"],
            l1_ratio=config["l1_ratio"],
            penalty="elasticnet",
            solver="saga",
            random_state=random_state,
            max_iter=2000,
        )


# --------------------------------------------------------------------------- #
# Registry (§6.2): keys must exactly match search_space.SEARCH_SPACES
# --------------------------------------------------------------------------- #
ADAPTERS: dict[str, ModelAdapter] = {
    "random_forest": RandomForestAdapter(),
    "lightgbm": LightGBMAdapter(),
    "elasticnet": ElasticNetAdapter(),
    "logistic_regression": LogisticRegressionAdapter(),
}


def get_adapter(model_family: str) -> ModelAdapter:
    """Look up a registered `ModelAdapter` by model family name."""
    try:
        return ADAPTERS[model_family]
    except KeyError:
        raise ValueError(
            f"Unknown model_family {model_family!r}. Valid options: {list(ADAPTERS)}"
        ) from None


def validate_family_for_task(model_family: str, task: str) -> None:
    """Raise if `model_family` does not support `task` (e.g. elasticnet + classification)."""
    adapter = get_adapter(model_family)
    if task not in adapter.supported_tasks:
        raise ValueError(
            f"model_family {model_family!r} does not support task {task!r}; "
            f"it supports {adapter.supported_tasks}."
        )
