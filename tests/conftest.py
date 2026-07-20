"""Make the package importable when running the tests without installing.

Also provides small synthetic-data factory fixtures shared across the suite
(§10: "synthetic data built inline or via a fixture rather than committed data
files"). Each fixture returns a *factory function* rather than fixed data, so
individual tests can control row count / seed / grouping without every test
module reinventing its own dataset builder.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# keep test output readable; the harness's numbers are unaffected
warnings.filterwarnings("ignore", message="X does not have valid feature names")


@pytest.fixture
def make_classification_df():
    """Factory: small synthetic binary-classification dataset -> (X, y)."""

    def _make(n: int = 300, n_features: int = 6, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=0,
            weights=[0.7, 0.3],
            random_state=seed,
        )
        cols = [f"f{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

    return _make


@pytest.fixture
def make_regression_df():
    """Factory: small synthetic regression dataset -> (X, y)."""

    def _make(n: int = 300, n_features: int = 6, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=n,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=5.0,
            random_state=seed,
        )
        cols = [f"f{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

    return _make


@pytest.fixture
def make_grouped_classification_df():
    """Factory: classification dataset where each entity (group) contributes
    several rows -> (X, y, group) -- for GroupKFold leakage checks (§6.5)."""

    def _make(
        n_groups: int = 12, rows_per_group: int = 5, seed: int = 0
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = n_groups * rows_per_group
        group = np.repeat(np.arange(n_groups), rows_per_group)
        group_effect = rng.normal(size=n_groups)[group]
        X = pd.DataFrame({
            "f0": group_effect + rng.normal(scale=0.3, size=n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        })
        y = (group_effect + rng.normal(scale=0.2, size=n) > 0).astype(int)
        return X, pd.Series(y, name="target"), group

    return _make
