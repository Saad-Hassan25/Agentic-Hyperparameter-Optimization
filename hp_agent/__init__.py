"""Agentic Hyperparameter Optimization agent -- see `Agent 3.md` for the design doc.

Public API is kept to one call (§5): construct a `TuningConfig`, hand it to
`HPAgent`, and call `.run(X, y, task=...)`. Every other module in this package
is an internal implementation detail of that one call.
"""

from __future__ import annotations

from .config import TuningConfig
from .orchestrator import HPAgent, HPAgentResult
from .evaluate import TrialOutcome
from .select import ConvergenceDecision
from .propose import HyperparamProposal
from .report import TuningReport
from .search_space import ParamSpec, SearchSpace

__all__ = [
    "TuningConfig",
    "HPAgent",
    "HPAgentResult",
    "TrialOutcome",
    "ConvergenceDecision",
    "HyperparamProposal",
    "TuningReport",
    "ParamSpec",
    "SearchSpace",
]

__version__ = "0.1.0"
