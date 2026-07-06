"""Training callbacks: observer protocol and implementations."""

from core.training.callbacks.diagnostics.base import (
    REGISTRY,
    DiagBase,
    Diagnostics,
    parse_log,
)
from core.training.callbacks.diagnostics.codes_eval import (
    compute_calibration,
    compute_per_domain_auc,
)
from core.training.callbacks.protocol import ObserverProtocol
from core.training.callbacks.run_writer import RunWriter

__all__ = [
    "REGISTRY",
    "DiagBase",
    "Diagnostics",
    "ObserverProtocol",
    "RunWriter",
    "compute_calibration",
    "compute_per_domain_auc",
    "parse_log",
]
