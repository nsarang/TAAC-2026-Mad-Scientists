"""Structured diagnostics using the ``DIAG|EVENT|context;;CODE:payload`` line protocol."""

from core.training.callbacks.diagnostics.base import (
    REGISTRY,
    DiagBase,
    Diagnostics,
    parse_log,
)
from core.training.callbacks.diagnostics.codes_compass import CompassCode
from core.training.callbacks.diagnostics.codes_data import (
    DenseStatsCode,
    LabelDistCode,
    OobCode,
    SeqLensCode,
)
from core.training.callbacks.diagnostics.codes_din import DINAttnCode
from core.training.callbacks.diagnostics.codes_eval import (
    LogitDistCode,
    LossConcCode,
    MetricsCode,
    PredCode,
    compute_calibration,
    compute_per_domain_auc,
)
from core.training.callbacks.diagnostics.codes_gdcn import GdcnCrossCode, GdcnGateCode
from core.training.callbacks.diagnostics.codes_meta import (
    DatasetCode,
    DoneCode,
    EnvCode,
    ModelCode,
    SchemaCode,
)
from core.training.callbacks.diagnostics.codes_model import (
    AttnCode,
    DomainGeomCode,
    EffRankCode,
    EmbRankCode,
    EmbUtilCode,
    GateStatsCode,
    GradFlowCode,
    LaneCosineCode,
    LayerHealthCode,
    TinStatsCode,
)
from core.training.callbacks.diagnostics.codes_optim import (
    GradCode,
    LrCode,
    OptStateCode,
    ReinitCode,
)
from core.training.callbacks.diagnostics.codes_perf import (
    ThroughputCode,
    TimingCode,
)
from core.training.callbacks.diagnostics.codes_repr import RepresentationProbeCode
from core.training.callbacks.diagnostics.codes_sage import SageCode
from core.training.callbacks.diagnostics.context import (
    EpochContext,
    StepContext,
    _compute_grouped_norms,
)

__all__ = [
    "REGISTRY",
    "AttnCode",
    "CompassCode",
    "DINAttnCode",
    "DatasetCode",
    "DenseStatsCode",
    "DiagBase",
    "Diagnostics",
    "DomainGeomCode",
    "DoneCode",
    "EffRankCode",
    "EmbRankCode",
    "EmbUtilCode",
    "EnvCode",
    "EpochContext",
    "GateStatsCode",
    "GdcnCrossCode",
    "GdcnGateCode",
    "GradCode",
    "GradFlowCode",
    "LabelDistCode",
    "LaneCosineCode",
    "LayerHealthCode",
    "LogitDistCode",
    "LossConcCode",
    "LrCode",
    "MetricsCode",
    "ModelCode",
    "OobCode",
    "OptStateCode",
    "PredCode",
    "ReinitCode",
    "RepresentationProbeCode",
    "SageCode",
    "SchemaCode",
    "SeqLensCode",
    "StepContext",
    "ThroughputCode",
    "TimingCode",
    "TinStatsCode",
    "_compute_grouped_norms",
    "compute_calibration",
    "compute_per_domain_auc",
    "parse_log",
]
