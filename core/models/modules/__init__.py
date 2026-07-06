"""Neural network building blocks for TAAC models."""

# Re-export everything so `from core.models.modules import X` keeps working.

from core.models.modules.adaptive_domain_scaling import SimpleQueryBooster
from core.models.modules.context_moe import ExpertMLP
from core.models.modules.cross_network import FieldSENET, GDCNNetwork, GDCNSource, GTCLiteMixer
from core.models.modules.din import (
    MultiChunkBidirectionalDIN,
    TargetAwareDINHead,
)
from core.models.modules.heads import (
    AntiSignalCrossHead,
    ClassificationHead,
    CrossFusionHead,
    GroupHeads,
    ProfileExtraCrossHead,
    ProfileItemCrossHead,
    SemanticRouteHeads,
)
from core.models.modules.primitives import (
    FeedForwardNetwork,
    GatedFusion,
    GatedResidualNetwork,
    InstanceGuidedMask,
    RMSNorm,
    SwiGLU,
    VariableSelectionNetwork,
    build_activation,
    build_norm,
    ffn_activation,
)
from core.models.modules.routing import (
    ItemBankSourceProjector,
    ItemDenseRouter,
    RouteFeatureProjector,
)
from core.models.modules.seq_writer import SeqLocalWriter

__all__ = [
    "AntiSignalCrossHead",
    "ClassificationHead",
    "CrossFusionHead",
    "ExpertMLP",
    "FeedForwardNetwork",
    "FieldSENET",
    "GDCNNetwork",
    "GDCNSource",
    "GTCLiteMixer",
    "GatedFusion",
    "GatedResidualNetwork",
    "GroupHeads",
    "InstanceGuidedMask",
    "ItemBankSourceProjector",
    "ItemDenseRouter",
    "MultiChunkBidirectionalDIN",
    "ProfileExtraCrossHead",
    "ProfileItemCrossHead",
    "RMSNorm",
    "RouteFeatureProjector",
    "SemanticRouteHeads",
    "SeqLocalWriter",
    "SimpleQueryBooster",
    "SwiGLU",
    "TargetAwareDINHead",
    "VariableSelectionNetwork",
    "build_activation",
    "build_norm",
    "ffn_activation",
]
