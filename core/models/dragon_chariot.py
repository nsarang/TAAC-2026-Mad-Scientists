"""DragonChariot: DIN-based CVR model with format-polymorphic embeddings.

Architecture:
1. NS tokenization: static categoricals -> per-entity embedder -> grouped projection -> tokens
2. Dense projection: numerical features -> token
3. Per-domain sequence embedding -> [B, L, D] tokens + masks
4. DIN target query: item router query or mean item token
5. Per-domain DIN attention -> context vectors + logits
6. Context fusion (concat_proj) -> seq_repr
7. Semantic routes or legacy multi-head aggregation

Format selection happens at init via the `format` arg. The embedding helpers
(StaticEmbedder, SeqEmbedder) are pure Tensor->Tensor modules. Schema
extraction happens here in forward().
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.data.masking import FeatureMaskingScheduler
from core.data.schema import FeatureSchema
from core.models.base import TrainableModel
from core.models.modules.adaptive_domain_scaling import (
    AdaptiveSequenceScaler,
    PersonalizedCandidateQueryGenerator,
    SimpleQueryBooster,
)
from core.models.modules.cross_network import GDCNSource
from core.models.modules.din import MultiChunkBidirectionalDIN, TargetAwareDINHead
from core.models.modules.embedders import ChunkedProjection, SeqEmbedder, StaticEmbedder
from core.models.modules.heads import (
    AntiSignalCrossHead,
    FlatGDSLHead,
    GroupHeads,
    ProfileExtraCrossHead,
    ProfileItemCrossHead,
    SemanticRouteHeads,
    two_layer_mlp,
)
from core.models.modules.pretext import PretextHeadV2
from core.models.modules.routing import (
    ItemBankSourceProjector,
    ItemDenseRouter,
    RouteFeatureProjector,
)
from core.models.modules.st_cnn import (
    _SHORT_TERM_CNN_PARAMS,
    ShortTermCausalCNN,
    resolve_short_term_cnn_params,
)

LOG = logging.getLogger(__name__)


def _normalise_context_slices(raw: dict[str, Any] = None) -> dict[str, tuple[int, int]]:
    """Normalize feature slice config into integer `(start, end)` pairs."""
    if raw is None:
        return {}
    return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in raw.items()}


def _make_repr_probes(names: list[str]) -> nn.ModuleDict:
    """Create named identity modules for hook-based representation diagnostics."""
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate representation probe names: {names}")
    return nn.ModuleDict({name: nn.Identity() for name in names})


def _build_adaptive_sequence_scaler(
    d_model: int,
    domains: list[str],
    cfg: dict[str, Any] = None,
) -> AdaptiveSequenceScaler | None:
    """Build the optional ADS-lite sequence scaler."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    enabled = bool(cfg.pop("enabled"))
    if not enabled:
        if cfg:
            raise ValueError(f"unknown adaptive_sequence_scaler keys: {sorted(cfg)}")
        return None
    scaler = AdaptiveSequenceScaler(
        d_model=d_model,
        num_domains=len(domains),
        hidden_mult=int(cfg.pop("hidden_mult")),
        dropout=float(cfg.pop("dropout")),
        cap=float(cfg.pop("cap")),
        scale_init=float(cfg.pop("scale_init")),
    )
    if cfg:
        raise ValueError(f"unknown adaptive_sequence_scaler keys: {sorted(cfg)}")
    return scaler


def _build_candidate_query_generator(
    d_model: int,
    domains: list[str],
    cfg: dict[str, Any] = None,
) -> PersonalizedCandidateQueryGenerator | None:
    """Build the optional PCRG-lite candidate query generator."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    enabled = bool(cfg.pop("enabled"))
    if not enabled:
        if cfg:
            raise ValueError(f"unknown candidate_query_generator keys: {sorted(cfg)}")
        return None
    generator = PersonalizedCandidateQueryGenerator(
        d_model=d_model,
        num_domains=len(domains),
        num_chunks=int(cfg.pop("num_chunks")),
        hidden_mult=int(cfg.pop("hidden_mult")),
        dropout=float(cfg.pop("dropout")),
        cap=float(cfg.pop("cap")),
        scale_init=float(cfg.pop("scale_init")),
        generator_type=str(cfg.pop("type", "shared")),
        private_cap=float(cfg.pop("private_cap")) if "private_cap" in cfg else None,
        private_hidden_mult=int(cfg.pop("private_hidden_mult"))
        if "private_hidden_mult" in cfg
        else None,
    )
    if cfg:
        raise ValueError(f"unknown candidate_query_generator keys: {sorted(cfg)}")
    return generator


def _build_query_booster(
    d_model: int,
    domains: list[str],
    cfg: dict[str, Any] = None,
    use_context: bool = False,
) -> SimpleQueryBooster | None:
    """Build the optional lightweight DIN score-query booster."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    enabled = bool(cfg.pop("enabled"))
    if not enabled:
        return None
    booster = SimpleQueryBooster(
        d_model=d_model,
        num_domains=len(domains),
        mode=str(cfg.pop("mode")),
        cap=float(cfg.pop("cap")),
        scale_init=float(cfg.pop("scale_init")),
        dropout=float(cfg.pop("dropout")),
        hidden_mult=int(cfg.pop("hidden_mult", 2)),
        apply_to_full=bool(cfg.pop("apply_to_full")),
        apply_to_windowed=bool(cfg.pop("apply_to_windowed")),
        recent_policy=str(cfg.pop("recent_policy")),
        zero_init_delta=bool(cfg.pop("zero_init_delta")),
        use_layernorm=bool(cfg.pop("use_layernorm")),
        use_context=use_context,
    )
    if cfg:
        raise ValueError(f"unknown query_boost keys: {sorted(cfg)}")
    return booster


def _masked_mean(tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool padded sequence tokens over a boolean keep mask."""
    w = keep_mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def _query_boost_score_queries(
    model: "DragonChariot",
    target_query: torch.Tensor,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    tb_ids: torch.Tensor,
    domain_idx: int,
    domain: str,
    context: torch.Tensor = None,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Build optional full and windowed score queries for one sequence domain."""
    booster = model.query_boost
    if booster is None:
        return None, None
    if tokens.dim() != 3:
        raise RuntimeError("query_boost currently requires padded sequence tokens")

    valid = ~padding_mask.bool()
    full_pool = _masked_mean(tokens, valid) if booster.mode == "sequence_pool" else None
    score_query = None
    if booster.apply_to_full:
        score_query = booster.forward_window(
            target_query,
            domain_idx=domain_idx,
            window_id="full",
            seq_pool=full_pool,
            context=context,
        )

    window_score_queries = None
    head = model.din_heads[domain]
    if booster.apply_to_windowed and getattr(head, "_windowed", False) and tb_ids is not None:
        valid_time = valid & (tb_ids > 0)
        recent_pool = None
        mid_pool = None
        old_pool = None
        if booster.mode == "sequence_pool":
            recent = valid_time & (tb_ids <= head._b_edges[0])
            mid = valid_time & (tb_ids > head._b_edges[0]) & (tb_ids <= head._b_edges[1])
            old = valid_time & (tb_ids > head._b_edges[-1])
            recent_pool = _masked_mean(tokens, recent)
            mid_pool = _masked_mean(tokens, mid)
            old_pool = _masked_mean(tokens, old)
        q_recent = booster.forward_window(
            target_query,
            domain_idx=domain_idx,
            window_id="recent",
            seq_pool=recent_pool,
            context=context,
        )
        q_mid = booster.forward_window(
            target_query,
            domain_idx=domain_idx,
            window_id="mid",
            seq_pool=mid_pool,
            context=context,
        )
        q_old = booster.forward_window(
            target_query,
            domain_idx=domain_idx,
            window_id="old",
            seq_pool=old_pool,
            context=context,
        )
        window_score_queries = (q_recent, q_mid, q_old)

    return score_query, window_score_queries


def _seq_repr_inner(
    model: "DragonChariot",
    seq_ids_list: list[torch.Tensor],
    lengths_list: list[torch.Tensor],
    tb_ids_list: list[torch.Tensor | None],
    target_query: torch.Tensor,
    state_list: list[torch.Tensor | None] = None,
    ads_context: torch.Tensor = None,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[tuple[torch.Tensor, torch.Tensor]],
    list[torch.Tensor],
]:
    """Pure-tensor seq repr computation (compile-friendly, no schema access).

    `state_list` carries pre-built writer state tensors [B, L, D] per domain
    (or None for domains without a writer or in content mode). It is computed
    outside this function to keep Python-heavy session loops out of the
    compiled region.
    """
    din_logits = []
    din_contexts = []
    seq_tokens_list: list[tuple[torch.Tensor, torch.Tensor]] = []

    for i, domain in enumerate(model.domains):
        # Embed sequence features into token vectors
        tokens, seq_info = model.seq_embedders[domain](seq_ids_list[i], lengths_list[i])

        # Add recency (time-bucket) embedding
        tb_ids = None
        if model.time_embedding is not None and tb_ids_list[i] is not None:
            tb_ids = tb_ids_list[i]
            if tokens.dim() == 3:
                tb_ids = tb_ids[:, : tokens.shape[1]]
            tokens = tokens + model.time_embedding(tb_ids)

        if model.adaptive_sequence_scaler is not None:
            tokens = model.adaptive_sequence_scaler(
                tokens,
                ads_context,
                domain_idx=i,
                lengths=lengths_list[i],
            )

        if model.training:
            tokens = model.emb_dropout(tokens)

        # Local writer: Conv1D gating conditioned on state
        score_bias = None
        if model.seq_local_writer is not None:
            if domain in model.seq_local_writer.convs:
                state = state_list[i] if state_list is not None else None
                tokens, score_bias = model.seq_local_writer.apply_conv(
                    domain, tokens, seq_info, state=state
                )

        score_query = None
        window_score_queries = None
        if model.candidate_query_generator is not None:
            score_query = model.candidate_query_generator(
                target_query,
                ads_context,
                domain_idx=i,
                seq_tokens=tokens,
                lengths=lengths_list[i],
            )
        elif model.query_boost is not None:
            score_query, window_score_queries = _query_boost_score_queries(
                model,
                target_query,
                tokens,
                seq_info,
                tb_ids,
                domain_idx=i,
                domain=domain,
                context=ads_context,
            )

        # Sequence encoding: Mamba/LSTM/Titans or DIN
        if model._seq_mode in ("mamba", "lstm", "titans"):
            if tokens.dim() != 3:
                raise RuntimeError(
                    f"{model._seq_mode.upper()} seq encoder requires padded collator "
                    "(3D tokens). Set data.collator_type=padded."
                )
            ctx = model.seq_encoders[domain](tokens, seq_info, target_query, score_bias=score_bias)
            logit = model.target_interactions[domain](ctx, target_query).squeeze(-1)
        else:
            logit, ctx, _entropy = model.din_heads[domain](
                target_query,
                tokens,
                seq_info,
                time_bucket_ids=tb_ids,
                score_bias=score_bias,
                score_query=score_query,
                window_score_queries=window_score_queries,
            )

        seq_tokens_list.append((tokens, seq_info))
        din_logits.append(logit)
        din_contexts.append(ctx)

    # Context fusion (MoE handled outside compiled boundary in _compute_seq_repr)
    if model._context_fusion == "add":
        seq_repr = sum(model.domain_projs[d](ctx) for d, ctx in zip(model.domains, din_contexts))
    elif model._context_fusion == "moe":
        seq_repr = torch.cat(din_contexts, dim=-1)  # defer MoE to caller
    else:
        seq_repr = model.context_proj(torch.cat(din_contexts, dim=-1))

    if model._short_term_cnn_domains:
        seq_repr_base = seq_repr
        for i, domain in enumerate(model.domains):
            if domain in model._short_term_cnn_domains:
                tokens, seq_info = seq_tokens_list[i]
                seq_repr = seq_repr + model.short_term_cnn[domain](tokens, seq_info, seq_repr_base)

    return seq_repr, din_logits, seq_tokens_list, din_contexts


class DragonChariot(TrainableModel):
    """DIN-based CVR model consuming FeatureSchema.

    Parameters
    ----------
    schema
        FeatureSchema with layout already set.
    d_model
        Token/hidden dimension.
    emb_dim
        Per-feature embedding dimension.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        emb_dim: int,
        hash_cardinality_threshold: int,
        hash_buckets: int,
        data_format: str,
        din_cfg: dict,
        logit_weight: float,
        head_mode: dict,
        learned_missingness: bool,
        user_cat_ns_bypass: bool,
        user_ns_tokens: int,
        item_ns_tokens: int,
        dropout_rate: float,
        user_dense_bypass: bool = False,
        use_tbe: bool = False,
        tbe_learning_rate: float = 0.01,
        pretext: dict = None,
        feature_masker: FeatureMaskingScheduler = None,
        output_bias_init: float = None,
        profile_item_cross: dict = None,
        item_dense_cfg: dict = None,
        dense_bn: str = None,
        seq_local_writer: dict = None,
        decorrelation: dict = None,
        moe: dict = None,
        sequence_chunked_projection: dict = None,
        learned_missingness_dense: list[str] = None,
    ) -> None:
        super().__init__()
        learned_missingness_dense = learned_missingness_dense or []
        self.schema = schema
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.user_cat_ns_bypass = user_cat_ns_bypass
        self.is_jagged = data_format == "flat"
        self._moe_balance_loss = None
        self._route_moe_balance_loss = None
        self._flat_gdsl_aux_losses: dict[str, torch.Tensor] = {}
        self.use_tbe = use_tbe and self.is_jagged
        self._tbe_learning_rate = tbe_learning_rate
        self.gdcn_source = None
        self._gdcn_source_name = None

        # Per-feature BatchNorm on selected continuous features
        # Uses BatchNorm1d(1) so all dims share one mean/var across B*L values
        if dense_bn:
            bn_specs = schema.query(dense_bn)
            self._dense_bns = nn.ModuleDict({spec.name: nn.BatchNorm1d(1) for spec in bn_specs})
        else:
            self._dense_bns = None

        # Static categorical embedders
        user_cat_specs = schema.query(
            "entity = 'user' and dtype = 'categorical' and scope = 'static' and source != 'metadata'"
        )
        item_cat_specs = schema.query(
            "entity = 'item' and dtype = 'categorical' and scope = 'static' and source != 'metadata'"
        )

        self.user_embedder = StaticEmbedder(
            user_cat_specs,
            emb_dim,
            use_ebc=self.is_jagged,
            hash_cardinality_threshold=hash_cardinality_threshold,
            hash_buckets=hash_buckets,
            learned_missingness=learned_missingness,
        )
        self.item_embedder = StaticEmbedder(
            item_cat_specs,
            emb_dim,
            use_ebc=self.is_jagged,
            hash_cardinality_threshold=hash_cardinality_threshold,
            hash_buckets=hash_buckets,
            learned_missingness=learned_missingness,
        )

        # Static categorical tokenization: chunked projection -> multiple d_model tokens.
        self.user_ns_proj = ChunkedProjection(
            emb_dim * len(user_cat_specs), d_model, user_ns_tokens
        )
        self.item_ns_proj = ChunkedProjection(
            emb_dim * len(item_cat_specs), d_model, item_ns_tokens
        )

        # Group heads (exclusive routing)
        hm = head_mode or {
            "context_fusion": "concat_proj",
            "base": True,
            "item_only": False,
            "fusion": False,
            "group_heads": [],
        }
        self._logit_head_activation = hm["logit_head_activation"]
        self._logit_head_tanh_scale = hm["logit_head_tanh_scale"]
        self._context_fusion = hm["context_fusion"]
        group_head_cfgs = hm["group_heads"]
        if group_head_cfgs:
            self.group_heads = GroupHeads(
                schema,
                group_head_cfgs,
                d_model,
                dropout_rate,
                logit_head_activation=self._logit_head_activation,
                logit_head_tanh_scale=self._logit_head_tanh_scale,
            )
        else:
            self.group_heads = None
        self._semantic_routes_cfg = hm.get("semantic_routes")
        semantic_cfg = self._semantic_routes_cfg or {}
        semantic_sources_cfg = hm.get("semantic_sources") or {}
        self._semantic_source_cfgs, self._semantic_route_sources = self._normalize_semantic_cfg(
            semantic_sources_cfg, semantic_cfg
        )
        self._flat_gdsl_cfg = hm.get("flat_gdsl")
        self.flat_gdsl_head = None
        self.repr_probes = _make_repr_probes(
            [
                "user_cat_ns",
                "item_cat_ns",
                "base_repr",
                "target_query_initial",
                "target_query",
                "seq_repr_pre_bypass",
                "seq_repr",
                "user_dense_repr",
                "item_router_all_repr",
                "item_sparse_repr",
                "item_dense_emb_repr",
                "item_dense_count_repr",
            ]
        )
        self.semantic_source_probes = _make_repr_probes(list(self._semantic_source_cfgs))

        # Profile-item cross head
        if profile_item_cross is not None:
            profile_item_cross_cfg = dict(profile_item_cross)
            self.profile_cross = ProfileItemCrossHead(
                schema=schema,
                d_model=d_model,
                dropout_rate=dropout_rate,
                logit_head_activation=self._logit_head_activation,
                logit_head_tanh_scale=self._logit_head_tanh_scale,
                **profile_item_cross_cfg,
            )
        else:
            self.profile_cross = None

        # Propagate din_cfg.query_sources into item_dense_cfg.routing so the full
        # DIN query is declared in one place (din_cfg) rather than split across configs.
        _din_query_sources_override = (din_cfg or {}).get("query_sources")
        if _din_query_sources_override is not None and item_dense_cfg is not None:
            item_dense_cfg = dict(item_dense_cfg)
            item_dense_cfg["routing"] = {
                **dict(item_dense_cfg["routing"]),
                "din_query_sources": list(_din_query_sources_override),
            }

        # Item dense router (split item dense fids with routing)
        if item_dense_cfg is not None:
            self.item_router = ItemDenseRouter(
                schema=schema,
                d_model=d_model,
                dropout_rate=dropout_rate,
                **item_dense_cfg,
            )
        else:
            self.item_router = None

        # Dense projection (exclude exclusive group features and profile cross features)
        cont_specs = schema.query(
            "dtype = 'numerical' and scope = 'static' and source != 'metadata'"
        )
        exclusive_spec_names = set()
        if self.group_heads is not None:
            exclusive_spec_names.update(self.group_heads.exclusive_feature_names)
        if self.profile_cross is not None:
            exclusive_spec_names.update(self.profile_cross.excluded_names)
        if self.item_router is not None:
            exclusive_spec_names.update(self.item_router.excluded_feature_names)

        self.semantic_feature_sources = nn.ModuleDict()
        self.semantic_profile_sources = nn.ModuleDict()
        self.semantic_profile_extra_sources = nn.ModuleDict()
        self.semantic_item_bank_sources = nn.ModuleDict()
        self._semantic_model_tensor_sources: dict[str, str] = {}
        self._ads_context_features = None
        self._ads_context_feature_names: list[str] = []
        self._ads_context_dims: list[int] = []
        self._ads_context_slices: dict[str, tuple[int, int]] = {}
        self.ads_context_projs = nn.ModuleDict()
        self.ads_context_merge = None
        for name, source_cfg in self._semantic_source_cfgs.items():
            source_cfg = dict(source_cfg)
            source_type = source_cfg.pop("type")
            if source_type == "model_tensor":
                self._semantic_model_tensor_sources[name] = source_cfg["tensor"]
            elif source_type == "feature_projector":
                projector = RouteFeatureProjector(
                    schema,
                    list(source_cfg.pop("exprs")),
                    d_model,
                    dropout_rate,
                    **source_cfg,
                )
                self.semantic_feature_sources[name] = projector
                exclusive_spec_names.update(projector.feature_names)
            elif source_type == "profile_item_cross":
                profile_source = ProfileItemCrossHead(
                    schema=schema,
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    logit_head_activation=self._logit_head_activation,
                    logit_head_tanh_scale=self._logit_head_tanh_scale,
                    **source_cfg,
                )
                self.semantic_profile_sources[name] = profile_source
                exclusive_spec_names.update(profile_source.excluded_names)
            elif source_type == "profile_extra_cross":
                profile_extra_source = ProfileExtraCrossHead(
                    schema=schema,
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    logit_head_activation=self._logit_head_activation,
                    logit_head_tanh_scale=self._logit_head_tanh_scale,
                    **source_cfg,
                )
                self.semantic_profile_extra_sources[name] = profile_extra_source
                exclusive_spec_names.update(profile_extra_source.excluded_names)
            elif source_type == "profile_anti_cross":
                anti_source = AntiSignalCrossHead(
                    schema=schema,
                    d_model=d_model,
                    dropout_rate=dropout_rate,
                    **source_cfg,
                )
                self.semantic_profile_extra_sources[name] = anti_source
                exclusive_spec_names.update(anti_source.excluded_names)
            elif source_type == "item_router_bank":
                bank_sources = list(source_cfg.get("sources") or source_cfg.get("keys"))
                self.semantic_item_bank_sources[name] = ItemBankSourceProjector(
                    bank_sources,
                    d_model,
                    dropout_rate,
                )
            elif source_type == "gdcn":
                self.gdcn_source = GDCNSource(
                    schema=schema,
                    emb_dim=emb_dim,
                    d_model=d_model,
                    n_user_tokens=user_ns_tokens,
                    n_item_tokens=item_ns_tokens,
                    **source_cfg,
                )
                self._gdcn_source_name = name
            else:
                raise ValueError(f"Unknown semantic source type {source_type!r} for {name!r}")

        # Validate item_router_bank sources against router availability.
        # emb/count keys only exist in item_bank when item_dense_cfg is set.
        _router_only_keys = {"emb", "count"}
        for _name, _bank_proj in self.semantic_item_bank_sources.items():
            _missing = set(_bank_proj.sources) & _router_only_keys
            if _missing and self.item_router is None:
                raise ValueError(
                    f"semantic source {_name!r} requests item_router_bank keys "
                    f"{sorted(_missing)} but item_dense_cfg is not set"
                )

        non_exclusive_cont = [s for s in cont_specs if s.name not in exclusive_spec_names]
        self._non_exclusive_cont_specs = non_exclusive_cont
        cont_dim = sum(s.dim for s in non_exclusive_cont)
        self.has_dense = cont_dim > 0
        if self.has_dense:
            _user_cont_names = {
                s.name
                for s in schema.query(
                    "entity = 'user' and dtype = 'numerical' "
                    "and scope = 'static' and source = 'original'"
                )
            }
            user_cont_specs = [s for s in non_exclusive_cont if s.name in _user_cont_names]
            other_cont_specs = [s for s in non_exclusive_cont if s.name not in _user_cont_names]
            self._user_dense_spec_names = [s.name for s in user_cont_specs]
            self._user_dense_spec_dims = [s.dim for s in user_cont_specs]
            self._other_dense_spec_names = [s.name for s in other_cont_specs]
            self._other_dense_spec_dims = [s.dim for s in other_cont_specs]
            user_cont_dim = sum(s.dim for s in user_cont_specs)
            other_cont_dim = sum(s.dim for s in other_cont_specs)

            self.has_user_dense = user_cont_dim > 0
            self.has_other_dense = other_cont_dim > 0
            if self.has_user_dense:
                user_names = ", ".join(f"'{s.name}'" for s in user_cont_specs)
                self._user_dense_expr = f"name in ({user_names}) and scope = 'static'"
                self.dense_proj_user = nn.Linear(
                    user_cont_dim, d_model, bias=not self.has_other_dense
                )
            if self.has_other_dense:
                other_names = ", ".join(f"'{s.name}'" for s in other_cont_specs)
                self._other_dense_expr = f"name in ({other_names}) and scope = 'static'"
                self.dense_proj_other = nn.Linear(other_cont_dim, d_model)
            self.dense_norm = nn.LayerNorm(d_model)

            # Learned null embeddings for high-null dense features.
            _offset = 0
            self._dense_null_offsets: dict[str, tuple[int, int]] = {}
            for spec in non_exclusive_cont:
                if spec.name in learned_missingness_dense:
                    self._dense_null_offsets[spec.name] = (_offset, spec.dim)
                _offset += spec.dim
            self._dense_null_params = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.zeros(dim))
                    for name, (_, dim) in self._dense_null_offsets.items()
                }
            )
        else:
            self.has_user_dense = False
            self.has_other_dense = False
            self._dense_null_offsets = {}
            self._dense_null_params = nn.ParameterDict()
            self._user_dense_spec_names = []
            self._user_dense_spec_dims = []
            self._other_dense_spec_names = []
            self._other_dense_spec_dims = []

        seq_domains = sorted(
            {s.domain for s in schema.query("scope = 'seq' and source != 'metadata'")}
        )

        if self._flat_gdsl_cfg is not None:
            flat_cfg = self._flat_gdsl_cfg
            flat_pathways = list(flat_cfg["pathways"])

            candidate_dims: dict[str, int] = {
                "seq_repr": d_model,
                "seq_item_repr": d_model,
                "user_cat_ns": d_model,
                "user_prior_repr": d_model,
                "base_repr": d_model,
                "base_user_item_repr": d_model,
                "item_cat_ns": d_model,
                "target_query": d_model,
                "item_router_all_repr": d_model,
                "item_sparse_repr": d_model,
                "__user_cat_flat": emb_dim * len(user_cat_specs),
                "__item_cat_flat": emb_dim * len(item_cat_specs),
                "__user_ns_tokens": user_ns_tokens * d_model,
                "__item_ns_tokens": item_ns_tokens * d_model,
            }
            for domain in seq_domains:
                candidate_dims[f"din_logit_{domain}"] = 1
            if self.item_router is not None:
                candidate_dims["item_dense_emb_repr"] = d_model
                candidate_dims["item_dense_count_repr"] = d_model
            if self.profile_cross is not None:
                candidate_dims["profile_item_cross_repr"] = d_model

            for source_name, source_cfg in self._semantic_source_cfgs.items():
                source_type = source_cfg["type"]
                if source_type == "model_tensor":
                    tensor_name = source_cfg["tensor"]
                    if tensor_name not in candidate_dims:
                        raise ValueError(
                            f"flat_gdsl semantic source {source_name!r} references unknown "
                            f"tensor {tensor_name!r}"
                        )
                    candidate_dims[source_name] = candidate_dims[tensor_name]
                else:
                    candidate_dims[source_name] = d_model

            unknown_pathways = [name for name in flat_pathways if name not in candidate_dims]
            if unknown_pathways:
                raise ValueError(
                    "Unknown head_mode.flat_gdsl.pathways entries: "
                    f"{unknown_pathways}. Available keys: {sorted(candidate_dims)}"
                )

            self.flat_gdsl_head = FlatGDSLHead(
                pathways=flat_pathways,
                pathway_dims=candidate_dims,
                mixer_cfg=flat_cfg["mixer"],
                mixer_type=flat_cfg["mixer_type"] if "mixer_type" in flat_cfg else "gdcn",
                path_norm=flat_cfg["path_norm"] if "path_norm" in flat_cfg else "none",
                d_cross=flat_cfg["d_cross"] if "d_cross" in flat_cfg else None,
                senet=bool(flat_cfg["senet"]) if "senet" in flat_cfg else False,
                senet_reduction=flat_cfg["senet_reduction"] if "senet_reduction" in flat_cfg else 3,
                senet_groups=flat_cfg["senet_groups"] if "senet_groups" in flat_cfg else 2,
            )

        # Per-domain sequence embedders
        self.domains = seq_domains
        self.seq_embedders = nn.ModuleDict()
        for domain in self.domains:
            specs = schema.query(f"scope = 'seq' and domain = '{domain}' and source = 'original'")
            self.seq_embedders[domain] = SeqEmbedder(
                specs,
                domain,
                emb_dim,
                d_model,
                use_ec=self.is_jagged and not self.use_tbe,
                use_tbe=self.use_tbe,
                tbe_learning_rate=tbe_learning_rate,
                hash_cardinality_threshold=hash_cardinality_threshold,
                hash_buckets=hash_buckets,
                return_jagged=self.is_jagged,
                learned_missingness=learned_missingness,
                chunked_projection=sequence_chunked_projection,
            )

        # Shared time bucket embedding (additive, post-projection, matches V1)
        # TODO (nsarang): consider per-domain time embeddings instead of shared
        tb_specs = schema.query(
            "name matches '*_time_bucket' and scope = 'seq' and source != 'metadata'"
        )
        if tb_specs:
            self.time_embedding = nn.Embedding(tb_specs[0].vocab_size, d_model, padding_idx=0)
        else:
            self.time_embedding = None

        # Per-domain sequence encoder: Mamba or DIN
        din_cfg = dict(din_cfg)
        seq_encoder_cfg = din_cfg.pop("seq_encoder", None)
        self._seq_mode = seq_encoder_cfg["type"] if seq_encoder_cfg else "din"
        adaptive_sequence_scaler_cfg = din_cfg.pop("adaptive_sequence_scaler", None)
        candidate_query_generator_cfg = din_cfg.pop("candidate_query_generator", None)
        query_boost_cfg = din_cfg.pop("query_boost", None)
        adaptive_sequence_scaler_cfg = (
            dict(adaptive_sequence_scaler_cfg) if adaptive_sequence_scaler_cfg is not None else None
        )
        candidate_query_generator_cfg = (
            dict(candidate_query_generator_cfg)
            if candidate_query_generator_cfg is not None
            else None
        )
        query_boost_cfg = dict(query_boost_cfg) if query_boost_cfg is not None else None
        candidate_enabled = bool(
            candidate_query_generator_cfg is not None and candidate_query_generator_cfg["enabled"]
        )
        query_boost_enabled = bool(query_boost_cfg is not None and query_boost_cfg["enabled"])
        if candidate_enabled and query_boost_enabled:
            raise ValueError(
                "din_cfg.query_boost and din_cfg.candidate_query_generator cannot both be enabled"
            )
        if query_boost_enabled and self.is_jagged:
            raise ValueError("din_cfg.query_boost currently requires padded data_format")

        context_cfgs = []
        for cfg in (adaptive_sequence_scaler_cfg, candidate_query_generator_cfg, query_boost_cfg):
            if cfg is None or not cfg["enabled"]:
                continue
            if "context_features" not in cfg:
                continue
            context_features = cfg.pop("context_features")
            context_slices = cfg.pop("context_slices", {})
            if context_features is not None:
                context_cfgs.append(
                    (str(context_features), _normalise_context_slices(context_slices))
                )
            elif context_slices:
                raise ValueError("context_slices requires context_features")
        if context_cfgs:
            self._ads_context_features = context_cfgs[0][0]
            self._ads_context_slices = context_cfgs[0][1]
            for other_features, other_slices in context_cfgs[1:]:
                if (
                    other_features != self._ads_context_features
                    or other_slices != self._ads_context_slices
                ):
                    raise ValueError(
                        "ADS/PCRG/query_boost context modules "
                        "must use the same context_features/context_slices"
                    )
            ads_specs = schema.query(self._ads_context_features)
            self._ads_context_feature_names = [spec.name for spec in ads_specs]
            self._ads_context_dims = [
                self._ads_context_slices[spec.name][1] - self._ads_context_slices[spec.name][0]
                if spec.name in self._ads_context_slices
                else spec.dim
                for spec in ads_specs
            ]
            if sum(self._ads_context_dims) <= 0:
                raise ValueError("adaptive_sequence_scaler context_features matched no features")
            self.ads_context_projs = nn.ModuleDict(
                {
                    spec.name: nn.Sequential(
                        nn.Linear(dim, d_model),
                        nn.LayerNorm(d_model),
                    )
                    for spec, dim in zip(ads_specs, self._ads_context_dims, strict=True)
                }
            )
            self.ads_context_merge = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
        self._short_term_cnn_domains: tuple[str, ...] = ()
        self.short_term_cnn = nn.ModuleDict()
        self._din_query_extra_cross: list[str] = []  # set below for DIN path

        if self._seq_mode in ("mamba", "lstm", "titans"):
            raise RuntimeError(f"Sequence encoder '{self._seq_mode}' is deprecated.")
        else:
            din_cfg.pop("query_sources", None)  # already propagated to item_dense_cfg.routing
            self._din_query_extra_cross: list[str] = list(
                din_cfg.pop("query_extra_cross", None) or []
            )
            for _cross_name in self._din_query_extra_cross:
                if _cross_name not in self.semantic_profile_extra_sources:
                    raise ValueError(
                        f"din_cfg.query_extra_cross source {_cross_name!r} not found in "
                        f"semantic_sources (must be profile_extra_cross or profile_anti_cross)"
                    )
            if self._din_query_extra_cross:
                self.din_query_extra_cross_scales = nn.ParameterDict(
                    {name: nn.Parameter(torch.zeros(1)) for name in self._din_query_extra_cross}
                )
            din_type = din_cfg.pop("type", "standard")
            domain_overrides = din_cfg.pop("domain_overrides", None)
            short_term_cnn_cfg = din_cfg.pop("short_term_cnn", None)
            windowed_cfg = din_cfg.pop("windowed", None)
            din_logit_head_activation = din_cfg.pop("logit_head_activation")
            din_logit_head_tanh_scale = din_cfg.pop("logit_head_tanh_scale")
            if windowed_cfg is not None:
                windowed_cfg = dict(windowed_cfg)
                domain_edges = windowed_cfg.pop("domain_edges_sec")
            use_time_bias = din_cfg.pop("time_bias", False)
            _raw_num_time_buckets = din_cfg.pop("num_time_buckets", 0)
            din_num_time_buckets = _raw_num_time_buckets if use_time_bias else 0
            time_in_score = din_cfg.pop("time_in_score", False)
            time_emb_dim = din_cfg.pop("time_emb_dim", 0)
            if short_term_cnn_cfg is not None:
                short_term_cnn_cfg = dict(short_term_cnn_cfg)
                st_domains = list(short_term_cnn_cfg["domains"])
                missing = [d for d in st_domains if d not in self.domains]
                if missing:
                    raise ValueError(f"short_term_cnn domains not present: {missing}")
                if self.is_jagged:
                    raise ValueError(
                        "short_term_cnn requires padded layout (data_format != 'flat')"
                    )
                if "defaults" in short_term_cnn_cfg or "per_domain" in short_term_cnn_cfg:
                    st_defaults = dict(short_term_cnn_cfg.get("defaults", {}))
                    st_per_domain = dict(short_term_cnn_cfg.get("per_domain", {}))
                else:
                    st_defaults = {k: short_term_cnn_cfg[k] for k in _SHORT_TERM_CNN_PARAMS}
                    st_per_domain = {}
                self._short_term_cnn_domains = tuple(st_domains)
                self.short_term_cnn = nn.ModuleDict(
                    {
                        domain: ShortTermCausalCNN(
                            d_model,
                            **resolve_short_term_cnn_params(st_defaults, st_per_domain, domain),
                        )
                        for domain in st_domains
                    }
                )
            self.din_heads = nn.ModuleDict()
            for domain in self.domains:
                if din_type == "multi_chunk":
                    head_cfg = {
                        **din_cfg,
                        "num_time_buckets": din_num_time_buckets,
                        "time_in_score": time_in_score,
                        "time_emb_dim": time_emb_dim,
                        "logit_head_activation": din_logit_head_activation,
                        "logit_head_tanh_scale": din_logit_head_tanh_scale,
                    }
                    if domain_overrides and domain in domain_overrides:
                        head_cfg.update(domain_overrides[domain])
                    self.din_heads[domain] = MultiChunkBidirectionalDIN(d_model, **head_cfg)
                else:
                    domain_windowed = None
                    if windowed_cfg is not None:
                        domain_windowed = {**windowed_cfg, "edges_sec": domain_edges[domain]}
                    self.din_heads[domain] = TargetAwareDINHead(
                        d_model,
                        windowed=domain_windowed,
                        num_time_buckets=din_num_time_buckets,
                        logit_head_activation=din_logit_head_activation,
                        logit_head_tanh_scale=din_logit_head_tanh_scale,
                        **din_cfg,
                    )

        self.adaptive_sequence_scaler = _build_adaptive_sequence_scaler(
            d_model=d_model,
            domains=self.domains,
            cfg=adaptive_sequence_scaler_cfg,
        )
        self.candidate_query_generator = _build_candidate_query_generator(
            d_model=d_model,
            domains=self.domains,
            cfg=candidate_query_generator_cfg,
        )
        self.query_boost = _build_query_booster(
            d_model=d_model,
            domains=self.domains,
            cfg=query_boost_cfg,
            use_context=self.ads_context_merge is not None,
        )
        if self.query_boost is not None and self.query_boost.apply_to_windowed:
            for domain, head in self.din_heads.items():
                if getattr(head, "_windowed", False) and len(head._b_edges) != 2:
                    raise ValueError(
                        "din_cfg.query_boost windowed mode expects exactly two "
                        f"window edges for {domain!r}, got {len(head._b_edges)}"
                    )

        # Context fusion
        if self._context_fusion == "add":
            self.domain_projs = nn.ModuleDict(
                {domain: nn.Linear(d_model, d_model) for domain in self.domains}
            )
        elif self._context_fusion == "moe":
            from core.models.modules.context_moe import ExpertMLP

            ctx_moe_cfg = moe["context_fusion"]
            self.context_proj = ExpertMLP(
                in_dim=d_model * len(self.domains),
                hidden_dim=d_model,
                out_dim=d_model,
                n_experts=ctx_moe_cfg["n_experts"],
                top_k=ctx_moe_cfg["top_k"],
                dropout_rate=dropout_rate,
                balance_weight=ctx_moe_cfg["balance_weight"],
            )
        else:
            self.context_proj = nn.Sequential(
                nn.Linear(d_model * len(self.domains), d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )

        # Multi-head logits (seq_head only used in non-semantic-routes/non-flat path)
        if self._semantic_routes_cfg is None and self.flat_gdsl_head is None:
            self.seq_head = two_layer_mlp(
                d_model,
                d_model,
                1,
                dropout_rate,
                activation=self._logit_head_activation,
                activation_scale=self._logit_head_tanh_scale,
            )
        self._base_enabled = hm["base"]
        if self._base_enabled:
            self.base_head = two_layer_mlp(
                d_model,
                d_model,
                1,
                dropout_rate,
                activation=self._logit_head_activation,
                activation_scale=self._logit_head_tanh_scale,
            )
        self._item_enabled = hm["item_only"]
        if self._item_enabled:
            self.item_head = two_layer_mlp(
                d_model,
                d_model,
                1,
                dropout_rate,
                activation=self._logit_head_activation,
                activation_scale=self._logit_head_tanh_scale,
            )

        # Fusion head: sees all repr branches concatenated
        self._fusion_enabled = hm["fusion"]
        if self._fusion_enabled:
            n_group_reprs = self.group_heads.n_reprs if self.group_heads is not None else 0
            n_reprs = 1 + int(self._base_enabled) + int(self._item_enabled) + n_group_reprs
            self.fusion_head = two_layer_mlp(
                d_model * n_reprs,
                d_model,
                1,
                dropout_rate,
                activation=self._logit_head_activation,
                activation_scale=self._logit_head_tanh_scale,
            )

        self.semantic_route_heads = None
        if self._semantic_routes_cfg is not None and self.flat_gdsl_head is None:
            route_moe_cfg = moe["route_heads"] if moe and moe["route_heads"] else None
            self.semantic_route_heads = SemanticRouteHeads(
                d_model=d_model,
                dropout_rate=dropout_rate,
                route_sources=self._semantic_route_sources,
                fusion=semantic_cfg["fusion"],
                bilinear_fusion=semantic_cfg["bilinear_fusion"],
                source_bilinear_fusion=semantic_cfg["source_bilinear_fusion"],
                moe_cfg=route_moe_cfg,
                logit_head_activation=self._logit_head_activation,
                logit_head_tanh_scale=self._logit_head_tanh_scale,
            )

        # Embedding dropout (applied to ns_tokens and seq_tokens before DIN)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # user_cat_ns_bypass: user categorical token bypasses base_repr, gets added to
        # seq_repr before heads instead (decouples user emb grads from DIN dynamics).
        if user_cat_ns_bypass:
            self.user_cat_ns_bypass_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.user_cat_ns_bypass_proj = None

        self.user_dense_bypass = user_dense_bypass and self.has_user_dense
        if self.user_dense_bypass:
            self.user_dense_bypass_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

        # DIN logit weight (0.0 disables DIN shortcut logit contribution)
        self._logit_weight = logit_weight

        # Xavier init on embedding tables (matches V1 EmbeddingMixin._init_params)
        self._init_embeddings(learned_missingness)

        # Initialize output bias from empirical log-odds (target final Linear in Sequential).
        # Skipped when route heads use MoE (each expert initializes its own bias).
        if output_bias_init is not None:
            if self.flat_gdsl_head is not None:
                self.flat_gdsl_head.init_output_bias(output_bias_init)
            elif self.semantic_route_heads is not None:
                if not self.semantic_route_heads._use_moe:
                    bias_route = (
                        "interest"
                        if "interest" in self.semantic_route_heads.route_heads
                        else self.semantic_route_heads.route_names[0]
                    )
                    nn.init.constant_(
                        self.semantic_route_heads.route_heads[bias_route][-1].bias,
                        output_bias_init,
                    )
            else:
                nn.init.constant_(self.seq_head[-1].bias, output_bias_init)

        # Feature masking
        self.feature_masker = feature_masker

        # Pretext head
        self.pretext_enabled = pretext is not None
        if self.pretext_enabled:
            self.pretext_head = PretextHeadV2(
                schema=schema,
                d_model=d_model,
                seq_domains=self.domains,
                **pretext,
            )
        else:
            self.pretext_head = None

        # Seq local writer: Conv1D gate applied to sequence tokens before DIN.
        # Only supported on the padded format path (tokens.dim() == 3).
        # SeqLocalWriter owns both modes (state and content). In state mode
        # it auto-discovers feature embeddings from the schema; in content mode
        # it creates convs without any state encoder.
        self.seq_local_writer = None
        self._slw_action_features: dict[str, str] = {}

        if seq_local_writer is not None:
            from core.models.modules.seq_writer import SeqLocalWriter

            slw = dict(seq_local_writer)
            action_feature = slw.pop("action_feature", {})
            enabled_domains = slw.pop("enabled_domains")

            self._slw_action_features = dict(action_feature)
            enabled_domains = [d for d in self.domains if d in enabled_domains]

            self.seq_local_writer = SeqLocalWriter(
                schema=schema,
                d_model=d_model,
                emb_dim=emb_dim,
                domains=enabled_domains,
                time_embedding=self.time_embedding,
                **slw,
            )

        # Decorrelation regularization on semantic source representations
        if decorrelation:
            self._decorr_weight = decorrelation["weight"]
            self._decorr_exclude_model_tensors = decorrelation["exclude_model_tensors"]
        else:
            self._decorr_weight = 0.0
            self._decorr_exclude_model_tensors = True

    @staticmethod
    def _normalize_semantic_cfg(
        source_cfgs: dict[str, Any],
        route_cfgs: dict[str, Any],
    ) -> tuple[dict[str, dict], dict[str, list[str]]]:
        """Normalize semantic source/route config.

        Each route entry must declare either ``sources`` (list of source names) or
        ``exprs`` (list of schema query strings auto-wrapped as a feature_projector source).
        """
        # cfg=None means the ablation config explicitly nulled out this source.
        sources = {name: dict(cfg) for name, cfg in (source_cfgs or {}).items() if cfg is not None}
        route_sources: dict[str, list[str]] = {}

        _non_route_keys = {"fusion", "bilinear_fusion", "source_bilinear_fusion"}
        for route_name, route_cfg in (route_cfgs or {}).items():
            if route_name in _non_route_keys:
                continue
            # route_cfg=None means the ablation config nulled out this route.
            if route_cfg is None:
                continue
            route_cfg = dict(route_cfg)
            if "sources" in route_cfg:
                route_sources[route_name] = list(route_cfg["sources"])
            elif "exprs" in route_cfg:
                source_name = f"{route_name}_features"
                sources[source_name] = {
                    "type": "feature_projector",
                    "exprs": list(route_cfg["exprs"]),
                }
                route_sources[route_name] = [source_name]
            else:
                raise ValueError(
                    f"semantic route {route_name!r} requires either 'sources' or 'exprs'"
                )

        preferred_order = ["interest", "attention", "creative", "convenience"]
        ordered = {name: route_sources[name] for name in preferred_order if name in route_sources}
        ordered.update({name: srcs for name, srcs in route_sources.items() if name not in ordered})
        return sources, ordered

    def _init_embeddings(self, learned_missingness: bool) -> None:
        """Xavier-normal init on all embedding tables.

        Zeros row 0 only when learned_missingness is False (row 0 = padding,
        should be inert). When learned_missingness is True, row 0 represents a
        trainable "missing" token and keeps its xavier-initialized value.
        """
        self.user_embedder.init_weights()
        self.item_embedder.init_weights()
        for embedder in self.seq_embedders.values():
            embedder.init_weights()
        if self.time_embedding is not None:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def enable_compile(self) -> None:
        """Compile the seq_repr inner loop (the model's compute-heavy region).

        Hoists schema.extract() outside the compiled boundary so dynamo sees
        only pure tensor ops — 0 graph breaks. Also compiles pretext DIN
        attention when pretext has its own heads (share_din=False).
        """
        compiled_components: list[str] = []
        self._compiled_seq_repr = torch.compile(_seq_repr_inner, dynamic=True)
        compiled_components.append("_seq_repr_inner")

        # GDCN input assembly uses schema extraction (outside compile-friendly region),
        # but the inner cross network is pure tensor compute and benefits from compile.
        if self.gdcn_source is not None:
            self.gdcn_source.network = torch.compile(self.gdcn_source.network, dynamic=True)
            compiled_components.append("gdcn_source.network")

        # Non-MoE context projection is part of the hot path and can be compiled directly.
        # MoE context projection remains outside compile due tuple return + side state.
        if self._context_fusion != "moe":
            self.context_proj = torch.compile(self.context_proj, dynamic=True)
            compiled_components.append("context_proj")

        if self.pretext_head is not None and not self.pretext_head.share_din:
            self.pretext_head.enable_compile()
            compiled_components.append("pretext_head")

        self._compiled_components = tuple(compiled_components)
        LOG.info("torch.compile enabled for: %s", ", ".join(compiled_components))

    def _lookup_action_embs(
        self,
        seq_ids_list: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Look up action embeddings from seq embedders for the writer.

        Returns a dict mapping domain → action embedding tensor [B, L, emb_dim]
        for domains that have an action_feature configured. Domains without one
        (or where the feature isn't present in the batch) are omitted.
        """
        action_embs: dict[str, torch.Tensor] = {}
        for i, domain in enumerate(self.domains):
            feat_name = self._slw_action_features.get(domain)
            if feat_name is None:
                continue
            tables = getattr(self.seq_embedders[domain], "tables", None)
            if tables is not None and feat_name in tables and feat_name in seq_ids_list[i]:
                action_embs[domain] = tables[feat_name](seq_ids_list[i][feat_name])
        return action_embs

    def _compute_ads_context(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Project configured user profile features for ADS-lite."""
        if self.ads_context_merge is None:
            return None
        parts = []
        for name, dim in zip(self._ads_context_feature_names, self._ads_context_dims, strict=True):
            x = self.schema.extract(
                batch,
                expr=f"name = '{name}' and scope = 'static' and source != 'metadata'",
                cat=True,
            )
            if name in self._ads_context_slices:
                start, end = self._ads_context_slices[name]
                x = x[:, start:end]
            if x.shape[-1] != dim:
                raise RuntimeError(
                    f"ADS context feature {name!r} expected dim {dim}, got {x.shape[-1]}"
                )
            parts.append(F.silu(self.ads_context_projs[name](x)))
        return self.ads_context_merge(torch.stack(parts, dim=0).sum(dim=0))

    def _compute_seq_repr(
        self,
        batch: dict[str, Any],
        target_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
    ]:
        """Per-domain DIN attention, context fusion, and user bypass.

        Extracts sequence tensors from `batch` (via schema, outside compiled
        region) then delegates to `_seq_repr_inner` for the GPU compute.

        Returns
        -------
        seq_repr
            Fused sequence representation [B, D].
        din_logits
            Per-domain DIN shortcut logits.
        seq_tokens_list
            Per-domain ``(tokens, seq_info)`` pairs. ``seq_info`` is a padding
            mask ``[B, L]`` (padded) or cu_seqlens ``[B+1]`` (flat/jagged).
            Consumers dispatch on ``tokens.dim()`` (3D vs 2D).
        din_contexts
            Per-domain DIN context vectors [B, D] (for pretext share_din).
        """
        schema = self.schema
        seq_ids_list = []
        lengths_list = []
        tb_ids_list = []
        for domain in self.domains:
            seq_ids_list.append(
                schema.extract(
                    batch, expr=f"scope = 'seq' and domain = '{domain}' and source = 'original'"
                )
            )
            lengths_list.append(batch[f"{domain}_len"])
            tb_ids_list.append(
                batch[f"{domain}_time_bucket"] if self.time_embedding is not None else None
            )

        state_list = None
        if self.seq_local_writer is not None:
            action_embs = self._lookup_action_embs(seq_ids_list)
            state_list = self.seq_local_writer.encode_states(
                self.domains, tb_ids_list, batch, action_embs=action_embs
            )

        ads_context = self._compute_ads_context(batch)
        needs_ads_context = (
            self.adaptive_sequence_scaler is not None
            or self.candidate_query_generator is not None
            or (
                self.query_boost is not None
                and self.query_boost.mode == "sequence_pool"
                and self.query_boost.context_proj is not None
            )
        )
        if needs_ads_context and ads_context is None:
            raise RuntimeError(
                "ADS/PCRG/query_boost context module is enabled but context is missing"
            )

        fn = self._compiled_seq_repr if hasattr(self, "_compiled_seq_repr") else _seq_repr_inner
        seq_repr, din_logits, seq_tokens_list, din_contexts = fn(
            self, seq_ids_list, lengths_list, tb_ids_list, target_query, state_list, ads_context
        )

        # MoE fusion runs outside compiled boundary (returns tuple, mutates state)
        if self._context_fusion == "moe":
            seq_repr, self._moe_balance_loss = self.context_proj(seq_repr)
        else:
            self._moe_balance_loss = None

        return seq_repr, din_logits, seq_tokens_list, din_contexts

    def _build_semantic_source_reprs(
        self,
        batch: dict[str, Any],
        tensor_reprs: dict[str, torch.Tensor],
        item_bank: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Materialize configured semantic sources for route/flat composition."""
        source_reprs = dict(tensor_reprs)

        for name, tensor_key in self._semantic_model_tensor_sources.items():
            if tensor_key not in tensor_reprs:
                raise ValueError(f"semantic source {name!r} missing tensor {tensor_key!r}")
            source_reprs[name] = tensor_reprs[tensor_key]

        for name, projector in self.semantic_feature_sources.items():
            source_reprs[name] = projector(batch)

        for name, profile_source in self.semantic_profile_sources.items():
            # TODO (siyang): cross_head and scale in ProfileItemCrossHead receive no gradient
            # here because _logit is discarded. Fix by either (a) adding a repr-only method that
            # stops before cross_head, or (b) adding _logit as a direct logit contribution.
            _logit, repr_ = profile_source.forward_with_repr(batch, tensor_reprs["target_query"])
            source_reprs[name] = repr_

        for name, profile_extra_source in self.semantic_profile_extra_sources.items():
            logit, repr_ = profile_extra_source.forward_with_repr(batch, item_bank)
            source_reprs[name] = repr_
            if profile_extra_source.logit_residual:
                source_reprs[f"__residual_logit__{name}"] = logit

        for name, bank_source in self.semantic_item_bank_sources.items():
            source_reprs[name] = bank_source(item_bank)

        if self.gdcn_source is not None:
            source_reprs[self._gdcn_source_name] = self.gdcn_source(tensor_reprs, batch)

        # Legacy top-level profile_item_cross still exposes the old semantic source id.
        if self.profile_cross is not None:
            _logit, profile_repr = self.profile_cross.forward_with_repr(
                batch, tensor_reprs["target_query"]
            )
            source_reprs["profile_item_cross_repr"] = profile_repr

        for name, probe in self.semantic_source_probes.items():
            if name in source_reprs:
                source_reprs[name] = probe(source_reprs[name])

        return source_reprs

    def _decorrelation_loss(self, source_reprs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean pairwise cosine similarity between semantic source representations.

        When `exclude_model_tensors` is True, model-tensor-vs-model-tensor pairs
        are excluded (their correlation is structural), but model-tensor-vs-projector
        pairs are still penalized — catching projectors that duplicate backbone signals.
        """
        model_tensor_names = set(self._semantic_model_tensor_sources.keys())
        names = []
        reprs = []
        for k, v in source_reprs.items():
            if k.startswith("__"):
                continue
            names.append(k)
            reprs.append(v)
        if len(reprs) < 2:
            return torch.tensor(0.0, device=reprs[0].device)
        stacked = torch.stack(reprs, dim=0)  # [N, B, D]
        normed = F.normalize(stacked, dim=-1)
        # Gram matrix averaged over batch: [N, N]
        sim = torch.einsum("nbd,mbd->nm", normed, normed) / stacked.shape[1]
        n = len(reprs)
        mask = torch.triu(torch.ones(n, n, device=sim.device), diagonal=1).bool()
        # Exclude model-tensor-vs-model-tensor pairs (structural correlation)
        if self._decorr_exclude_model_tensors:
            for i in range(n):
                for j in range(i + 1, n):
                    if names[i] in model_tensor_names and names[j] in model_tensor_names:
                        mask[i, j] = False
        if not mask.any():
            return torch.tensor(0.0, device=sim.device)
        return sim[mask].mean()

    def _compute_head_logits(
        self,
        seq_repr: torch.Tensor,
        base_repr: torch.Tensor,
        item_cat_ns: torch.Tensor,
        din_logits: list[torch.Tensor],
        batch: dict[str, Any],
        target_query: torch.Tensor,
        semantic_source_reprs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate all head logits into a single [B, 1] tensor.

        Sums conditional contributions from: seq_head, base_head, item_head,
        DIN attention logit (domain-averaged, scaled by ``_logit_weight``),
        group_heads, fusion_head, profile_cross, or flat_gdsl. Each branch activates
        based on its config flag. Also collects repr tensors from active
        branches for the optional fusion head's concatenated input.
        """
        if self.flat_gdsl_head is not None:
            if len(din_logits) != len(self.domains):
                raise ValueError(f"Expected {len(self.domains)} DIN logits, got {len(din_logits)}")
            flat_pathway_reprs = dict(semantic_source_reprs)
            for domain, din_logit in zip(self.domains, din_logits, strict=True):
                flat_pathway_reprs[f"din_logit_{domain}"] = din_logit.unsqueeze(-1)
            logits = self.flat_gdsl_head(flat_pathway_reprs)
            self._flat_gdsl_aux_losses = self.flat_gdsl_head.consume_aux_losses()
            return logits

        if self.semantic_route_heads is not None:
            self._flat_gdsl_aux_losses = {}
            head_out = self.semantic_route_heads(
                source_reprs=semantic_source_reprs,
            )
            if isinstance(head_out, tuple):
                logits, self._route_moe_balance_loss = head_out
            else:
                logits = head_out
                self._route_moe_balance_loss = None
            if self._logit_weight != 0.0:
                din_logit = sum(din_logits) / len(din_logits)
                logits = logits + self._logit_weight * din_logit.unsqueeze(-1)
            for _name, _residual_logit in semantic_source_reprs.items():
                if _name.startswith("__residual_logit__"):
                    logits = logits + _residual_logit
            return logits

        reprs = [seq_repr]
        logits = self.seq_head(seq_repr)
        if self._base_enabled:
            reprs.append(base_repr)
            logits = logits + self.base_head(base_repr)
        if self._item_enabled:
            reprs.append(item_cat_ns)
            logits = logits + self.item_head(item_cat_ns)
        if self._logit_weight != 0.0:
            din_logit = sum(din_logits) / len(din_logits)
            logits = logits + self._logit_weight * din_logit.unsqueeze(-1)

        # Group head logits
        if self.group_heads is not None:
            group_logit, group_reprs = self.group_heads(batch)
            logits = logits + group_logit
            reprs.extend(group_reprs)

        # Fusion head: concatenate all branch reprs → MLP → logit
        if self._fusion_enabled:
            fusion_repr = torch.cat(reprs, dim=-1)
            logits = logits + self.fusion_head(fusion_repr)

        # Profile-item cross interaction
        if self.profile_cross is not None:
            logits = logits + self.profile_cross(batch, target_query)

        return logits

    def forward(
        self, batch: dict[str, Any], labels: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass returning conversion logits [B] and auxiliary losses.

        Parameters
        ----------
        batch
            Collated batch dict from the data pipeline.
        labels
            Target labels, passed through for auxiliary losses that need them.

        Returns
        -------
        tuple
            ``(logits, aux_losses)`` where aux_losses is a dict of named loss
            tensors (empty when not training or pretext is disabled).

        Pipeline stages:
        1. Feature masking (training only) — stochastic input dropout, keeps
           unmasked copy for pretext targets.
        2. Static categorical embedding → non-sequence (NS) tokenization via
           chunked projection into [B, num_tokens, d_model].
        3. Token-level masking on NS tokens (training only).
        4. Dense feature projection + base repr assembly.
        5. Per-domain DIN attention with time embedding → fused seq_repr.
        6. Flat GDSL, semantic route, or legacy multi-head aggregation.
        7. Pretext auxiliary loss computation (training only).
        """
        schema = self.schema
        self._flat_gdsl_aux_losses = {}

        # Per-feature BatchNorm on selected continuous features
        # Reshape (B, dim) → (B, 1, dim) for BN1d(1), then back
        if self._dense_bns is not None:
            for name, bn in self._dense_bns.items():
                x = schema.extract(batch, names=name)
                x = bn(x.unsqueeze(1)).squeeze(1)
                schema.update(batch, f"name = '{name}'", x)

        # Feature masking: apply input-level masking, keep originals for pretext
        unmasked_batch = batch
        if self.training and self.feature_masker is not None:
            self.feature_masker.tick()
            batch = self.feature_masker.apply(batch)

        # Extract static categorical IDs
        user_ids = schema.extract(
            batch,
            expr="entity = 'user' and dtype = 'categorical' and scope = 'static' and source != 'metadata'",
        )
        item_ids = schema.extract(
            batch,
            expr="entity = 'item' and dtype = 'categorical' and scope = 'static' and source != 'metadata'",
        )

        # Embed static categoricals
        user_embs = self.user_embedder(user_ids)
        item_embs = self.item_embedder(item_ids)

        user_cat = torch.cat(list(user_embs.values()), dim=-1)
        item_cat = torch.cat(list(item_embs.values()), dim=-1)

        # NS tokenization: [B, num_tokens, d_model]
        user_ns_tokens = self.user_ns_proj(user_cat)  # [B, user_ns_tokens, D]
        item_ns_tokens = self.item_ns_proj(item_cat)  # [B, item_ns_tokens, D]
        user_cat_ns = user_ns_tokens.mean(dim=1)  # [B, D]
        item_cat_ns = item_ns_tokens.mean(dim=1)  # [B, D]

        # Token-level masking
        if self.training and self.feature_masker is not None:
            B = user_cat_ns.shape[0]
            keep = self.feature_masker.sample(["user", "item"], B=B, device=user_cat_ns.device)
            user_ns_tokens = user_ns_tokens * keep.get(
                "user", torch.ones(B, 1, 1, device=user_cat_ns.device)
            )
            item_ns_tokens = item_ns_tokens * keep.get(
                "item", torch.ones(B, 1, 1, device=item_cat_ns.device)
            )
            user_cat_ns = user_ns_tokens.mean(dim=1)
            item_cat_ns = item_ns_tokens.mean(dim=1)
        user_cat_ns = self.repr_probes["user_cat_ns"](user_cat_ns)
        item_cat_ns = self.repr_probes["item_cat_ns"](item_cat_ns)

        # Assemble ns_tokens as [B, N, D] then mean — keeps V1 weighting
        ns_parts = []
        if not self.user_cat_ns_bypass:
            ns_parts.append(user_ns_tokens)
        if self.has_dense:
            dense_parts = []
            if self.has_user_dense:
                user_cont = schema.extract(batch, expr=self._user_dense_expr, cat=True)
                if self._dense_null_params and self._user_dense_spec_dims:
                    user_chunks = list(torch.split(user_cont, self._user_dense_spec_dims, dim=-1))
                    for idx, name in enumerate(self._user_dense_spec_names):
                        if name not in self._dense_null_params:
                            continue
                        feat = user_chunks[idx]
                        null_mask = (feat == 0).all(-1)
                        if null_mask.any():
                            feat = feat.clone()
                            feat[null_mask] = self._dense_null_params[name]
                            user_chunks[idx] = feat
                    user_cont = torch.cat(user_chunks, dim=-1)
                user_dense_repr = self.dense_proj_user(user_cont)
                user_dense_repr = self.repr_probes["user_dense_repr"](user_dense_repr)
                dense_parts.append(user_dense_repr)
            if self.has_other_dense:
                other_cont = schema.extract(batch, expr=self._other_dense_expr, cat=True)
                if self._dense_null_params and self._other_dense_spec_dims:
                    other_chunks = list(
                        torch.split(other_cont, self._other_dense_spec_dims, dim=-1)
                    )
                    for idx, name in enumerate(self._other_dense_spec_names):
                        if name not in self._dense_null_params:
                            continue
                        feat = other_chunks[idx]
                        null_mask = (feat == 0).all(-1)
                        if null_mask.any():
                            feat = feat.clone()
                            feat[null_mask] = self._dense_null_params[name]
                            other_chunks[idx] = feat
                    other_cont = torch.cat(other_chunks, dim=-1)
                dense_parts.append(self.dense_proj_other(other_cont))
            dense_tok = F.silu(self.dense_norm(sum(dense_parts)))
            ns_parts.append(dense_tok.unsqueeze(1))

        # Item representation: routing bank or raw tokens
        if self.item_router is not None:
            item_token, target_query, item_bank = self.item_router.forward_with_bank(
                item_ns_tokens, batch
            )
            ns_parts.append(item_token)
        else:
            ns_parts.append(item_ns_tokens)
            target_query = item_cat_ns
            item_bank = {
                "all": item_cat_ns,
                "sparse": item_cat_ns,
                "query": target_query,
            }
        item_bank["all"] = self.repr_probes["item_router_all_repr"](item_bank["all"])
        item_bank["sparse"] = self.repr_probes["item_sparse_repr"](item_bank["sparse"])
        if "emb" in item_bank:
            item_bank["emb"] = self.repr_probes["item_dense_emb_repr"](item_bank["emb"])
        if "count" in item_bank:
            item_bank["count"] = self.repr_probes["item_dense_count_repr"](item_bank["count"])
        target_query = self.repr_probes["target_query_initial"](target_query)
        item_bank["query"] = target_query

        ns_tokens = torch.cat(ns_parts, dim=1)  # [B, total_ns_tokens, D]

        # Embedding dropout on ns_tokens (matches V1)
        if self.training:
            ns_tokens = self.emb_dropout(ns_tokens)

        base_repr = self.repr_probes["base_repr"](ns_tokens.mean(dim=1))

        # Inject extra cross-head reprs into target_query before DIN.
        # Each source is evaluated early (only needs batch + item_bank) and added
        # as a scaled residual. Scale init=0 means the base run is reproduced exactly
        # at step 0 — the scales learn to activate if the signal helps attention.
        if self._din_query_extra_cross:
            for aug_name in self._din_query_extra_cross:
                _, aug_repr = self.semantic_profile_extra_sources[aug_name].forward_with_repr(
                    batch, item_bank
                )
                target_query = target_query + self.din_query_extra_cross_scales[aug_name] * aug_repr
        target_query = self.repr_probes["target_query"](target_query)

        # Per-domain DIN attention → context fusion → seq_repr
        seq_repr, din_logits, seq_tokens_list, din_contexts = self._compute_seq_repr(
            batch, target_query
        )
        seq_repr = self.repr_probes["seq_repr_pre_bypass"](seq_repr)

        # TODO (nsarang): is adding user_cat_ns to seq_repr the right target?
        # It couples user identity into the sequence-attention output — consider
        # adding to base_repr or a dedicated pathway instead.
        if self.user_cat_ns_bypass:
            seq_repr = seq_repr + F.silu(self.user_cat_ns_bypass_proj(user_cat_ns))
        if self.user_dense_bypass:
            seq_repr = seq_repr + F.silu(self.user_dense_bypass_proj(user_dense_repr))
        seq_repr = self.repr_probes["seq_repr"](seq_repr)

        tensor_reprs = {
            "seq_repr": seq_repr,
            "seq_item_repr": seq_repr,
            "user_cat_ns": user_cat_ns,
            "user_prior_repr": user_cat_ns,
            "base_repr": base_repr,
            "base_user_item_repr": base_repr,
            "item_cat_ns": item_cat_ns,
            "target_query": target_query,
            "item_router_all_repr": item_bank["all"],
            "item_sparse_repr": item_bank["sparse"],
            "__user_cat_flat": user_cat,
            "__item_cat_flat": item_cat,
            "__user_ns_tokens": user_ns_tokens,
            "__item_ns_tokens": item_ns_tokens,
        }
        if self.has_user_dense:
            tensor_reprs["user_dense_repr"] = user_dense_repr
        if "emb" in item_bank:
            tensor_reprs["item_dense_emb_repr"] = item_bank["emb"]
        if "count" in item_bank:
            tensor_reprs["item_dense_count_repr"] = item_bank["count"]
        need_semantic_reprs = (
            self.semantic_route_heads is not None or self.flat_gdsl_head is not None
        )
        semantic_source_reprs = (
            self._build_semantic_source_reprs(batch, tensor_reprs, item_bank)
            if need_semantic_reprs
            else {}
        )

        # Aggregate head logits
        logits = self._compute_head_logits(
            seq_repr,
            base_repr,
            item_cat_ns,
            din_logits,
            batch,
            target_query,
            semantic_source_reprs,
        )

        logits = logits.squeeze(-1)

        # Pretext: predict static features from sequence representations
        if self.pretext_enabled and self.training:
            tokens_list = [t for t, _ in seq_tokens_list]
            info_list = [info for _, info in seq_tokens_list]
            aux = self.pretext_head(
                tokens_list,
                info_list,
                unmasked_batch,
                domain_contexts=din_contexts if self.pretext_head.share_din else None,
            )
        else:
            aux = {}

        # Decorrelation: penalize cosine similarity between semantic sources
        if self.training and self._decorr_weight > 0 and semantic_source_reprs:
            aux["decorrelation"] = self._decorr_weight * self._decorrelation_loss(
                semantic_source_reprs
            )

        # MoE load balance losses
        if self.training and self._moe_balance_loss is not None:
            aux["moe_context_balance"] = self._moe_balance_loss
        if self.training and self._route_moe_balance_loss is not None:
            aux["moe_route_balance"] = self._route_moe_balance_loss
        if self.training and self._flat_gdsl_aux_losses:
            aux.update(self._flat_gdsl_aux_losses)

        return logits, aux

    def pretext_trainable_params(self) -> set[int]:
        """Return data_ptrs of params that should remain unfrozen during pretext phase."""
        ptrs: set[int] = set()
        for emb in self.seq_embedders.values():
            for p in emb.parameters():
                ptrs.add(p.data_ptr())
        if self.pretext_head is not None:
            for p in self.pretext_head.parameters():
                ptrs.add(p.data_ptr())
            if self.pretext_head.share_din:
                for p in self.din_heads.parameters():
                    ptrs.add(p.data_ptr())
        return ptrs

    def update_learning_rate(self, lr: float = None) -> None:
        """Sync learning rate to all TBE instances (must be called before forward).

        When `lr` is None, uses the model's configured tbe_learning_rate.
        No-op when use_tbe is False.
        """
        if not self.use_tbe:
            return
        effective_lr = lr if lr is not None else self._tbe_learning_rate
        for embedder in self.seq_embedders.values():
            if embedder.use_tbe:
                embedder.tbe.set_learning_rate(effective_lr)

    def get_sparse_params(self) -> list[nn.Parameter]:
        """Return all embedding parameters (for Adagrad).

        TBE weights are self-optimized buffers and excluded — they don't appear
        as nn.Parameters.
        """
        sparse_ptrs: set[int] = set()
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                sparse_ptrs.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_ptrs]

    def get_dense_params(self) -> list[nn.Parameter]:
        """Return all non-embedding parameters (for AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int) -> set[int]:
        """Reinitialize embeddings for features above `cardinality_threshold`.

        Returns set of data_ptr() for reinitialized parameters (for EMA reset).
        """
        reinitialized: set[int] = set()
        reinitialized |= self.user_embedder.reinit_high_cardinality(cardinality_threshold)
        reinitialized |= self.item_embedder.reinit_high_cardinality(cardinality_threshold)
        for embedder in self.seq_embedders.values():
            reinitialized |= embedder.reinit_high_cardinality(cardinality_threshold)
        if self.seq_local_writer is not None:
            reinitialized |= self.seq_local_writer.reinit_high_cardinality(cardinality_threshold)
        return reinitialized

    def _embedding_submodules(self) -> list:
        """All submodules that participate in snapshot/restore/reinit."""
        subs = list(self.seq_embedders.values())
        if self.seq_local_writer is not None:
            subs.append(self.seq_local_writer)
        return subs

    def snapshot_low_cardinality_embs(self, vocab_threshold: int) -> dict[str, torch.Tensor]:
        """Clone embedding weights for tables with vocab <= threshold.

        Each submodule owns its own key namespace and prefixes keys itself.
        """
        snapshot = {}
        for sub in self._embedding_submodules():
            sub_snapshot = sub.snapshot_weights(vocab_threshold)
            collisions = snapshot.keys() & sub_snapshot.keys()
            if collisions:
                raise RuntimeError(
                    f"Snapshot key collision: {collisions}. "
                    "Submodules must use disjoint key prefixes."
                )
            snapshot.update(sub_snapshot)
        return snapshot

    def restore_emb_snapshot(self, snapshot: dict[str, torch.Tensor]) -> set[int]:
        """Restore previously snapshotted embedding weights.

        Each submodule filters the snapshot for its own prefix. Returns
        restored data_ptrs.
        """
        restored_ptrs: set[int] = set()
        for sub in self._embedding_submodules():
            restored_ptrs |= sub.restore_weights(snapshot)
        return restored_ptrs
