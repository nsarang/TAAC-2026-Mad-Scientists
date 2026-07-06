"""Classification and prediction heads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.data.schema import FeatureSchema
from core.models.modules.cross_network import FieldSENET, GDCNNetwork, GTCLiteMixer
from core.models.modules.primitives import build_activation


def two_layer_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    *,
    activation: str = "silu",
    activation_scale: float = None,
) -> nn.Sequential:
    """2-layer MLP: Linear → LN → activation → Dropout → Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        build_activation(activation, scaled_tanh_scale=activation_scale),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _make_repr_probes(names: list[str]) -> nn.ModuleDict:
    """Create named identity modules for hook-based representation diagnostics."""
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate representation probe names: {names}")
    return nn.ModuleDict({name: nn.Identity() for name in names})


class ClassificationHead(nn.Module):
    """MLP classification head with optional layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | Sequence[int],
        *,
        output_dim: int = 1,
        activation: str = "gelu",
        dropout: float | Sequence[float] = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        resolved_hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
        if not resolved_hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
            dropout_schedule = [float(v) for v in dropout]
        else:
            dropout_schedule = [float(dropout)] * len(resolved_hidden_dims)

        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        current_dim = input_dim
        for next_dim, next_dropout in zip(resolved_hidden_dims, dropout_schedule, strict=True):
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(build_activation(activation))
            layers.append(nn.Dropout(next_dropout))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project `hidden_states` to logits."""
        return self.layers(hidden_states)


class CrossFusionHead(nn.Module):
    """Target-aware domain gating + DCN-v2 cross layers + MLP classifier.

    Replaces the additive multi-head design with:
    1. Per-domain Q pooling (mean over queries per domain)
    2. Separate user/item NS mean-pooling
    3. Item-gated domain weighting (which domains matter for this item)
    4. DCN-v2 cross layers on [user_repr, item_repr, seq_repr]
    5. Single MLP classifier

    Parameters
    ----------
    d_model
        Token dimension.
    num_sequences
        Number of sequence domains (e.g. 4).
    action_num
        Number of output logits.
    num_cross_layers
        Number of stacked DCN-v2 cross layers.
    dropout
        Dropout rate in classifier MLP and cross layers.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        action_num: int,
        num_cross_layers: int = 2,
        cross_rank: int = 0,
        dropout: float = 0.1,
        use_seq_evo: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.use_seq_evo = use_seq_evo

        # Item-conditioned domain gate: item_repr → gate per domain
        self.domain_gate = nn.Linear(d_model, num_sequences)

        # Evolved-sequence domain gate (separate from Q-token gate)
        if use_seq_evo:
            self.evo_domain_gate = nn.Linear(d_model, num_sequences)

        # DCN-v2 cross layers on [user, item, seq_q, (seq_evo)] vector
        cross_dim = d_model * (4 if use_seq_evo else 3)
        if cross_rank > 0:
            # Low-rank factorization: W = U @ V instead of full (cross_dim, cross_dim)
            self.cross_U = nn.ModuleList(
                [nn.Linear(cross_dim, cross_rank, bias=False) for _ in range(num_cross_layers)]
            )
            self.cross_V = nn.ModuleList(
                [nn.Linear(cross_rank, cross_dim) for _ in range(num_cross_layers)]
            )
        else:
            self.cross_weights = nn.ModuleList(
                [nn.Linear(cross_dim, cross_dim) for _ in range(num_cross_layers)]
            )
        self.cross_rank = cross_rank
        self.cross_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_cross_layers)])
        self.cross_norm = nn.LayerNorm(cross_dim)

        # Single MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(cross_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_num),
        )

    def forward(
        self,
        final_qs: list[torch.Tensor],
        final_ns: torch.Tensor,
        num_user_tokens: int,
        final_seqs: list[torch.Tensor] = None,
        final_seq_masks: list[torch.Tensor] = None,
        domain_reprs: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        final_qs
            List of S tensors, each (B, Nq, D) — per-domain Q tokens after blocks.
            Ignored when `domain_reprs` is provided directly.
        final_ns
            (B, num_ns, D) — NS tokens after blocks.
        num_user_tokens
            Number of user-side NS tokens (user_ns + user_dense).
        final_seqs
            List of S tensors, each (B, L_i, D) — evolved sequence tokens after blocks.
            Only used when `use_seq_evo=True`.
        final_seq_masks
            List of S tensors, each (B, L_i) — True = padding.
            Only used when `use_seq_evo=True`.
        domain_reprs
            (B, S, D) — pre-computed per-domain representations (e.g. from DIN
            contexts). When provided, skips Q-token pooling.

        Returns
        -------
        logits
            (B, action_num)
        seq_repr
            (B, D) for downstream consumers (contrastive, DIN gate, return_embedding).
        """
        # Per-domain mean pool: (B, S, D)
        if domain_reprs is None:
            domain_reprs = torch.stack([q.mean(dim=1) for q in final_qs], dim=1)

        # Separate user/item NS pooling
        user_repr = final_ns[:, :num_user_tokens].mean(dim=1)  # (B, D)
        item_repr = final_ns[:, num_user_tokens:].mean(dim=1)  # (B, D)

        # Item-gated domain weighting for Q tokens
        gate = torch.sigmoid(self.domain_gate(item_repr))  # (B, S)
        seq_repr = (gate.unsqueeze(-1) * domain_reprs).sum(1)  # (B, D)

        if self.use_seq_evo and final_seqs is not None:
            # Masked mean-pool evolved sequence tokens per domain
            evo_reprs = []
            for seq_tok, seq_mask in zip(final_seqs, final_seq_masks):
                valid = (~seq_mask).unsqueeze(-1).float()  # (B, L, 1)
                pooled = (seq_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                evo_reprs.append(pooled)
            evo_domain_reprs = torch.stack(evo_reprs, dim=1)  # (B, S, D)
            evo_gate = torch.sigmoid(self.evo_domain_gate(item_repr))  # (B, S)
            evo_repr = (evo_gate.unsqueeze(-1) * evo_domain_reprs).sum(1)  # (B, D)
            x = torch.cat([user_repr, item_repr, seq_repr, evo_repr], dim=-1)  # (B, 4D)
        else:
            x = torch.cat([user_repr, item_repr, seq_repr], dim=-1)  # (B, 3D)

        # DCN-v2 cross layers
        x_0 = x
        if self.cross_rank > 0:
            for U, V, drop in zip(self.cross_U, self.cross_V, self.cross_dropouts):
                x = x_0 * drop(V(U(x))) + x
        else:
            for W, drop in zip(self.cross_weights, self.cross_dropouts):
                x = x_0 * drop(W(x)) + x
        x = self.cross_norm(x)

        logits = self.classifier(x)
        return logits, seq_repr


class ProfileItemCrossHead(nn.Module):
    """Profile-item cross-interaction head.

    Projects per-feature profile vectors, forms interactions with the item
    vector, and produces a scaled logit.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        dropout_rate: float,
        features: str,
        include_profile_vec: bool,
        include_item_vec: bool,
        use_product: bool,
        use_abs_diff: bool,
        hidden_mult: int,
        scale_init: float,
        remove_from_dense_ns: bool,
        slices: dict[str, Any] = None,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        """Build profile projections, interaction layers, and scoring MLP.

        When `remove_from_dense_ns` is True, the features selected by
        `features` are excluded from the shared dense NS projection in the
        parent model (exposed via ``excluded_names``).
        """
        super().__init__()
        self._schema = schema

        profile_specs = schema.query(features)
        self._spec_names = [s.name for s in profile_specs]
        self._slices = ProfileExtraCrossHead._normalise_slices(slices or {})
        self._raw_dims = [s.dim for s in profile_specs]
        self._dims = [
            ProfileExtraCrossHead._effective_dim(s.name, s.dim, self._slices) for s in profile_specs
        ]
        self._remove_from_dense_ns = remove_from_dense_ns

        self.norms = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        for spec, eff_dim in zip(profile_specs, self._dims):
            self.norms[spec.name] = nn.LayerNorm(eff_dim)
            self.projs[spec.name] = nn.Linear(eff_dim, d_model)

        n_cross_parts = (
            int(include_profile_vec) + int(include_item_vec) + int(use_product) + int(use_abs_diff)
        )
        self._include_profile_vec = include_profile_vec
        self._include_item_vec = include_item_vec
        self._use_product = use_product
        self._use_abs_diff = use_abs_diff

        self.cross_repr = nn.Sequential(
            nn.Linear(d_model * n_cross_parts, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        hidden = d_model * hidden_mult
        self.cross_head = two_layer_mlp(
            d_model,
            hidden,
            1,
            dropout_rate,
            activation=logit_head_activation,
            activation_scale=logit_head_tanh_scale,
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    @property
    def excluded_names(self) -> set[str]:
        """Feature names to exclude from the dense projection, if configured."""
        if self._remove_from_dense_ns:
            return set(self._spec_names)
        return set()

    def forward_with_repr(
        self, batch: dict[str, Any], item_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute profile-item cross logit and representation."""
        names_str = ", ".join(f"'{n}'" for n in self._spec_names)
        raw = self._schema.extract(
            batch,
            expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
            cat=True,
        )
        splits = torch.split(raw, self._raw_dims, dim=-1)
        parts = [
            self.projs[name](
                self.norms[name](ProfileExtraCrossHead._apply_slice(name, x, self._slices))
            )
            for name, x in zip(self._spec_names, splits)
        ]
        profile_vec = sum(parts) / len(parts) if len(parts) > 1 else parts[0]
        profile_vec = F.silu(profile_vec)

        pieces = []
        if self._include_profile_vec:
            pieces.append(profile_vec)
        if self._include_item_vec:
            pieces.append(item_vec)
        if self._use_product:
            pieces.append(profile_vec * item_vec)
        if self._use_abs_diff:
            pieces.append(torch.abs(profile_vec - item_vec))
        cross = torch.cat(pieces, dim=-1)
        repr_ = self.cross_repr(cross)
        return self.scale * self.cross_head(repr_), repr_

    def forward(self, batch: dict[str, Any], item_vec: torch.Tensor) -> torch.Tensor:
        """Compute profile-item cross interaction logit [B, 1]."""
        logit, _repr = self.forward_with_repr(batch, item_vec)
        return logit


class ProfileExtraCrossHead(nn.Module):
    """Low-impact parallel profile-item interaction for extra pretrained user embeddings.

    Supports two clean ablation patterns:
    - user feature slicing (e.g. user_cont_f130[:259], user_cont_f131[:256])
    - item feature slicing when ``item_source: feature`` (e.g. item_cont_f129[:128])

    Design:
    - per-fid LayerNorm -> Linear for direction stabilization
    - learned user/item alignment before product, avoiding a hard same-space assumption
    - small gate bias and small output scale for safe residual addition
    - optional norm side channel only for gating, not for the main vector
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        dropout_rate: float,
        features: str,
        hidden_mult: int = 2,
        align_dim: int = None,
        item_source: str = "emb",
        item_features: str = None,
        use_norm_side: bool = False,
        norm_scale: float = 6.0,
        gate_bias_init: float = -3.0,
        scale_init: float = 0.01,
        remove_from_dense_ns: bool = False,
        logit_residual: bool = True,
        slices: dict[str, Any] = None,
        item_slices: dict[str, Any] = None,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        super().__init__()
        self._schema = schema
        profile_specs = schema.query(features)
        if not profile_specs:
            raise ValueError(f"ProfileExtraCrossHead matched no features for expr: {features!r}")

        self._spec_names = [s.name for s in profile_specs]
        self._raw_dims = [s.dim for s in profile_specs]
        self._slices = self._normalise_slices(slices or {})
        self._dims = [self._effective_dim(s.name, s.dim, self._slices) for s in profile_specs]
        self._remove_from_dense_ns = remove_from_dense_ns
        self.item_source = item_source
        self.use_norm_side = use_norm_side
        self.norm_scale = float(norm_scale)
        self.logit_residual = bool(logit_residual)
        align_dim = align_dim or d_model

        self.norms = nn.ModuleDict()
        self.user_projs = nn.ModuleDict()
        self.user_aligns = nn.ModuleDict()
        self.item_aligns = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.cross_mlps = nn.ModuleDict()

        # Optional direct item feature projection. This is intentionally separate
        # from item_bank so a CR branch can use item_cont_f129[:128] without
        # changing the main item router or the existing EA 123/132 branch.
        self._item_feature_expr = item_features
        self._item_feature_names: list[str] = []
        self._item_raw_dims: list[int] = []
        self._item_slices = self._normalise_slices(item_slices or {})
        self._item_dims: list[int] = []
        self.item_norms = nn.ModuleDict()
        self.item_projs = nn.ModuleDict()
        if self.item_source == "feature" or item_features is not None:
            if not item_features:
                raise ValueError(
                    "ProfileExtraCrossHead with item_source='feature' requires item_features"
                )
            item_specs = schema.query(item_features)
            if not item_specs:
                raise ValueError(
                    f"ProfileExtraCrossHead matched no item features for expr: {item_features!r}"
                )
            self.item_source = "feature"
            self._item_feature_names = [s.name for s in item_specs]
            self._item_raw_dims = [s.dim for s in item_specs]
            self._item_dims = [
                self._effective_dim(s.name, s.dim, self._item_slices) for s in item_specs
            ]
            for spec, eff_dim in zip(item_specs, self._item_dims):
                self.item_norms[spec.name] = nn.LayerNorm(eff_dim)
                self.item_projs[spec.name] = nn.Linear(eff_dim, d_model)

        gate_in_dim = d_model * 2 + (1 if self.use_norm_side else 0)
        cross_in_dim = d_model * 2 + align_dim
        for spec, eff_dim in zip(profile_specs, self._dims):
            self.norms[spec.name] = nn.LayerNorm(eff_dim)
            self.user_projs[spec.name] = nn.Linear(eff_dim, d_model)
            self.user_aligns[spec.name] = nn.Linear(d_model, align_dim)
            self.item_aligns[spec.name] = nn.Linear(d_model, align_dim)
            self.gates[spec.name] = nn.Sequential(
                nn.Linear(gate_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, 1),
            )
            # sigmoid(-3) ~= 0.047 — branch must earn its way in.
            nn.init.constant_(self.gates[spec.name][-1].bias, gate_bias_init)
            self.cross_mlps[spec.name] = nn.Sequential(
                nn.Linear(cross_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )

        self.output_norm = nn.LayerNorm(d_model)
        hidden = d_model * hidden_mult
        self.cross_head = two_layer_mlp(
            d_model,
            hidden,
            1,
            dropout_rate,
            activation=logit_head_activation,
            activation_scale=logit_head_tanh_scale,
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    @staticmethod
    def _normalise_slices(raw: dict[str, Any]) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        for name, val in raw.items():
            if val is None:
                continue
            if isinstance(val, dict):
                start = int(val.get("start", 0))
                end = int(val["end"])
            else:
                start = int(val[0])
                end = int(val[1])
            result[str(name)] = (start, end)
        return result

    @staticmethod
    def _effective_dim(name: str, raw_dim: int, slices: dict[str, tuple[int, int]]) -> int:
        if name not in slices:
            return raw_dim
        start, end = slices[name]
        if start < 0 or end <= start or end > raw_dim:
            raise ValueError(f"Invalid slice for {name}: [{start}, {end}) with raw_dim={raw_dim}")
        return end - start

    @staticmethod
    def _apply_slice(
        name: str, x: torch.Tensor, slices: dict[str, tuple[int, int]]
    ) -> torch.Tensor:
        if name not in slices:
            return x
        start, end = slices[name]
        return x[..., start:end]

    @property
    def excluded_names(self) -> set[str]:
        """Feature names to exclude from dense projection, only when explicitly requested."""
        if self._remove_from_dense_ns:
            return set(self._spec_names)
        return set()

    def _select_item_vec(
        self, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.item_source == "feature":
            names_str = ", ".join(f"'{n}'" for n in self._item_feature_names)
            raw = self._schema.extract(
                batch,
                expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
                cat=True,
            )
            splits = torch.split(raw, self._item_raw_dims, dim=-1)
            parts = []
            for name, x in zip(self._item_feature_names, splits):
                x = self._apply_slice(name, x, self._item_slices)
                parts.append(F.silu(self.item_projs[name](self.item_norms[name](x))))
            return sum(parts) / len(parts) if len(parts) > 1 else parts[0]

        if self.item_source not in item_bank:
            raise ValueError(
                f"ProfileExtraCrossHead requested item_source={self.item_source!r}, "
                f"but item_bank has keys {sorted(item_bank)}"
            )
        return item_bank[self.item_source]

    def forward_with_repr(
        self, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute small residual logit and representation for extra user embeddings."""
        item_vec = self._select_item_vec(batch, item_bank)
        names_str = ", ".join(f"'{n}'" for n in self._spec_names)
        raw = self._schema.extract(
            batch,
            expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
            cat=True,
        )
        splits = torch.split(raw, self._raw_dims, dim=-1)

        weighted_parts = []
        gates = []
        for name, x in zip(self._spec_names, splits):
            x = self._apply_slice(name, x, self._slices)
            # Keep direction stable via LayerNorm; handle norm magnitude only through
            # an explicit scalar side channel when enabled.
            u = F.silu(self.user_projs[name](self.norms[name](x)))
            aligned_prod = self.user_aligns[name](u) * self.item_aligns[name](item_vec)
            cross = self.cross_mlps[name](torch.cat([u, item_vec, aligned_prod], dim=-1))

            gate_inputs = [u, item_vec]
            if self.use_norm_side:
                norm_feat = torch.log1p(torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True))
                norm_feat = (norm_feat / self.norm_scale).to(dtype=u.dtype, device=u.device)
                gate_inputs.append(norm_feat)
            gate = torch.sigmoid(self.gates[name](torch.cat(gate_inputs, dim=-1)))
            weighted_parts.append(gate * cross)
            gates.append(gate)

        denom = torch.stack(gates, dim=0).sum(dim=0).clamp_min(1e-3)
        repr_ = sum(weighted_parts) / denom
        repr_ = self.output_norm(repr_)
        logit = self.scale * self.cross_head(repr_)
        return logit, repr_

    def forward(self, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logit only, discarding intermediate representations."""
        logit, _repr = self.forward_with_repr(batch, item_bank)
        return logit


class AntiSignalCrossHead(nn.Module):
    """Anti-signal head via cosine similarity with a negative logit contribution.

    Implements:

    .. code-block:: text

        u = normalize(Wu(LN(user_feature)))      [B, align_dim]
        v = normalize(Wv(LN(item_feature)))      [B, align_dim]
        s = dot(u, v)                            [B, 1]  (= cosine, both normalized)
        gate = sigmoid(Linear(cat[u, v]))        [B, 1]
        anti_logit = -softplus(alpha) * gate * s [B, 1]

    High user-item alignment → negative logit contribution.
    `softplus(alpha)` is always positive; alpha is a learnable scalar.
    Gate starts near-closed (gate_bias_init < 0) and must earn its way in.
    Representation is LN(Linear(u * v)) — hadamard product captures per-dim
    alignment and feeds the route attention like any other semantic source.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        dropout_rate: float,
        features: str,
        align_dim: int = None,
        item_source: str = "emb",
        item_features: str = None,
        gate_bias_init: float = -3.0,
        alpha_init: float = 0.0,
        remove_from_dense_ns: bool = False,
        logit_residual: bool = True,
        slices: dict[str, Any] = None,
        item_slices: dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self._schema = schema
        profile_specs = schema.query(features)
        if not profile_specs:
            raise ValueError(f"AntiSignalCrossHead matched no features for expr: {features!r}")

        self._spec_names = [s.name for s in profile_specs]
        self._raw_dims = [s.dim for s in profile_specs]
        self._slices = ProfileExtraCrossHead._normalise_slices(slices or {})
        self._dims = [
            ProfileExtraCrossHead._effective_dim(s.name, s.dim, self._slices) for s in profile_specs
        ]
        self._remove_from_dense_ns = remove_from_dense_ns
        self.item_source = item_source
        self.logit_residual = bool(logit_residual)
        align_dim = align_dim or d_model

        # Learnable amplitude — softplus keeps it positive
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.norms = nn.ModuleDict()
        self.user_projs = nn.ModuleDict()  # eff_dim → align_dim
        self.gates = nn.ModuleDict()  # 2*align_dim → 1
        self.repr_projs = nn.ModuleDict()  # align_dim → d_model

        # Item feature projection (projects to align_dim, then normalized in forward)
        self._item_feature_names: list[str] = []
        self._item_raw_dims: list[int] = []
        self._item_slices = ProfileExtraCrossHead._normalise_slices(item_slices or {})
        self._item_dims: list[int] = []
        self.item_norms = nn.ModuleDict()
        self.item_projs = nn.ModuleDict()  # item_dim → align_dim
        if self.item_source == "feature" or item_features is not None:
            if not item_features:
                raise ValueError(
                    "AntiSignalCrossHead with item_source='feature' requires item_features"
                )
            item_specs = schema.query(item_features)
            if not item_specs:
                raise ValueError(
                    f"AntiSignalCrossHead matched no item features for expr: {item_features!r}"
                )
            self.item_source = "feature"
            self._item_feature_names = [s.name for s in item_specs]
            self._item_raw_dims = [s.dim for s in item_specs]
            self._item_dims = [
                ProfileExtraCrossHead._effective_dim(s.name, s.dim, self._item_slices)
                for s in item_specs
            ]
            for spec, eff_dim in zip(item_specs, self._item_dims):
                self.item_norms[spec.name] = nn.LayerNorm(eff_dim)
                self.item_projs[spec.name] = nn.Linear(eff_dim, align_dim)

        # Per-user-feature modules; also one item_align for emb/non-feature item sources
        self.item_aligns = nn.ModuleDict()
        for spec, eff_dim in zip(profile_specs, self._dims):
            self.norms[spec.name] = nn.LayerNorm(eff_dim)
            self.user_projs[spec.name] = nn.Linear(eff_dim, align_dim)
            if self.item_source != "feature":
                self.item_aligns[spec.name] = nn.Linear(d_model, align_dim)
            gate = nn.Linear(2 * align_dim, 1, bias=True)
            nn.init.constant_(gate.bias, gate_bias_init)
            self.gates[spec.name] = gate
            self.repr_projs[spec.name] = nn.Sequential(
                nn.Linear(align_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.repr_norm = nn.LayerNorm(d_model)

    @property
    def excluded_names(self) -> set[str]:
        """Feature names this head removes from the shared dense NS path."""
        if self._remove_from_dense_ns:
            return set(self._spec_names)
        return set()

    def _get_item_vec(
        self, spec_name: str, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return normalized item vector in align_dim space."""
        if self.item_source == "feature":
            names_str = ", ".join(f"'{n}'" for n in self._item_feature_names)
            raw = self._schema.extract(
                batch,
                expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
                cat=True,
            )
            splits = torch.split(raw, self._item_raw_dims, dim=-1)
            parts = []
            for name, x in zip(self._item_feature_names, splits):
                x = ProfileExtraCrossHead._apply_slice(name, x, self._item_slices)
                parts.append(self.item_projs[name](self.item_norms[name](x)))
            v_raw = sum(parts) / len(parts) if len(parts) > 1 else parts[0]
        else:
            if self.item_source not in item_bank:
                raise ValueError(
                    f"AntiSignalCrossHead requested item_source={self.item_source!r}, "
                    f"but item_bank has keys {sorted(item_bank)}"
                )
            v_raw = self.item_aligns[spec_name](item_bank[self.item_source])
        return F.normalize(v_raw, dim=-1)

    def forward_with_repr(
        self, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anti-signal logit and hadamard-product representation."""
        names_str = ", ".join(f"'{n}'" for n in self._spec_names)
        raw = self._schema.extract(
            batch,
            expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
            cat=True,
        )
        splits = torch.split(raw, self._raw_dims, dim=-1)

        logit_acc = torch.zeros(raw.shape[0], 1, device=raw.device, dtype=raw.dtype)
        repr_acc = torch.zeros(
            raw.shape[0],
            self.repr_projs[self._spec_names[0]][0].out_features,
            device=raw.device,
            dtype=raw.dtype,
        )
        scale = F.softplus(self.alpha)

        for name, x in zip(self._spec_names, splits):
            x = ProfileExtraCrossHead._apply_slice(name, x, self._slices)
            u = F.normalize(self.user_projs[name](self.norms[name](x)), dim=-1)  # [B, align_dim]
            v = self._get_item_vec(name, batch, item_bank)  # [B, align_dim]
            s = (u * v).sum(dim=-1, keepdim=True)  # [B, 1]
            gate = torch.sigmoid(self.gates[name](torch.cat([u, v], dim=-1)))  # [B, 1]
            logit_acc = logit_acc + (-scale * gate * s)
            repr_acc = repr_acc + self.repr_projs[name](u * v)

        n = len(self._spec_names)
        logit = logit_acc / n
        repr_ = self.repr_norm(repr_acc / n)
        return logit, repr_

    def forward(self, batch: dict[str, Any], item_bank: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the anti-signal logit, discarding the representation."""
        logit, _ = self.forward_with_repr(batch, item_bank)
        return logit


class GroupHeads(nn.Module):
    """Independent feature-group heads with exclusive routing.

    Each group head projects a subset of features (selected by a schema query)
    through LN+Linear, then a 2-layer MLP to produce a logit. Groups marked
    "exclusive" claim their features — they're excluded from the shared dense
    projection in the parent model.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        group_cfgs: list[dict],
        d_model: int,
        dropout_rate: float,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        """Build per-group projection and MLP heads from config.

        Each entry in `group_cfgs` has keys ``name``, ``expr`` (schema query
        selecting features), and ``exclusive`` (bool). Exclusive groups remove
        their features from the parent's shared dense path, exposed via
        ``exclusive_feature_names``.
        """
        super().__init__()
        self._schema = schema
        self._d_model = d_model
        self._exclusive_names: set[str] = set()
        self._group_names: list[str] = []
        self._group_spec_names: dict[str, list[str]] = {}

        self.projs = nn.ModuleDict()
        self.linear_heads = nn.ModuleDict()
        for gh in group_cfgs:
            name = gh["name"]
            specs = schema.query(gh["expr"])
            gh_dim = sum(s.dim for s in specs)
            if gh_dim > 0:
                self.projs[name] = nn.Sequential(
                    nn.Linear(gh_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.linear_heads[name] = two_layer_mlp(
                    d_model,
                    d_model,
                    1,
                    dropout_rate,
                    activation=logit_head_activation,
                    activation_scale=logit_head_tanh_scale,
                )
                self._group_names.append(name)
                self._group_spec_names[name] = [s.name for s in specs]
            if gh["exclusive"]:
                self._exclusive_names.add(name)

    @property
    def exclusive_feature_names(self) -> set[str]:
        """All feature names claimed exclusively across groups."""
        names = set()
        for group_name in self._exclusive_names:
            if group_name in self._group_spec_names:
                names.update(self._group_spec_names[group_name])
        return names

    @property
    def n_reprs(self) -> int:
        """Number of repr tensors returned by forward (for fusion head sizing)."""
        return len(self._group_names)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute group head logits and projected representations.

        Returns
        -------
        logit
            Summed logit contribution [B, 1].
        reprs
            List of projected repr tensors [B, d_model], one per active group.
        """
        logit = None
        reprs = []
        for name in self._group_names:
            spec_names = self._group_spec_names[name]
            names_str = ", ".join(f"'{n}'" for n in spec_names)
            feats = self._schema.extract(
                batch,
                expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
                cat=True,
            )
            proj = F.silu(self.projs[name](feats))
            reprs.append(proj)
            head_logit = self.linear_heads[name](proj)
            logit = head_logit if logit is None else logit + head_logit
        return logit, reprs


class FlatGDSLHead(nn.Module):
    """Flatten declared pathways, mix with selected mixer, emit one logit.

    Pathways are explicit and ordered. No auto-injection or deduplication.
    """

    def __init__(
        self,
        pathways: list[str],
        pathway_dims: dict[str, int],
        mixer_cfg: dict[str, Any],
        mixer_type: str = "gdcn",
        path_norm: str = "none",
        d_cross: int = None,
        senet: bool = False,
        senet_reduction: int = 3,
        senet_groups: int = 2,
    ) -> None:
        super().__init__()
        if not pathways:
            raise ValueError("head_mode.flat_gdsl.pathways must be non-empty")

        missing = [name for name in pathways if name not in pathway_dims]
        if missing:
            raise ValueError(f"flat_gdsl pathways missing dims: {missing}")

        self.pathways = list(pathways)
        self.path_dims = [int(pathway_dims[name]) for name in self.pathways]
        if any(dim <= 0 for dim in self.path_dims):
            raise ValueError(f"flat_gdsl pathway dims must be > 0, got {self.path_dims}")

        valid_norms = {"none", "layernorm"}
        if path_norm not in valid_norms:
            raise ValueError(f"head_mode.flat_gdsl.path_norm must be one of {sorted(valid_norms)}")
        if path_norm == "none":
            self.path_norms = nn.ModuleList([nn.Identity() for _ in self.path_dims])
        else:
            self.path_norms = nn.ModuleList(
                [
                    # LayerNorm(1) collapses scalars (e.g., per-domain DIN logits) to
                    # a constant, so keep scalar pathways unnormalized.
                    nn.Identity() if dim == 1 else nn.LayerNorm(dim)
                    for dim in self.path_dims
                ]
            )

        self.path_projs = None
        if d_cross is not None:
            if d_cross <= 0:
                raise ValueError("head_mode.flat_gdsl.d_cross must be > 0 when set")
            self.path_projs = nn.ModuleList(
                [nn.Linear(dim, d_cross, bias=False) for dim in self.path_dims]
            )
            effective_field_dim = d_cross
            field_dims = [d_cross] * len(self.pathways)
        else:
            effective_field_dim = None
            field_dims = self.path_dims

        self.senet = None
        if senet:
            if senet_reduction <= 0:
                raise ValueError("head_mode.flat_gdsl.senet_reduction must be > 0")
            if senet_groups <= 0:
                raise ValueError("head_mode.flat_gdsl.senet_groups must be > 0")
            if effective_field_dim is not None and effective_field_dim % senet_groups != 0:
                raise ValueError(
                    "head_mode.flat_gdsl.senet_groups must divide effective field dim "
                    f"({effective_field_dim})"
                )
            self.senet = FieldSENET(
                n_fields=len(self.pathways),
                d_field=effective_field_dim,
                reduction=senet_reduction,
                n_groups=senet_groups,
            )

        self._last_aux_losses: dict[str, torch.Tensor] = {}
        valid_mixers = {"gdcn", "tabm", "gtc_lite"}
        if mixer_type not in valid_mixers:
            raise ValueError(
                f"head_mode.flat_gdsl.mixer_type must be one of {sorted(valid_mixers)}"
            )
        self.mixer_type = mixer_type

        if mixer_type == "gdcn":
            mixer_kwargs = dict(mixer_cfg)
            if mixer_kwargs["field_regulation"]:
                mixer_kwargs["field_dims"] = field_dims
            else:
                mixer_kwargs["field_dims"] = None
            self.mixer = GDCNNetwork(
                input_dim=sum(field_dims),
                output_dim=1,
                **mixer_kwargs,
            )
        elif mixer_type == "tabm":
            raise RuntimeError("head_mode.flat_gdsl.mixer_type='tabm' is deprecated.")
        else:
            if len(set(field_dims)) != 1:
                raise ValueError(
                    "head_mode.flat_gdsl.mixer_type='gtc_lite' requires equal field dims. "
                    "Set head_mode.flat_gdsl.d_cross to project all pathways to a shared width."
                )
            self.mixer = GTCLiteMixer(
                n_fields=len(field_dims),
                d_field=field_dims[0],
                output_dim=1,
                **mixer_cfg,
            )

    def init_output_bias(self, bias_value: float) -> None:
        """Initialize final output bias regardless of mixer choice."""
        if self.mixer_type == "gdcn":
            nn.init.constant_(self.mixer.output_proj.bias, bias_value)
        else:
            self.mixer.init_output_bias(bias_value)

    def consume_aux_losses(self) -> dict[str, torch.Tensor]:
        """Return and clear mixer auxiliary losses from the latest forward pass."""
        aux = self._last_aux_losses
        self._last_aux_losses = {}
        return aux

    def forward(
        self,
        pathway_reprs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return a single logit tensor [B, 1]."""
        parts: list[torch.Tensor] = []
        for idx, name in enumerate(self.pathways):
            if name not in pathway_reprs:
                raise ValueError(f"flat_gdsl pathway {name!r} is missing at runtime")
            x = pathway_reprs[name]

            if x.dim() == 3:
                x = x.reshape(x.shape[0], -1)
            elif x.dim() != 2:
                raise ValueError(
                    f"flat_gdsl pathway {name!r} must be rank-2 or rank-3, got shape {tuple(x.shape)}"
                )

            expected_dim = self.path_dims[idx]
            if x.shape[-1] != expected_dim:
                raise ValueError(
                    f"flat_gdsl pathway {name!r} has dim {x.shape[-1]}, expected {expected_dim}"
                )

            x = self.path_norms[idx](x)
            if self.path_projs is not None:
                x = self.path_projs[idx](x)
            parts.append(x)

        if self.senet is not None:
            parts = self.senet(parts)
        if self.mixer_type == "gtc_lite":
            x = torch.stack(parts, dim=1)
            logits = self.mixer(x)
            self._last_aux_losses = self.mixer.consume_aux_losses()
            return logits
        self._last_aux_losses = {}
        return self.mixer(torch.cat(parts, dim=-1))


class SemanticRouteHeads(nn.Module):
    """IACC semantic routes for interpretable logit branches.

    Routes:
    - interest: sequence match + user prior + base user-item token + profile-item cross
    - attention: schema-selected activation/history signals
    - creative: item sparse+dense routed representation
    - convenience: recent activity/window concentration signals

    When `moe_cfg` is provided, each route head becomes a top-k expert pool
    instead of a single fixed MLP.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        route_sources: dict[str, list[str]],
        fusion: bool,
        bilinear_fusion: bool = False,
        source_bilinear_fusion: bool = False,
        moe_cfg: dict = None,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        super().__init__()
        if not route_sources:
            raise ValueError("SemanticRouteHeads requires at least one route")
        self.route_sources = route_sources
        self.route_names = list(route_sources.keys())
        self.fusion = fusion
        self.bilinear_fusion = bilinear_fusion
        self.source_bilinear_fusion = source_bilinear_fusion
        self._use_moe = moe_cfg is not None
        self.route_repr_probes = _make_repr_probes(self.route_names)
        self.fused_route_probe = nn.Identity() if fusion else None

        self.route_projs = nn.ModuleDict()
        self.route_heads = nn.ModuleDict()
        for route, sources in route_sources.items():
            if not sources:
                raise ValueError(f"semantic route {route!r} requires at least one source")
            self.route_projs[route] = self._route_proj(
                d_model * len(sources), d_model, dropout_rate
            )
            if self._use_moe:
                from core.models.modules.context_moe import ExpertMLP

                self.route_heads[route] = ExpertMLP(
                    in_dim=d_model,
                    hidden_dim=d_model,
                    out_dim=1,
                    n_experts=moe_cfg["n_experts"],
                    top_k=moe_cfg["top_k"],
                    dropout_rate=dropout_rate,
                    balance_weight=moe_cfg["balance_weight"],
                )
            else:
                self.route_heads[route] = two_layer_mlp(
                    d_model,
                    d_model,
                    1,
                    dropout_rate,
                    activation=logit_head_activation,
                    activation_scale=logit_head_tanh_scale,
                )
        if fusion:
            self.fusion_head = two_layer_mlp(
                d_model * len(route_sources),
                d_model,
                1,
                dropout_rate,
                activation=logit_head_activation,
                activation_scale=logit_head_tanh_scale,
            )

        if bilinear_fusion:
            n_routes = len(self.route_names)
            self.bilinear_pairs = nn.ParameterList()
            self._bilinear_pair_indices = []
            for i in range(n_routes):
                for j in range(i + 1, n_routes):
                    W = nn.Parameter(torch.empty(d_model, d_model))
                    nn.init.normal_(W, std=0.01)
                    self.bilinear_pairs.append(W)
                    self._bilinear_pair_indices.append((i, j))

        self.source_bilinear_pairs = nn.ParameterDict()
        self._source_bilinear_pair_indices: dict[str, list[tuple[int, int, str]]] = {}
        if source_bilinear_fusion:
            for route, sources in self.route_sources.items():
                pair_specs: list[tuple[int, int, str]] = []
                for i in range(len(sources)):
                    for j in range(i + 1, len(sources)):
                        key = f"{route}__{i}__{j}"
                        W = nn.Parameter(torch.empty(d_model, d_model))
                        nn.init.normal_(W, std=0.01)
                        self.source_bilinear_pairs[key] = W
                        pair_specs.append((i, j, key))
                self._source_bilinear_pair_indices[route] = pair_specs

    @staticmethod
    def _route_proj(in_dim: int, d_model: int, dropout_rate: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        *,
        source_reprs: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate semantic route logits.

        Returns
        -------
        torch.Tensor or tuple
            Logits [B, 1] when MoE is off. ``(logits, balance_loss)`` when
            MoE is on.
        """
        route_reprs = []
        logits = None
        balance_losses = []
        for route in self.route_names:
            sources = self.route_sources[route]
            missing = [name for name in sources if name not in source_reprs]
            if missing:
                raise ValueError(f"semantic route {route!r} missing sources: {missing}")
            parts = [source_reprs[name] for name in sources]
            route_repr = self.route_projs[route](torch.cat(parts, dim=-1))
            route_repr = self.route_repr_probes[route](route_repr)
            route_reprs.append(route_repr)
            if self._use_moe:
                route_logit, bl = self.route_heads[route](route_repr)
                balance_losses.append(bl)
            else:
                route_logit = self.route_heads[route](route_repr)
            if self.source_bilinear_fusion:
                for i, j, key in self._source_bilinear_pair_indices[route]:
                    # s_i^T W s_j → [B, 1]
                    Ws_j = torch.matmul(parts[j], self.source_bilinear_pairs[key].t())
                    route_logit = route_logit + (parts[i] * Ws_j).sum(dim=-1, keepdim=True)
            logits = route_logit if logits is None else logits + route_logit
        if self.fusion:
            fused_route_repr = self.fused_route_probe(torch.cat(route_reprs, dim=-1))
            logits = logits + self.fusion_head(fused_route_repr)
        if self.bilinear_fusion:
            for idx, (i, j) in enumerate(self._bilinear_pair_indices):
                # r_i^T W r_j → [B, 1]
                Wr_j = torch.matmul(route_reprs[j], self.bilinear_pairs[idx].t())
                logits = logits + (route_reprs[i] * Wr_j).sum(dim=-1, keepdim=True)
        if self._use_moe:
            return logits, sum(balance_losses)
        return logits
