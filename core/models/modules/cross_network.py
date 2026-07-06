"""Gated Deep Cross Network (GDCN) for learned feature interactions.

Implements the gated cross layer from "GDCN: Gated Deep Cross Network for
Feature Interaction" (arXiv:2311.04635). Each layer computes:

    c_{l+1} = c_0 ⊙ (W_c(c_l) + b) ⊙ sigmoid(W_g(c_l)) + c_l

where W_c/W_g are full-rank or low-rank factorized projections.
The per-element sigmoid gate suppresses noisy interactions, allowing
stacking to 3+ layers without degradation.

Optional enhancements:
- SENET field gating: per-sample field importance before crossing (FiBiNET).
- Factorized post-layer: compositional interactions on crossed output (FINAL).
- Learned anchor blend: adaptive interpolation between x_0 and x_l
  (internal extension from anchor-blend experiments, not from the original
  GDCN paper).
- Field regulation: per-layer field-wise routing prior inspired by EDCN's
  regulation/bridge design (Chen et al., DLP-KDD 2021).
- Cross experts: per-layer mixture of cross operators with input-conditioned
  expert routing.
- Directional pair residual: per-feature user-item aligned residual branch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from core.data.schema import FeatureSchema
from core.models.modules.primitives import build_activation


class GDCNLayer(nn.Module):
    """Single GDCN layer with optional low-rank factorization and experts."""

    def __init__(
        self,
        input_dim: int,
        rank: int = 0,
        dropout: float = 0.0,
        anchor_mode: str = "fixed_x0",
        anchor_rank: int = 0,
        anchor_init: float = 2.0,
        cross_experts: int = 1,
        cross_expert_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.anchor_mode = anchor_mode
        self.anchor_rank = anchor_rank
        self.cross_experts = cross_experts
        if cross_experts < 1:
            raise ValueError("cross_experts must be >= 1")
        if cross_expert_hidden < 0:
            raise ValueError("cross_expert_hidden must be >= 0")
        if rank > 0:
            if cross_experts > 1:
                self.cross_U_experts = nn.ModuleList(
                    [nn.Linear(input_dim, rank, bias=False) for _ in range(cross_experts)]
                )
                self.cross_V_experts = nn.ModuleList(
                    [nn.Linear(rank, input_dim) for _ in range(cross_experts)]
                )
                self.gate_U_experts = nn.ModuleList(
                    [nn.Linear(input_dim, rank, bias=False) for _ in range(cross_experts)]
                )
                self.gate_V_experts = nn.ModuleList(
                    [nn.Linear(rank, input_dim) for _ in range(cross_experts)]
                )
            else:
                self.cross_U = nn.Linear(input_dim, rank, bias=False)
                self.cross_V = nn.Linear(rank, input_dim)
                self.gate_U = nn.Linear(input_dim, rank, bias=False)
                self.gate_V = nn.Linear(rank, input_dim)
        elif cross_experts > 1:
            self.cross_W_experts = nn.ModuleList(
                [nn.Linear(input_dim, input_dim) for _ in range(cross_experts)]
            )
            self.gate_W_experts = nn.ModuleList(
                [nn.Linear(input_dim, input_dim) for _ in range(cross_experts)]
            )
        else:
            self.cross_W = nn.Linear(input_dim, input_dim)
            self.gate_W = nn.Linear(input_dim, input_dim)
        if cross_experts > 1:
            if cross_expert_hidden > 0:
                self.expert_router = nn.Sequential(
                    nn.Linear(input_dim, cross_expert_hidden),
                    nn.SiLU(),
                    nn.Linear(cross_expert_hidden, cross_experts),
                )
            else:
                self.expert_router = nn.Linear(input_dim, cross_experts)
        self.dropout = nn.Dropout(dropout)

        valid_anchor_modes = {"fixed_x0", "learned_scalar", "learned_vector"}
        if anchor_mode not in valid_anchor_modes:
            raise ValueError(
                f"Unknown anchor_mode {anchor_mode!r}. "
                f"Expected one of {sorted(valid_anchor_modes)}."
            )
        if anchor_mode != "learned_vector" and anchor_rank > 0:
            raise ValueError("anchor_rank is only supported when anchor_mode='learned_vector'")

        if anchor_mode == "learned_scalar":
            self.anchor_scalar = nn.Linear(input_dim, 1)
            nn.init.zeros_(self.anchor_scalar.weight)
            nn.init.constant_(self.anchor_scalar.bias, anchor_init)
        elif anchor_mode == "learned_vector":
            if anchor_rank > 0:
                self.anchor_U = nn.Linear(input_dim, anchor_rank, bias=False)
                self.anchor_V = nn.Linear(anchor_rank, input_dim)
                # Avoid dead low-rank gate path: all-zero U/V blocks gradients to
                # both factors on the first backward pass.
                nn.init.normal_(self.anchor_U.weight, mean=0.0, std=1e-3)
                nn.init.normal_(self.anchor_V.weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.anchor_V.bias, anchor_init)
            else:
                self.anchor_W = nn.Linear(input_dim, input_dim)
                nn.init.zeros_(self.anchor_W.weight)
                nn.init.constant_(self.anchor_W.bias, anchor_init)

    def forward(
        self,
        x_0: torch.Tensor,
        x_l: torch.Tensor,
        projection_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute one cross layer.

        Parameters
        ----------
        x_0
            Original input (fixed across layers) [B, D].
        x_l
            Current layer input [B, D].
        projection_input
            Optional alternate input for cross/gate projections [B, D].
            When provided, `x_l` is still used for residual and anchor blending.
        """
        x_proj = projection_input if projection_input is not None else x_l
        if self.cross_experts > 1:
            expert_weights = torch.softmax(self.expert_router(x_proj), dim=-1)  # [B, E]
            if self.rank > 0:
                cross_terms = []
                gate_terms = []
                for i in range(self.cross_experts):
                    cross_terms.append(self.cross_V_experts[i](self.cross_U_experts[i](x_proj)))
                    gate_terms.append(
                        torch.sigmoid(self.gate_V_experts[i](self.gate_U_experts[i](x_proj)))
                    )
            else:
                cross_terms = [layer(x_proj) for layer in self.cross_W_experts]
                gate_terms = [torch.sigmoid(layer(x_proj)) for layer in self.gate_W_experts]
            cross = (torch.stack(cross_terms, dim=1) * expert_weights.unsqueeze(-1)).sum(dim=1)
            gate = (torch.stack(gate_terms, dim=1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        elif self.rank > 0:
            cross = self.cross_V(self.cross_U(x_proj))
            gate = torch.sigmoid(self.gate_V(self.gate_U(x_proj)))
        else:
            cross = self.cross_W(x_proj)
            gate = torch.sigmoid(self.gate_W(x_proj))

        if self.anchor_mode == "fixed_x0":
            base = x_0
        elif self.anchor_mode == "learned_scalar":
            alpha = torch.sigmoid(self.anchor_scalar(x_l))  # [B, 1]
            base = alpha * x_0 + (1.0 - alpha) * x_l
        else:
            if self.anchor_rank > 0:
                alpha_logits = self.anchor_V(self.anchor_U(x_l))
            else:
                alpha_logits = self.anchor_W(x_l)
            alpha = torch.sigmoid(alpha_logits)  # [B, D]
            base = alpha * x_0 + (1.0 - alpha) * x_l

        return base * self.dropout(cross) * gate + x_l


class FieldSENET(nn.Module):
    """SENet+ field gating (FiBiNET++, CIKM 2023).

    Two modes depending on whether field dimensions are uniform:

    Uniform (d_field set): Full SENet+ with grouped max+avg squeeze,
    per-dimension excitation, additive skip, and LayerNorm.

    Mixed (d_field=None): Original FiBiNET per-field scalar squeeze-excitation
    with additive skip. No LayerNorm (dims vary per field).

    Parameters
    ----------
    n_fields
        Number of input fields.
    d_field
        Dimension per field when uniform. None for mixed dims.
    reduction
        Bottleneck reduction ratio for excitation.
    n_groups
        Number of groups to split each field embedding for squeeze (uniform mode).
    """

    def __init__(
        self,
        n_fields: int,
        d_field: int = None,
        reduction: int = 3,
        n_groups: int = 2,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.d_field = d_field
        self.n_groups = n_groups

        if d_field is not None:
            # SENet+ mode: grouped squeeze, per-dimension excitation
            squeeze_dim = n_fields * n_groups * 2  # max + avg per group
            reduced = max(1, squeeze_dim // reduction)
            excite_out = n_fields * d_field

            self.excitation = nn.Sequential(
                nn.Linear(squeeze_dim, reduced, bias=False),
                nn.ReLU(),
                nn.Linear(reduced, excite_out, bias=False),
                nn.ReLU(),
            )
            self.norm = nn.LayerNorm(d_field)
        else:
            # Original mode: mean squeeze, per-field scalar excitation
            reduced = max(1, n_fields // reduction)
            self.excitation = nn.Sequential(
                nn.Linear(n_fields, reduced, bias=False),
                nn.ReLU(),
                nn.Linear(reduced, n_fields, bias=False),
                nn.ReLU(),
            )
            self.norm = None

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply field-wise squeeze-excitation.

        Parameters
        ----------
        features
            List of `n_fields` tensors, each [B, D]. When `d_field` is set,
            all D must equal `d_field`.

        Returns
        -------
        list[torch.Tensor]
            Reweighted features, same shapes.
        """
        if self.d_field is not None:
            stacked = torch.stack(features, dim=1)  # [B, F, D]
            B, F, D = stacked.shape
            g = self.n_groups
            grouped = stacked.reshape(B, F, g, D // g)
            group_max = grouped.max(dim=-1).values  # [B, F, g]
            group_avg = grouped.mean(dim=-1)  # [B, F, g]
            z = torch.cat([group_max, group_avg], dim=-1).reshape(B, -1)  # [B, F*2g]

            a = self.excitation(z).reshape(B, F, D)  # [B, F, D]
            out = self.norm(stacked + a * stacked)
            return list(out.unbind(dim=1))
        else:
            z = torch.stack([f.mean(dim=-1) for f in features], dim=-1)  # [B, F]
            a = self.excitation(z)  # [B, F]
            return [f + a[:, i : i + 1] * f for i, f in enumerate(features)]


class FieldNorm(nn.Module):
    """Per-field LayerNorm before crossing.

    Each field gets its own LayerNorm at its native dimension. Ensures all
    fields enter the cross at consistent scale regardless of upstream
    magnitude differences.

    Categoricals get the same LayerNorm treatment as numericals. The original
    FiBiNET++ design uses BatchNorm for categoricals — may need to revisit if
    LN underperforms on categorical embeddings.

    Parameters
    ----------
    field_dims
        List of per-field dimensions (may be mixed).
    """

    def __init__(self, field_dims: list[int]) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in field_dims])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Normalize each field independently.

        Parameters
        ----------
        features
            List of tensors, each [B, D_i].

        Returns
        -------
        list[torch.Tensor]
            Normalized features, same shapes.
        """
        return [norm(f) for norm, f in zip(self.norms, features)]


class FieldRegulation(nn.Module):
    """Learned per-field softmax routing prior to cross projections.

    Two modes are supported:

    - static: stores one learnable logit per field.
    - dynamic: predicts per-sample field logits from the current layer input.

    At forward time, logits are temperature-scaled, softmax-normalized,
    expanded to per-dimension weights, and applied elementwise to the flat
    cross input.

    Inspiration: EDCN regulation module ("Enhancing Explicit and Implicit
    Feature Interactions via Information Sharing for Parallel Deep CTR Models",
    Chen et al., DLP-KDD 2021). This is a lightweight adaptation for a pure
    cross stack (no parallel deep tower).
    """

    def __init__(
        self,
        field_dims: list[int],
        tau: float = 1.0,
        mode: str = "static",
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if not field_dims:
            raise ValueError("field_dims must be non-empty when field regulation is enabled")
        if any(d <= 0 for d in field_dims):
            raise ValueError("all field dims must be > 0 for field regulation")
        if tau <= 0:
            raise ValueError("field_regulation_tau must be > 0")
        valid_modes = {"static", "dynamic"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown field_regulation_mode {mode!r}. Expected one of {sorted(valid_modes)}."
            )
        if hidden_dim < 0:
            raise ValueError("field_regulation_hidden must be >= 0")
        self.mode = mode
        self.tau = tau
        self.n_fields = len(field_dims)
        self.input_dim = sum(field_dims)
        if mode == "static":
            self.logits = nn.Parameter(torch.ones(self.n_fields))
            self.router = None
        else:
            self.logits = None
            if hidden_dim > 0:
                self.router = nn.Sequential(
                    nn.Linear(self.input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.n_fields),
                )
            else:
                self.router = nn.Linear(self.input_dim, self.n_fields)
        dim_to_field = torch.repeat_interleave(
            torch.arange(len(field_dims), dtype=torch.long),
            torch.tensor(field_dims, dtype=torch.long),
        )
        self.register_buffer("_dim_to_field", dim_to_field)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply field-wise softmax weighting to flat cross input [B, D]."""
        if self.mode == "dynamic":
            weights = torch.softmax(self.router(x) / self.tau, dim=-1)  # [B, F]
            per_dim = weights.index_select(1, self._dim_to_field)  # [B, D]
        else:
            weights = torch.softmax(self.logits / self.tau, dim=0)  # [F]
            per_dim = weights.index_select(0, self._dim_to_field).unsqueeze(0)  # [1, D]
        return x * per_dim


class FactorizedPostLayer(nn.Module):
    """Factorized interaction post-layer (FINAL, SIGIR 2023).

    Projects to 2x width, chunks, and adds a scaled multiplicative interaction
    back to the input residual:
        x' = x + alpha * (h1 * h2)
    followed by normalization.

    This keeps the block near-identity at initialization (small alpha), which
    avoids destabilizing the parent representation at epoch start.

    Parameters
    ----------
    dim
        Input and output dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.residual_scale = nn.Parameter(torch.tensor(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            [B, dim] input.

        Returns
        -------
        torch.Tensor
            [B, dim] factorized interaction output.
        """
        h = self.linear(x)
        h1, h2 = h.chunk(2, dim=-1)
        out = x + self.residual_scale * (h1 * h2)
        return self.norm(out)


class DirectionalPairResidual(nn.Module):
    """Per-feature directional user-item residual interaction branch.

    Builds an item anchor vector from all item-aligned fields, then for each
    user-aligned field computes a feature-specific aligned product and gate.
    The gated feature interactions are averaged and added back as a small
    residual vector.
    """

    def __init__(
        self,
        field_dims: list[int],
        field_groups: list[str],
        d_model: int,
        dropout: float = 0.0,
        align_dim: int = 0,
        hidden_mult: int = 2,
        scale_init: float = 1e-2,
        use_abs_diff: bool = False,
    ) -> None:
        super().__init__()
        if len(field_dims) != len(field_groups):
            raise ValueError("field_dims and field_groups must have the same length")
        if d_model <= 0:
            raise ValueError("directional_pair_residual d_model must be > 0")
        if hidden_mult <= 0:
            raise ValueError("directional_pair_hidden_mult must be > 0")
        self.user_indices = [i for i, g in enumerate(field_groups) if g.startswith("user_")]
        self.item_indices = [i for i, g in enumerate(field_groups) if g.startswith("item_")]
        if not self.user_indices:
            raise ValueError(
                "directional_pair_residual requires at least one user_* field in GDCN input"
            )
        if not self.item_indices:
            raise ValueError(
                "directional_pair_residual requires at least one item_* field in GDCN input"
            )
        align_dim = align_dim if align_dim > 0 else d_model
        self.use_abs_diff = use_abs_diff

        self.item_norms = nn.ModuleList([nn.LayerNorm(field_dims[i]) for i in self.item_indices])
        self.item_projs = nn.ModuleList(
            [nn.Linear(field_dims[i], d_model) for i in self.item_indices]
        )

        self.user_norms = nn.ModuleList([nn.LayerNorm(field_dims[i]) for i in self.user_indices])
        self.user_projs = nn.ModuleList(
            [nn.Linear(field_dims[i], d_model) for i in self.user_indices]
        )
        self.user_aligns = nn.ModuleList([nn.Linear(d_model, align_dim) for _ in self.user_indices])
        self.item_aligns = nn.ModuleList([nn.Linear(d_model, align_dim) for _ in self.user_indices])

        cross_in_dim = d_model * 2 + align_dim + (d_model if use_abs_diff else 0)
        hidden_dim = d_model * hidden_mult
        self.cross_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cross_in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in self.user_indices
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in self.user_indices
            ]
        )
        for gate in self.gates:
            nn.init.constant_(gate[-1].bias, -2.0)
        self.output_norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Compute directional per-feature residual representation [B, D]."""
        item_parts = []
        for item_idx, item_norm, item_proj in zip(
            self.item_indices, self.item_norms, self.item_projs, strict=True
        ):
            item_parts.append(torch.nn.functional.silu(item_proj(item_norm(features[item_idx]))))
        item_vec = sum(item_parts) / len(item_parts)

        weighted_parts = []
        gates = []
        for user_idx, user_norm, user_proj, user_align, item_align, cross_mlp, gate_mlp in zip(
            self.user_indices,
            self.user_norms,
            self.user_projs,
            self.user_aligns,
            self.item_aligns,
            self.cross_mlps,
            self.gates,
            strict=True,
        ):
            user_vec = torch.nn.functional.silu(user_proj(user_norm(features[user_idx])))
            aligned_prod = user_align(user_vec) * item_align(item_vec)
            cross_inputs = [user_vec, item_vec, aligned_prod]
            if self.use_abs_diff:
                cross_inputs.append(torch.abs(user_vec - item_vec))
            cross_repr = cross_mlp(torch.cat(cross_inputs, dim=-1))
            gate = torch.sigmoid(gate_mlp(torch.cat([user_vec, item_vec], dim=-1)))
            weighted_parts.append(gate * cross_repr)
            gates.append(gate)

        denom = torch.stack(gates, dim=0).sum(dim=0).clamp_min(1e-3)
        repr_ = sum(weighted_parts) / denom
        repr_ = self.output_norm(repr_)
        return self.scale * repr_


class GDCNNetwork(nn.Module):
    """Stacked GDCN layers with output projection.

    Parameters
    ----------
    input_dim
        Dimension of the flat input vector.
    output_dim
        Dimension of the output representation.
    n_layers
        Number of stacked cross layers.
    rank
        Low-rank factorization rank. 0 = full-rank (only viable for
        input_dim < ~500). For larger inputs, use rank 32-128.
    dropout
        Dropout applied to cross term before residual addition.
    factorized_post
        When True, adds a FINAL-style factorized interaction layer after
        the cross stack (before output projection).
    factorized_post_output
        When True, adds FINAL-style factorized post-layer(s) after output
        projection at output_dim width.
    factorized_post_output_layers
        Number of stacked output-side factorized post-layers.
    anchor_mode
        Anchor strategy inside each cross layer:
        - "fixed_x0": original GDCN behavior (default)
        - "learned_scalar": scalar blend between x_0 and x_l
        - "learned_vector": per-dimension blend between x_0 and x_l
    anchor_rank
        Low-rank factorization rank for learned_vector anchor gate.
        0 = full-rank.
    anchor_init
        Initial logit bias for anchor blend gate (sigmoid space). Higher means
        closer to x_0 at initialization.
    field_regulation
        Enables per-layer field-wise softmax regulation before cross projections,
        inspired by EDCN's regulation idea (Chen et al., DLP-KDD 2021).
    field_regulation_tau
        Temperature for field regulation softmax.
    field_regulation_mode
        Regulation mode:
        - "static": one learned field prior shared by all samples
        - "dynamic": per-sample field routing from current layer input
    field_regulation_hidden
        Hidden width for dynamic regulation router MLP. 0 uses a single linear
        router.
    field_dims
        Per-field dimensions corresponding to the flattened cross input. Required
        when `field_regulation=True`.
    cross_experts
        Number of expert cross operators per layer. 1 preserves baseline behavior.
    cross_expert_hidden
        Hidden width for expert router MLP. 0 uses a single linear router.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 3,
        rank: int = 0,
        dropout: float = 0.0,
        factorized_post: bool = False,
        factorized_post_output: bool = False,
        factorized_post_output_layers: int = 1,
        anchor_mode: str = "fixed_x0",
        anchor_rank: int = 0,
        anchor_init: float = 2.0,
        field_regulation: bool = False,
        field_regulation_tau: float = 1.0,
        field_regulation_mode: str = "static",
        field_regulation_hidden: int = 0,
        field_dims: list[int] | None = None,
        cross_experts: int = 1,
        cross_expert_hidden: int = 0,
    ) -> None:
        super().__init__()
        if factorized_post_output and factorized_post_output_layers < 1:
            raise ValueError("factorized_post_output_layers must be >= 1")
        if cross_experts < 1:
            raise ValueError("cross_experts must be >= 1")
        if cross_expert_hidden < 0:
            raise ValueError("cross_expert_hidden must be >= 0")
        if field_regulation_hidden < 0:
            raise ValueError("field_regulation_hidden must be >= 0")
        if field_regulation:
            if field_dims is None:
                raise ValueError("field_dims is required when field_regulation=True")
            if sum(field_dims) != input_dim:
                raise ValueError(
                    f"sum(field_dims)={sum(field_dims)} must match input_dim={input_dim}"
                )
        self.input_norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList(
            [
                GDCNLayer(
                    input_dim,
                    rank,
                    dropout,
                    anchor_mode=anchor_mode,
                    anchor_rank=anchor_rank,
                    anchor_init=anchor_init,
                    cross_experts=cross_experts,
                    cross_expert_hidden=cross_expert_hidden,
                )
                for _ in range(n_layers)
            ]
        )
        self.field_regulators = (
            nn.ModuleList(
                [
                    FieldRegulation(
                        field_dims,
                        tau=field_regulation_tau,
                        mode=field_regulation_mode,
                        hidden_dim=field_regulation_hidden,
                    )
                    for _ in range(n_layers)
                ]
            )
            if field_regulation
            else None
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.factorized_post = FactorizedPostLayer(input_dim) if factorized_post else None
        self.output_proj = nn.Linear(input_dim, output_dim)
        self.factorized_post_output = (
            nn.ModuleList(
                [FactorizedPostLayer(output_dim) for _ in range(factorized_post_output_layers)]
            )
            if factorized_post_output
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run stacked cross layers and project to output_dim.

        Parameters
        ----------
        x
            Flat feature vector [B, input_dim].

        Returns
        -------
        torch.Tensor
            Output representation [B, output_dim].
        """
        x = self.input_norm(x)
        x_0 = x
        for i, layer in enumerate(self.layers):
            projection_input = (
                self.field_regulators[i](x) if self.field_regulators is not None else None
            )
            x = layer(x_0, x, projection_input=projection_input)
        x = self.output_norm(x)
        if self.factorized_post is not None:
            x = self.factorized_post(x)
        x = self.output_proj(x)
        if self.factorized_post_output is not None:
            for post in self.factorized_post_output:
                x = post(x)
        return x


def _field_hierarchy_groups_from_schema_expr(
    schema: FeatureSchema,
    feature_names: list[str],
) -> list[str]:
    """Assign coarse hierarchy groups from schema query expressions.

    Uses the existing FeatureSchema DSL to keep grouping logic aligned with the
    rest of DragonChariot config/query conventions.
    """
    group_exprs: list[tuple[str, str]] = [
        (
            "user_cat",
            "entity = 'user' and dtype = 'categorical' and scope = 'static' and source != 'metadata'",
        ),
        (
            "item_cat",
            "entity = 'item' and dtype = 'categorical' and scope = 'static' and source != 'metadata'",
        ),
        (
            "context_cat",
            "entity = 'context' and dtype = 'categorical' and scope = 'static' and source != 'metadata'",
        ),
        (
            "derived_cont",
            "dtype = 'numerical' and scope = 'static' and source = 'derived'",
        ),
        (
            "user_cont",
            "entity = 'user' and dtype = 'numerical' and scope = 'static' and source = 'original'",
        ),
        (
            "item_cont",
            "entity = 'item' and dtype = 'numerical' and scope = 'static' and source = 'original'",
        ),
        (
            "context_cont",
            "entity = 'context' and dtype = 'numerical' and scope = 'static' and source = 'original'",
        ),
    ]
    feature_to_group: dict[str, str] = {}
    for group_name, expr in group_exprs:
        for spec in schema.query(expr):
            if spec.name not in feature_to_group:
                feature_to_group[spec.name] = group_name

    missing = [name for name in feature_names if name not in feature_to_group]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Could not assign field hierarchy group for features: {preview}{suffix}")
    return [feature_to_group[name] for name in feature_names]


class GDCNSource(nn.Module):
    """Semantic source wrapping GDCN with schema-aware input assembly.

    Consumes the full raw embedding vectors (user + item categoricals) plus
    all static numerical features. Does NOT claim features as exclusive — the
    same features remain available to dense_proj and other pathways.

    Parameters
    ----------
    schema
        Feature schema for numerical feature lookup.
    emb_dim
        Per-feature embedding dimension (for computing categorical input width).
    d_model
        Output dimension (model hidden size).
    n_layers
        Number of stacked GDCN layers.
    rank
        Low-rank factorization rank (0 = full-rank).
    dropout
        Dropout rate within cross layers.
    input_mode
        "raw" uses raw embedding vectors (default, pre-NS).
        "ns_tokenized" uses post-NS token representations.
    d_cross
        Per-feature projection dimension. None = no projection (pass native dims).
    field_type_emb
        When True, adds a hierarchical learned embedding to each feature before
        crossing: entity-type (user/item/numerical) + per-field identity.
        Requires `d_cross` to be set.
    field_type_emb_granularity
        Controls field identity embedding granularity when `field_type_emb=True`.
        - `per_field`: one embedding per feature (default, existing behavior)
        - `entity_dtype_plus_derived_cont`: coarse hierarchy
          (`user_cat`, `item_cat`, `user_cont`, `item_cont`, `derived_cont`)
    factorized_post
        Legacy FINAL-style factorized post-layer at cross_input_dim (before output projection).
    factorized_post_output
        FINAL-style factorized post-layer(s) at d_model (after output projection).
    factorized_post_output_layers
        Number of stacked output-side factorized post-layers.
    anchor_mode
        Anchor strategy inside each cross layer:
        - "fixed_x0": original GDCN behavior
        - "learned_scalar": scalar blend between x_0 and x_l
        - "learned_vector": per-dimension blend between x_0 and x_l
    anchor_rank
        Low-rank rank for learned_vector anchor blending. 0 = full-rank.
    anchor_init
        Initial anchor blend logit bias (sigmoid space). Higher => closer to x_0
        at initialization.
    field_regulation
        Enables EDCN-lite style per-layer field regulation before cross
        projections (inspired by Chen et al., DLP-KDD 2021).
    field_regulation_tau
        Temperature for field regulation softmax.
    field_regulation_mode
        Regulation mode:
        - `static`: one learned field prior shared by all samples
        - `dynamic`: per-sample field routing from current layer input
    field_regulation_hidden
        Hidden width for dynamic regulation router MLP. 0 uses a single linear
        router.
    cross_experts
        Number of expert cross operators per layer. 1 preserves baseline.
    cross_expert_hidden
        Hidden width for expert router MLP. 0 uses a single linear router.
    directional_pair_residual
        Enables an additive per-feature directional user-item residual branch.
    directional_pair_align_dim
        Alignment dimension inside the directional residual branch. 0 uses
        `d_model`.
    directional_pair_hidden_mult
        Hidden multiplier for directional residual per-feature MLPs.
    directional_pair_scale_init
        Initial residual scale for directional branch output.
    directional_pair_use_abs_diff
        When True, includes |user-item| in directional per-feature inputs.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        emb_dim: int,
        d_model: int,
        n_layers: int = 3,
        rank: int = 64,
        dropout: float = 0.01,
        input_mode: str = "raw",
        d_cross: int = None,
        field_type_emb: bool = False,
        field_type_emb_granularity: str = "per_field",
        n_user_tokens: int = None,
        n_item_tokens: int = None,
        senet: bool = False,
        senet_reduction: int = 3,
        senet_groups: int = 2,
        field_norm: bool = False,
        factorized_post: bool = False,
        factorized_post_output: bool = False,
        factorized_post_output_layers: int = 1,
        anchor_mode: str = "fixed_x0",
        anchor_rank: int = 0,
        anchor_init: float = 2.0,
        field_regulation: bool = False,
        field_regulation_tau: float = 1.0,
        field_regulation_mode: str = "static",
        field_regulation_hidden: int = 0,
        cross_experts: int = 1,
        cross_expert_hidden: int = 0,
        directional_pair_residual: bool = False,
        directional_pair_align_dim: int = 0,
        directional_pair_hidden_mult: int = 2,
        directional_pair_scale_init: float = 1e-2,
        directional_pair_use_abs_diff: bool = False,
    ) -> None:
        super().__init__()
        self._schema = schema
        self._input_mode = input_mode

        if field_type_emb and d_cross is None:
            raise ValueError("field_type_emb requires d_cross to be set")
        valid_field_granularities = {"per_field", "entity_dtype_plus_derived_cont"}
        if field_type_emb_granularity not in valid_field_granularities:
            raise ValueError(
                f"field_type_emb_granularity must be one of {sorted(valid_field_granularities)}"
            )
        if input_mode == "ns_tokenized" and n_user_tokens is None:
            raise ValueError("input_mode='ns_tokenized' requires n_user_tokens and n_item_tokens")
        if field_regulation_tau <= 0:
            raise ValueError("field_regulation_tau must be > 0")
        if field_regulation_hidden < 0:
            raise ValueError("field_regulation_hidden must be >= 0")
        if cross_experts < 1:
            raise ValueError("cross_experts must be >= 1")
        if cross_expert_hidden < 0:
            raise ValueError("cross_expert_hidden must be >= 0")
        if directional_pair_hidden_mult <= 0:
            raise ValueError("directional_pair_hidden_mult must be > 0")

        _entity_type_map = {"user_cat": 0, "item_cat": 1, "numerical": 2}

        # Numericals are the same in both modes: per-field at native dims
        cont_specs = schema.query(
            "dtype = 'numerical' and scope = 'static' and source != 'metadata'"
        )
        self._cont_dims = [s.dim for s in cont_specs]
        self._has_cont = len(cont_specs) > 0
        if self._has_cont:
            names_str = ", ".join(f"'{s.name}'" for s in cont_specs)
            self._cont_expr = f"name in ({names_str}) and scope = 'static' and source != 'metadata'"

        if input_mode == "raw":
            user_cat_specs = schema.query(
                "entity = 'user' and dtype = 'categorical' and scope = 'static' and source != 'metadata'"
            )
            item_cat_specs = schema.query(
                "entity = 'item' and dtype = 'categorical' and scope = 'static' and source != 'metadata'"
            )

            self._n_user_cat = len(user_cat_specs)
            self._n_item_cat = len(item_cat_specs)
            self._user_cat_splits = [emb_dim] * len(user_cat_specs)
            self._item_cat_splits = [emb_dim] * len(item_cat_specs)

            feature_dims = [emb_dim] * (self._n_user_cat + self._n_item_cat) + [
                s.dim for s in cont_specs
            ]
            feature_types = (
                [_entity_type_map["user_cat"]] * self._n_user_cat
                + [_entity_type_map["item_cat"]] * self._n_item_cat
                + [_entity_type_map["numerical"]] * len(cont_specs)
            )
            per_feature_names = (
                [spec.name for spec in user_cat_specs]
                + [spec.name for spec in item_cat_specs]
                + [spec.name for spec in cont_specs]
            )
            per_feature_hierarchy_groups = _field_hierarchy_groups_from_schema_expr(
                schema, per_feature_names
            )

        elif input_mode == "ns_tokenized":
            self._n_user_tokens = n_user_tokens
            self._n_item_tokens = n_item_tokens

            feature_dims = [d_model] * (n_user_tokens + n_item_tokens) + [s.dim for s in cont_specs]
            feature_types = (
                [_entity_type_map["user_cat"]] * n_user_tokens
                + [_entity_type_map["item_cat"]] * n_item_tokens
                + [_entity_type_map["numerical"]] * len(cont_specs)
            )
            per_feature_names = (
                [f"user_token_{i}" for i in range(n_user_tokens)]
                + [f"item_token_{i}" for i in range(n_item_tokens)]
                + [spec.name for spec in cont_specs]
            )
            per_feature_hierarchy_groups = (
                ["user_cat"] * n_user_tokens
                + ["item_cat"] * n_item_tokens
                + _field_hierarchy_groups_from_schema_expr(
                    schema,
                    [spec.name for spec in cont_specs],
                )
            )
        else:
            raise ValueError(f"Unknown input_mode {input_mode!r}")

        self._n_features = len(feature_dims)
        self._feature_groups = list(per_feature_hierarchy_groups)
        self._field_names = list(per_feature_names)

        # Per-feature projections
        if d_cross is not None:
            self.projections = nn.ModuleList(
                [nn.Linear(dim, d_cross, bias=False) for dim in feature_dims]
            )
        else:
            self.projections = None

        # Hierarchical field embedding: entity-type + field identity
        if field_type_emb:
            self.entity_type_emb = nn.Embedding(len(_entity_type_map), d_cross)
            self.register_buffer("_type_ids", torch.tensor(feature_types, dtype=torch.long))

            field_group_labels: list[str]
            field_ids: torch.Tensor
            if field_type_emb_granularity == "per_field":
                self.field_emb = nn.Embedding(self._n_features, d_cross)
                field_ids = torch.arange(self._n_features, dtype=torch.long)
                field_group_labels = ["per_field"]
            else:
                per_feature_groups = per_feature_hierarchy_groups
                field_group_labels = []
                for group_name in per_feature_groups:
                    if group_name not in field_group_labels:
                        field_group_labels.append(group_name)
                group_to_id = {group_name: idx for idx, group_name in enumerate(field_group_labels)}
                field_ids = torch.tensor(
                    [group_to_id[group_name] for group_name in per_feature_groups],
                    dtype=torch.long,
                )
                self.field_emb = nn.Embedding(len(field_group_labels), d_cross)
            self.register_buffer("_field_ids", field_ids)
            self._field_emb_group_labels = field_group_labels
            self._field_emb_granularity = field_type_emb_granularity
        else:
            self.entity_type_emb = None
            self.field_emb = None
            self._field_emb_group_labels = []
            self._field_emb_granularity = "none"

        # Per-field LayerNorm (operates on effective dims: d_cross if projecting, else native)
        if field_norm:
            effective_dims = [d_cross] * self._n_features if d_cross else feature_dims
            self.field_norm = FieldNorm(effective_dims)
        else:
            self.field_norm = None

        # SENET field gating (operates on per-field embeddings before concat)
        # d_field is the effective per-field width: d_cross if projecting, d_model if
        # ns_tokenized (already uniform), None if raw with mixed native dims.
        if d_cross is not None:
            d_field_effective = d_cross
        elif input_mode == "ns_tokenized":
            d_field_effective = d_model
        else:
            d_field_effective = None

        if senet:
            if d_field_effective is not None and d_field_effective % senet_groups != 0:
                raise ValueError(
                    f"effective field dim ({d_field_effective}) must be divisible by senet_groups ({senet_groups})"
                )
            self.senet = FieldSENET(
                self._n_features,
                d_field=d_field_effective,
                reduction=senet_reduction,
                n_groups=senet_groups,
            )
        else:
            self.senet = None

        # Compute cross input dimension
        if d_cross is not None:
            cross_input_dim = self._n_features * d_cross
            field_dims_for_regulation = [d_cross] * self._n_features
            effective_feature_dims = [d_cross] * self._n_features
        else:
            cross_input_dim = sum(feature_dims)
            field_dims_for_regulation = feature_dims
            effective_feature_dims = feature_dims

        # Per-field widths in the flattened cross input, in concat order
        # (user cats, item cats, numericals) — consumed by GDCN diagnostics.
        self._effective_feature_dims = list(effective_feature_dims)

        if directional_pair_residual:
            self.directional_pair = DirectionalPairResidual(
                field_dims=effective_feature_dims,
                field_groups=self._feature_groups,
                d_model=d_model,
                dropout=dropout,
                align_dim=directional_pair_align_dim,
                hidden_mult=directional_pair_hidden_mult,
                scale_init=directional_pair_scale_init,
                use_abs_diff=directional_pair_use_abs_diff,
            )
        else:
            self.directional_pair = None

        self._needs_per_field = (
            d_cross is not None
            or field_type_emb
            or senet
            or field_norm
            or directional_pair_residual
        )
        self.network = GDCNNetwork(
            cross_input_dim,
            d_model,
            n_layers,
            rank,
            dropout,
            factorized_post=factorized_post,
            factorized_post_output=factorized_post_output,
            factorized_post_output_layers=factorized_post_output_layers,
            anchor_mode=anchor_mode,
            anchor_rank=anchor_rank,
            anchor_init=anchor_init,
            field_regulation=field_regulation,
            field_regulation_tau=field_regulation_tau,
            field_regulation_mode=field_regulation_mode,
            field_regulation_hidden=field_regulation_hidden,
            field_dims=field_dims_for_regulation if field_regulation else None,
            cross_experts=cross_experts,
            cross_expert_hidden=cross_expert_hidden,
        )

    def field_layout(self) -> dict[str, list]:
        """Field partition of the flattened cross input, for GDCN diagnostics.

        Returns per-field hierarchy-group labels and widths in concatenation
        order (user categoricals, item categoricals, numericals), plus the
        cumulative offsets that bound each field's slice in the cross vector.
        """
        dims = list(self._effective_feature_dims)
        offsets = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)
        return {
            "names": list(self._field_names),
            "groups": list(self._feature_groups),
            "dims": dims,
            "offsets": offsets,
        }

    def forward(
        self,
        tensor_reprs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Assemble cross input and run GDCN.

        Parameters
        ----------
        tensor_reprs
            Dictionary of precomputed tensor representations. Raw mode uses
            ``__user_cat_flat`` and ``__item_cat_flat``; ns_tokenized mode
            uses ``__user_ns_tokens`` and ``__item_ns_tokens``.
        batch
            Full batch dict for numerical feature extraction (raw mode only).

        Returns
        -------
        torch.Tensor
            Cross-interaction representation [B, d_model].
        """
        if self._input_mode == "ns_tokenized":
            features = self._assemble_ns(tensor_reprs, batch)
        else:
            features = self._assemble_raw(tensor_reprs, batch)

        if self.projections is not None:
            features = [proj(feat) for proj, feat in zip(self.projections, features)]

        if self.entity_type_emb is not None:
            entity_vecs = self.entity_type_emb(self._type_ids)
            field_vecs = self.field_emb(self._field_ids)
            features = [feat + entity_vecs[i] + field_vecs[i] for i, feat in enumerate(features)]

        if self.field_norm is not None:
            features = self.field_norm(features)

        if self.senet is not None:
            features = self.senet(features)

        network_out = self.network(torch.cat(features, dim=-1))
        if self.directional_pair is not None:
            network_out = network_out + self.directional_pair(features)
        return network_out

    def _assemble_raw(
        self,
        tensor_reprs: dict[str, Any],
        batch: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Assemble per-feature list from raw embeddings + numericals."""
        user_cat_flat = tensor_reprs["__user_cat_flat"]
        item_cat_flat = tensor_reprs["__item_cat_flat"]

        if not self._needs_per_field:
            parts = [user_cat_flat, item_cat_flat]
            if self._has_cont:
                cont = self._schema.extract(batch, expr=self._cont_expr, cat=True)
                parts.append(cont)
            return [torch.cat(parts, dim=-1)]

        features: list[torch.Tensor] = []
        if self._n_user_cat > 0:
            features.extend(user_cat_flat.split(self._user_cat_splits, dim=-1))
        if self._n_item_cat > 0:
            features.extend(item_cat_flat.split(self._item_cat_splits, dim=-1))
        if self._has_cont:
            cont = self._schema.extract(batch, expr=self._cont_expr, cat=True)
            features.extend(cont.split(self._cont_dims, dim=-1))
        return features

    def _assemble_ns(
        self,
        tensor_reprs: dict[str, Any],
        batch: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Assemble per-token list from post-NS representations + raw numericals."""
        user_ns = tensor_reprs["__user_ns_tokens"]  # [B, n_user, D]
        item_ns = tensor_reprs["__item_ns_tokens"]  # [B, n_item, D]

        features: list[torch.Tensor] = []
        features.extend(user_ns.unbind(dim=1))
        features.extend(item_ns.unbind(dim=1))
        if self._has_cont:
            cont = self._schema.extract(batch, expr=self._cont_expr, cat=True)
            features.extend(cont.split(self._cont_dims, dim=-1))
        return features


class GTCLiteMixer(nn.Module):
    """General Tabular Combiner (GTC) Lite grouped pairwise mixer.

    Parameters
    ----------
    n_fields
        Number of pathway fields.
    d_field
        Per-pathway feature width.
    output_dim
        Output width.
    n_groups
        Maximum number of latent groups.
    pair_rank
        Rank of the shared bilinear pair sketch.
    pair_hidden
        Hidden width for the shared pair kernel MLP.
    dropout
        Dropout used inside the pair kernel.
    activation
        Activation used in the pair kernel.
    activation_scale
        Optional scale for `scaled_tanh`.
    assignment_temperature
        Temperature for soft group assignment.
    assignment_scale
        Multiplicative scale on assignment logits before softmax.
    group_gate_init
        Initial logit for per-group gates.
    pair_gate_init
        Initial logit for per-pair gates.
    group_sparsity_weight
        Weight for mean group-gate activation auxiliary loss.
    pair_sparsity_weight
        Weight for mean pair-gate activation auxiliary loss.
    assignment_entropy_weight
        Weight for assignment entropy auxiliary loss.
    """

    def __init__(
        self,
        *,
        n_fields: int,
        d_field: int,
        output_dim: int = 1,
        n_groups: int = 8,
        pair_rank: int = 16,
        pair_hidden: int = 64,
        dropout: float = 0.0,
        activation: str = "silu",
        activation_scale: float = None,
        assignment_temperature: float = 1.0,
        assignment_scale: float = 1.0,
        group_gate_init: float = 0.0,
        pair_gate_init: float = 0.0,
        group_sparsity_weight: float = 1e-3,
        pair_sparsity_weight: float = 1e-3,
        assignment_entropy_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_fields <= 0:
            raise ValueError("n_fields must be > 0")
        if d_field <= 0:
            raise ValueError("d_field must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if n_groups <= 1:
            raise ValueError("n_groups must be > 1")
        if pair_rank <= 0:
            raise ValueError("pair_rank must be > 0")
        if pair_hidden <= 0:
            raise ValueError("pair_hidden must be > 0")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be > 0")
        if assignment_scale <= 0:
            raise ValueError("assignment_scale must be > 0")
        if group_sparsity_weight < 0:
            raise ValueError("group_sparsity_weight must be >= 0")
        if pair_sparsity_weight < 0:
            raise ValueError("pair_sparsity_weight must be >= 0")
        if assignment_entropy_weight < 0:
            raise ValueError("assignment_entropy_weight must be >= 0")

        self.n_fields = n_fields
        self.d_field = d_field
        self.n_groups = n_groups
        self.assignment_temperature = assignment_temperature
        self.assignment_scale = assignment_scale
        self.group_sparsity_weight = group_sparsity_weight
        self.pair_sparsity_weight = pair_sparsity_weight
        self.assignment_entropy_weight = assignment_entropy_weight

        self.assignment = nn.Linear(d_field, n_groups, bias=False)
        self.group_gate_logits = nn.Parameter(torch.full((n_groups,), group_gate_init))
        pair_indices = torch.triu_indices(n_groups, n_groups, offset=1)
        self.register_buffer("_pair_i", pair_indices[0], persistent=False)
        self.register_buffer("_pair_j", pair_indices[1], persistent=False)
        self.pair_gate_logits = nn.Parameter(torch.full((pair_indices.shape[1],), pair_gate_init))

        self.u_proj = nn.Linear(d_field, pair_rank, bias=False)
        self.v_proj = nn.Linear(d_field, pair_rank, bias=False)
        self.pair_kernel = nn.Sequential(
            nn.Linear(2 * d_field + pair_rank, pair_hidden),
            nn.LayerNorm(pair_hidden),
            build_activation(activation, scaled_tanh_scale=activation_scale),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, d_field),
        )
        self.output = nn.Linear(d_field, output_dim)
        self._last_aux_losses: dict[str, torch.Tensor] = {}

    def init_output_bias(self, value: float) -> None:
        """Initialize output bias.

        Parameters
        ----------
        value
            Scalar value assigned to all output bias terms.
        """
        nn.init.constant_(self.output.bias, value)

    def consume_aux_losses(self) -> dict[str, torch.Tensor]:
        """Return and clear auxiliary losses from the last forward pass."""
        aux = self._last_aux_losses
        self._last_aux_losses = {}
        return aux

    def _compute_aux_losses(
        self,
        assignments: torch.Tensor,
        group_gates: torch.Tensor,
        pair_gates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted structural auxiliary losses.

        Parameters
        ----------
        assignments
            Soft group assignments with shape `[B, F, K]`.
        group_gates
            Group gate activations with shape `[K]`.
        pair_gates
            Pair gate activations with shape `[num_pairs]`.
        """
        aux: dict[str, torch.Tensor] = {}
        if self.group_sparsity_weight > 0:
            aux["gtc_group_sparsity"] = self.group_sparsity_weight * group_gates.mean()
        if self.pair_sparsity_weight > 0:
            aux["gtc_pair_sparsity"] = self.pair_sparsity_weight * pair_gates.mean()
        if self.assignment_entropy_weight > 0:
            entropy = -(assignments * torch.log(assignments.clamp_min(1e-8))).sum(dim=-1).mean()
            aux["gtc_assignment_entropy"] = self.assignment_entropy_weight * entropy
        return aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix grouped pair interactions into a logit.

        Parameters
        ----------
        x
            Input tensor of shape `[B, F, D]`.
        """
        if x.dim() != 3:
            raise ValueError(f"GTCLiteMixer expects rank-3 input, got {tuple(x.shape)}")
        if x.shape[1] != self.n_fields:
            raise ValueError(f"GTCLiteMixer expects {self.n_fields} fields, got {x.shape[1]}")
        if x.shape[2] != self.d_field:
            raise ValueError(f"GTCLiteMixer expects field dim {self.d_field}, got {x.shape[2]}")

        assignment_logits = self.assignment_scale * self.assignment(x)
        assignments = torch.softmax(assignment_logits / self.assignment_temperature, dim=-1)
        group_mass = assignments.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        grouped = torch.einsum("bfk,bfd->bkd", assignments, x) / group_mass

        group_gates = torch.sigmoid(self.group_gate_logits)
        grouped = grouped * group_gates.view(1, self.n_groups, 1)

        u_group = self.u_proj(grouped)
        v_group = self.v_proj(grouped)

        pair_i = self._pair_i
        pair_j = self._pair_j
        gi = grouped.index_select(1, pair_i)
        gj = grouped.index_select(1, pair_j)
        ui = u_group.index_select(1, pair_i)
        vj = v_group.index_select(1, pair_j)
        pair_feat = torch.cat([gi, gj, ui * vj], dim=-1)

        pair_gates = torch.sigmoid(self.pair_gate_logits)
        pair_out = self.pair_kernel(pair_feat)
        mixed = (pair_out * pair_gates.view(1, -1, 1)).mean(dim=1)

        self._last_aux_losses = self._compute_aux_losses(assignments, group_gates, pair_gates)
        return self.output(mixed)
