"""SeqLocalWriter: schema-driven state encoder + per-domain Conv1D gating."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.data.schema import FeatureSchema


class ConvGate(nn.Module):
    """Depthwise-separable Conv1D local gating for [B, L, D] sequence tokens.

    Aggregates a k-wide local neighborhood via a depthwise+pointwise conv stack,
    derives a scalar gate and a state_add vector, then modulates the content tokens.

    In state mode, the conv operates on a separately built state vector rather
    than the content tokens directly. The gate and state_add still modulate content.

    When `ads_lite_enabled` is True and mode is "state", an optional bounded FiLM
    layer reshapes content via the local conv context before the gate is applied.
    At `film_zero_init=True`, the FiLM weights start at zero so initial output is
    identical to the non-ADS-lite path.

    TODO: far_conv — add a second dilated depthwise Conv1D that applies only to
    far-history positions (defined by a far_mask derived from time_bucket edges).
    Requires a clean interface to pass the far_mask into apply_conv; do not
    approximate it from padding alone.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        content_gate_mode: str,
        alpha_init: float,
        beta_init: float,
        gamma_init: float,
        gate_bias_init: float,
        conv_variant: str = "single",
        dilations: list[int] | None = None,
        score_eps: float = 1e-4,
        blend_content_state: bool = False,
        ads_lite_enabled: bool = False,
        film_cap: float = 0.1,
        film_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if content_gate_mode not in ("hard", "residual"):
            raise ValueError(
                f"content_gate_mode must be 'hard' or 'residual', got {content_gate_mode!r}"
            )
        if conv_variant not in ("single", "multi_dilated"):
            raise ValueError(
                f"conv_variant must be 'single' or 'multi_dilated', got {conv_variant!r}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for same-length output, got {kernel_size}")
        branch_dilations = [1]
        if conv_variant == "multi_dilated":
            if not dilations:
                raise ValueError("dilations must be non-empty when conv_variant='multi_dilated'")
            branch_dilations = list(dict.fromkeys(int(d) for d in dilations))
            if any(d <= 0 for d in branch_dilations):
                raise ValueError(f"all dilations must be > 0, got {branch_dilations}")
        self.content_gate_mode = content_gate_mode
        self.conv_variant = conv_variant
        self.branch_dilations = tuple(branch_dilations)
        self.score_eps = score_eps
        # When True, conv input = content + state (requires state mode).
        self.blend_content_state = blend_content_state
        pad = kernel_size // 2
        # Depthwise conv branch(es): per-channel temporal aggregation.
        self.dw_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size,
                    padding=pad * dilation,
                    dilation=dilation,
                    groups=d_model,
                )
                for dilation in branch_dilations
            ]
        )
        # Backward compatibility for diagnostics/tests that read `dw`.
        self.dw = self.dw_branches[0]
        self.branch_gates = (
            nn.Parameter(torch.zeros(len(self.dw_branches) - 1))
            if len(self.dw_branches) > 1
            else None
        )
        # Pointwise channel mix after SiLU
        self.pw = nn.Conv1d(d_model, d_model, 1)
        # Gate head: D → 1 scalar per position
        self.gate_conv = nn.Conv1d(d_model, 1, 1)
        nn.init.constant_(self.gate_conv.bias, gate_bias_init)
        # State add projection: D → D
        self.state_proj = nn.Conv1d(d_model, d_model, 1)

        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        if content_gate_mode == "residual":
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # ADS-lite FiLM: state-conditioned content reshaping (bounded, zero-init safe)
        self._film_cap = film_cap
        if ads_lite_enabled:
            self.film_proj = nn.Conv1d(d_model, 2 * d_model, 1)
            if film_zero_init:
                nn.init.zeros_(self.film_proj.weight)
                nn.init.zeros_(self.film_proj.bias)
        else:
            self.film_proj = None

        # Lightweight per-step diagnostic scalars (populated by _store_diag).
        self._diag: dict[str, float] = {}

    def forward(
        self,
        content: torch.Tensor,
        padding_mask: torch.Tensor,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate sequence tokens via local conv context.

        Parameters
        ----------
        content
            Content token sequence [B, L, D].
        padding_mask
            Bool [B, L]; True = padding position.
        state
            Optional state vector [B, L, D]. When provided, the conv operates
            on `state` instead of `content`. When None, content mode.

        Returns
        -------
        memory
            Gated token sequence [B, L, D].
        score_bias
            Per-position attention score bias [B, L].
        """
        if content.dim() != 3:
            raise RuntimeError(
                f"SeqLocalWriter requires padded [B, L, D] input, got {content.dim()}D"
            )
        if state is not None and self.blend_content_state:
            conv_input = content + state
        else:
            conv_input = state if state is not None else content
        x = conv_input.transpose(1, 2)  # [B, D, L]
        # Prevent padding positions from bleeding into valid neighbors via conv
        x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        branches = [F.silu(dw(x)) for dw in self.dw_branches]
        if self.branch_gates is None:
            local = branches[0]  # [B, D, L]
        else:
            # Residual-style branch merge: dilation-1 anchor + gated extras.
            gates = torch.sigmoid(self.branch_gates).to(dtype=x.dtype, device=x.device)
            local = branches[0]
            for gate, branch in zip(gates, branches[1:], strict=True):
                local = local + gate * branch
            local = local / (1.0 + gates.sum())
        local = self.pw(local)  # [B, D, L]

        gate = torch.sigmoid(self.gate_conv(local)).transpose(1, 2)  # [B, L, 1]
        state_add = self.state_proj(local).transpose(1, 2)  # [B, L, D]

        # ADS-lite FiLM: state-local context softly reshapes content before gating.
        # film_gamma / film_beta are bounded to [-film_cap, film_cap] via tanh.
        # At zero init both are 0, so mod_content == content and output is unchanged.
        if self.film_proj is not None:
            film = self.film_proj(local)  # [B, 2D, L]
            film_gamma, film_beta = film.chunk(2, dim=1)  # each [B, D, L]
            film_gamma = self._film_cap * torch.tanh(film_gamma).transpose(1, 2)  # [B, L, D]
            film_beta = self._film_cap * torch.tanh(film_beta).transpose(1, 2)  # [B, L, D]
            mod_content = content * (1.0 + film_gamma) + film_beta
        else:
            film_gamma = None
            film_beta = None
            mod_content = content

        if self.content_gate_mode == "hard":
            memory = mod_content * gate + self.beta * state_add
        else:
            memory = mod_content * (1.0 + self.alpha * (gate - 0.5)) + self.beta * state_add

        # Preserve padding positions unchanged (always restores to original content).
        valid = (~padding_mask).unsqueeze(-1)  # [B, L, 1]
        memory = torch.where(valid, memory, content)

        score_bias = self.gamma * torch.log(gate.squeeze(-1).clamp_min(self.score_eps))
        score_bias = score_bias.masked_fill(padding_mask, 0.0)

        self._store_diag(gate, film_gamma, film_beta)
        return memory, score_bias

    @torch.compiler.disable
    def _store_diag(
        self,
        gate: torch.Tensor,
        film_gamma: torch.Tensor = None,
        film_beta: torch.Tensor = None,
    ) -> None:
        """Cache per-step scalar stats for the diagnostics system.

        Runs outside the compiled graph (torch.compiler.disable) so .item() calls
        are safe without causing graph breaks in the main forward pass.
        """
        g = gate.detach().float()
        self._diag["gate_mean"] = g.mean().item()
        self._diag["gate_std"] = g.std().item()
        if film_gamma is not None:
            self._diag["film_gamma_abs_mean"] = film_gamma.detach().float().abs().mean().item()
            self._diag["film_beta_abs_mean"] = film_beta.detach().float().abs().mean().item()


class SeqLocalWriter(nn.Module):
    """Unified seq local writer: state encoder + Conv1D gate, one module.

    Handles both modes:

    - **state mode**: auto-discovers features by name pattern from the schema,
      creates one embedding table per matched spec, and sums their embeddings
      to produce a state vector. The Conv1D writer then uses that state to gate
      the content tokens.
    - **content mode**: no embeddings. The Conv1D writer operates on the content
      tokens directly (state=None).

    Parameters
    ----------
    schema
        FeatureSchema with all blocks' output_specs already registered.
    d_model
        Token / embedding dimension.
    emb_dim
        Input dimension of action embeddings (for projection).
    domains
        Sequence domains that have a writer enabled.
    time_embedding
        Shared time_bucket embedding (owned by DragonChariot, used by both the
        main seq path and the writer state). Passed in to avoid duplication.
    kernel_size
        Conv1D receptive field width (must be odd).
    conv_variant
        Writer conv variant: ``"single"`` (default) or ``"multi_dilated"``.
    dilations
        Dilation factors for ``"multi_dilated"`` conv variant.
    content_gate_mode
        ``'hard'`` or ``'residual'``.
    alpha_init
        Initial alpha scalar (residual mode).
    beta_init
        Initial beta scalar (state_add scale).
    gamma_init
        Initial gamma for score_bias.
    gate_bias_init
        Initial bias for the gate projection.
    mode
        ``'state'`` or ``'content'``.
    score_eps
        Clamp floor for gate before log in score_bias.
    feature_patterns
        Name-glob patterns to discover features from the schema. Each matched
        spec gets an embedding table sized by its `vocab_size`. Patterns use
        ``{domain}`` as a placeholder expanded per-domain. Required in state
        mode.
    ads_lite
        Optional ADS-lite FiLM config dict. Recognized keys:

        - ``enabled`` (bool, default False): activate FiLM path.
        - ``film_cap`` (float, default 0.1): tanh cap for gamma/beta.
        - ``film_zero_init`` (bool, default True): zero-init film_proj so
          initial output is identical to the non-ADS-lite path.

        When absent or ``enabled=False``, ConvGate behavior is unchanged.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        emb_dim: int,
        domains: list[str],
        time_embedding: nn.Embedding,
        kernel_size: int,
        content_gate_mode: str,
        alpha_init: float,
        beta_init: float,
        gamma_init: float,
        gate_bias_init: float,
        conv_variant: str = "single",
        dilations: list[int] | None = None,
        mode: str = "content",
        score_eps: float = 1e-4,
        feature_patterns: list[str] = None,
        blend_content_state: bool = False,
        ads_lite: dict = None,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._domains = domains
        self._mode = mode
        self._time_embedding = time_embedding if mode == "state" else None

        # --- State encoder: schema-driven feature embeddings (state mode only) ---
        # Embeddings are SHARED across domains (keyed by domain-stripped suffix).
        # E.g. seq_a_slw_gap and seq_b_slw_gap both use the "slw_gap" table.
        self._domain_specs: dict[str, list[tuple[str, str]]] = {}
        self.embeddings = nn.ModuleDict()
        self.action_proj = nn.ModuleDict()
        self.domain_emb = nn.ParameterDict()

        if mode == "state":
            if not feature_patterns:
                raise ValueError("feature_patterns required for state mode")
            for domain in domains:
                domain_specs: list[tuple[str, str]] = []
                for pattern in feature_patterns:
                    expanded = pattern.replace("{domain}", domain)
                    matched = schema.query(f"name matches '{expanded}' and domain = '{domain}'")
                    for spec in matched:
                        shared_key = spec.name.removeprefix(f"{domain}_")
                        if shared_key not in self.embeddings:
                            self.embeddings[shared_key] = nn.Embedding(
                                spec.vocab_size, d_model, padding_idx=0
                            )
                        domain_specs.append((shared_key, spec.batch_key))
                self._domain_specs[domain] = domain_specs

            for domain in domains:
                self.action_proj[domain] = nn.Linear(emb_dim, d_model)
                self.domain_emb[domain] = nn.Parameter(torch.zeros(d_model))

        # Parse ADS-lite FiLM config (all keys optional, default disabled).
        _ads = ads_lite or {}
        _ads_enabled = bool(_ads.get("enabled", False))
        _film_cap = float(_ads.get("film_cap", 0.1))
        _film_zero_init = bool(_ads.get("film_zero_init", True))

        # --- Per-domain Conv1D gates ---
        self.convs = nn.ModuleDict()
        for domain in domains:
            self.convs[domain] = ConvGate(
                d_model=d_model,
                kernel_size=kernel_size,
                conv_variant=conv_variant,
                dilations=dilations,
                content_gate_mode=content_gate_mode,
                alpha_init=alpha_init,
                beta_init=beta_init,
                gamma_init=gamma_init,
                gate_bias_init=gate_bias_init,
                score_eps=score_eps,
                blend_content_state=blend_content_state,
                ads_lite_enabled=_ads_enabled,
                film_cap=_film_cap,
                film_zero_init=_film_zero_init,
            )

    def encode_states(
        self,
        domains: list[str],
        tb_ids_list: list[torch.Tensor | None],
        batch: dict[str, Any],
        action_embs: dict[str, torch.Tensor] = None,
    ) -> list[torch.Tensor | None]:
        """Build writer state vectors for all domains.

        In content mode, returns a list of Nones (conv operates on content).

        Parameters
        ----------
        domains
            Ordered domain names matching tb_ids_list indices.
        tb_ids_list
            Per-domain recency bucket IDs [B, L], or None.
        batch
            Full batch dict with all feature tensors.
        action_embs
            Pre-looked-up action embeddings keyed by domain. Each value is
            [B, L, emb_dim], or absent/None for domains without an action slot.
        """
        if self._mode != "state":
            return [None] * len(domains)

        action_embs = action_embs or {}
        state_list: list[torch.Tensor | None] = []

        for i, domain in enumerate(domains):
            if domain not in self.convs or tb_ids_list[i] is None:
                state_list.append(None)
                continue

            action_emb = action_embs.get(domain)
            if action_emb is not None:
                state = self.action_proj[domain](action_emb)
            else:
                B, L = tb_ids_list[i].shape
                state = tb_ids_list[i].new_zeros(B, L, self._d_model, dtype=torch.float)

            if self._time_embedding is not None:
                state = state + self._time_embedding(tb_ids_list[i])

            for spec_name, batch_key in self._domain_specs[domain]:
                state = state + self.embeddings[spec_name](batch[batch_key])

            state = state + self.domain_emb[domain].to(dtype=state.dtype, device=state.device)
            state_list.append(state)

        return state_list

    def apply_conv(
        self,
        domain: str,
        content: torch.Tensor,
        padding_mask: torch.Tensor,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Conv1D gating to content tokens (compile-friendly).

        Parameters
        ----------
        domain
            Sequence domain name.
        content
            Content token sequence [B, L, D].
        padding_mask
            Bool [B, L]; True = padding position.
        state
            Pre-built state vector [B, L, D], or None for content mode.

        Returns
        -------
        memory
            Gated token sequence [B, L, D].
        score_bias
            Per-position attention score bias [B, L].
        """
        return self.convs[domain](content, padding_mask, state=state)

    def reinit_high_cardinality(self, cardinality_threshold: int) -> set[int]:
        """Reinit embedding tables above threshold. Returns data_ptrs of reinitialized weights."""
        reinitialized: set[int] = set()
        for name, emb in self.embeddings.items():
            if emb.num_embeddings > cardinality_threshold:
                nn.init.normal_(emb.weight, std=0.02)
                if emb.padding_idx is not None:
                    emb.weight.data[emb.padding_idx].zero_()
                reinitialized.add(emb.weight.data_ptr())
        return reinitialized

    def snapshot_weights(self, vocab_threshold: int) -> dict[str, torch.Tensor]:
        """Clone embedding weights for tables with vocab <= threshold.

        Keys are prefixed with ``slw.`` so the model can merge all submodule
        snapshots into a single flat dict without worrying about collisions.
        """
        snapshot = {}
        for name, emb in self.embeddings.items():
            if emb.num_embeddings <= vocab_threshold:
                snapshot[f"slw.{name}"] = emb.weight.data.clone()
        return snapshot

    def restore_weights(self, snapshot: dict[str, torch.Tensor]) -> set[int]:
        """Restore previously snapshotted weights. Returns restored data_ptrs.

        Accepts the same ``slw.``-prefixed keys that `snapshot_weights` produces.
        Ignores keys that don't start with ``slw.``.
        """
        ptrs: set[int] = set()
        for key, weight in snapshot.items():
            if not key.startswith("slw."):
                continue
            name = key.removeprefix("slw.")
            if name in self.embeddings:
                self.embeddings[name].weight.data.copy_(weight)
                ptrs.add(self.embeddings[name].weight.data_ptr())
        return ptrs
