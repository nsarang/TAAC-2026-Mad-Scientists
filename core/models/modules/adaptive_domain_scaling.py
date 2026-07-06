"""ADS-lite sequence personalization modules."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveSequenceScaler(nn.Module):
    """Apply user-conditioned FiLM residuals to sequence tokens before DIN.

    Parameters
    ----------
    d_model
        Token and context width.
    num_domains
        Number of sequence domains. Each domain gets a learned embedding.
    hidden_mult
        Hidden multiplier for the context MLP.
    dropout
        Dropout inside the FiLM generator.
    cap
        Maximum residual gate after `tanh`.
    scale_init
        Initial residual scale. `0.0` makes the module an exact no-op.
    """

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        hidden_mult: int,
        dropout: float,
        cap: float,
        scale_init: float,
    ) -> None:
        super().__init__()
        self._cap = float(cap)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.domain_emb = nn.Embedding(num_domains, d_model)
        hidden_dim = d_model * int(hidden_mult)
        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.last_gate_abs_mean = 0.0
        self.last_update_norm_ratio = 0.0
        self.gate_probe = _ScalarProbe()
        self.update_ratio_probe = _ScalarProbe()

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        domain_idx: int,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return user-personalized tokens with the same shape as `tokens`."""
        domain = torch.full(
            (context.shape[0],),
            int(domain_idx),
            dtype=torch.long,
            device=context.device,
        )
        film_context = context + self.domain_emb(domain)
        film_scale, film_bias = self.generator(film_context).chunk(2, dim=-1)
        gate = self._cap * torch.tanh(self.scale)

        if tokens.dim() == 3:
            update = tokens * film_scale.unsqueeze(1) + film_bias.unsqueeze(1)
        elif tokens.dim() == 2:
            if lengths is None:
                raise ValueError("lengths required for jagged ADS-lite tokens")
            context_idx = torch.repeat_interleave(
                torch.arange(context.shape[0], device=tokens.device),
                lengths,
            )
            update = tokens * film_scale[context_idx] + film_bias[context_idx]
        else:
            raise RuntimeError(f"expected 2D or 3D tokens, got {tokens.dim()}D")

        out = tokens + gate * self.dropout(update)
        self._record_diagnostics(tokens, out, gate)
        return out

    def _record_diagnostics(
        self,
        tokens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        """Store scalar values for diagnostics hooks."""
        if torch.compiler.is_compiling():
            return
        update_ratio = (out - tokens).detach().float().norm()
        update_ratio = update_ratio / tokens.detach().float().norm().clamp_min(1e-8)
        gate_abs_mean = gate.detach().abs().float().reshape(1)
        update_ratio_t = update_ratio.detach().float().reshape(1)
        self.last_gate_abs_mean = float(gate_abs_mean.item())
        self.last_update_norm_ratio = float(update_ratio_t.item())
        self.gate_probe(gate_abs_mean)
        self.update_ratio_probe(update_ratio_t)


class PersonalizedCandidateQueryGenerator(nn.Module):
    """Generate chunk-specific target queries for PCRG-lite DIN scoring.

    The original target query remains the shared base. A small residual branch
    generates private chunk queries from target, user context, and domain.
    `scale_init=0.0` makes the generated per-token queries exactly equal to the
    original target query at initialization.
    """

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        num_chunks: int,
        hidden_mult: int,
        dropout: float,
        cap: float,
        scale_init: float,
        generator_type: str = "shared",
        private_cap: float = None,
        private_hidden_mult: int = None,
    ) -> None:
        super().__init__()
        if generator_type not in ("shared", "shared_private"):
            raise ValueError(
                f"generator_type must be 'shared' or 'shared_private', got {generator_type!r}"
            )
        self.generator_type = generator_type
        self.num_chunks = int(num_chunks)
        self._cap = float(cap)
        self._private_cap = float(private_cap) if private_cap is not None else float(cap)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.private_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.domain_emb = nn.Embedding(num_domains, d_model)
        hidden_dim = d_model * int(hidden_mult)
        shared_in_dim = d_model * 2 if generator_type == "shared_private" else d_model * 3
        self.generator = nn.Sequential(
            nn.LayerNorm(shared_in_dim),
            nn.Linear(shared_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_chunks * d_model),
        )
        private_hidden_dim = d_model * int(private_hidden_mult or hidden_mult)
        self.private_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model * 3),
                    nn.Linear(d_model * 3, private_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(private_hidden_dim, self.num_chunks * d_model),
                )
                for _ in range(num_domains)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.last_gate_abs_mean = 0.0
        self.last_private_gate_abs_mean = 0.0
        self.last_update_norm_ratio = 0.0
        self.gate_probe = _ScalarProbe()
        self.private_gate_probe = _ScalarProbe()
        self.update_ratio_probe = _ScalarProbe()

    def forward(
        self,
        target_query: torch.Tensor,
        context: torch.Tensor,
        domain_idx: int,
        seq_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-token score queries matching `seq_tokens` layout."""
        domain = torch.full(
            (context.shape[0],),
            int(domain_idx),
            dtype=torch.long,
            device=context.device,
        )
        domain_context = self.domain_emb(domain)
        if self.generator_type == "shared_private":
            shared_input = torch.cat([target_query, context], dim=-1)
            private_input = torch.cat([target_query, context, domain_context], dim=-1)
            shared = self._reshape_generated(self.generator(shared_input), target_query)
            private = self._reshape_generated(
                self.private_generators[int(domain_idx)](private_input),
                target_query,
            )
            gate = self._cap * torch.tanh(self.scale)
            private_gate = self._private_cap * torch.tanh(self.private_scale)
            delta = gate * self.dropout(shared) + private_gate * self.dropout(private)
        else:
            generator_input = torch.cat([target_query, context, domain_context], dim=-1)
            private = self._reshape_generated(self.generator(generator_input), target_query)
            gate = self._cap * torch.tanh(self.scale)
            private_gate = torch.zeros_like(gate)
            delta = gate * self.dropout(private)
        chunks = target_query.unsqueeze(1) + delta

        if seq_tokens.dim() == 3:
            out = self._expand_padded(chunks, seq_tokens.shape[1], lengths)
            base = target_query.unsqueeze(1).expand_as(out)
        elif seq_tokens.dim() == 2:
            out = self._expand_jagged(chunks, lengths)
            base = target_query.repeat_interleave(lengths.long(), dim=0)
        else:
            raise RuntimeError(f"expected 2D or 3D tokens, got {seq_tokens.dim()}D")

        self._record_diagnostics(base, out, gate, private_gate)
        return out

    def _reshape_generated(
        self,
        generated: torch.Tensor,
        target_query: torch.Tensor,
    ) -> torch.Tensor:
        """Reshape generated chunk deltas to `[B, num_chunks, D]`."""
        return generated.reshape(
            target_query.shape[0],
            self.num_chunks,
            target_query.shape[-1],
        )

    def _expand_padded(
        self,
        chunks: torch.Tensor,
        seq_len: int,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Map each padded position to one generated chunk query."""
        B, _G, D = chunks.shape
        positions = torch.arange(seq_len, device=chunks.device).unsqueeze(0).expand(B, -1)
        safe_lengths = lengths.to(chunks.device).long().clamp_min(1).unsqueeze(1)
        chunk_idx = (positions * self.num_chunks // safe_lengths).clamp(max=self.num_chunks - 1)
        return chunks.gather(1, chunk_idx.unsqueeze(-1).expand(-1, -1, D))

    def _expand_jagged(self, chunks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Map each flat jagged position to one generated chunk query."""
        lengths = lengths.to(chunks.device).long()
        seg_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=chunks.device), lengths
        )
        if seg_ids.numel() == 0:
            return chunks.new_empty((0, chunks.shape[-1]))
        starts = torch.cumsum(
            torch.cat([lengths.new_zeros(1), lengths[:-1]]),
            dim=0,
        )
        local_pos = torch.arange(seg_ids.numel(), device=chunks.device) - starts[seg_ids]
        safe_lengths = lengths[seg_ids].clamp_min(1)
        chunk_idx = (local_pos * self.num_chunks // safe_lengths).clamp(max=self.num_chunks - 1)
        return chunks[seg_ids, chunk_idx]

    def _record_diagnostics(
        self,
        base: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor,
        private_gate: torch.Tensor,
    ) -> None:
        """Store scalar values for diagnostics hooks."""
        if torch.compiler.is_compiling():
            return
        update_ratio = (out - base).detach().float().norm()
        update_ratio = update_ratio / base.detach().float().norm().clamp_min(1e-8)
        gate_abs_mean = gate.detach().abs().float().reshape(1)
        private_gate_abs_mean = private_gate.detach().abs().float().reshape(1)
        update_ratio_t = update_ratio.detach().float().reshape(1)
        self.last_gate_abs_mean = float(gate_abs_mean.item())
        self.last_private_gate_abs_mean = float(private_gate_abs_mean.item())
        self.last_update_norm_ratio = float(update_ratio_t.item())
        self.gate_probe(gate_abs_mean)
        self.private_gate_probe(private_gate_abs_mean)
        self.update_ratio_probe(update_ratio_t)


class SimpleQueryBooster(nn.Module):
    """Small gated residual branch for DIN score-query enrichment.

    The module only returns alternate attention score queries. It does not
    mutate sequence tokens or the downstream target representation.
    """

    _WINDOW_TO_INDEX: ClassVar[dict[str, int]] = {"full": 0, "recent": 1, "mid": 2, "old": 3}

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        mode: str,
        cap: float,
        scale_init: float,
        dropout: float,
        hidden_mult: int,
        apply_to_full: bool,
        apply_to_windowed: bool,
        recent_policy: str,
        zero_init_delta: bool,
        use_layernorm: bool,
        use_context: bool,
    ) -> None:
        super().__init__()
        if mode not in ("domain_add", "window_domain_add", "sequence_pool"):
            raise ValueError(f"unknown query_boost mode: {mode!r}")
        if recent_policy not in ("identity", "boosted"):
            raise ValueError(f"unknown query_boost recent_policy: {recent_policy!r}")

        self.mode = mode
        self.apply_to_full = bool(apply_to_full)
        self.apply_to_windowed = bool(apply_to_windowed)
        self.recent_policy = recent_policy
        self._cap = float(cap)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.domain_emb = nn.Parameter(torch.empty(num_domains, d_model))
        self.window_emb = nn.Parameter(torch.empty(len(self._WINDOW_TO_INDEX), d_model))
        self.delta_norm = nn.LayerNorm(d_model) if use_layernorm else None
        self.dropout = nn.Dropout(dropout)

        self.context_proj = nn.Linear(d_model, d_model) if use_context else None
        if mode == "sequence_pool":
            nn.init.normal_(self.domain_emb, mean=0.0, std=d_model**-0.5)
            nn.init.normal_(self.window_emb, mean=0.0, std=d_model**-0.5)
            hidden_dim = d_model * int(hidden_mult)
            self.generator = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            if zero_init_delta:
                nn.init.zeros_(self.generator[-1].weight)
                nn.init.zeros_(self.generator[-1].bias)
        else:
            self.generator = None
            if zero_init_delta:
                nn.init.zeros_(self.domain_emb)
                nn.init.zeros_(self.window_emb)
            else:
                nn.init.normal_(self.domain_emb, mean=0.0, std=d_model**-0.5)
                nn.init.normal_(self.window_emb, mean=0.0, std=d_model**-0.5)

        self.last_gate_abs_mean = 0.0
        self.last_update_norm_ratio = 0.0
        self.last_cosine_to_target_mean = 1.0
        self.gate_probe = _ScalarProbe()
        self.update_ratio_probe = _ScalarProbe()
        self.cosine_to_target_probe = _ScalarProbe()

    def forward_domain(
        self,
        target_query: torch.Tensor,
        domain_idx: int,
    ) -> torch.Tensor:
        """Return a domain-specific boosted score query."""
        domain_delta = self._domain_delta(target_query, domain_idx)
        return self._apply_delta(target_query, domain_delta)

    def forward_window(
        self,
        target_query: torch.Tensor,
        domain_idx: int,
        window_id: str,
        seq_pool: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return a window/domain-specific boosted score query."""
        if window_id == "recent" and self.recent_policy == "identity":
            self._record_diagnostics(target_query, target_query, self._gate())
            return target_query
        if self.mode == "domain_add":
            return self.forward_domain(target_query, domain_idx)
        window_delta = self._domain_delta(target_query, domain_idx) + self._window_delta(
            target_query,
            window_id,
        )
        if self.mode == "sequence_pool":
            if seq_pool is None:
                raise ValueError("seq_pool is required for sequence_pool query_boost")
            generator_input = window_delta + seq_pool
            if context is not None and self.context_proj is not None:
                generator_input = generator_input + self.context_proj(context)
            window_delta = self.generator(generator_input)
        return self._apply_delta(target_query, window_delta)

    def _domain_delta(self, target_query: torch.Tensor, domain_idx: int) -> torch.Tensor:
        domain = torch.full(
            (target_query.shape[0],),
            int(domain_idx),
            dtype=torch.long,
            device=target_query.device,
        )
        return self.domain_emb[domain]

    def _window_delta(self, target_query: torch.Tensor, window_id: str) -> torch.Tensor:
        if window_id not in self._WINDOW_TO_INDEX:
            raise ValueError(f"unknown query_boost window_id: {window_id!r}")
        window = torch.full(
            (target_query.shape[0],),
            self._WINDOW_TO_INDEX[window_id],
            dtype=torch.long,
            device=target_query.device,
        )
        return self.window_emb[window]

    def _gate(self) -> torch.Tensor:
        return self._cap * torch.tanh(self.scale)

    def _apply_delta(self, target_query: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.delta_norm is not None:
            delta = self.delta_norm(delta)
        gate = self._gate()
        out = target_query + gate * self.dropout(delta)
        self._record_diagnostics(target_query, out, gate)
        return out

    def _record_diagnostics(
        self,
        target_query: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        """Store scalar values for diagnostics hooks."""
        if torch.compiler.is_compiling():
            return
        update_ratio = (out - target_query).detach().float().norm()
        update_ratio = update_ratio / target_query.detach().float().norm().clamp_min(1e-8)
        cosine = F.cosine_similarity(
            out.detach().float(),
            target_query.detach().float(),
            dim=-1,
        ).mean()
        gate_abs_mean = gate.detach().abs().float().reshape(1)
        update_ratio_t = update_ratio.detach().float().reshape(1)
        cosine_t = cosine.detach().float().reshape(1)
        self.last_gate_abs_mean = float(gate_abs_mean.item())
        self.last_update_norm_ratio = float(update_ratio_t.item())
        self.last_cosine_to_target_mean = float(cosine_t.item())
        self.gate_probe(gate_abs_mean)
        self.update_ratio_probe(update_ratio_t)
        self.cosine_to_target_probe(cosine_t)


class _ScalarProbe(nn.Module):
    """Expose scalar diagnostics through normal forward hooks."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return `value` unchanged."""
        return value
