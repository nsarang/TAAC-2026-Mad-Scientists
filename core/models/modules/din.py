"""DIN attention heads for behavioral sequences.

Contains both the original TargetAwareDINHead (4-way interaction MLP scoring)
and MultiChunkBidirectionalDIN (position-based exponential chunking with
bidirectional multi-head scoring via stacked weight tensors).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.models.modules.primitives import build_activation
from core.models.modules.segment_ops import segment_softmax, segment_sum


class ExponentialChunker(nn.Module):
    """Converts variable-length sequences into fixed-shape [B, K*N, D] tensors.

    Uses Option B (vectorized scatter-based pooling) for compile safety.
    Emits forward-ordered chunks only; bidirectional flipping is handled by
    the caller.

    Parameters
    ----------
    num_chunks
        K — number of chunks.
    chunk_tokens
        N — output tokens per chunk.
    newton_iters
        Fixed iteration count for Newton's method to solve r.
    """

    def __init__(self, num_chunks: int, chunk_tokens: int, newton_iters: int = 12) -> None:
        super().__init__()
        self.K = num_chunks
        self.N = chunk_tokens
        self.KN = num_chunks * chunk_tokens
        self.newton_iters = newton_iters

    def _solve_r(self, L: torch.Tensor) -> torch.Tensor:
        """Solve for r given sequence lengths via Newton's method.

        Solves f(r) = N * (r^K - 1) / (r - 1) - L = 0 for r > 1.

        Parameters
        ----------
        L
            Sequence lengths, shape ``(...)``. Must be > K*N for meaningful r.

        Returns
        -------
        torch.Tensor
            Exponential ratio r, same shape as L. Values >= 1.
        """
        N, K = self.N, self.K
        # Initial guess: r ≈ (L / N)^(1/K) works when K*N << L
        r = (L.float() / N).clamp(min=1.0).pow(1.0 / K).clamp(min=1.001)

        for _ in range(self.newton_iters):
            r_K = r.pow(K)
            rm1 = r - 1.0
            # f(r) = N * (r^K - 1) / (r - 1) - L
            f = N * (r_K - 1.0) / rm1 - L.float()
            # f'(r) = N * (K * r^(K-1) * (r-1) - (r^K - 1)) / (r-1)^2
            fp = N * (K * r.pow(K - 1) * rm1 - (r_K - 1.0)) / (rm1 * rm1)
            r = r - f / fp.clamp_min(1e-6)
            r = r.clamp(min=1.0)

        return r

    def _compute_chunk_boundaries(self, r: torch.Tensor) -> torch.Tensor:
        """Compute cumulative input spans for K chunks.

        Parameters
        ----------
        r
            Exponential ratio, shape ``(B,)`` or scalar.

        Returns
        -------
        torch.Tensor
            Cumulative spans, shape ``(..., K+1)``. boundaries[..., k] is the
            start position of chunk k; boundaries[..., K] = sum of all spans.
        """
        N, K = self.N, self.K
        # spans[k] = floor(N * r^k)
        k_range = torch.arange(K, device=r.device, dtype=r.dtype)
        if r.dim() == 0:
            spans = (N * r.pow(k_range)).floor()
        else:
            spans = (N * r.unsqueeze(-1).pow(k_range.unsqueeze(0))).floor()
        # Cumulative boundaries: [0, span_0, span_0+span_1, ...]
        boundaries = F.pad(spans.cumsum(dim=-1), (1, 0), value=0.0)
        return boundaries

    def forward(
        self,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Chunk sequences into fixed-shape [B, K*N, D].

        Parameters
        ----------
        seq_tokens
            Shape ``[B, L, D]`` (padded) or ``(total, D)`` (jagged).
        padding_mask_or_cu_seqlens
            Bool mask ``[B, L]`` (padded, True=pad) or cu_seqlens ``(B+1,)`` (jagged).
        time_bucket_ids
            Optional int tensor, same layout as seq_tokens but last dim absent.

        Returns
        -------
        chunked_tokens
            Shape ``[B, K*N, D]``.
        chunked_mask
            Bool shape ``[B, K*N]``; True = no valid source events (should be masked).
        chunked_tb
            Int shape ``[B, K*N]`` or None if time_bucket_ids is None.
        """
        if seq_tokens.dim() == 3:
            return self._forward_padded(seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids)
        return self._forward_jagged(seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids)

    def _forward_padded(
        self,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, L, D = seq_tokens.shape
        device = seq_tokens.device

        # Edge case: L <= K*N → pad to K*N, r=1, identity
        if L <= self.KN:
            pad_len = self.KN - L
            out_tokens = F.pad(seq_tokens, (0, 0, 0, pad_len))  # [B, KN, D]
            out_mask = F.pad(padding_mask, (0, pad_len), value=True)  # [B, KN]
            out_tb = None
            if time_bucket_ids is not None:
                out_tb = F.pad(time_bucket_ids, (0, pad_len), value=0)
            return out_tokens, out_mask, out_tb

        # Compute effective lengths per sample (non-padded)
        valid = ~padding_mask.bool()  # [B, L], True = valid
        lengths = valid.sum(dim=-1)  # [B]

        # Solve r per sample
        r = self._solve_r(lengths)  # [B]
        boundaries = self._compute_chunk_boundaries(r)  # [B, K+1]

        # Assign each position to a chunk and output slot
        pos = (
            torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )  # [1, L, 1]
        # chunk_ends: cumulative boundary positions per sample
        chunk_ends = boundaries[:, 1:-1].unsqueeze(1)  # [B, 1, K-1] (K-1 internal boundaries)
        # Position p is in chunk k if it's past k boundaries → count boundaries <= p
        chunk_ids = (pos >= chunk_ends).sum(dim=-1).long()  # [B, L]
        chunk_ids = chunk_ids.clamp(max=self.K - 1)

        pos_flat = (
            torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        )  # [B, L]

        # Position within chunk
        chunk_starts = boundaries[:, :-1]  # [B, K]
        chunk_start_for_pos = chunk_starts.gather(1, chunk_ids)  # [B, L]
        pos_in_chunk = pos_flat - chunk_start_for_pos  # [B, L]

        # Span of each chunk
        spans = boundaries[:, 1:] - boundaries[:, :-1]  # [B, K]
        span_for_pos = spans.gather(1, chunk_ids)  # [B, L]

        # Output slot within chunk: maps [0, span) → [0, N)
        output_slot = (pos_in_chunk * self.N / span_for_pos.clamp(min=1.0)).long()  # [B, L]
        output_slot = output_slot.clamp(min=0, max=self.N - 1)

        # Global output index: chunk_id * N + output_slot
        output_idx = chunk_ids * self.N + output_slot  # [B, L]
        output_idx = output_idx.clamp(min=0, max=self.KN - 1)

        # Scatter-mean pooling into [B, KN, D]
        out_tokens = torch.zeros(B, self.KN, D, device=device, dtype=seq_tokens.dtype)
        counts = torch.zeros(B, self.KN, device=device, dtype=seq_tokens.dtype)

        # Mask invalid positions so they don't contribute
        valid_f = valid.float()  # [B, L]
        idx_expand = output_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, L, D]
        masked_tokens = seq_tokens * valid_f.unsqueeze(-1)
        out_tokens.scatter_add_(1, idx_expand, masked_tokens)
        counts.scatter_add_(1, output_idx, valid_f)

        # Average
        out_tokens = out_tokens / counts.unsqueeze(-1).clamp(min=1.0)
        out_mask = counts == 0  # [B, KN], True = no valid events pooled

        # Time bucket IDs
        out_tb = None
        if time_bucket_ids is not None:
            tb_out = torch.zeros(B, self.KN, device=device, dtype=torch.float32)
            tb_float = time_bucket_ids.float() * valid_f
            tb_out.scatter_add_(1, output_idx, tb_float)
            out_tb = (tb_out / counts.clamp(min=1.0)).round().long()

        return out_tokens, out_mask, out_tb

    def _forward_jagged(
        self,
        seq_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        total, D = seq_tokens.shape
        B = cu_seqlens.shape[0] - 1
        device = seq_tokens.device
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()  # [B]

        # Segment IDs and local positions
        seg_ids = torch.repeat_interleave(torch.arange(B, device=device), lengths)  # [total]
        offsets = cu_seqlens[:-1].long()  # [B]
        local_pos = torch.arange(total, device=device) - offsets[seg_ids]  # [total]
        local_pos_f = local_pos.float()

        # Edge case: all sequences <= K*N
        max_len = lengths.max()
        if max_len <= self.KN:
            # Pad each sequence to KN, identity mapping
            out_tokens = torch.zeros(B, self.KN, D, device=device, dtype=seq_tokens.dtype)
            counts = torch.zeros(B, self.KN, device=device, dtype=seq_tokens.dtype)
            output_idx_flat = seg_ids * self.KN + local_pos.clamp(max=self.KN - 1)
            idx_expand = output_idx_flat.unsqueeze(-1).expand(-1, D)
            out_flat = out_tokens.reshape(B * self.KN, D)
            out_flat.scatter_add_(0, idx_expand, seq_tokens)
            out_tokens = out_flat.reshape(B, self.KN, D)
            counts_flat = counts.reshape(B * self.KN)
            counts_flat.scatter_add_(
                0, output_idx_flat, torch.ones(total, device=device, dtype=counts.dtype)
            )
            counts = counts_flat.reshape(B, self.KN)
            out_tokens = out_tokens / counts.unsqueeze(-1).clamp(min=1.0)
            out_mask = counts == 0

            out_tb = None
            if time_bucket_ids is not None:
                tb_out = torch.zeros(B, self.KN, device=device, dtype=torch.float32)
                tb_flat = tb_out.reshape(B * self.KN)
                tb_flat.scatter_add_(0, output_idx_flat, time_bucket_ids.float())
                tb_out = tb_flat.reshape(B, self.KN)
                out_tb = (tb_out / counts.clamp(min=1.0)).round().long()
            return out_tokens, out_mask, out_tb

        # Solve r per sample
        r = self._solve_r(lengths.float())  # [B]
        boundaries = self._compute_chunk_boundaries(r)  # [B, K+1]

        # Per-token chunk assignment
        chunk_ends = boundaries[:, 1:]  # [B, K]
        per_token_ends = chunk_ends[seg_ids]  # [total, K]
        chunk_ids = (local_pos_f.unsqueeze(-1) >= per_token_ends).sum(dim=-1).long()  # [total]
        chunk_ids = chunk_ids.clamp(max=self.K - 1)

        # Position within chunk
        chunk_starts = boundaries[:, :-1]  # [B, K]
        per_token_start = (
            chunk_starts[seg_ids].gather(1, chunk_ids.unsqueeze(-1)).squeeze(-1)
        )  # [total]
        pos_in_chunk = local_pos_f - per_token_start

        # Span of assigned chunk
        spans = boundaries[:, 1:] - boundaries[:, :-1]  # [B, K]
        span_for_token = spans[seg_ids].gather(1, chunk_ids.unsqueeze(-1)).squeeze(-1)  # [total]

        # Output slot
        output_slot = (pos_in_chunk * self.N / span_for_token.clamp(min=1.0)).long()
        output_slot = output_slot.clamp(min=0, max=self.N - 1)

        # Global output index per token
        output_idx = chunk_ids * self.N + output_slot  # [total]
        output_idx = output_idx.clamp(min=0, max=self.KN - 1)

        # Scatter into [B*KN, D]
        flat_idx = seg_ids * self.KN + output_idx  # [total]
        out_flat = torch.zeros(B * self.KN, D, device=device, dtype=seq_tokens.dtype)
        counts_flat = torch.zeros(B * self.KN, device=device, dtype=seq_tokens.dtype)
        out_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, D), seq_tokens)
        counts_flat.scatter_add_(
            0, flat_idx, torch.ones(total, device=device, dtype=counts_flat.dtype)
        )

        out_tokens = out_flat.reshape(B, self.KN, D)
        counts = counts_flat.reshape(B, self.KN)
        out_tokens = out_tokens / counts.unsqueeze(-1).clamp(min=1.0)
        out_mask = counts == 0

        out_tb = None
        if time_bucket_ids is not None:
            tb_flat = torch.zeros(B * self.KN, device=device, dtype=torch.float32)
            tb_flat.scatter_add_(0, flat_idx, time_bucket_ids.float())
            out_tb = (tb_flat.reshape(B, self.KN) / counts.clamp(min=1.0)).round().long()

        return out_tokens, out_mask, out_tb


class MultiHeadDINCore(nn.Module):
    """M independent DIN scoring heads via stacked weights (single batched matmul).

    Parameters
    ----------
    d_model
        Token / query dimensionality.
    num_heads
        M — number of independent scoring heads.
    hidden_mult
        Expansion factor for scoring MLP hidden dim.
    dropout
        Dropout rate in scoring layers.
    logit_softcap
        Tanh softcap on attention scores. 0 = disabled.
    num_time_buckets
        Number of time bias buckets. 0 = no time bias.
    direction_specific
        If True, use 2M independent weight sets (M per direction).
    time_in_score
        If True, embed time bucket IDs and concatenate to the 4-way
        interaction input before the scoring MLP. Richer than additive
        bias — lets the MLP learn nonlinear time x item x query interactions.
    time_emb_dim
        Dimension of per-event time embedding when `time_in_score` is True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        logit_softcap: float = 0.0,
        num_time_buckets: int = 0,
        direction_specific: bool = False,
        time_in_score: bool = False,
        time_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.M = num_heads
        self.d_model = d_model
        self.logit_softcap = float(logit_softcap)
        self.direction_specific = direction_specific
        self.time_in_score = time_in_score and num_time_buckets > 0

        # Time embedding for scoring input (richer than additive bias)
        if self.time_in_score:
            t_dim = time_emb_dim if time_emb_dim > 0 else d_model // 4
            self.time_score_emb = nn.Embedding(num_time_buckets, t_dim, padding_idx=0)
            nn.init.normal_(self.time_score_emb.weight, std=0.02)
            self.time_bias = None
        elif num_time_buckets > 0:
            t_dim = 0
            self.time_score_emb = None
            self.time_bias = nn.Embedding(num_time_buckets, 1, padding_idx=0)
            nn.init.zeros_(self.time_bias.weight)
        else:
            t_dim = 0
            self.time_score_emb = None
            self.time_bias = None

        input_dim = d_model * 4 + t_dim  # [e, q, e*q, e-q, (t_emb)]
        hidden = max(d_model, int(d_model * hidden_mult))
        M_total = num_heads * 2 if direction_specific else num_heads

        # Stacked scoring weights: two linear layers per head
        self.score_w1 = nn.Parameter(torch.empty(M_total, input_dim, hidden))
        self.score_b1 = nn.Parameter(torch.zeros(M_total, 1, hidden))
        self.score_ln = nn.LayerNorm(hidden)
        self.score_w2 = nn.Parameter(torch.empty(M_total, hidden, 1))
        self.score_b2 = nn.Parameter(torch.zeros(M_total, 1, 1))
        self.score_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.score_w1)
        nn.init.kaiming_uniform_(self.score_w2)

    def forward(
        self,
        target: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and attend over chunked tokens with M heads.

        Parameters
        ----------
        target
            Shape ``[N, D]`` where N is B or 2B depending on caller.
        tokens
            Shape ``[N, KN, D]``.
        mask
            Bool ``[N, KN]``; True = masked (no valid events).
        time_bucket_ids
            Int ``[N, KN]``. Optional.

        Returns
        -------
        contexts
            Attended contexts, shape ``[N, M, D]``.
        attn
            Attention weights, shape ``[N, M, KN]``.
        """
        N, KN, D = tokens.shape
        M = self.M

        # 4-way interaction
        q = target.unsqueeze(1).expand(-1, KN, -1)  # [N, KN, D]
        parts = [tokens, q, tokens * q, tokens - q]
        if self.time_in_score and time_bucket_ids is not None:
            parts.append(self.time_score_emb(time_bucket_ids))  # [N, KN, t_dim]
        x = torch.cat(parts, dim=-1)  # [N, KN, 4D(+t_dim)]

        # Stacked scoring
        if self.direction_specific:
            B = N // 2
            x_fwd = x[:B]
            x_bwd = x[B:]
            scores_fwd = self._score_batch(x_fwd, slice(0, M))  # [B, M, KN]
            scores_bwd = self._score_batch(x_bwd, slice(M, 2 * M))  # [B, M, KN]
            scores = torch.cat([scores_fwd, scores_bwd], dim=0)  # [N, M, KN]
        else:
            scores = self._score_batch(x, slice(0, M))  # [N, M, KN]

        # Time bias (mutually exclusive with time_in_score)
        if self.time_bias is not None and time_bucket_ids is not None:
            tb = self.time_bias(time_bucket_ids).squeeze(-1)  # [N, KN]
            scores = scores + tb.unsqueeze(1)

        # Softcap
        if self.logit_softcap > 0:
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)

        # Mask and softmax
        scores = scores.masked_fill(mask.unsqueeze(1), -1e4)  # [N, M, KN]
        attn = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)  # [N, M, KN]

        # Zero out masked positions (safety for all-masked samples)
        valid = (~mask).unsqueeze(1).to(attn.dtype)  # [N, 1, KN]
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Weighted sum: [N, M, KN] @ [N, KN, D] → [N, M, D]
        contexts = torch.bmm(
            attn.reshape(N * M, 1, KN),
            tokens.unsqueeze(1).expand(-1, M, -1, -1).reshape(N * M, KN, D),
        ).reshape(N, M, D)

        return contexts, attn

    def _score_batch(self, x: torch.Tensor, head_slice: slice) -> torch.Tensor:
        """Apply stacked scoring MLP to a batch of interactions.

        Parameters
        ----------
        x
            Shape ``[batch, KN, 4D]``.
        head_slice
            Which heads' weights to use.

        Returns
        -------
        torch.Tensor
            Scores, shape ``[batch, M, KN]``.
        """
        # x: [batch, KN, 4D]
        # W1[heads]: [M, 4D, H], b1[heads]: [M, 1, H]
        w1 = self.score_w1[head_slice]  # [M, 4D, H]
        b1 = self.score_b1[head_slice]  # [M, 1, H]
        w2 = self.score_w2[head_slice]  # [M, H, 1]
        b2 = self.score_b2[head_slice]  # [M, 1, 1]

        # einsum: [batch, KN, 4D] x [M, 4D, H] → [batch, M, KN, H]
        h = torch.einsum("bld,mdf->bmlf", x, w1) + b1.unsqueeze(0)
        h = self.score_ln(h)
        h = F.silu(h)
        h = self.score_dropout(h)
        # [batch, M, KN, H] x [M, H, 1] → [batch, M, KN, 1] → [batch, M, KN]
        scores = torch.einsum("bmlf,mfk->bmlk", h, w2).squeeze(-1) + b2.squeeze(-1).unsqueeze(0)
        return scores

    @staticmethod
    def compute_entropy(attn: torch.Tensor) -> torch.Tensor:
        """Compute mean attention entropy across heads from cached weights.

        Parameters
        ----------
        attn
            Attention weights, shape ``[N, M, KN]``.

        Returns
        -------
        torch.Tensor
            Mean entropy across heads, shape ``[N]``.
        """
        p = attn.float().clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=-1)  # [N, M]
        return ent.mean(dim=-1)  # [N]


class MultiChunkBidirectionalDIN(nn.Module):
    """Multi-chunk bidirectional DIN head matching TargetAwareDINHead's interface.

    Parameters
    ----------
    d_model
        Token / query dimensionality.
    num_heads
        M — number of independent DIN scoring heads.
    chunk_tokens
        N — output tokens per chunk.
    num_chunks
        K — number of exponential chunks.
    hidden_mult
        Expansion factor for scoring MLP.
    dropout
        Dropout rate.
    logit_softcap
        Tanh softcap on attention scores.
    num_time_buckets
        Time bias buckets. 0 = no time bias.
    bidirectional
        If False, only forward direction (halves output).
    direction_specific
        If True, independent weights per direction.
    time_in_score
        If True, embed time bucket IDs and concatenate to the scoring MLP
        input instead of using additive scalar bias. Mutually exclusive with
        the default time_bias behavior.
    time_emb_dim
        Dimension of per-event time embedding for `time_in_score`. Defaults
        to d_model // 4 if 0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        chunk_tokens: int = 32,
        num_chunks: int = None,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        logit_softcap: float = 0.0,
        num_time_buckets: int = 0,
        bidirectional: bool = True,
        direction_specific: bool = False,
        time_in_score: bool = False,
        time_emb_dim: int = 0,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.chunking = num_chunks is not None

        self.chunker = ExponentialChunker(num_chunks, chunk_tokens) if self.chunking else None
        self.din_core = MultiHeadDINCore(
            d_model=d_model,
            num_heads=num_heads,
            hidden_mult=hidden_mult,
            dropout=dropout,
            logit_softcap=logit_softcap,
            num_time_buckets=num_time_buckets,
            direction_specific=direction_specific,
            time_in_score=time_in_score,
            time_emb_dim=time_emb_dim,
        )

        # Target-gated aggregation: softmax gate over head slots, weighted sum
        num_dir = 2 if bidirectional else 1
        num_slots = num_dir * num_heads
        self.agg_gate = nn.Linear(d_model, num_slots)

        # context_proj: same naming as TargetAwareDINHead for diagnostics compat
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        # TODO (nsarang): add a higher-capacity DIN output mode (e.g., multi-logit or
        # repr-preserving head) so downstream mixers can consume richer per-domain signal.
        self.logit_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            build_activation(logit_head_activation, scaled_tanh_scale=logit_head_tanh_scale),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _passthrough(
        self,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Bypass chunking: convert to [B, L, D] padded format if jagged."""
        if seq_tokens.dim() == 3:
            return seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
        else:
            # Jagged → padded
            cu = padding_mask_or_cu_seqlens
            B = cu.shape[0] - 1
            D = seq_tokens.shape[-1]
            max_len = (cu[1:] - cu[:-1]).max().item()
            device = seq_tokens.device

            padded = torch.zeros(B, max_len, D, device=device, dtype=seq_tokens.dtype)
            mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
            for i in range(B):
                length = (cu[i + 1] - cu[i]).item()
                padded[i, :length] = seq_tokens[cu[i] : cu[i + 1]]
                mask[i, :length] = False

            out_tb = None
            if time_bucket_ids is not None:
                tb_padded = torch.zeros(B, max_len, device=device, dtype=time_bucket_ids.dtype)
                for i in range(B):
                    length = (cu[i + 1] - cu[i]).item()
                    tb_padded[i, :length] = time_bucket_ids[cu[i] : cu[i + 1]]
                out_tb = tb_padded
            return padded, mask, out_tb

    @staticmethod
    def _reverse_valid(
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Reverse valid region of each padded sequence, keeping padding at end."""
        B, L, D = seq_tokens.shape
        valid = ~padding_mask  # [B, L], True = real
        lengths = valid.sum(dim=-1)  # [B]

        pos = torch.arange(L, device=seq_tokens.device).unsqueeze(0).expand(B, -1)
        rev_pos = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)

        rev_tokens = seq_tokens.gather(1, rev_pos.unsqueeze(-1).expand(-1, -1, D))
        rev_tokens = rev_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)

        rev_tb = None
        if time_bucket_ids is not None:
            rev_tb = time_bucket_ids.gather(1, rev_pos)
            rev_tb = rev_tb * valid.long()

        return rev_tokens, padding_mask, rev_tb

    def _chunk_reversed(
        self,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Reverse the raw sequence then chunk.

        After reversing, newest events sit at position 0 → chunk 0 (smallest
        span, full resolution). Oldest events land in chunk K-1 (largest span,
        heavy pooling).
        """
        if seq_tokens.dim() == 3:
            rev_tokens, rev_mask, rev_tb = self._reverse_valid(
                seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
            )
        else:
            cu = padding_mask_or_cu_seqlens
            B = cu.shape[0] - 1
            rev_tokens = seq_tokens.clone()
            rev_tb = time_bucket_ids.clone() if time_bucket_ids is not None else None
            for i in range(B):
                s, e = int(cu[i]), int(cu[i + 1])
                rev_tokens[s:e] = rev_tokens[s:e].flip(0)
                if rev_tb is not None:
                    rev_tb[s:e] = rev_tb[s:e].flip(0)
            rev_mask = cu

        if self.chunking:
            return self.chunker(rev_tokens, rev_mask, rev_tb)
        return self._passthrough(rev_tokens, rev_mask, rev_tb)

    def _chunk_raw(
        self,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Chunk the raw (oldest-first) sequence directly.

        Oldest events sit at position 0 → chunk 0 (full resolution).
        Newest events in chunk K-1 (heavy pooling).
        """
        if self.chunking:
            return self.chunker(seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids)
        return self._passthrough(seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids)

    def _aggregate(
        self,
        target: torch.Tensor,
        fwd_tokens: torch.Tensor,
        fwd_mask: torch.Tensor,
        fwd_tb: torch.Tensor | None,
        bwd_tokens: torch.Tensor = None,
        bwd_mask: torch.Tensor = None,
        bwd_tb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DIN core + target-gated aggregate across heads/directions → [B, D]."""
        B = target.shape[0]

        if self.bidirectional:
            tokens_in = torch.cat([fwd_tokens, bwd_tokens], dim=0)
            mask_in = torch.cat([fwd_mask, bwd_mask], dim=0)
            tb_in = torch.cat([fwd_tb, bwd_tb], dim=0) if fwd_tb is not None else None
            target_in = target.repeat(2, 1)
        else:
            tokens_in, mask_in, tb_in = fwd_tokens, fwd_mask, fwd_tb
            target_in = target

        contexts, attn = self.din_core(target_in, tokens_in, mask_in, tb_in)

        if self.bidirectional:
            contexts = torch.cat([contexts[:B], contexts[B:]], dim=1)  # [B, 2*M, D]

        gate = torch.softmax(self.agg_gate(target), dim=-1)
        attended = (gate.unsqueeze(-1) * contexts).sum(dim=1)
        return attended, attn

    def forward(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_bias: torch.Tensor = None,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward matching TargetAwareDINHead signature.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Shape ``[B, L, D]`` (padded) or ``(total, D)`` (jagged).
        padding_mask_or_cu_seqlens
            Bool mask ``[B, L]`` (padded) or cu_seqlens ``(B+1,)`` (jagged).
        time_bucket_ids
            Optional time bucket IDs matching seq_tokens layout.
        score_bias
            Optional additive bias ``[B, L]`` from SeqLocalWriter. Currently
            unused by multi-chunk DIN (accepted for interface compatibility).
        score_query
            Optional score-only query accepted for interface compatibility.

        Returns
        -------
        logit
            Shape ``[B]``.
        context
            Fused representation, shape ``[B, D]``.
        entropy
            Per-sample attention entropy, shape ``[B]``.
        """
        B = target.shape[0]

        # Forward: reverse → chunk (recent at full resolution)
        fwd_tokens, fwd_mask, fwd_tb = self._chunk_reversed(
            seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
        )

        # Backward: chunk raw (old at full resolution)
        bwd_tokens = bwd_mask = bwd_tb = None
        if self.bidirectional:
            bwd_tokens, bwd_mask, bwd_tb = self._chunk_raw(
                seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
            )

        attended, attn = self._aggregate(
            target, fwd_tokens, fwd_mask, fwd_tb, bwd_tokens, bwd_mask, bwd_tb
        )

        fused = self.context_proj(torch.cat([target, attended], dim=-1))
        logit = self.logit_head(fused).squeeze(-1)

        ent = MultiHeadDINCore.compute_entropy(attn)
        if self.bidirectional:
            entropy = (ent[:B] + ent[B:]) / 2.0
        else:
            entropy = ent

        return logit, fused, entropy

    def attend_only(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return raw attended context (pre context_proj) for pretext head."""
        fwd_tokens, fwd_mask, fwd_tb = self._chunk_reversed(
            seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
        )

        bwd_tokens = bwd_mask = bwd_tb = None
        if self.bidirectional:
            bwd_tokens, bwd_mask, bwd_tb = self._chunk_raw(
                seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
            )

        attended, _attn = self._aggregate(
            target, fwd_tokens, fwd_mask, fwd_tb, bwd_tokens, bwd_mask, bwd_tb
        )
        return attended


class TargetAwareDINHead(nn.Module):
    """Original DIN target-conditioned attention head over one behavior stream.

    Computes attention weights via a 4-way interaction MLP
    ``[event, query, event*query, event-query]``, attends over the sequence,
    then produces a scalar logit from the fused representation.

    Parameters
    ----------
    d_model
        Token / query dimensionality.
    hidden_mult
        Expansion factor for the attention-scoring MLP.
    dropout
        Dropout rate applied inside both MLPs.
    logit_softcap
        If > 0, applies ``softcap * tanh(score / softcap)`` to attention scores.
    query_mode
        How to construct the target query from raw inputs:
        - ``"item"`` — use item_repr directly (default, original behavior).
        - ``"film"`` — FiLM conditioning: item_repr modulated by user context
          via learned scale (gamma) and shift (beta).
        - ``"additive"`` — project item and user independently, sum.
        - ``"gated"`` — sigmoid gate blends item and user projections per-dim.
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        logit_softcap: float = 0.0,
        differential: bool = False,
        query_mode: str = "item",
        num_time_buckets: int = 0,
        windowed: dict = None,
        logit_head_activation: str = "silu",
        logit_head_tanh_scale: float = None,
    ) -> None:
        super().__init__()
        self.query_mode = query_mode
        self.num_time_buckets = num_time_buckets
        hidden = max(d_model, int(d_model * hidden_mult))
        self.logit_softcap = float(logit_softcap)
        self.differential = differential

        # Query builder layers
        if query_mode == "film":
            self.film_gamma = nn.Linear(d_model, d_model)
            self.film_beta = nn.Linear(d_model, d_model)
            # Init gamma bias=1 so query starts as item + small perturbation
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.bias)
        elif query_mode == "additive":
            self.user_query_proj = nn.Linear(d_model, d_model)
        elif query_mode == "gated":
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.user_query_proj = nn.Linear(d_model, d_model)
        elif query_mode != "item":
            raise ValueError(f"Unknown query_mode: {query_mode!r}")

        if num_time_buckets > 0:
            self.time_bias = nn.Embedding(num_time_buckets, 1, padding_idx=0)
            nn.init.zeros_(self.time_bias.weight)

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if differential:
            self.attn_mlp2 = nn.Sequential(
                nn.Linear(d_model * 4, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.diff_lambda = nn.Parameter(torch.tensor(0.5))
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            build_activation(logit_head_activation, scaled_tanh_scale=logit_head_tanh_scale),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.capture_diag: bool = False
        self._last_diag: dict | None = None

        # Windowed DIN: time-decomposed residual fused internally
        self._windowed = windowed is not None
        if self._windowed:
            from core.data.blocks import TimeDeltaBucketBlock

            boundaries = TimeDeltaBucketBlock.BUCKET_BOUNDARIES
            b_edges = []
            for e in sorted(windowed["edges_sec"]):
                b = int(np.searchsorted(boundaries, int(e)).clip(0, len(boundaries) - 1)) + 1
                if b_edges and b <= b_edges[-1]:
                    b = b_edges[-1] + 1
                b_edges.append(b)
            self._b_edges = b_edges

            self._fusion_mode = windowed["fusion"]
            self._residual_cap = windowed["residual_cap"]
            self._memory_cap = windowed["memory_cap"]
            if self._fusion_mode not in ("residual", "replace"):
                raise ValueError(f"Unknown windowed fusion mode: {self._fusion_mode!r}")
            self.windowed_residual_scale = nn.Parameter(
                torch.tensor(windowed["residual_scale_init"])
            )
            self.windowed_memory_scale = nn.Parameter(torch.tensor(windowed["memory_scale_init"]))
            self.windowed_fusion_proj = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(windowed["dropout"]),
            )

    def build_query(
        self,
        item_repr: torch.Tensor,
        user_repr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Construct the DIN target query from raw inputs.

        Parameters
        ----------
        item_repr
            Item representation, shape ``[B, D]``.
        user_repr
            User context (pre-selected/pooled by caller based on sources config),
            shape ``[B, D]``. Required for all modes except ``"item"``.

        Returns
        -------
        torch.Tensor
            Target query, shape ``[B, D]``.
        """
        if self.query_mode == "item":
            return item_repr
        if self.query_mode == "film":
            gamma = self.film_gamma(user_repr)
            beta = self.film_beta(user_repr)
            return gamma * item_repr + beta
        if self.query_mode == "additive":
            return item_repr + self.user_query_proj(user_repr)
        # gated
        gate = torch.sigmoid(self.gate_proj(torch.cat([item_repr, user_repr], dim=-1)))
        return gate * item_repr + (1 - gate) * self.user_query_proj(user_repr)

    @torch.compiler.disable
    def _store_attn_diag(
        self,
        attn: torch.Tensor,
        entropy: torch.Tensor,
        *,
        tb_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
    ) -> None:
        """Cache per-forward attention tensors for the DIN_ATTN diagnostic.

        Runs outside the compiled graph (`torch.compiler.disable`) so the
        module-attribute side effect is never traced into the compiled DIN
        forward — same pattern as `SeqLocalWriter._store_diag`. The `if
        self.capture_diag` guard stays at the call site so dynamo prunes it to
        zero cost when capture is off. `jagged` is inferred from whether
        `cu_seqlens` (jagged path) vs `padding_mask` (padded path) was supplied.
        """
        self._last_diag = {
            "attn": attn.detach(),
            "entropy": entropy.detach(),
            "tb_ids": tb_ids,
            "padding_mask": padding_mask,
            "cu_seqlens": cu_seqlens,
            "jagged": cu_seqlens is not None,
        }

    def _compute_attention(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_bias: torch.Tensor = None,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core attention: score, normalize, pool.

        Returns
        -------
        context
            Attended representation, shape ``[B, D]``.
        attn
            Attention weights, shape ``[B, L]``.
        """
        _, L, _ = seq_tokens.shape
        if score_query is None:
            q = target.unsqueeze(1).expand(-1, L, -1)
        elif score_query.dim() == 2:
            q = score_query.unsqueeze(1).expand(-1, L, -1)
        else:
            q = score_query
        x = torch.cat([seq_tokens, q, seq_tokens * q, seq_tokens - q], dim=-1)
        scores = self.attn_mlp(x).squeeze(-1)  # (B, L)
        if self.num_time_buckets > 0 and time_bucket_ids is not None:
            scores = scores + self.time_bias(time_bucket_ids).squeeze(-1)
        if score_bias is not None:
            scores = scores + score_bias
        if self.logit_softcap > 0:
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)

        valid = ~padding_mask.bool()
        scores = scores.masked_fill(~valid, -1.0e4)

        if self.differential:
            scores2 = self.attn_mlp2(x).squeeze(-1)
            if self.logit_softcap > 0:
                scores2 = self.logit_softcap * torch.tanh(scores2 / self.logit_softcap)
            scores2 = scores2.masked_fill(~valid, -1.0e4)
            w1 = torch.softmax(scores.float(), dim=-1)
            w2 = torch.softmax(scores2.float(), dim=-1)
            attn = (w1 - self.diff_lambda * w2).to(seq_tokens.dtype)
            attn = attn * valid.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        else:
            attn = torch.softmax(scores.float(), dim=-1).to(seq_tokens.dtype)
            attn = attn * valid.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        context = torch.bmm(attn.unsqueeze(1), seq_tokens).squeeze(1)  # (B, D)
        return context, attn

    def forward_padded(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_bias: torch.Tensor = None,
        score_query: torch.Tensor = None,
        window_score_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded-path DIN attention.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Shape ``[B, L, D]``.
        padding_mask
            Bool tensor ``[B, L]``; ``True`` = padding position.
        time_bucket_ids
            Int tensor ``[B, L]``; time bucket index per event. Optional.
        score_bias
            Float tensor ``[B, L]``; additive bias applied to attention scores
            before softmax. Optional. Produced by SeqLocalWriter when gamma > 0.
        score_query
            Optional score-only query, shape ``[B, D]`` or ``[B, L, D]``.
        window_score_queries
            Optional ``(recent, mid, old)`` score-only queries for windowed DIN.

        Returns
        -------
        logit
            Shape ``[B]``.
        context
            Fused attended representation, shape ``[B, D]``.
        entropy
            Per-sample attention entropy, shape ``[B]``.
        """
        context, attn = self._compute_attention(
            target, seq_tokens, padding_mask, time_bucket_ids, score_bias, score_query
        )
        fused = self.context_proj(torch.cat([target, context], dim=-1))  # (B, D)
        logit = self.logit_head(fused).squeeze(-1)  # (B,)

        p = attn.float().clamp_min(1.0e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # (B,)

        if self.capture_diag:
            self._store_attn_diag(attn, entropy, tb_ids=time_bucket_ids, padding_mask=padding_mask)

        if self._windowed and time_bucket_ids is not None:
            logit, fused = self._apply_windowed(
                target,
                seq_tokens,
                padding_mask,
                time_bucket_ids,
                logit,
                fused,
                window_score_queries,
            )

        return logit, fused, entropy

    def _forward_padded_raw(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention + context_proj without windowed recursion."""
        context, attn = self._compute_attention(
            target, seq_tokens, padding_mask, time_bucket_ids, score_query=score_query
        )
        fused = self.context_proj(torch.cat([target, context], dim=-1))
        logit = self.logit_head(fused).squeeze(-1)
        p = attn.float().clamp_min(1.0e-8)
        entropy = -(p * p.log()).sum(dim=-1)
        return logit, fused, entropy

    def _apply_windowed(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        tb_ids: torch.Tensor,
        full_logit: torch.Tensor,
        full_ctx: torch.Tensor,
        window_score_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Time-decomposed residual: recent full attention + N memory tokens from older windows."""
        valid = (~padding_mask.bool()) & (tb_ids > 0)
        recent = valid & (tb_ids <= self._b_edges[0])
        recent_score_query = None
        memory_score_query = None
        if window_score_queries is not None:
            q_recent, q_mid, q_old = window_score_queries
            recent_score_query = q_recent
            if q_mid is not None and q_old is not None:
                memory_score_query = torch.stack([q_mid, q_old], dim=1)

        # Recent: full attention over recent positions only
        recent_logit, recent_ctx, _ = self._forward_padded_raw(
            target,
            seq_tokens,
            ~recent,
            tb_ids,
            score_query=recent_score_query,
        )

        # Memory tokens: one mean-pool per window between consecutive edges, plus tail
        mem_ctxs = []
        mem_has = []
        prev_b = self._b_edges[0]
        for next_b in self._b_edges[1:]:
            w = valid & (tb_ids > prev_b) & (tb_ids <= next_b)
            mem_ctxs.append(self._masked_mean(seq_tokens, w))
            mem_has.append(w.any(dim=1))
            prev_b = next_b
        tail = valid & (tb_ids > self._b_edges[-1])
        mem_ctxs.append(self._masked_mean(seq_tokens, tail))
        mem_has.append(tail.any(dim=1))

        memory_tokens = torch.stack(mem_ctxs, dim=1)
        memory_mask = torch.stack([~h for h in mem_has], dim=1)
        mem_logit, mem_ctx, _ = self._forward_padded_raw(
            target,
            memory_tokens,
            memory_mask,
            None,
            score_query=memory_score_query,
        )

        return self._fuse_windowed(
            target, recent_logit, recent_ctx, mem_logit, mem_ctx, full_logit, full_ctx
        )

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool sequence tokens over boolean keep mask (True = keep)."""
        w = keep_mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def _compute_attention_jagged(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jagged attention: score, normalize, pool without padding.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Shape ``(total_tokens, D)``.
        cu_seqlens
            Cumulative sequence lengths, shape ``(B+1,)``.
        score_query
            Optional score-only query matching the flat token layout.

        Returns
        -------
        context
            Attended representation, shape ``[B, D]``.
        attn
            Attention weights, shape ``(total_tokens,)``.
        """
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        q = (
            target.repeat_interleave(lengths.long(), dim=0) if score_query is None else score_query
        )  # (total_tokens, D)

        x = torch.cat([seq_tokens, q, seq_tokens * q, seq_tokens - q], dim=-1)
        scores = self.attn_mlp(x).squeeze(-1)  # (total_tokens,)

        if self.logit_softcap > 0:
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)

        if self.differential:
            scores2 = self.attn_mlp2(x).squeeze(-1)
            if self.logit_softcap > 0:
                scores2 = self.logit_softcap * torch.tanh(scores2 / self.logit_softcap)
            w1 = segment_softmax(scores.float(), cu_seqlens)
            w2 = segment_softmax(scores2.float(), cu_seqlens)
            attn = (w1 - self.diff_lambda * w2).to(seq_tokens.dtype)
            # Re-normalize
            attn_sum = segment_sum(attn, cu_seqlens)
            expanded_sum = attn_sum.repeat_interleave(lengths.long(), dim=0).clamp_min(1e-6)
            attn = attn / expanded_sum
        else:
            attn = segment_softmax(scores.float(), cu_seqlens).to(seq_tokens.dtype)

        weighted = seq_tokens * attn.unsqueeze(-1)  # (total_tokens, D)
        context = segment_sum(weighted, cu_seqlens)  # (B, D)
        return context, attn

    def forward_jagged(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jagged-path DIN attention.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Shape ``(total_tokens, D)``.
        cu_seqlens
            Cumulative sequence lengths, shape ``(B+1,)``.
        time_bucket_ids
            Int tensor ``(total_tokens,)``; time bucket index per event. Optional.
        score_query
            Optional score-only query matching the flat token layout.

        Returns
        -------
        logit
            Shape ``[B]``.
        context
            Fused attended representation, shape ``[B, D]``.
        entropy
            Per-sample attention entropy, shape ``[B]``.
        """
        context, attn = self._compute_attention_jagged(target, seq_tokens, cu_seqlens, score_query)
        fused = self.context_proj(torch.cat([target, context], dim=-1))
        logit = self.logit_head(fused).squeeze(-1)

        p = attn.float().clamp_min(1e-8)
        token_entropy = -(p * p.log())  # (total_tokens,)
        entropy = segment_sum(token_entropy, cu_seqlens)  # (B,)

        if self.capture_diag:
            self._store_attn_diag(attn, entropy, tb_ids=time_bucket_ids, cu_seqlens=cu_seqlens)

        if self._windowed and time_bucket_ids is not None:
            logit, fused = self._apply_windowed_jagged(
                target, seq_tokens, cu_seqlens, time_bucket_ids, logit, fused
            )

        return logit, fused, entropy

    def _forward_jagged_raw(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        score_query: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jagged attention + context_proj without windowed recursion."""
        context, attn = self._compute_attention_jagged(target, seq_tokens, cu_seqlens, score_query)
        fused = self.context_proj(torch.cat([target, context], dim=-1))
        logit = self.logit_head(fused).squeeze(-1)
        p = attn.float().clamp_min(1e-8)
        token_entropy = -(p * p.log())
        entropy = segment_sum(token_entropy, cu_seqlens)
        return logit, fused, entropy

    def _apply_windowed_jagged(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tb_ids: torch.Tensor,
        full_logit: torch.Tensor,
        full_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Time-decomposed residual for jagged (flat) tensors.

        Same windowed logic as `_apply_windowed` but operates on flat
        ``(total_tokens, D)`` tensors with ``cu_seqlens`` instead of a
        ``[B, L, D]`` padded grid. Partitions positions by time bucket,
        runs recent through jagged attention and mid/old through padded
        memory-token attention.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Flat sequence tokens ``(total_tokens, D)``.
        cu_seqlens
            Cumulative lengths ``(B+1,)``.
        tb_ids
            Time bucket IDs ``(total_tokens,)``.
        full_logit
            Full-sequence DIN logit ``[B]`` to fuse with.
        full_ctx
            Full-sequence DIN context ``[B, D]`` to fuse with.
        """
        B = cu_seqlens.shape[0] - 1
        seg_ids = torch.repeat_interleave(
            torch.arange(B, device=cu_seqlens.device),
            (cu_seqlens[1:] - cu_seqlens[:-1]).long(),
        )

        valid = tb_ids > 0
        recent_mask = valid & (tb_ids <= self._b_edges[0])

        # Recent: sub-jagged attention over recent positions only
        recent_tokens, recent_cu = self._subset_jagged(
            seq_tokens, B, seg_ids, recent_mask, cu_seqlens.dtype
        )
        recent_logit, recent_ctx, _ = self._forward_jagged_raw(target, recent_tokens, recent_cu)

        # Memory tokens: one mean-pool per window between consecutive edges, plus tail
        mem_ctxs = []
        mem_has = []
        prev_b = self._b_edges[0]
        for next_b in self._b_edges[1:]:
            w = valid & (tb_ids > prev_b) & (tb_ids <= next_b)
            mem_ctxs.append(self._jagged_masked_mean(seq_tokens, B, seg_ids, w))
            mem_has.append(self._segment_any(B, seg_ids, w))
            prev_b = next_b
        tail_mask = valid & (tb_ids > self._b_edges[-1])
        mem_ctxs.append(self._jagged_masked_mean(seq_tokens, B, seg_ids, tail_mask))
        mem_has.append(self._segment_any(B, seg_ids, tail_mask))

        memory_tokens = torch.stack(mem_ctxs, dim=1)
        memory_mask = torch.stack([~h for h in mem_has], dim=1)
        mem_logit, mem_ctx, _ = self._forward_padded_raw(target, memory_tokens, memory_mask, None)

        return self._fuse_windowed(
            target, recent_logit, recent_ctx, mem_logit, mem_ctx, full_logit, full_ctx
        )

    def _fuse_windowed(
        self,
        target: torch.Tensor,
        recent_logit: torch.Tensor,
        recent_ctx: torch.Tensor,
        mem_logit: torch.Tensor,
        mem_ctx: torch.Tensor,
        full_logit: torch.Tensor,
        full_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine windowed components with full DIN output.

        Scales memory context by ``memory_cap * tanh(memory_scale)``, projects
        ``[recent_ctx, scaled_mem_ctx, target]`` through ``windowed_fusion_proj``,
        then either replaces or residually adds to ``full_logit``/``full_ctx``
        depending on ``_fusion_mode``.
        """
        mem_scale = self._memory_cap * torch.tanh(self.windowed_memory_scale)
        win_ctx = self.windowed_fusion_proj(
            torch.cat([recent_ctx, mem_scale * mem_ctx, target], dim=-1)
        )
        win_logit = recent_logit + mem_scale * mem_logit

        if self._fusion_mode == "replace":
            return win_logit, win_ctx
        res_scale = self._residual_cap * torch.tanh(self.windowed_residual_scale)
        return full_logit + res_scale * win_logit, full_ctx + res_scale * win_ctx

    @staticmethod
    def _subset_jagged(
        tokens: torch.Tensor,
        B: int,
        seg_ids: torch.Tensor,
        mask: torch.Tensor,
        cu_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract masked positions into a new jagged tensor with recomputed cu_seqlens.

        Parameters
        ----------
        tokens
            Flat tokens ``(total, D)``.
        B
            Batch size.
        seg_ids
            Pre-computed segment assignment ``(total,)`` mapping each position
            to its batch index.
        mask
            Boolean ``(total,)`` selecting positions to keep.
        cu_dtype
            Dtype for the output cu_seqlens tensor.

        Returns
        -------
        sub_tokens
            Flat tokens for kept positions ``(sum(mask), D)``.
        sub_cu_seqlens
            New cumulative lengths ``(B+1,)``.
        """
        sub_tokens = tokens[mask]
        kept_per_seg = torch.zeros(B, dtype=torch.long, device=tokens.device)
        kept_per_seg.scatter_add_(0, seg_ids[mask], torch.ones_like(seg_ids[mask]))
        sub_cu = torch.zeros(B + 1, dtype=cu_dtype, device=tokens.device)
        sub_cu[1:] = kept_per_seg.cumsum(0)
        return sub_tokens, sub_cu

    @staticmethod
    def _jagged_masked_mean(
        tokens: torch.Tensor,
        B: int,
        seg_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool positions selected by `mask` within each segment.

        Parameters
        ----------
        tokens
            Flat tokens ``(total, D)``.
        B
            Batch size.
        seg_ids
            Pre-computed segment assignment ``(total,)``.
        mask
            Boolean ``(total,)`` selecting which positions to include.

        Returns
        -------
        torch.Tensor
            ``[B, D]`` with zeros for segments with no selected positions.
        """
        D = tokens.shape[1]
        w = mask.to(tokens.dtype).unsqueeze(-1)  # (total, 1)
        weighted = tokens * w  # (total, D)
        sum_out = torch.zeros(B, D, dtype=tokens.dtype, device=tokens.device)
        idx = seg_ids.unsqueeze(1).expand(-1, D)
        sum_out.scatter_add_(0, idx, weighted)
        count = torch.zeros(B, dtype=tokens.dtype, device=tokens.device)
        count.scatter_add_(0, seg_ids, w.squeeze(-1))
        return sum_out / count.unsqueeze(-1).clamp_min(1.0)

    @staticmethod
    def _segment_any(B: int, seg_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Check if any position in each segment satisfies `mask`.

        Returns
        -------
        torch.Tensor
            Bool ``[B]``; True if at least one position in the segment is masked.
        """
        out = torch.zeros(B, dtype=torch.bool, device=mask.device)
        out.scatter_reduce_(0, seg_ids, mask, reduce="amax", include_self=False)
        return out

    def forward(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
        score_bias: torch.Tensor = None,
        score_query: torch.Tensor = None,
        window_score_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to padded or jagged path based on seq_tokens shape.

        - 3D ``[B, L, D]``: padded path.
        - 2D ``(total_tokens, D)``: jagged path.

        `score_bias` is only supported on the padded path.
        `window_score_queries` is only supported on the padded path.
        """
        if seq_tokens.dim() == 3:
            return self.forward_padded(
                target,
                seq_tokens,
                padding_mask_or_cu_seqlens,
                time_bucket_ids,
                score_bias,
                score_query,
                window_score_queries,
            )
        if window_score_queries is not None:
            raise ValueError("window_score_queries are only supported for padded DIN")
        return self.forward_jagged(
            target,
            seq_tokens,
            padding_mask_or_cu_seqlens,
            time_bucket_ids,
            score_query,
        )

    def attend_only(
        self,
        target: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask_or_cu_seqlens: torch.Tensor,
        time_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return raw attended context without context_proj or logit_head.

        Dispatches to padded (3D seq_tokens) or jagged (2D seq_tokens) path.

        Parameters
        ----------
        target
            Shape ``[B, D]``.
        seq_tokens
            Shape ``[B, L, D]`` (padded) or ``(total_tokens, D)`` (jagged).
        padding_mask_or_cu_seqlens
            Bool mask ``[B, L]`` (padded) or cu_seqlens ``(B+1,)`` (jagged).
        time_bucket_ids
            Int tensor ``[B, L]``; time bucket index per event. Only for padded path.

        Returns
        -------
        torch.Tensor
            Attended context, shape ``[B, D]``.
        """
        if seq_tokens.dim() == 3:
            context, _ = self._compute_attention(
                target, seq_tokens, padding_mask_or_cu_seqlens, time_bucket_ids
            )
        else:
            context, _ = self._compute_attention_jagged(
                target, seq_tokens, padding_mask_or_cu_seqlens
            )
        return context
