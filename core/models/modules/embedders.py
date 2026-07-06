"""Format-polymorphic embedding helpers for DragonChariot.

StaticEmbedder: embeds static categoricals via nn.Embedding (padded) or torchrec EBC (flat).
SeqEmbedder: embeds a sequence domain via nn.Embedding (padded), torchrec EC (flat),
             or FBGEMM TBE in PoolingMode.NONE (flat, fused).

Format choice is config-driven (use_ebc / use_ec / use_tbe constructor args).
Both produce the same output shapes so the model forward is format-agnostic.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    EmbeddingLocation,
    OptimType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    PoolingType,
)

from core.data.schema import FeatureSpec


def _zero_row0_grad(grad: torch.Tensor) -> torch.Tensor:
    """Hook that zeros row 0 of an embedding gradient (emulates padding_idx=0)."""
    grad = grad.clone()
    grad[0] = 0
    return grad


def _should_skip(spec: FeatureSpec, hash_cardinality_threshold: int, hash_buckets: int) -> bool:
    """Return True if this feature should be skipped (zeroed) rather than embedded."""
    return (
        hash_cardinality_threshold > 0
        and spec.vocab_size > hash_cardinality_threshold
        and hash_buckets == 0
    )


def _effective_vocab(spec: FeatureSpec, hash_cardinality_threshold: int, hash_buckets: int) -> int:
    """Return the embedding table size for a feature, applying hash bucketing if needed."""
    if hash_cardinality_threshold > 0 and spec.vocab_size > hash_cardinality_threshold:
        if hash_buckets == 0:
            return 1  # placeholder — forward will zero these out
        return hash_buckets
    return spec.vocab_size + 1


class ChunkedProjection(nn.Module):
    """NS tokenizer: chunk embeddings into N tokens.

    Matches V1's RankMixerNSTokenizer:
    1. Pad concatenated embeddings to divisible length
    2. Split into num_tokens equal chunks
    3. Each chunk -> Linear + LayerNorm + SiLU -> [B, d_model]

    Output is [B, num_tokens, d_model].

    Optional RankMixer-style token mixing (`token_mixing=True`) adds a stage
    after projection: parameter-free channel-permutation mixing across the
    num_tokens tokens, a shared per-token FFN, and a residual. The permutation
    splits d_model into num_tokens equal blocks; when d_model is not divisible
    by num_tokens the channels are zero-padded to the next multiple. With
    `token_mixing=False` (or when already divisible) no padding is added and
    the module is identical to the plain chunked projection.

    Parameters
    ----------
    input_dim
        Width of the concatenated input embedding vector.
    d_model
        Output token dimension.
    num_tokens
        Number of NS tokens to produce.
    token_mixing
        If True, apply the RankMixer mixing stage after projection.
    mixing_hidden_mult
        Expansion factor for the shared per-token FFN (mixing stage only).
    mixing_dropout
        Dropout inside the mixing FFN.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_tokens: int,
        token_mixing: bool = False,
        mixing_hidden_mult: int = 2,
        mixing_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_dim = math.ceil(input_dim / num_tokens)
        self._pad_size = self.chunk_dim * num_tokens - input_dim
        self.token_projs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model))
                for _ in range(num_tokens)
            ]
        )

        self.token_mixing = token_mixing
        if token_mixing:
            # Mixing splits d_model into num_tokens equal blocks; pad up to the
            # next multiple when not divisible. mix_pad == d_model in the
            # divisible case, so no padding is applied.
            self.mix_pad = d_model + (-d_model % num_tokens)
            self.mix_sub = self.mix_pad // num_tokens
            self.mix_norm = nn.LayerNorm(self.mix_pad)
            self.mix_fc1 = nn.Linear(self.mix_pad, d_model * mixing_hidden_mult)
            self.mix_fc2 = nn.Linear(d_model * mixing_hidden_mult, d_model)
            self.mix_dropout = nn.Dropout(mixing_dropout)
            self.mix_post_norm = nn.LayerNorm(d_model)

    def _mix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free token mixing + shared per-token FFN + residual.

        Parameters
        ----------
        tokens
            Projected tokens, shape ``[..., num_tokens, d_model]``.

        Returns
        -------
        torch.Tensor
            Mixed tokens, shape ``[..., num_tokens, d_model]``.
        """
        leading_shape = tokens.shape[:-2]
        T, D = tokens.shape[-2], tokens.shape[-1]
        flat_tokens = tokens.reshape(-1, T, D)

        # Parameter-free token mixing: split each token's channels into T blocks
        # and swap the token/block axes so output token h gathers block h from
        # every input token. Pad to a multiple of T first (no-op when divisible).
        x = flat_tokens
        if self.mix_pad != D:
            x = F.pad(x, (0, self.mix_pad - D))
        x = (
            x.view(flat_tokens.shape[0], T, T, self.mix_sub)
            .transpose(1, 2)
            .contiguous()
            .view(flat_tokens.shape[0], T, self.mix_pad)
        )

        # Shared per-token FFN with residual on the original (unmixed) tokens
        x = self.mix_norm(x)
        x = self.mix_fc1(x)
        x = F.gelu(x)
        x = self.mix_dropout(x)
        x = self.mix_fc2(x)
        mixed = self.mix_post_norm(flat_tokens + x)
        return mixed.view(*leading_shape, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [..., input_dim] -> [..., num_tokens, d_model]."""
        if self._pad_size > 0:
            x = F.pad(x, (0, self._pad_size))
        # Reshape instead of split — avoids SymInt error under torch.compile(dynamic=True)
        x = x.view(*x.shape[:-1], self.num_tokens, self.chunk_dim)
        tokens = [F.silu(proj(x[..., i, :])) for i, proj in enumerate(self.token_projs)]
        tokens = torch.stack(tokens, dim=-2)
        if self.token_mixing:
            tokens = self._mix_tokens(tokens)
        return tokens


class StaticEmbedder(nn.Module):
    """Embeds static categoricals. nn.Embedding (padded) or EBC (flat).

    Parameters
    ----------
    specs
        Sorted list of FeatureSpec for this entity's static categoricals.
    emb_dim
        Embedding dimension per feature.
    use_ebc
        If True, use torchrec EmbeddingBagCollection (flat/KJT path).
        If False, use plain nn.Embedding with mean pooling.
    hash_cardinality_threshold
        Features with vocab above this get hashed. 0 disables.
    hash_buckets
        Number of hash buckets for high-cardinality features.
    """

    def __init__(
        self,
        specs: list[FeatureSpec],
        emb_dim: int,
        use_ebc: bool,
        hash_cardinality_threshold: int = 0,
        hash_buckets: int = 50000,
        learned_missingness: bool = False,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.emb_dim = emb_dim
        self.use_ebc = use_ebc
        self.hash_cardinality_threshold = hash_cardinality_threshold
        self.hash_buckets = hash_buckets
        self.learned_missingness = learned_missingness

        _pad_idx = 0 if not learned_missingness else None

        if use_ebc:
            self.ebc = EmbeddingBagCollection(
                tables=[
                    EmbeddingBagConfig(
                        name=spec.name,
                        embedding_dim=emb_dim,
                        num_embeddings=_effective_vocab(
                            spec, hash_cardinality_threshold, hash_buckets
                        ),
                        feature_names=[spec.name],
                        pooling=PoolingType.MEAN,
                    )
                    for spec in specs
                ]
            )
        else:
            self.tables = nn.ModuleDict(
                {
                    spec.name: nn.Embedding(
                        _effective_vocab(spec, hash_cardinality_threshold, hash_buckets),
                        emb_dim,
                        padding_idx=_pad_idx,
                    )
                    for spec in specs
                    if not _should_skip(spec, hash_cardinality_threshold, hash_buckets)
                }
            )

    def _hash_ids(self, ids: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        """Apply modular hashing for high-cardinality features, preserving 0 as padding."""
        if (
            self.hash_cardinality_threshold > 0
            and spec.vocab_size > self.hash_cardinality_threshold
        ):
            mask = ids > 0
            ids = ids.clone()
            ids[mask] = (ids[mask] % (self.hash_buckets - 1)) + 1
        return ids

    def forward(self, feat_ids: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed pre-extracted feature ID tensors.

        Parameters
        ----------
        feat_ids
            Mapping ``{feature_name: tensor}`` where tensors are [B] or [B, dim].

        Returns
        -------
        dict
            ``{feature_name: [B, emb_dim]}`` pooled embeddings.
        """
        if self.use_ebc:
            kjt = self._build_kjt(feat_ids)
            kt = self.ebc(kjt)
            return {spec.name: kt[spec.name] for spec in self.specs}

        result = {}
        for spec in self.specs:
            ids = feat_ids[spec.name].long()
            if _should_skip(spec, self.hash_cardinality_threshold, self.hash_buckets):
                result[spec.name] = ids.new_zeros(ids.shape[0], self.emb_dim, dtype=torch.float)
                continue
            ids = self._hash_ids(ids, spec)
            if ids.dim() == 1:
                ids = ids.unsqueeze(-1)
            emb = self.tables[spec.name](ids)
            # Multi-hot: zero positions are unused slots regardless of learned_missingness.
            # Only single-slot features (dim=1) treat index 0 as a valid "missing" token.
            if self.learned_missingness and ids.shape[1] == 1:
                result[spec.name] = emb.mean(dim=1)
            else:
                mask = (ids != 0).unsqueeze(-1).float()
                result[spec.name] = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return result

    def _build_kjt(self, feat_ids: dict[str, torch.Tensor]) -> KeyedJaggedTensor:
        """Build KJT from static IDs.

        When learned_missingness=False, zeros are stripped so EBC mean-pools only
        non-padding entries. When True, all entries (including 0) are kept.
        """
        all_values, all_lengths = [], []
        for spec in self.specs:
            if _should_skip(spec, self.hash_cardinality_threshold, self.hash_buckets):
                ids = torch.zeros_like(feat_ids[spec.name])
            else:
                ids = self._hash_ids(feat_ids[spec.name], spec)
            if ids.dim() == 1:
                ids = ids.unsqueeze(-1)
            if self.learned_missingness:
                B, D = ids.shape
                all_values.append(ids.reshape(-1))
                all_lengths.append(torch.full((B,), D, dtype=torch.int32, device=ids.device))
            else:
                mask = ids != 0
                all_values.append(ids[mask])
                all_lengths.append(mask.sum(dim=-1))
        return KeyedJaggedTensor(
            keys=[s.name for s in self.specs],
            values=torch.cat(all_values) if all_values else torch.zeros(0, dtype=torch.long),
            lengths=torch.cat(all_lengths).to(torch.int32),
        )

    def embedding_tables(self) -> dict[str, nn.Module]:
        """Return ``{name: embedding_module}`` regardless of backend path."""
        if self.use_ebc:
            return dict(self.ebc.embedding_bags)
        return dict(self.tables)

    def init_weights(self) -> None:
        """Xavier-normal init on all tables. Zeros row 0 when not learned_missingness."""
        if self.use_ebc:
            for emb in self.ebc.embedding_bags.values():
                nn.init.xavier_normal_(emb.weight.data)
                if not self.learned_missingness:
                    emb.weight.data[0, :] = 0
        else:
            for emb in self.tables.values():
                nn.init.xavier_normal_(emb.weight.data)
                if not self.learned_missingness and emb.padding_idx is not None:
                    emb.weight.data[0, :] = 0

    def reinit_high_cardinality(self, cardinality_threshold: int) -> set[int]:
        """Reinit tables above threshold. Returns data_ptrs of reinitialized weights."""
        reinitialized: set[int] = set()
        if self.use_ebc:
            for spec in self.specs:
                vocab = _effective_vocab(spec, self.hash_cardinality_threshold, self.hash_buckets)
                if vocab > cardinality_threshold:
                    emb = self.ebc.embedding_bags[spec.name]
                    nn.init.normal_(emb.weight, std=0.02)
                    if not self.learned_missingness:
                        emb.weight.data[0, :] = 0
                    reinitialized.add(emb.weight.data_ptr())
        else:
            for name, table in self.tables.items():
                if table.num_embeddings > cardinality_threshold:
                    nn.init.normal_(table.weight, std=0.02)
                    if table.padding_idx is not None:
                        table.weight.data[table.padding_idx].zero_()
                    reinitialized.add(table.weight.data_ptr())
        return reinitialized


class SeqEmbedder(nn.Module):
    """Embeds a sequence domain. Padded path, EC path, or TBE path.

    Always outputs (tokens [B, max_L, d_model], mask [B, max_L] where True=padded).
    The jagged path uses EC/TBE then optionally to_padded_dense() to normalize output shape.

    Parameters
    ----------
    specs
        ORIGINAL-origin FeatureSpecs for this domain, sorted by name.
    domain
        Sequence domain name.
    emb_dim
        Per-feature embedding dimension.
    d_model
        Output token dimension (after projection).
    use_ec
        If True, use torchrec EmbeddingCollection (flat/KJT path).
    use_tbe
        If True, use FBGEMM TBE with PoolingMode.NONE (flat, fused kernel).
        Mutually exclusive with use_ec.
    tbe_learning_rate
        Initial LR for TBE's fused EXACT_ROWWISE_ADAGRAD optimizer.
    hash_cardinality_threshold
        Features with vocab above this get hashed. 0 disables.
    hash_buckets
        Number of hash buckets for high-cardinality features.
    chunked_projection
        Optional chunked projection config for sequence event projection. When
        set, replaces the default per-event ``Linear+LayerNorm+GELU`` with:
        ``ChunkedProjection(...)->mean over chunk tokens``.
    """

    def __init__(
        self,
        specs: list[FeatureSpec],
        domain: str,
        emb_dim: int,
        d_model: int,
        use_ec: bool,
        use_tbe: bool = False,
        tbe_learning_rate: float = 0.01,
        hash_cardinality_threshold: int = 0,
        hash_buckets: int = 50000,
        return_jagged: bool = False,
        learned_missingness: bool = False,
        chunked_projection: dict = None,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.emb_dim = emb_dim
        self.use_ec = use_ec
        self.use_tbe = use_tbe
        self.hash_cardinality_threshold = hash_cardinality_threshold
        self.hash_buckets = hash_buckets
        self.return_jagged = return_jagged
        self.learned_missingness = learned_missingness
        self.specs = specs
        total_emb = len(self.specs) * emb_dim

        _pad_idx = 0 if not learned_missingness else None

        if use_tbe:
            self.tbe = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        _effective_vocab(spec, hash_cardinality_threshold, hash_buckets),
                        emb_dim,
                        EmbeddingLocation.DEVICE,
                        ComputeDevice.CUDA,
                    )
                    for spec in self.specs
                ],
                pooling_mode=PoolingMode.NONE,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                learning_rate=tbe_learning_rate,
            )
        elif use_ec:
            self.ec = EmbeddingCollection(
                tables=[
                    EmbeddingConfig(
                        name=spec.name,
                        embedding_dim=emb_dim,
                        num_embeddings=_effective_vocab(
                            spec, hash_cardinality_threshold, hash_buckets
                        ),
                        feature_names=[spec.name],
                    )
                    for spec in self.specs
                ]
            )
            if not learned_missingness:
                self._register_padding_hooks()
        else:
            self.tables = nn.ModuleDict(
                {
                    spec.name: nn.Embedding(
                        _effective_vocab(spec, hash_cardinality_threshold, hash_buckets),
                        emb_dim,
                        padding_idx=_pad_idx,
                    )
                    for spec in self.specs
                    if not _should_skip(spec, hash_cardinality_threshold, hash_buckets)
                }
            )

        self.use_chunked_projection = chunked_projection is not None
        if self.use_chunked_projection:
            cfg = dict(chunked_projection)
            self.chunked_proj = ChunkedProjection(
                input_dim=total_emb,
                d_model=d_model,
                num_tokens=cfg["num_tokens"],
                token_mixing=cfg["token_mixing"],
                mixing_hidden_mult=cfg["mixing_hidden_mult"],
                mixing_dropout=cfg["mixing_dropout"],
            )
            self.proj = None
        else:
            self.chunked_proj = None
            self.proj = nn.Sequential(
                nn.Linear(total_emb, d_model), nn.LayerNorm(d_model), nn.GELU()
            )

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project concatenated per-event embeddings to ``d_model`` tokens."""
        if self.chunked_proj is None:
            return self.proj(x)
        # chunked_proj returns [..., num_tokens, d_model]; collapse chunk tokens
        # back to one representation per event to keep sequence length unchanged.
        return self.chunked_proj(x).mean(dim=-2)

    def _register_padding_hooks(self) -> None:
        """Zero row-0 gradients on EC embedding weights (emulates padding_idx=0)."""
        for module in self.ec.modules():
            if isinstance(module, nn.Embedding):
                module.weight.register_hook(_zero_row0_grad)

    def forward(
        self, feat_ids: dict[str, torch.Tensor], lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed pre-extracted sequence feature tensors.

        Parameters
        ----------
        feat_ids
            Mapping ``{feature_name: tensor}``. Padded path: each is [B, max_L].
            Flat path: each is [total_events] (1D jagged values).
        lengths
            Per-sample sequence lengths, shape [B].

        Returns
        -------
        tuple
            If ``return_jagged=False``: ``(tokens [B, max_L, d_model], mask [B, max_L])``.
            If ``return_jagged=True``: ``(tokens [total_tokens, d_model], cu_seqlens [B+1])``.
        """
        B = lengths.shape[0]

        if self.return_jagged:
            return self._forward_jagged(feat_ids, B, lengths)

        if self.use_ec:
            max_L = int(lengths.max())
            emb_list = self._embed_ec(feat_ids, B, max_L, lengths)
        else:
            emb_list = self._embed_padded(feat_ids)

        tokens = self._project_tokens(torch.cat(emb_list, dim=-1))
        # Mask from actual token length (may differ from max_L when inputs are
        # pre-masked with zeroed lengths but original-shape tensors)
        actual_L = tokens.shape[1]
        mask = torch.arange(actual_L, device=tokens.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return tokens, mask

    def _forward_jagged(
        self,
        feat_ids: dict[str, torch.Tensor],
        B: int,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jagged path: embed flat, project flat, return (total_tokens, d_model) + cu_seqlens."""
        if self.use_tbe:
            return self._forward_tbe(feat_ids, B, lengths)

        kjt = self._build_kjt(feat_ids, lengths)
        embedded = self.ec(kjt)

        # Concatenate per-feature embeddings at each token position,
        # zeroing positions where the original ID was 0 (missing sentinel)
        emb_list = []
        for spec in self.specs:
            jt = embedded[spec.name]
            emb = jt.values()
            if not self.learned_missingness:
                mask = (feat_ids[spec.name] == 0).unsqueeze(-1)
                emb = emb.masked_fill(mask, 0.0)
            emb_list.append(emb)
        flat_cat = torch.cat(emb_list, dim=-1)  # (total_tokens, num_specs * emb_dim)
        tokens = self._project_tokens(flat_cat)  # (total_tokens, d_model)

        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=lengths.device)
        cu_seqlens[1:] = lengths.cumsum(0)
        return tokens, cu_seqlens

    def _forward_tbe(
        self,
        feat_ids: dict[str, torch.Tensor],
        B: int,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TBE path: single fused kernel for all tables, output feature-major."""
        T = len(self.specs)
        total_tokens = int(lengths.sum())

        all_ids = []
        all_weights = []
        for spec in self.specs:
            ids = feat_ids[spec.name].long()
            if _should_skip(spec, self.hash_cardinality_threshold, self.hash_buckets):
                ids = torch.zeros_like(ids)
            else:
                ids = self._hash_ids(ids, spec)
            if self.learned_missingness:
                w = torch.ones(ids.shape[0], dtype=torch.float, device=ids.device)
            else:
                w = (ids != 0).float()
            all_ids.append(ids)
            all_weights.append(w)

        indices = torch.cat(all_ids)  # (T * total_tokens,)
        per_sample_weights = torch.cat(all_weights)  # (T * total_tokens,)

        # Offsets: T*B+1 entries. Each feature t, sample b spans
        # offsets[t*B + b] to offsets[t*B + b + 1].
        # Since all features share the same lengths, offset for feature t is
        # base_offsets shifted by t * total_tokens.
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=lengths.device)
        cu_seqlens[1:] = lengths.cumsum(0)
        offsets_parts = [cu_seqlens[:-1] + t * total_tokens for t in range(T)]
        # Sentinel: cu_seqlens[-1] = total_tokens, so this equals T * total_tokens
        offsets_parts.append(cu_seqlens[-1:] + (T - 1) * total_tokens)
        offsets = torch.cat(offsets_parts)  # (T * B + 1,)

        # One fused kernel: output is (T * total_tokens, emb_dim), feature-major.
        # per_sample_weights is passed for gradient masking in backward, but NONE
        # mode ignores it during forward — we apply the mask manually below.
        out = self.tbe(indices, offsets, per_sample_weights)

        # Reshape: (T, total_tokens, emb_dim) → (total_tokens, T * emb_dim)
        out = out.view(T, total_tokens, self.emb_dim)
        if not self.learned_missingness:
            # Zero embeddings at padding positions (ID==0). per_sample_weights is
            # (T * total_tokens,) with 0 at padding positions — reshape to (T, total_tokens, 1).
            mask = per_sample_weights.view(T, total_tokens).unsqueeze(-1)
            out = out * mask
        out = out.permute(1, 0, 2).reshape(total_tokens, -1)

        tokens = self._project_tokens(out)
        cu_seqlens_out = cu_seqlens.to(torch.int32)
        return tokens, cu_seqlens_out

    def _hash_ids(self, ids: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        """Apply modular hashing for high-cardinality features, preserving 0 as padding."""
        if (
            self.hash_cardinality_threshold > 0
            and spec.vocab_size > self.hash_cardinality_threshold
        ):
            mask = ids > 0
            ids = ids.clone()
            ids[mask] = (ids[mask] % (self.hash_buckets - 1)) + 1
        return ids

    def _embed_padded(self, feat_ids: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Embed via nn.Embedding on padded [B, max_L] IDs."""
        embs = []
        for spec in self.specs:
            ids = feat_ids[spec.name].long()
            if _should_skip(spec, self.hash_cardinality_threshold, self.hash_buckets):
                embs.append(ids.new_zeros(*ids.shape, self.emb_dim, dtype=torch.float))
                continue
            ids = self._hash_ids(ids, spec)
            embs.append(self.tables[spec.name](ids))
        return embs

    # TODO (nima): optimize padded path — concatenate flat feature embeddings first,
    # call to_padded_dense() once on the fused (total_tokens, total_emb) tensor,
    # then slice. Avoids N independent pad operations.
    def _embed_ec(
        self,
        feat_ids: dict[str, torch.Tensor],
        B: int,
        max_L: int,
        lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Embed via torchrec EC on flat values, then pad to dense."""
        kjt = self._build_kjt(feat_ids, lengths)
        embedded = self.ec(kjt)
        embs = []
        for spec in self.specs:
            jt = embedded[spec.name]
            dense = jt.to_padded_dense(desired_length=max_L, padding_value=0.0)
            if not self.learned_missingness:
                # Zero embeddings where original ID was 0 (EC has no padding_idx).
                # Build padded ID mask using the same scatter pattern as to_padded_dense.
                ids_flat = feat_ids[spec.name].long()
                ids_padded = torch.zeros(B, max_L, dtype=ids_flat.dtype, device=ids_flat.device)
                offsets = torch.zeros(B, dtype=torch.long, device=lengths.device)
                offsets[1:] = lengths[:-1].cumsum(0)
                for i in range(B):
                    L = int(lengths[i])
                    ids_padded[i, :L] = ids_flat[offsets[i] : offsets[i] + L]
                id_mask = (ids_padded == 0).unsqueeze(-1)
                dense = dense.masked_fill(id_mask, 0.0)
            embs.append(dense)
        return embs

    def _build_kjt(
        self, feat_ids: dict[str, torch.Tensor], lengths: torch.Tensor
    ) -> KeyedJaggedTensor:
        """Build KJT from flat per-feature values + shared lengths."""
        all_values, all_lengths = [], []
        for spec in self.specs:
            if _should_skip(spec, self.hash_cardinality_threshold, self.hash_buckets):
                all_values.append(torch.zeros_like(feat_ids[spec.name]))
            else:
                all_values.append(self._hash_ids(feat_ids[spec.name], spec))
            all_lengths.append(lengths)
        return KeyedJaggedTensor(
            keys=[s.name for s in self.specs],
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        )

    def embedding_tables(self) -> dict[str, nn.Module]:
        """Return ``{name: embedding_module}`` regardless of backend path.

        For TBE path, returns the TBE module itself (not per-table modules).
        Use ``tbe_weight_views()`` for per-table weight access.
        """
        if self.use_tbe:
            # TBE weights are buffers, not nn.Embedding modules. Return a dict
            # mapping spec names to the TBE for compatibility with callers that
            # just need to iterate "which tables exist". Direct weight access
            # goes through tbe_weight_views().
            return {spec.name: self.tbe for spec in self.specs}
        if self.use_ec:
            return dict(self.ec.embeddings)
        return dict(self.tables)

    def tbe_weight_views(self) -> dict[str, torch.Tensor]:
        """Return per-table weight views from TBE's split_embedding_weights().

        Only valid when use_tbe=True. Returns ``{spec_name: weight_tensor}``.
        """
        weights = self.tbe.split_embedding_weights()
        return {spec.name: weights[i] for i, spec in enumerate(self.specs)}

    def init_weights(self) -> None:
        """Xavier-normal init on all tables. Zeros row 0 when not learned_missingness."""
        if self.use_tbe:
            for weight in self.tbe.split_embedding_weights():
                nn.init.xavier_normal_(weight)
                if not self.learned_missingness:
                    weight[0, :] = 0
        elif self.use_ec:
            for emb in self.ec.embeddings.values():
                nn.init.xavier_normal_(emb.weight.data)
                if not self.learned_missingness:
                    emb.weight.data[0, :] = 0
        else:
            for emb in self.tables.values():
                nn.init.xavier_normal_(emb.weight.data)
                if not self.learned_missingness and emb.padding_idx is not None:
                    emb.weight.data[0, :] = 0

    def reinit_high_cardinality(self, cardinality_threshold: int) -> set[int]:
        """Reinit tables above threshold. Returns data_ptrs of reinitialized weights."""
        reinitialized: set[int] = set()
        if self.use_tbe:
            weights = self.tbe.split_embedding_weights()
            opt_state = self.tbe.get_optimizer_state()
            for i, spec in enumerate(self.specs):
                if weights[i].shape[0] > cardinality_threshold:
                    with torch.no_grad():
                        nn.init.normal_(weights[i], std=0.02)
                        if not self.learned_missingness:
                            weights[i][0, :] = 0
                        if "sum" in opt_state[i]:
                            opt_state[i]["sum"].zero_()
                    reinitialized.add(weights[i].data_ptr())
        elif self.use_ec:
            for name, emb in self.ec.embeddings.items():
                if emb.num_embeddings > cardinality_threshold:
                    nn.init.normal_(emb.weight, std=0.02)
                    if not self.learned_missingness:
                        emb.weight.data[0, :] = 0
                    reinitialized.add(emb.weight.data_ptr())
        else:
            for name, table in self.tables.items():
                if table.num_embeddings > cardinality_threshold:
                    nn.init.normal_(table.weight, std=0.02)
                    if table.padding_idx is not None:
                        table.weight.data[table.padding_idx].zero_()
                    reinitialized.add(table.weight.data_ptr())
        return reinitialized

    def snapshot_weights(self, vocab_threshold: int) -> dict[str, torch.Tensor]:
        """Clone weights for tables with vocab <= threshold.

        Keys are fully qualified: ``seq.{domain}.{table_name}``.
        """
        prefix = f"seq.{self.domain}."
        snapshot = {}
        if self.use_tbe:
            views = self.tbe_weight_views()
            for name, weight in views.items():
                if weight.shape[0] <= vocab_threshold:
                    snapshot[f"{prefix}{name}"] = weight.clone()
        elif self.use_ec:
            for name, emb in self.ec.embeddings.items():
                if emb.num_embeddings <= vocab_threshold:
                    snapshot[f"{prefix}{name}"] = emb.weight.data.clone()
        else:
            for name, table in self.tables.items():
                if table.num_embeddings <= vocab_threshold:
                    snapshot[f"{prefix}{name}"] = table.weight.data.clone()
        return snapshot

    def restore_weights(self, snapshot: dict[str, torch.Tensor]) -> set[int]:
        """Restore previously snapshotted weights. Returns restored data_ptrs.

        Filters for keys matching ``seq.{self.domain}.*`` and ignores the rest.
        """
        prefix = f"seq.{self.domain}."
        ptrs: set[int] = set()
        if self.use_tbe:
            views = self.tbe_weight_views()
            for key, weight in snapshot.items():
                if not key.startswith(prefix):
                    continue
                name = key.removeprefix(prefix)
                with torch.no_grad():
                    views[name].copy_(weight)
                ptrs.add(views[name].data_ptr())
        elif self.use_ec:
            for key, weight in snapshot.items():
                if not key.startswith(prefix):
                    continue
                name = key.removeprefix(prefix)
                self.ec.embeddings[name].weight.data.copy_(weight)
                ptrs.add(self.ec.embeddings[name].weight.data_ptr())
        else:
            for key, weight in snapshot.items():
                if not key.startswith(prefix):
                    continue
                name = key.removeprefix(prefix)
                self.tables[name].weight.data.copy_(weight)
                ptrs.add(self.tables[name].weight.data_ptr())
        return ptrs
