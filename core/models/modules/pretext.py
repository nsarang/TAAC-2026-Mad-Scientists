"""PretextHead v2: self-supervised prediction of static features from sequences.

Predicts categorical IDs and numerical values of static features using
attended sequence representations. Targets are resolved from FeatureSchema
at init time — no raw offset tuples or manual coordinate resolution.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.data.schema import FeatureSchema, FeatureSpec
from core.models.modules.din import TargetAwareDINHead
from core.models.modules.segment_ops import segment_sum


def _attend_inner(
    heads: nn.ModuleDict,
    din_mode: str,
    queries: list[torch.Tensor],
    seq_tokens_list: list[torch.Tensor],
    seq_masks_list: list[torch.Tensor],
    domain_keys: list[str],
) -> list[torch.Tensor]:
    """Pure-tensor pretext DIN attention (compile-friendly, no schema access)."""
    contexts = []
    if din_mode == "all":
        head = heads["all"]
        for q, seq_tokens, mask_or_cu in zip(queries, seq_tokens_list, seq_masks_list):
            contexts.append(head.attend_only(q, seq_tokens, mask_or_cu))
    else:
        for domain, q, seq_tokens, mask_or_cu in zip(
            domain_keys, queries, seq_tokens_list, seq_masks_list
        ):
            contexts.append(heads[domain].attend_only(q, seq_tokens, mask_or_cu))
    return contexts


class PretextHeadV2(nn.Module):
    """Self-supervised head predicting static features from sequence contexts.

    Parameters
    ----------
    schema
        FeatureSchema with all features registered.
    d_model
        Hidden / token dimensionality.
    seq_domains
        Ordered list of sequence domain names.
    cat_targets
        DSL expression string for categorical prediction targets.
    num_targets
        DSL expression string for numerical prediction targets.
    per_domain_cat_targets
        Dict mapping domain name → DSL expression string for
        domain-specific categorical prediction.
    per_domain_num_targets
        Dict mapping domain name → DSL expression string for
        domain-specific numerical prediction.
    weight
        Loss multiplier.
    dropout
        Dropout rate in DIN MLPs and projection.
    hidden_mult
        DIN attention MLP expansion factor.
    query_type
        ``"mean_pool"`` or ``"learned"``.
    share_din
        If True, receives pre-attended domain contexts from the main model
        instead of running internal attention.
    din_mode
        ``"all"`` (one shared DIN head) or ``"per_domain"``.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        seq_domains: list[str],
        cat_targets: str = None,
        num_targets: str = None,
        per_domain_cat_targets: dict[str, str] = None,
        per_domain_num_targets: dict[str, str] = None,
        weight: float = 0.1,
        dropout: float = 0.1,
        hidden_mult: int = 2,
        query_type: str = "mean_pool",
        share_din: bool = False,
        din_mode: str = "all",
        shared_num_head: bool = False,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.weight = weight
        self.d_model = d_model
        self.seq_domains = seq_domains
        self.num_domains = len(seq_domains)
        self.query_type = query_type
        self.share_din = share_din
        self.din_mode = din_mode
        self._shared_num_head = shared_num_head

        # Resolve categorical targets
        if cat_targets:
            self._cat_specs: list[FeatureSpec] = schema.query(cat_targets)
        else:
            self._cat_specs: list[FeatureSpec] = []

        # Resolve numerical targets
        if num_targets:
            self._num_specs: list[FeatureSpec] = schema.query(num_targets)
        else:
            self._num_specs: list[FeatureSpec] = []

        # Per-domain categorical targets
        self._per_domain_cat_specs: dict[str, list[FeatureSpec]] = {}
        for domain, expr in (per_domain_cat_targets or {}).items():
            specs = schema.query(expr)
            if specs:
                self._per_domain_cat_specs[domain] = specs

        # Per-domain numerical targets
        self._per_domain_num_specs: dict[str, list[FeatureSpec]] = {}
        for domain, expr in (per_domain_num_targets or {}).items():
            specs = schema.query(expr)
            if specs:
                self._per_domain_num_specs[domain] = specs

        # Query parameter
        if query_type == "learned":
            if din_mode == "per_domain" and not share_din:
                self.query = nn.Parameter(torch.randn(self.num_domains, d_model) * 0.02)
            else:
                self.query = nn.Parameter(torch.randn(d_model) * 0.02)

        # DIN attention heads (when not sharing)
        if not share_din:
            self.din_heads: nn.ModuleDict = nn.ModuleDict()
            if din_mode == "all":
                self.din_heads["all"] = TargetAwareDINHead(d_model, hidden_mult, dropout)
            elif din_mode == "per_domain":
                for domain in seq_domains:
                    self.din_heads[domain] = TargetAwareDINHead(d_model, hidden_mult, dropout)
        else:
            self.din_heads = None

        # Domain fusion
        self.domain_proj = nn.Sequential(
            nn.Linear(d_model * self.num_domains, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Shared categorical classifiers
        self.cat_classifiers = nn.ModuleList()
        for spec in self._cat_specs:
            self.cat_classifiers.append(nn.Linear(d_model, spec.vocab_size * spec.dim))

        # Shared numerical heads
        self.num_heads = nn.ModuleDict()
        if shared_num_head and self._num_specs:
            total_dim = sum(s.dim for s in self._num_specs)
            self.num_heads["raw"] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, total_dim),
            )
        else:
            for spec in self._num_specs:
                self.num_heads[spec.name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, spec.dim),
                )

        # Per-domain categorical classifiers
        self.domain_cat_classifiers: nn.ModuleDict = nn.ModuleDict()
        for domain, specs in self._per_domain_cat_specs.items():
            clfs = nn.ModuleList()
            for spec in specs:
                clfs.append(nn.Linear(d_model, spec.vocab_size * spec.dim))
            self.domain_cat_classifiers[domain] = clfs

        # Per-domain numerical heads
        self.domain_num_heads: nn.ModuleDict = nn.ModuleDict()
        for domain, specs in self._per_domain_num_specs.items():
            heads = nn.ModuleDict()
            for spec in specs:
                heads[spec.name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, spec.dim),
                )
            self.domain_num_heads[domain] = heads

    def enable_compile(self) -> None:
        """Compile the pretext DIN attention loop."""
        self._compiled_attend = torch.compile(_attend_inner, dynamic=True)

    def _get_query(
        self, seq_tokens_list: list[torch.Tensor], seq_masks_list: list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        jagged = seq_tokens_list[0].dim() == 2
        if jagged:
            B = seq_masks_list[0].shape[0] - 1
        else:
            B = seq_tokens_list[0].shape[0]
        if self.query_type == "learned":
            if self.query.dim() == 2:
                return [self.query[i].unsqueeze(0).expand(B, -1) for i in range(self.num_domains)]
            return self.query.unsqueeze(0).expand(B, -1)
        pooled = []
        if jagged:
            for seq_tokens, cu_seqlens in zip(seq_tokens_list, seq_masks_list):
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).float().clamp(min=1)
                seg_sum = segment_sum(seq_tokens, cu_seqlens)  # (B, D)
                pooled.append(seg_sum / lengths.unsqueeze(-1))
        else:
            for seq_tokens, mask in zip(seq_tokens_list, seq_masks_list):
                valid = (~mask).unsqueeze(-1).float()
                pooled.append((seq_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1))
        return torch.stack(pooled, dim=0).mean(dim=0)

    def forward(
        self,
        seq_tokens_list: list[torch.Tensor],
        seq_masks_list: list[torch.Tensor],
        batch: dict[str, Any],
        domain_contexts: list[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute pretext losses.

        Parameters
        ----------
        seq_tokens_list
            Per-domain sequence token tensors, each ``[B, L_i, D]``.
        seq_masks_list
            Per-domain padding masks, each ``[B, L_i]``, True = pad.
        batch
            Unmasked batch dict for extracting ground-truth targets.
        domain_contexts
            Pre-attended per-domain contexts (required when share_din=True).

        Returns
        -------
        Dict of named scalar losses (already weighted).
        """
        if not self.training:
            return {}

        if self.share_din:
            assert domain_contexts is not None
        else:
            queries = self._get_query(seq_tokens_list, seq_masks_list)
            domain_contexts = self._attend(queries, seq_tokens_list, seq_masks_list)

        # Fuse domain contexts
        h = self.domain_proj(torch.cat(domain_contexts, dim=-1))

        losses: dict[str, torch.Tensor] = {}
        schema = self.schema

        # Shared categorical loss
        if self.cat_classifiers:
            cat_loss = h.new_tensor(0.0)
            for clf, spec in zip(self.cat_classifiers, self._cat_specs):
                target = schema.extract(batch, spec.name).long()
                logits = clf(h).view(-1, spec.dim, spec.vocab_size)
                cat_loss = cat_loss + F.cross_entropy(
                    logits.reshape(-1, spec.vocab_size), target.reshape(-1), ignore_index=0
                )
            losses["pretext_cat"] = self.weight * cat_loss / len(self.cat_classifiers)

        # Shared numerical loss
        if self.num_heads:
            num_loss = h.new_tensor(0.0)
            if self._shared_num_head:
                pred_all = self.num_heads["raw"](h)
                offset = 0
                for spec in self._num_specs:
                    pred = pred_all[:, offset : offset + spec.dim].contiguous().float()
                    target = schema.extract(batch, spec.name).contiguous().float()
                    num_loss = num_loss + F.smooth_l1_loss(pred, target)
                    offset += spec.dim
            else:
                for spec in self._num_specs:
                    pred = self.num_heads[spec.name](h)
                    target = schema.extract(batch, spec.name).float()
                    num_loss = num_loss + F.smooth_l1_loss(pred, target)
            losses["pretext_num"] = self.weight * num_loss / len(self._num_specs)

        # Per-domain categorical losses
        if self.domain_cat_classifiers:
            domain_idx = {d: i for i, d in enumerate(self.seq_domains)}
            for domain, specs in self._per_domain_cat_specs.items():
                ctx = domain_contexts[domain_idx[domain]]
                clfs = self.domain_cat_classifiers[domain]
                dom_loss = ctx.new_tensor(0.0)
                for clf, spec in zip(clfs, specs):
                    target = schema.extract(batch, spec.name).long()
                    logits = clf(ctx).view(-1, spec.dim, spec.vocab_size)
                    dom_loss = dom_loss + F.cross_entropy(
                        logits.reshape(-1, spec.vocab_size), target.reshape(-1), ignore_index=0
                    )
                losses[f"pretext_{domain}"] = self.weight * dom_loss / len(specs)

        # Per-domain numerical losses
        if self.domain_num_heads:
            domain_idx = {d: i for i, d in enumerate(self.seq_domains)}
            for domain, specs in self._per_domain_num_specs.items():
                ctx = domain_contexts[domain_idx[domain]]
                heads = self.domain_num_heads[domain]
                dom_loss = ctx.new_tensor(0.0)
                for spec in specs:
                    pred = heads[spec.name](ctx)
                    target = schema.extract(batch, spec.name).float()
                    dom_loss = dom_loss + F.smooth_l1_loss(pred, target)
                losses[f"pretext_num_{domain}"] = self.weight * dom_loss / len(specs)

        return losses

    def _attend(
        self,
        queries: torch.Tensor | list[torch.Tensor],
        seq_tokens_list: list[torch.Tensor],
        seq_masks_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run internal DIN attention to produce per-domain contexts.

        `attend_only` dispatches on seq_tokens dim: 3D → padded, 2D → jagged.
        `seq_masks_list` is padding masks (padded) or cu_seqlens (jagged).
        """
        # Normalize queries to a list (one per domain)
        if not isinstance(queries, list):
            queries = [queries] * len(seq_tokens_list)

        fn = self._compiled_attend if hasattr(self, "_compiled_attend") else _attend_inner
        return fn(
            self.din_heads,
            self.din_mode,
            queries,
            seq_tokens_list,
            seq_masks_list,
            self.seq_domains,
        )
