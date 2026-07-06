"""Item feature routing and projection modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.data.schema import FeatureSchema


class RouteFeatureProjector(nn.Module):
    """Project a semantic route's schema-selected static numerical features.

    Two modes:
    - concat (default): concatenate all features, single Linear projection.
    - per_feature: independent Linear+LN per feature, SiLU, sum.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        exprs: list[str],
        d_model: int,
        dropout_rate: float,
        per_feature: bool = False,
    ) -> None:
        super().__init__()
        self._schema = schema
        self._per_feature = per_feature
        self._spec_names: list[str] = []
        self._dims: list[int] = []
        seen = set()
        for expr in exprs:
            for spec in schema.query(expr):
                if spec.name not in seen:
                    self._spec_names.append(spec.name)
                    self._dims.append(spec.dim)
                    seen.add(spec.name)

        in_dim = sum(self._dims)
        if in_dim <= 0:
            raise ValueError(f"RouteFeatureProjector matched no features for exprs={exprs!r}")

        if per_feature:
            self.feature_projs = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model))
                    for dim in self._dims
                ]
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
        self.dropout = nn.Dropout(dropout_rate)

    @property
    def feature_names(self) -> set[str]:
        """Feature names consumed by this route projector."""
        return set(self._spec_names)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Extract and project route features to [B, D]."""
        names_str = ", ".join(f"'{n}'" for n in self._spec_names)
        feats = self._schema.extract(
            batch,
            expr=f"name in ({names_str}) and scope = 'static' and source != 'metadata'",
            cat=True,
        )
        if self._per_feature:
            splits = torch.split(feats, self._dims, dim=-1)
            parts = [F.silu(proj(x)) for proj, x in zip(self.feature_projs, splits)]
            return self.dropout(sum(parts))
        return self.proj(feats)


class ItemBankSourceProjector(nn.Module):
    """Project selected ItemDenseRouter bank entries into a semantic source."""

    def __init__(self, sources: list[str], d_model: int, dropout_rate: float) -> None:
        super().__init__()
        if not sources:
            raise ValueError("ItemBankSourceProjector requires at least one source")
        self.sources = sources
        self.proj = (
            nn.Identity()
            if len(sources) == 1
            else nn.Sequential(
                nn.Linear(d_model * len(sources), d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
        )

    def forward(self, item_bank: dict[str, torch.Tensor]) -> torch.Tensor:
        """Merge selected item bank entries."""
        missing = [name for name in self.sources if name not in item_bank]
        if missing:
            raise ValueError(f"semantic item bank source missing entries: {missing}")
        return self.proj(torch.cat([item_bank[name] for name in self.sources], dim=-1))


class ItemDenseRouter(nn.Module):
    """Split item dense features into semantic groups, then route
    purpose-built representations to different consumers.

    Normalization (L2-norm for embeddings, RSSC for counts) is handled
    by data-pipeline blocks before the model sees the features.

    Groups:
    - emb: per-fid Linear+LN -> sum
    - count: concat + Linear+LN

    forward() returns (ns_token [B,1,D], target_query [B,D]).
    """

    def __init__(
        self,
        schema: FeatureSchema,
        d_model: int,
        dropout_rate: float,
        groups: dict[str, dict],
        routing: dict = None,
    ) -> None:
        super().__init__()
        self._schema = schema
        self._d_model = d_model

        # Emb group: per-fid projection
        emb_cfg = groups["emb"]
        emb_specs = schema.query(emb_cfg["expr"])

        self._emb_fid_names: list[str] = []
        self.emb_projs = nn.ModuleDict()
        for spec in emb_specs:
            self.emb_projs[spec.name] = nn.Sequential(
                nn.Linear(spec.dim, d_model), nn.LayerNorm(d_model)
            )
            self._emb_fid_names.append(spec.name)

        # Count group: single concat projection
        count_cfg = groups["count"]
        count_specs = schema.query(count_cfg["expr"])
        self._count_fid_names = [s.name for s in count_specs]
        count_dim = sum(s.dim for s in count_specs)
        self.count_proj = (
            nn.Sequential(nn.Linear(count_dim, d_model), nn.LayerNorm(d_model))
            if count_dim > 0
            else None
        )

        # Routing: merge subsets of {sparse, emb, count} into named outputs
        route_dropout = routing["dropout"]
        self._all_sources = self._resolve_sources(routing["all_sources"])
        self._query_sources = self._resolve_sources(
            routing.get("din_query_sources", routing["all_sources"])
        )
        self._embsem_sources = self._resolve_sources(routing["embsem_query_sources"])
        self.all_merge_proj = self._make_merger(self._all_sources, d_model, route_dropout)
        self.query_merge_proj = self._make_merger(self._query_sources, d_model, route_dropout)
        self.embsem_merge_proj = self._make_merger(self._embsem_sources, d_model, route_dropout)

    def _resolve_sources(self, sources: list[str]) -> list[str]:
        """Filter source list to only those backed by data."""
        active = []
        for src in sources:
            if src == "sparse":
                active.append(src)
            elif src == "emb" and self._emb_fid_names:
                active.append(src)
            elif src == "count" and self.count_proj is not None:
                active.append(src)
        return list(dict.fromkeys(active))

    def _make_merger(self, sources: list[str], d_model: int, dropout: float) -> nn.Module:
        if not sources:
            return None
        return nn.Sequential(
            nn.Linear(d_model * len(sources), d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _build_bank(
        self, sparse_repr: torch.Tensor, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Build internal representation bank from feature groups."""
        bank: dict[str, torch.Tensor] = {"sparse": sparse_repr}

        emb_reprs = []
        for name in self._emb_fid_names:
            x = self._schema.extract(
                batch,
                expr=f"name = '{name}' and scope = 'static' and source != 'metadata'",
                cat=True,
            )
            emb_reprs.append(F.silu(self.emb_projs[name](x)))
        if emb_reprs:
            bank["emb"] = torch.stack(emb_reprs, dim=0).sum(dim=0)

        if self.count_proj is not None and self._count_fid_names:
            count_parts = [
                self._schema.extract(
                    batch,
                    expr=f"name = '{n}' and scope = 'static' and source != 'metadata'",
                    cat=True,
                )
                for n in self._count_fid_names
            ]
            bank["count"] = F.silu(self.count_proj(torch.cat(count_parts, dim=-1)))

        return bank

    def _merge_sources(
        self,
        bank: dict[str, torch.Tensor],
        sources: list[str],
        merger: nn.Module,
    ) -> torch.Tensor:
        """Merge selected sources via cat_proj."""
        vecs = [bank[src] for src in sources]
        return merger(torch.cat(vecs, dim=-1))

    def forward(
        self, item_ns_tokens: torch.Tensor, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce item token for ns_parts and DIN target query.

        Parameters
        ----------
        item_ns_tokens
            Raw chunked-projection tokens [B, N, D].
        batch
            Full batch dict for schema extraction.

        Returns
        -------
        ns_token
            Shape [B, 1, D] — ready to cat into ns_parts.
        target_query
            Shape [B, D] — DIN target.
        """
        item_cat_ns = item_ns_tokens.mean(dim=1)
        bank = self._build_bank(item_cat_ns, batch)
        all_repr = self._merge_sources(bank, self._all_sources, self.all_merge_proj)
        query_repr = self._merge_sources(bank, self._query_sources, self.query_merge_proj)
        return all_repr.unsqueeze(1), query_repr

    def forward_with_bank(
        self, item_ns_tokens: torch.Tensor, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Produce item token, DIN target query, and named route representations."""
        item_cat_ns = item_ns_tokens.mean(dim=1)
        bank = self._build_bank(item_cat_ns, batch)
        all_repr = self._merge_sources(bank, self._all_sources, self.all_merge_proj)
        query_repr = self._merge_sources(bank, self._query_sources, self.query_merge_proj)
        bank["all"] = all_repr
        bank["query"] = query_repr
        if self.embsem_merge_proj is not None:
            bank["embsem_query"] = self._merge_sources(
                bank, self._embsem_sources, self.embsem_merge_proj
            )
        return all_repr.unsqueeze(1), query_repr, bank

    @property
    def excluded_feature_names(self) -> set[str]:
        """Feature names consumed by router (exclude from shared dense projection)."""
        return set(self._emb_fid_names) | set(self._count_fid_names)
