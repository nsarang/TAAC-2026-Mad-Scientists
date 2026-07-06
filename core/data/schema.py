"""Dataset schema: JSON parser, typed feature specs, and tag-based selection.

Two layers:
- ``DatasetSchema`` parses schema.json and provides parquet column names for I/O.
- ``FeatureSchema`` is the typed, queryable registry that blocks and models consume.
  Built from DatasetSchema + block output declarations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Flag, auto
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
from lark import Lark, Transformer, v_args

# ─────────────────────── Tag Enums ────────────────────────────────────


class Dtype(Flag):
    """What kind of values this feature carries."""

    CATEGORICAL = auto()
    NUMERICAL = auto()


class Entity(Flag):
    """Who/what this feature describes."""

    USER = auto()
    ITEM = auto()
    CONTEXT = auto()


class Scope(Flag):
    """Whether the feature is static (one value per sample) or sequential."""

    STATIC = auto()
    SEQ = auto()


class Source(Flag):
    """What kind of feature this is."""

    ORIGINAL = auto()
    DERIVED = auto()
    METADATA = auto()


# Backward compat alias — external code may import Origin
Origin = Source


# ─────────────────────── Filter DSL ──────────────────────────────────

_GRAMMAR = r"""
    ?start: expr

    ?expr: or_expr
    ?or_expr: and_expr (_OR and_expr)*
    ?and_expr: atom (_AND atom)*

    ?atom: comparison
         | _NOT atom              -> not_expr
         | "(" expr ")"

    ?comparison: field "=" value            -> eq
              | field "!=" value            -> neq
              | field "<>" value            -> neq
              | field _IN "(" values ")"    -> in_expr
              | field _MATCHES STRING       -> glob
              | field ">" NUMBER            -> gt
              | field ">=" NUMBER           -> gte
              | field "<" NUMBER            -> lt
              | field "<=" NUMBER           -> lte

    field: CNAME
    values: value ("," value)*
    ?value: STRING              -> string_val
          | CNAME               -> bare_val

    _OR: /[Oo][Rr]/
    _AND: /[Aa][Nn][Dd]/
    _NOT: /[Nn][Oo][Tt]/
    _IN: /[Ii][Nn]/
    _MATCHES: /[Mm][Aa][Tt][Cc][Hh][Ee][Ss]/

    STRING: "'" /[^']*/ "'"
    CNAME: /(?!(?:[Aa][Nn][Dd]|[Oo][Rr]|[Nn][Oo][Tt]|[Ii][Nn]|[Mm][Aa][Tt][Cc][Hh][Ee][Ss])\b)[a-z_][a-z_0-9]*/
    NUMBER: /[0-9]+/

    COMMENT: /--[^\n]*/

    %import common.WS
    %ignore WS
    %ignore COMMENT
"""

_ENUM_FIELDS: dict[str, dict[str, str]] = {
    "dtype": {m.upper(): m.upper() for m in Dtype.__members__},
    "entity": {m.upper(): m.upper() for m in Entity.__members__},
    "source": {m.upper(): m.upper() for m in Source.__members__},
    "scope": {m.upper(): m.upper() for m in Scope.__members__},
}

_NUMERIC_FIELDS: set[str] = {"dim", "vocab_size"}
_VALID_FIELDS: set[str] = {"name", "group", "domain"} | set(_ENUM_FIELDS) | _NUMERIC_FIELDS


def _get_str(spec: "FeatureSpec", field_name: str) -> str:
    """Get field value as string for comparisons."""
    if field_name == "scope":
        val = spec.scope
    else:
        val = getattr(spec, field_name, None)
    if val is None:
        return ""
    if isinstance(val, Flag):
        return val.name
    return str(val)


def _get_num(spec: "FeatureSpec", field_name: str) -> float:
    """Get field value as number for numeric comparisons."""
    val = getattr(spec, field_name, None)
    return float(val) if val is not None else 0.0


def _validate_field(field_name: str) -> None:
    if field_name not in _VALID_FIELDS:
        raise ValueError(f"Unknown field '{field_name}'. Valid fields: {sorted(_VALID_FIELDS)}")


def _validate_enum_value(field_name: str, val: str) -> None:
    if field_name in _ENUM_FIELDS:
        upper = val.upper()
        valid = _ENUM_FIELDS[field_name]
        if upper not in valid:
            raise ValueError(
                f"Invalid value '{val}' for field '{field_name}'. Valid: {sorted(valid)}"
            )


@v_args(inline=True)
class _FilterCompiler(Transformer):
    """Compiles filter AST into a predicate closure with compile-time validation."""

    def string_val(self, token):
        return str(token)[1:-1]

    def bare_val(self, token):
        return str(token)

    def field(self, token):
        name = str(token)
        _validate_field(name)
        return name

    def values(self, *args):
        return set(args)

    def eq(self, field_name, val):
        _validate_enum_value(field_name, val)
        cmp_val = val.upper() if field_name in _ENUM_FIELDS else val
        return lambda spec: _get_str(spec, field_name) == cmp_val

    def neq(self, field_name, val):
        _validate_enum_value(field_name, val)
        cmp_val = val.upper() if field_name in _ENUM_FIELDS else val
        return lambda spec: _get_str(spec, field_name) != cmp_val

    def in_expr(self, field_name, val_set):
        for v in val_set:
            _validate_enum_value(field_name, v)
        cmp_set = {v.upper() for v in val_set} if field_name in _ENUM_FIELDS else val_set
        return lambda spec: _get_str(spec, field_name) in cmp_set

    def glob(self, field_name, pattern):
        pat = str(pattern)[1:-1]
        return lambda spec: fnmatch(_get_str(spec, field_name), pat)

    def gt(self, field_name, num):
        n = float(num)
        return lambda spec: _get_num(spec, field_name) > n

    def gte(self, field_name, num):
        n = float(num)
        return lambda spec: _get_num(spec, field_name) >= n

    def lt(self, field_name, num):
        n = float(num)
        return lambda spec: _get_num(spec, field_name) < n

    def lte(self, field_name, num):
        n = float(num)
        return lambda spec: _get_num(spec, field_name) <= n

    def not_expr(self, pred):
        return lambda spec: not pred(spec)

    def and_expr(self, *preds):
        return lambda spec: all(p(spec) for p in preds)

    def or_expr(self, *preds):
        return lambda spec: any(p(spec) for p in preds)

    def start(self, pred):
        return pred


_parser = Lark(_GRAMMAR, parser="lalr", transformer=_FilterCompiler())


@lru_cache(maxsize=1024)
def compile_query(expr: str) -> Callable[["FeatureSpec"], bool]:
    """Compile a filter expression into a predicate function.

    Parameters
    ----------
    expr
        DSL expression string (e.g. ``"entity == 'user' and dtype == 'numerical'"``).

    Returns
    -------
    Callable that accepts a FeatureSpec and returns bool.

    Raises
    ------
    ValueError
        On unknown fields or invalid enum values.
    lark.exceptions.UnexpectedInput
        On syntax errors.
    """
    return _parser.parse(expr)


# ─────────────────────── FeatureSpec ──────────────────────────────────


@dataclass(slots=True)
class FeatureSpec:
    """One logical feature in the schema, including its physical batch location.

    Parameters
    ----------
    `name`
        Unique feature name, used as the addressing key.
    `dtype`
        CATEGORICAL (int IDs for embedding lookup) or NUMERICAL (floats).
    `entity`
        USER, ITEM, or CONTEXT.
    `dim`
        Output width: number of float columns for NUMERICAL, number of
        ID slots per sample for CATEGORICAL.
    `vocab_size`
        Only for CATEGORICAL. Max ID + 1 (embedding table rows).
    `domain`
        If set, this feature is sequence-level (one value per timestep)
        and belongs to this domain. None means static (one value per sample).
    `source`
        What kind of feature: ORIGINAL (from raw data), DERIVED (produced
        by a BatchTransform), or METADATA (infrastructure like lengths and
        timestamps). Pretext head should only predict ORIGINAL features.
    `group`
        Optional selection label for config-level feature selection, SAGE
        groups, and ablation studies. Multiple features can share a group.
    `batch_key`
        Key in the collated batch dict where this feature's data lives.
        None before the collator assigns layout for seq features.
    `col_range`
        For features packed into a shared tensor, the column range
        ``(start, end)`` in the last dimension.
    `source_col`
        Parquet column name that this feature reads from during fitting.
    `source_offset`
        For split features: offset within the source column's list values
        where this feature's data begins. E.g. a sub-feature starting at
        position 128 in a 130-dim list column has ``source_offset=128``.
    """

    name: str
    dtype: Dtype
    entity: Entity
    dim: int
    vocab_size: int = None
    domain: str = None
    source: Source = Source.ORIGINAL
    group: str = None
    batch_key: str = None
    col_range: tuple[int, int] = None
    source_col: str = None
    source_offset: int = 0

    @property
    def scope(self) -> Scope:
        """SEQ if this feature belongs to a sequence domain, else STATIC."""
        return Scope.SEQ if self.domain else Scope.STATIC

    def __post_init__(self) -> None:
        if self.dtype == Dtype.CATEGORICAL and self.vocab_size is None:
            raise ValueError(f"CATEGORICAL feature {self.name!r} requires vocab_size")


# ─────────────────────── FeatureSchema ────────────────────────────────


class FeatureSchema:
    """Central registry that maps logical features to physical batch locations.

    Every feature in the system is registered here with two kinds of info:

    1. **Logical metadata** — name, dtype, entity, vocab_size, domain,
       source. Used for querying and selection.
    2. **Physical layout** — batch_key and col_range fields on the spec
       itself. Tells extract where to slice from the collated batch.

    Features are registered at construction time by `build_feature_schema`
    (raw data features) and by each BatchTransform's `output_specs` (derived
    features).

    Two API layers:

    - **Query** (`select`) — find specs by name/glob and/or tag filters.
    - **Extraction** (`extract`) — slice tensors out of a collated batch by
      name or predicate. Used in model forward passes.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        """Register or overwrite a feature spec."""
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a feature from the schema."""
        del self._specs[name]

    # ──── Query API ────

    def query(self, expr: str) -> list[FeatureSpec]:
        """Filter features using a DSL expression string.

        Fully explicit semantics — only filters on fields mentioned in the
        expression. Unmentioned fields are unconstrained.

        Parameters
        ----------
        expr
            Filter expression, e.g.
            ``"entity = 'user' and dtype = 'numerical' and source = 'original'"``

        Supported operators: ``=``, ``!=``/``<>``, ``in``, ``matches`` (fnmatch glob),
        ``>``, ``>=``, ``<``, ``<=``, ``and``, ``or``, ``not``, parentheses.
        Keywords (and, or, not, in, matches) are case-insensitive.

        Valid fields: name, group, domain, dtype, entity, source, scope, dim, vocab_size.
        """
        predicate = compile_query(expr)
        return [spec for spec in self._specs.values() if predicate(spec)]

    # ──── Extraction API ────

    @staticmethod
    def _cat(arrays: list):
        """Concatenate along last dim, numpy or torch."""
        import numpy as np

        if isinstance(arrays[0], np.ndarray):
            return np.concatenate(arrays, axis=-1)
        return torch.cat(arrays, dim=-1)

    def extract(
        self,
        batch: dict[str, Any],
        names: str | list[str] = None,
        *,
        expr: str = None,
        cat: bool = False,
    ):
        """Extract array views from batch by name or expression.

        Parameters
        ----------
        batch
            Batch dict (numpy pre-collation or torch post-collation).
        names
            A feature name or list of names (exact match only).
            When a single string resolves to one feature and ``cat=False``,
            returns the raw array. Otherwise returns a dict.
        expr
            DSL expression string. Returns a dict of matched arrays,
            or a single concatenated array if ``cat=True``.
        cat
            If True, concatenate all matched arrays along the last dim.
        """
        if expr is not None:
            matched = self.query(expr)
            views = {s.name: self._slice(batch, s.name) for s in matched}
            if cat:
                if not views:
                    return None
                return self._cat(list(views.values()))
            return views

        if names is not None:
            if isinstance(names, str):
                return self._slice(batch, names)
            views = {n: self._slice(batch, n) for n in names}
            if cat:
                if not views:
                    return None
                return self._cat(list(views.values()))
            return views

        raise ValueError("extract() requires either `names` or `expr`")

    def update(
        self,
        batch: dict[str, Any],
        expr: str,
        data,
    ) -> None:
        """Write data back into batch columns matching expr. In-place.

        Parameters
        ----------
        batch
            Batch dict (numpy or torch). Modified in-place.
        expr
            DSL expression selecting target features.
        data
            Replacement array with last-dim width equal to the sum of
            matched col_range widths, in query-result order.
        """
        matched = self.query(expr)
        if not matched:
            return
        offset = 0
        for spec in matched:
            width = spec.col_range[1] - spec.col_range[0] if spec.col_range else None
            if width is not None:
                batch[spec.batch_key][..., spec.col_range[0] : spec.col_range[1]] = data[
                    ..., offset : offset + width
                ]
                offset += width
            else:
                batch[spec.batch_key] = data[..., offset : offset + spec.dim]
                offset += spec.dim

    def _slice(self, batch: dict[str, Any], feat_name: str):
        """Return a view for a single feature (numpy or torch).

        Dispatches on layout type:
        - col_range set: slice last dim of shared tensor -> ``[B, cols]``
        - otherwise: return the entire tensor at batch_key
        """
        spec = self._specs[feat_name]
        if spec.col_range is not None:
            s, e = spec.col_range
            return batch[spec.batch_key][..., s:e]
        return batch[spec.batch_key]

    # ──── Layout accessors ────

    @property
    def seq_domains(self) -> list[str]:
        """Sorted unique domain values across all registered specs."""
        return sorted({s.domain for s in self._specs.values() if s.domain is not None})

    def static_batch_keys(self) -> list[str]:
        """Unique batch keys for static (non-sequence) features, in registration order.

        Returns deduplicated keys like ``["user_cat", "item_cat", "user_cont"]``.
        Multiple features may share a batch key (e.g. all user categoricals
        pack into ``"user_cat"``), so this returns each key only once.
        """
        seen: set[str] = set()
        keys: list[str] = []
        for spec in self._specs.values():
            if spec.domain is not None:
                continue
            if spec.batch_key is None:
                continue
            if spec.batch_key not in seen:
                seen.add(spec.batch_key)
                keys.append(spec.batch_key)
        return keys

    # ──── Dunder ────

    def __getitem__(self, name: str) -> FeatureSpec:
        return self._specs[name]


# ─────────────────────── DatasetSchema (I/O layer) ────────────────────


@dataclass(frozen=True, slots=True)
class SeqDomainConfig:
    """Configuration for a single sequence domain."""

    prefix: str
    ts_fid: int | None
    features: list[tuple[int, int]]  # [(fid, vocab_size), ...]

    @property
    def sideinfo_fids(self) -> list[int]:
        """Feature IDs excluding the timestamp feature."""
        return [fid for fid, _ in self.features if fid != self.ts_fid]

    @property
    def vocab_sizes(self) -> dict[int, int]:
        """Mapping fid -> vocab_size."""
        return dict(self.features)


class DatasetSchema:
    """Low-level parser for ``schema.json``.

    Maps the JSON structure to Python tuples and provides parquet column
    name generation. Only ``dataset.py`` and ``build_feature_schema``
    should use this directly — everything else goes through
    ``FeatureSchema``.

    Parameters
    ----------
    schema_path
        Path to the JSON schema file.
    """

    def __init__(self, schema_path: str | Path) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.user_cat: list[tuple[int, int, int]] = [
            (fid, vs, dim) for fid, vs, dim in raw["user_int"]
        ]
        self.item_cat: list[tuple[int, int, int]] = [
            (fid, vs, dim) for fid, vs, dim in raw["item_int"]
        ]
        self.user_cont: list[tuple[int, int]] = [(fid, dim) for fid, dim in raw["user_dense"]]
        self.item_cont: list[tuple[int, int]] = [
            (fid, dim) for fid, dim in raw.get("item_dense", [])
        ]

        self._seq_configs: dict[str, SeqDomainConfig] = {}
        for domain, cfg in raw["seq"].items():
            self._seq_configs[domain] = SeqDomainConfig(
                prefix=cfg["prefix"],
                ts_fid=cfg["ts_fid"],
                features=[(fid, vs) for fid, vs in cfg["features"]],
            )

    @property
    def seq_domains(self) -> list[str]:
        """Sorted list of sequence domain names."""
        return sorted(self._seq_configs.keys())

    def seq_config(self, domain: str) -> SeqDomainConfig:
        """Get config for a sequence domain."""
        return self._seq_configs[domain]

    @property
    def user_cat_columns(self) -> list[str]:
        """Parquet column names for user categorical features."""
        return [f"user_int_feats_{fid}" for fid, _, _ in self.user_cat]

    @property
    def user_cont_columns(self) -> list[str]:
        """Parquet column names for user continuous features."""
        return [f"user_dense_feats_{fid}" for fid, _ in self.user_cont]

    @property
    def item_cat_columns(self) -> list[str]:
        """Parquet column names for item categorical features."""
        return [f"item_int_feats_{fid}" for fid, _, _ in self.item_cat]

    @property
    def item_cont_columns(self) -> list[str]:
        """Parquet column names for item continuous features."""
        return [f"item_dense_feats_{fid}" for fid, _ in self.item_cont]

    def seq_columns(self, domain: str) -> list[str]:
        """Parquet column names for a sequence domain (all features incl. ts)."""
        cfg = self._seq_configs[domain]
        return [f"{cfg.prefix}_{fid}" for fid, _ in cfg.features]


# ─────────────────────── Factory ──────────────────────────────────────


def build_feature_schema(dataset_schema: DatasetSchema, split_f129: bool = True) -> FeatureSchema:
    """Construct a FeatureSchema from a DatasetSchema.

    Registers all features with their physical layout. Static categoricals are
    packed into shared tensors (user_cat, item_cat) with col_range. Static
    continuous features each get their own batch_key (one tensor per feature).
    Sequence features each get ``batch_key = name``.

    Parameters
    ----------
    dataset_schema
        Parsed schema.json.
    split_f129
        When True, splits item_cont_f129 (128-dim emb + 1 count + 1 dead pad)
        into two named features: ``item_cont_f129_emb`` and ``item_cont_f129_count``.
        Set False to register it as a single ``item_cont_f129`` feature.
    """
    schema = FeatureSchema()

    # -- Static user categoricals (one spec per fid, packed into "user_cat") --
    offset = 0
    for fid, vocab_size, dim in dataset_schema.user_cat:
        schema.register(
            FeatureSpec(
                name=f"user_cat_f{fid}",
                dtype=Dtype.CATEGORICAL,
                entity=Entity.USER,
                dim=dim,
                vocab_size=vocab_size,
                batch_key="user_cat",
                col_range=(offset, offset + dim),
                source_col=f"user_int_feats_{fid}",
            )
        )
        offset += dim

    # -- Static item categoricals (one spec per fid, packed into "item_cat") --
    offset = 0
    for fid, vocab_size, dim in dataset_schema.item_cat:
        schema.register(
            FeatureSpec(
                name=f"item_cat_f{fid}",
                dtype=Dtype.CATEGORICAL,
                entity=Entity.ITEM,
                dim=dim,
                vocab_size=vocab_size,
                batch_key="item_cat",
                col_range=(offset, offset + dim),
                source_col=f"item_int_feats_{fid}",
            )
        )
        offset += dim

    # -- Static user continuous (one tensor per feature) --
    for fid, dim in dataset_schema.user_cont:
        schema.register(
            FeatureSpec(
                name=f"user_cont_f{fid}",
                dtype=Dtype.NUMERICAL,
                entity=Entity.USER,
                dim=dim,
                batch_key=f"user_cont_f{fid}",
                source_col=f"user_dense_feats_{fid}",
            )
        )

    # -- Static item continuous (one tensor per feature) --
    for fid, dim in dataset_schema.item_cont:
        if fid == 129 and split_f129:
            # f129 is 128 embedding dims + 1 count scalar + 1 dead padding.
            # Split into two features; discard the dead trailing dim.
            schema.register(
                FeatureSpec(
                    name="item_cont_f129_emb",
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.ITEM,
                    dim=128,
                    batch_key="item_cont_f129_emb",
                    source_col="item_dense_feats_129",
                    source_offset=0,
                )
            )
            schema.register(
                FeatureSpec(
                    name="item_cont_f129_count",
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.ITEM,
                    dim=1,
                    batch_key="item_cont_f129_count",
                    source_col="item_dense_feats_129",
                    source_offset=128,
                )
            )
        else:
            schema.register(
                FeatureSpec(
                    name=f"item_cont_f{fid}",
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.ITEM,
                    dim=dim,
                    batch_key=f"item_cont_f{fid}",
                    source_col=f"item_dense_feats_{fid}",
                )
            )

    # -- Sequence features (per domain, sideinfo + infrastructure) --
    for domain in dataset_schema.seq_domains:
        cfg = dataset_schema.seq_config(domain)
        sideinfo = [(fid, vs) for fid, vs in cfg.features if fid != cfg.ts_fid]

        for fid, vocab_size in sideinfo:
            feat_name = f"{domain}_f{fid}"
            schema.register(
                FeatureSpec(
                    name=feat_name,
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=vocab_size,
                    domain=domain,
                    batch_key=feat_name,
                    source_col=f"{cfg.prefix}_{fid}",
                )
            )

        # Per-sample sequence length — static scalar, no domain field
        schema.register(
            FeatureSpec(
                name=f"{domain}_len",
                dtype=Dtype.NUMERICAL,
                entity=Entity.USER,
                dim=1,
                source=Source.METADATA,
                batch_key=f"{domain}_len",
            )
        )

        # Per-timestep timestamps — SEQ scope, registered for collation and masking
        if cfg.ts_fid is not None:
            schema.register(
                FeatureSpec(
                    name=f"{domain}_ts",
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    domain=domain,
                    source=Source.METADATA,
                    batch_key=f"{domain}_ts",
                )
            )

    return schema
