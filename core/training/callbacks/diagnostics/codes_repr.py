"""Representation-probe diagnostic: bucketed hidden-state MI and relevance cosine.

Each epoch, this diagnostic runs a bounded validation gather, captures configured
representation probes, computes KMeans MI/AMI and optional user-item cosine
inside the diagnostic, then emits compact DIAG rows. It never emits full
representation vectors. Optional projection emits 2-D coordinates only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.data.loader import batch_to_device, clone_loader
from core.data.schema import FeatureSchema
from core.training.callbacks.diagnostics.base import DiagBase, _try_numeric
from core.training.callbacks.diagnostics.codec import decode_array, encode_array
from core.training.callbacks.diagnostics.context import EpochContext, StepContext

LOG = logging.getLogger(__name__)

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _tag(name: str) -> str:
    """Flatten a dotted module path into a payload-safe key."""
    return name.replace(".", "_")


def _resolve_modules(model: torch.nn.Module, names: list[str]) -> dict[str, torch.nn.Module]:
    """Map probe names to submodules, warning on any that do not resolve."""
    table = dict(model.named_modules())
    resolved: dict[str, torch.nn.Module] = {}
    for name in names:
        mod = table.get(name)
        if mod is None:
            LOG.warning("REPR_PROBE: probe %r not found in model; skipping", name)
            continue
        resolved[name] = mod
    return resolved


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def _kv_payload(prefix: str, values: dict[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={_format_float(value)}")
        else:
            parts.append(f"{key}={value}")
    return prefix + ",".join(parts)


def _parse_kv_payload(payload: str, prefix: str) -> dict[str, Any]:
    text = payload[len(prefix) :]
    row: dict[str, Any] = {}
    for kv in text.split(","):
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        row[key] = _try_numeric(value)
    return row


def _normalise_slices(raw: dict[str, list[int]] = None) -> dict[str, tuple[int, int]]:
    """Normalize feature slice config into `(start, end)` pairs."""
    if raw is None:
        return {}
    return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in raw.items()}


def _apply_slice(
    name: str, tensor: torch.Tensor, slices: dict[str, tuple[int, int]]
) -> torch.Tensor:
    """Apply an optional feature slice to one feature tensor."""
    if name not in slices:
        return tensor
    start, end = slices[name]
    if start < 0 or end <= start or end > tensor.shape[-1]:
        raise ValueError(
            f"invalid slice for {name!r}: [{start}, {end}) with dim {tensor.shape[-1]}"
        )
    return tensor[..., start:end]


class RepresentationProbeCode(DiagBase):
    """Bucketed hidden-state MI + user/item relevance cosine over validation.

    ``probes`` lists module names to capture. ``pairings`` lists user/item
    feature groups whose continuous vectors are compared by cosine. ``buckets``
    defines explicit row partitions used for both MI and cosine summaries.
    """

    code = "REPR_PROBE"
    config_key = "repr_probe"
    emit = frozenset({"epoch"})
    accumulate = frozenset()
    init_params: ClassVar[tuple[str, ...]] = (
        "schema",
        "val_loader",
        "device",
        "amp_dtype",
    )

    def __init__(
        self,
        *,
        schema: FeatureSchema = None,
        val_loader: DataLoader = None,
        device: str = None,
        amp_dtype: Any = None,
        probes: list[str] = None,
        pairings: list[dict[str, Any]] = None,
        max_samples: int = 10000,
        batch_size: int = None,
        cluster_counts: list[int] = None,
        kmeans_seeds: list[int] = None,
        standardize: bool = True,
        n_init: int = 3,
        minibatch_size: int = 4096,
        min_bucket_n: int = 1,
        min_bucket_pos: int = 0,
        buckets: dict[str, Any] = None,
        projection: dict[str, Any] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("writer", "accumulate_freq", "warmup_steps")
            }
        )
        self._schema = schema
        self._val_loader = val_loader
        self._device = device
        self._amp_dtype = _AMP_DTYPES.get(amp_dtype) if isinstance(amp_dtype, str) else amp_dtype
        self._probes = probes or []
        self._pairings = pairings or []
        self._max_samples = int(max_samples)
        self._batch_size = batch_size
        self._cluster_counts = cluster_counts or [4, 16, 48, 96]
        self._kmeans_seeds = kmeans_seeds or [seed]
        self._standardize = standardize
        self._n_init = int(n_init)
        self._minibatch_size = int(minibatch_size)
        self._min_bucket_n = int(min_bucket_n)
        self._min_bucket_pos = int(min_bucket_pos)
        self._buckets_cfg = buckets or {}
        self._projection_cfg = projection or {"enabled": False}
        self._seed = seed

    # ── Gather ────────────────────────────────────────────────────────────────

    def _gather(self, model: torch.nn.Module) -> tuple[dict, dict, dict]:
        """Run one val pass; return (per_sample, reps, pairings) aligned by row."""
        device = torch.device(self._device) if self._device else torch.device("cpu")
        loader = (
            clone_loader(self._val_loader, batch_size=self._batch_size)
            if self._batch_size
            else self._val_loader
        )
        modules = _resolve_modules(model, self._probes)

        captured: dict[str, torch.Tensor] = {}
        handles = []
        for name, mod in modules.items():

            def _hook(_m: Any, _a: Any, out: Any, _n: str = name) -> None:
                t = out[0] if isinstance(out, tuple) else out
                captured[_n] = t.detach()

            handles.append(mod.register_forward_hook(_hook))

        reps: dict[str, list[torch.Tensor]] = {n: [] for n in modules}
        labels: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        pair_vecs: dict[str, dict[str, list[torch.Tensor]]] = {
            p["name"]: {"user": [], "item": []} for p in self._pairings
        }
        dropped: set[str] = set()
        meta: dict[str, list[torch.Tensor]] = {}
        total = 0

        try:
            with torch.no_grad():
                for batch in loader:
                    gpu = batch_to_device(batch, device)
                    captured.clear()
                    if self._amp_dtype is not None:
                        with torch.autocast(device_type=device.type, dtype=self._amp_dtype):
                            out = model(gpu)
                    else:
                        out = model(gpu)
                    logit = (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
                    label = gpu["label"].reshape(-1).float()
                    for n in modules:
                        if n in captured:
                            reps[n].append(captured[n].reshape(label.shape[0], -1).float().cpu())
                    logits.append(logit.cpu())
                    labels.append(label.cpu())
                    for key, val in gpu.items():
                        if key == "label" or not torch.is_tensor(val):
                            continue
                        if val.ndim == 1 and val.shape[0] == label.shape[0]:
                            meta.setdefault(key, []).append(val.detach().cpu())
                    for p in self._pairings:
                        if p["name"] in dropped:
                            continue
                        u = self._extract_feature_side(
                            gpu, p["user_features"], p.get("user_slices")
                        )
                        it = self._extract_feature_side(
                            gpu, p["item_features"], p.get("item_slices")
                        )
                        if u is None or it is None:
                            dropped.add(p["name"])
                            LOG.warning(
                                "REPR_PROBE pairing %r: a side matched no features; dropping",
                                p["name"],
                            )
                            continue
                        pair_vecs[p["name"]]["user"].append(
                            u.reshape(label.shape[0], -1).float().cpu()
                        )
                        pair_vecs[p["name"]]["item"].append(
                            it.reshape(label.shape[0], -1).float().cpu()
                        )
                    total += label.shape[0]
                    if total >= self._max_samples:
                        break
        finally:
            for h in handles:
                h.remove()

        if not labels:
            return {}, {}, {}

        label_np = torch.cat(labels).numpy()
        logit_np = torch.cat(logits).numpy()
        prob_np = 1.0 / (1.0 + np.exp(-logit_np))
        p = np.clip(prob_np, 1e-7, 1.0 - 1e-7)
        loss_np = -(label_np * np.log(p) + (1.0 - label_np) * np.log(1.0 - p))
        per_sample = {"label": label_np, "logit": logit_np, "prob": prob_np, "loss": loss_np}
        for key, vals in meta.items():
            arr = torch.cat(vals).numpy()
            if arr.shape[0] == label_np.shape[0]:
                per_sample[key] = arr

        reps_np = {n: torch.cat(v).numpy() for n, v in reps.items() if v}
        pairs_np = {
            name: (torch.cat(ui["user"]).numpy(), torch.cat(ui["item"]).numpy())
            for name, ui in pair_vecs.items()
            if name not in dropped and ui["user"]
        }
        return per_sample, reps_np, pairs_np

    def _extract_feature_side(
        self,
        batch: dict[str, Any],
        expr: str,
        raw_slices: dict[str, list[int]] = None,
    ) -> torch.Tensor | None:
        """Extract a schema-selected continuous vector side with optional per-feature slices."""
        specs = self._schema.query(expr)
        if not specs:
            return None
        raw = self._schema.extract(batch, expr=expr, cat=True)
        if raw is None:
            return None
        slices = _normalise_slices(raw_slices)
        chunks = torch.split(raw, [spec.dim for spec in specs], dim=-1)
        parts = [
            _apply_slice(spec.name, chunk, slices)
            for spec, chunk in zip(specs, chunks, strict=True)
        ]
        return torch.cat(parts, dim=-1)

    # ── Buckets ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quantile_masks(
        values: np.ndarray, cfg: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, str]]:
        quantiles = list(cfg["quantiles"])
        labels = list(cfg["labels"])
        if len(labels) != len(quantiles) + 1:
            raise ValueError("bucket labels must have len(quantiles) + 1 entries")
        edges = np.quantile(values, quantiles)
        ids = np.digitize(values, edges, right=True).astype(np.int64)
        masks = {label: ids == idx for idx, label in enumerate(labels)}
        return masks, ids, dict(enumerate(labels))

    def _seq_len_keys(self, per_sample: dict[str, np.ndarray]) -> list[str]:
        seq_cfg = self._buckets_cfg.get("seq_len") or {}
        specs = seq_cfg.get("specs") or {}
        if specs:
            return [key for key in specs if key in per_sample]
        return sorted(
            key
            for key in per_sample
            if key.endswith("_len") and "_raw_len" not in key and "_recency" not in key
        )

    def _bucket_masks(
        self, per_sample: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, dict[int, str]],
        dict[str, str],
    ]:
        n = per_sample["label"].shape[0]
        masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
        derived: dict[str, np.ndarray] = {}
        categories: dict[str, dict[int, str]] = {}
        sources: dict[str, str] = {"all": "all"}

        loss_cfg = self._buckets_cfg.get("loss")
        if loss_cfg:
            loss_masks, loss_ids, loss_cats = self._quantile_masks(per_sample["loss"], loss_cfg)
            masks.update(loss_masks)
            sources.update(dict.fromkeys(loss_masks, "loss"))
            derived["loss_bucket"] = loss_ids.astype(np.float32)
            categories["loss_bucket"] = loss_cats

        score_cfg = self._buckets_cfg.get("score_band")
        if score_cfg:
            labels = per_sample["label"] > 0.5
            prob = per_sample["prob"]
            pos_desc = score_cfg["positive_order"] == "desc"
            neg_asc = score_cfg["negative_order"] == "asc"
            pos_difficulty = -prob if pos_desc else prob
            neg_difficulty = prob if neg_asc else -prob
            difficulty = np.where(labels, pos_difficulty, neg_difficulty)
            score_masks, score_ids, score_cats = self._quantile_masks(difficulty, score_cfg)
            masks.update(score_masks)
            sources.update(dict.fromkeys(score_masks, "score_band"))
            derived["score_bucket"] = score_ids.astype(np.float32)
            categories["score_bucket"] = score_cats

        seq_cfg = self._buckets_cfg.get("seq_len") or {}
        for key, edges in (seq_cfg.get("specs") or {}).items():
            if key not in per_sample:
                continue
            values = per_sample[key]
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if i == 0:
                    mask = (values >= lo) & (values <= hi)
                else:
                    mask = (values > lo) & (values <= hi)
                bucket = f"{key}_{lo}_to_{hi}"
                masks[bucket] = mask
                sources[bucket] = "seq_len"
            if np.any(values > edges[-1]):
                bucket = f"{key}_gt_{edges[-1]}"
                masks[bucket] = values > edges[-1]
                sources[bucket] = "seq_len"

        len_keys = self._seq_len_keys(per_sample)
        if self._buckets_cfg.get("active_domain_count") and len_keys:
            counts = sum((per_sample[key] > 0).astype(np.int64) for key in len_keys)
            derived["active_domain_count"] = counts.astype(np.float32)
            for count in sorted(np.unique(counts)):
                bucket = f"active_domains_{int(count)}"
                masks[bucket] = counts == count
                sources[bucket] = "active_domain_count"

        if self._buckets_cfg.get("primary_domain") and len_keys:
            lengths = np.stack([per_sample[key] for key in len_keys], axis=1)
            primary = lengths.argmax(axis=1)
            none = lengths.max(axis=1) <= 0
            primary[none] = -1
            category_map = {
                -1: "none",
                **{idx: key.removesuffix("_len") for idx, key in enumerate(len_keys)},
            }
            derived["primary_domain"] = primary.astype(np.float32)
            categories["primary_domain"] = category_map
            for idx, label in category_map.items():
                bucket = f"primary_domain_{label}"
                masks[bucket] = primary == idx
                sources[bucket] = "primary_domain"

        return masks, derived, categories, sources

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _bucket_is_valid(self, labels: np.ndarray, mask: np.ndarray) -> tuple[bool, int, int]:
        n = int(mask.sum())
        pos = int((labels[mask] > 0.5).sum()) if n else 0
        valid = n >= self._min_bucket_n and pos >= self._min_bucket_pos
        return valid, n, pos

    def _bucket_skip_reason(self, n: int, pos: int) -> str:
        """Explain why a bucket was not used for MI/cosine summaries."""
        if n < self._min_bucket_n:
            return "min_bucket_n"
        if pos < self._min_bucket_pos:
            return "min_bucket_pos"
        return ""

    def _bucket_outcome_rows(
        self,
        per_sample: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        sources: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Summarize denominator and outcome metrics for every attempted bucket."""
        from sklearn.metrics import roc_auc_score

        auc_sources = {"seq_len", "active_domain_count", "primary_domain"}
        labels = per_sample["label"]
        probs = per_sample["prob"]
        losses = per_sample["loss"]
        rows: list[dict[str, Any]] = []
        for bucket, mask in sorted(masks.items()):
            source = sources[bucket]
            valid, n, pos = self._bucket_is_valid(labels, mask)
            row: dict[str, Any] = {
                "bucket": bucket,
                "source": source,
                "n": n,
                "pos": pos,
                "pos_rate": pos / n if n else 0.0,
                "valid": int(valid),
                "skip": "" if valid else self._bucket_skip_reason(n, pos),
            }
            if n > 0:
                bucket_labels = labels[mask]
                bucket_probs = probs[mask]
                row["logloss"] = float(losses[mask].mean())
                row["mean_pred"] = float(bucket_probs.mean())
                row["actual_rate"] = float(bucket_labels.mean())
                if source in auc_sources and len(np.unique(bucket_labels)) == 2:
                    row["auc"] = float(roc_auc_score(bucket_labels, bucket_probs))
            rows.append(row)
        return rows

    def _mutual_information_rows(
        self,
        reps_np: dict[str, np.ndarray],
        labels: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score

        rows: list[dict[str, Any]] = []
        for probe_name, x_all in reps_np.items():
            x_all = np.nan_to_num(x_all, copy=False)
            for bucket, mask in masks.items():
                valid, n, pos = self._bucket_is_valid(labels, mask)
                if not valid:
                    continue
                x = x_all[mask]
                y = labels[mask].astype(int)
                if self._standardize:
                    std = x.std(axis=0, keepdims=True)
                    x = (x - x.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)
                for k in self._cluster_counts:
                    if n < k:
                        continue
                    mi_vals = []
                    ami_vals = []
                    for seed in self._kmeans_seeds:
                        clusters = MiniBatchKMeans(
                            n_clusters=k,
                            n_init=self._n_init,
                            batch_size=self._minibatch_size,
                            random_state=seed,
                        ).fit_predict(x)
                        mi_vals.append(float(mutual_info_score(y, clusters)))
                        ami_vals.append(float(adjusted_mutual_info_score(y, clusters)))
                    rows.append(
                        {
                            "probe": _tag(probe_name),
                            "bucket": bucket,
                            "k": int(k),
                            "n": n,
                            "pos": pos,
                            "pos_rate": pos / n,
                            "mi_mean": float(np.mean(mi_vals)),
                            "ami_mean": float(np.mean(ami_vals)),
                            "mi_std": float(np.std(mi_vals)),
                            "ami_std": float(np.std(ami_vals)),
                        }
                    )
        return rows

    def _relevance_cosine_rows(
        self,
        pairs_np: dict[str, tuple[np.ndarray, np.ndarray]],
        labels: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        pos_mask_all = labels > 0.5
        for name, (u, it) in pairs_np.items():
            if u.shape[-1] != it.shape[-1]:
                LOG.warning(
                    "REPR_PROBE pairing %r: dim mismatch (user=%d, item=%d); skipping cosine",
                    name,
                    u.shape[-1],
                    it.shape[-1],
                )
                continue
            u = np.nan_to_num(u, copy=False)
            it = np.nan_to_num(it, copy=False)
            un = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
            itn = it / (np.linalg.norm(it, axis=-1, keepdims=True) + 1e-8)
            cos = (un * itn).sum(axis=-1)
            for bucket, mask in masks.items():
                valid, n, pos = self._bucket_is_valid(labels, mask)
                if not valid:
                    continue
                pos_values = cos[mask & pos_mask_all]
                neg_values = cos[mask & ~pos_mask_all]
                if pos_values.size == 0 or neg_values.size == 0:
                    continue
                rows.append(
                    {
                        "name": name,
                        "bucket": bucket,
                        "n": n,
                        "pos": pos,
                        "pos_mean": float(pos_values.mean()),
                        "neg_mean": float(neg_values.mean()),
                        "gap": float(pos_values.mean() - neg_values.mean()),
                    }
                )
        return rows

    # ── Projection ────────────────────────────────────────────────────────────

    @staticmethod
    def _project(x: np.ndarray, method: str, seed: int) -> np.ndarray:
        from sklearn.decomposition import PCA

        x = np.nan_to_num(x)
        std = x.std(axis=0, keepdims=True)
        x = (x - x.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)
        if method == "pca":
            return PCA(n_components=2, random_state=seed).fit_transform(x)
        if method == "tsne":
            from sklearn.manifold import TSNE

            if x.shape[1] > 50:
                x = PCA(n_components=50, random_state=seed).fit_transform(x)
            perplexity = float(min(30, max(5, x.shape[0] // 4)))
            return TSNE(
                n_components=2, random_state=seed, perplexity=perplexity, init="pca"
            ).fit_transform(x)
        raise ValueError(f"unknown projection method {method!r}")

    def _projection_payloads(
        self,
        per_sample: dict[str, np.ndarray],
        reps_np: dict[str, np.ndarray],
        derived: dict[str, np.ndarray],
        categories: dict[str, dict[int, str]],
    ) -> list[str]:
        cfg = self._projection_cfg
        if not cfg["enabled"]:
            return []
        n = per_sample["label"].shape[0]
        max_points = cfg["max_points"]
        method = cfg["method"]
        probes = cfg["probes"]
        color_by = cfg["color_by"]
        rng = np.random.default_rng(self._seed)
        idx = rng.choice(n, max_points, replace=False) if n > max_points else np.arange(n)

        payloads = [
            "proj_manifest:"
            + json.dumps(
                {
                    "method": method,
                    "n": len(idx),
                    "probes": [_tag(name) for name in probes if name in reps_np],
                    "meta": list(color_by),
                    "categories": {
                        k: {str(ck): cv for ck, cv in v.items()} for k, v in categories.items()
                    },
                },
                separators=(",", ":"),
            )
        ]
        for name in probes:
            if name not in reps_np:
                continue
            coords = self._project(reps_np[name][idx], method, self._seed)
            payloads.append(f"proj_coords:{_tag(name)}:" + encode_array(coords, "float16"))
        for field in color_by:
            if field in per_sample:
                arr = per_sample[field][idx]
            elif field in derived:
                arr = derived[field][idx]
            else:
                continue
            payloads.append(f"proj_meta:{field}:" + encode_array(arr, "float16"))
        return payloads

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Run the gather pass and emit bucketed MI/cosine/projection summaries."""
        if phase != "epoch":
            return []
        if not self._probes and not self._pairings:
            return []
        if self._schema is None or self._val_loader is None:
            LOG.warning("REPR_PROBE: schema or val_loader missing; skipping")
            return []
        model = ctx.get("model") if isinstance(ctx, dict) else getattr(ctx, "model", None)
        if model is None:
            LOG.warning("REPR_PROBE: no model in context; skipping")
            return []

        was_training = model.training
        model.eval()
        try:
            per_sample, reps_np, pairs_np = self._gather(model)
        finally:
            if was_training:
                model.train()

        if not per_sample:
            return []

        masks, derived, categories, sources = self._bucket_masks(per_sample)
        mi_rows = self._mutual_information_rows(reps_np, per_sample["label"], masks)
        payloads = [
            _kv_payload("bucket:", row)
            for row in self._bucket_outcome_rows(per_sample, masks, sources)
        ]
        payloads += [_kv_payload("mi:", row) for row in mi_rows]
        payloads += [
            _kv_payload("cos:", row)
            for row in self._relevance_cosine_rows(pairs_np, per_sample["label"], masks)
        ]
        if self.writer:
            for row in (row for row in mi_rows if row["bucket"] == "all"):
                self.writer.add_scalar(
                    f"ReprProbe/{row['probe']}/k{row['k']}/ami",
                    row["ami_mean"],
                    self._tb_step(phase, ctx)[1],
                )
        payloads += self._projection_payloads(per_sample, reps_np, derived, categories)
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse MI rows, cosine rows, and compact projection coordinates."""
        if payload.startswith("bucket:"):
            accum.setdefault("buckets", []).append(_parse_kv_payload(payload, "bucket:"))
        elif payload.startswith("mi:"):
            accum.setdefault("mi", []).append(_parse_kv_payload(payload, "mi:"))
        elif payload.startswith("cos:"):
            accum.setdefault("cosine", []).append(_parse_kv_payload(payload, "cos:"))
        elif payload.startswith("proj_manifest:"):
            accum.setdefault("projection", {})["manifest"] = json.loads(
                payload[len("proj_manifest:") :]
            )
        elif payload.startswith("proj_coords:"):
            _, name, blob = payload.split(":", 2)
            accum.setdefault("projection", {}).setdefault("coords", {})[name] = decode_array(blob)
        elif payload.startswith("proj_meta:"):
            _, field, blob = payload.split(":", 2)
            accum.setdefault("projection", {}).setdefault("meta", {})[field] = decode_array(blob)
