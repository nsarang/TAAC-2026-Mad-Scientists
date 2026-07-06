"""SAGE feature-group importance diagnostic code (v2 data pipeline).

Estimates Shapley-based importance for feature groups by walking random
permutations and measuring the marginal loss/AUC delta when each group
is added. Operates directly on batch dicts + FeatureSchema.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from core.data.loader import batch_to_device, clone_loader
from core.data.masking import mask_batch
from core.data.schema import FeatureSchema
from core.training.callbacks.diagnostics.base import DiagBase
from core.training.callbacks.diagnostics.context import EpochContext, StepContext

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------


class ImportanceTracker:
    """Running mean + variance via Welford's algorithm."""

    def __init__(self, num_groups: int) -> None:
        self.mean = np.zeros(num_groups)
        self.sum_squares = np.zeros(num_groups)
        self.n = 0

    def update(self, scores: np.ndarray) -> None:
        """Incorporate one new observation."""
        self.n += 1
        diff = scores - self.mean
        self.mean += diff / self.n
        diff2 = scores - self.mean
        self.sum_squares += diff * diff2

    @property
    def values(self) -> np.ndarray:
        """Current running mean per group."""
        return self.mean

    @property
    def std(self) -> np.ndarray:
        """Standard error of the mean estimate."""
        if self.n < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.sum_squares / (self.n * (self.n - 1)))


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------


def resolve_groups(
    schema: FeatureSchema,
    groups_cfg: dict[str, Any],
    seq_domains: list[str],
) -> list[dict[str, Any]]:
    """Resolve group config to concrete group definitions.

    Parameters
    ----------
    schema
        FeatureSchema for resolving groups.
    groups_cfg
        Group definitions. Keys are group names, values are DSL expression
        strings resolved via ``schema.query()``.
    seq_domains
        Sequence domain names (auto-added if not in groups_cfg).

    Returns
    -------
    List of group defs: ``{"name": str, "specs": list[FeatureSpec]}``
    """
    defs = []
    for name, cfg in groups_cfg.items():
        specs = schema.query(cfg)
        defs.append({"name": name, "specs": specs})

    covered_domains = {s.domain for d in defs for s in d["specs"] if s.domain}
    for domain in seq_domains:
        if domain not in covered_domains:
            specs = schema.query(f"domain = '{domain}' and source != 'metadata'")
            defs.append({"name": domain, "specs": specs})

    return defs


# ---------------------------------------------------------------------------
# Masking and forward helpers
# ---------------------------------------------------------------------------


def _mask_groups(
    batch: dict[str, Any],
    group_defs: list[dict[str, Any]],
    active: np.ndarray,
    schema: FeatureSchema,
) -> dict[str, Any]:
    """Apply group masking based on active flags."""
    out = batch
    for i, gdef in enumerate(group_defs):
        if active[i]:
            continue
        out = mask_batch(out, gdef["specs"], schema)
    return out


def _compute_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    labels: torch.Tensor,
    group_defs: list[dict[str, Any]],
    active: np.ndarray,
    schema: FeatureSchema,
    amp_dtype: torch.dtype = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass with masked input. Returns (per_sample_loss, per_sample_pred)."""
    masked = _mask_groups(batch, group_defs, active, schema)
    device = labels.device
    if amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out = model(masked)
    else:
        out = model(masked)
    logits = (out[0] if isinstance(out, tuple) else out).reshape(-1)
    logits = logits.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    preds = torch.sigmoid(logits)
    return loss.cpu().numpy(), preds.cpu().numpy()


def _safe_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, preds))


# ---------------------------------------------------------------------------
# Permutation sampling
# ---------------------------------------------------------------------------


def _sage_permutation(
    model: torch.nn.Module,
    batches: list[tuple[dict[str, Any], torch.Tensor]],
    group_defs: list[dict[str, Any]],
    schema: FeatureSchema,
    rng: np.random.Generator,
    device: torch.device,
    amp_dtype: torch.dtype = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One permutation over all batches."""
    num_groups = len(group_defs)
    perm = rng.permutation(num_groups)
    active = np.zeros(num_groups, dtype=bool)

    all_prev_loss = []
    all_prev_preds = []
    all_labels = []
    for batch, labels in batches:
        gpu_batch = batch_to_device(batch, device)
        gpu_labels = labels.to(device, non_blocking=device.type == "cuda")
        loss, preds = _compute_loss(
            model, gpu_batch, gpu_labels, group_defs, active, schema, amp_dtype
        )
        all_prev_loss.append(loss)
        all_prev_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    prev_loss = np.concatenate(all_prev_loss)
    prev_preds = np.concatenate(all_prev_preds)
    labels_np = np.concatenate(all_labels)
    prev_auc = _safe_auc(labels_np, prev_preds)

    loss_scores = np.zeros(num_groups)
    auc_scores = np.zeros(num_groups)

    for idx in perm:
        active[idx] = True
        curr_loss_parts = []
        curr_preds_parts = []
        for batch, labels in batches:
            gpu_batch = batch_to_device(batch, device)
            gpu_labels = labels.to(device, non_blocking=device.type == "cuda")
            loss, preds = _compute_loss(
                model, gpu_batch, gpu_labels, group_defs, active, schema, amp_dtype
            )
            curr_loss_parts.append(loss)
            curr_preds_parts.append(preds)

        curr_loss = np.concatenate(curr_loss_parts)
        curr_preds = np.concatenate(curr_preds_parts)
        curr_auc = _safe_auc(labels_np, curr_preds)

        loss_scores[idx] = float(np.mean(prev_loss - curr_loss))
        auc_scores[idx] = curr_auc - prev_auc

        prev_loss = curr_loss
        prev_preds = curr_preds
        prev_auc = curr_auc

    return loss_scores, auc_scores


def estimate_sage(
    model: torch.nn.Module,
    batches: list[tuple[dict[str, Any], torch.Tensor]],
    group_defs: list[dict[str, Any]],
    schema: FeatureSchema,
    n_permutations: int = 100,
    convergence_thresh: float = 0.025,
    seed: int = 42,
    amp_dtype: torch.dtype = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate SAGE values via permutation sampling.

    Returns
    -------
    tuple
        ``(loss_values, loss_std, auc_values, auc_std, n_permutations_run)``.
    """
    num_groups = len(group_defs)
    rng = np.random.default_rng(seed)
    loss_tracker = ImportanceTracker(num_groups)
    auc_tracker = ImportanceTracker(num_groups)
    params = list(model.parameters())
    device = params[0].device if params else batches[0][1].device

    for i in range(n_permutations):
        loss_scores, auc_scores = _sage_permutation(
            model, batches, group_defs, schema, rng, device, amp_dtype
        )
        loss_tracker.update(loss_scores)
        auc_tracker.update(auc_scores)

        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            max_std = max(np.max(loss_tracker.std), np.max(auc_tracker.std))
            loss_gap = max(loss_tracker.values.max() - loss_tracker.values.min(), 1e-12)
            auc_gap = max(auc_tracker.values.max() - auc_tracker.values.min(), 1e-12)
            ratio = max(max_std / loss_gap, max_std / auc_gap)
            LOG.info(
                "  perm %d/%d: ratio=%.4f (thresh=%.4f)",
                i + 1,
                n_permutations,
                ratio,
                convergence_thresh,
            )
            if ratio < convergence_thresh:
                LOG.info("  Converged after %d permutations", i + 1)
                break

    return (
        loss_tracker.values,
        loss_tracker.std,
        auc_tracker.values,
        auc_tracker.std,
        loss_tracker.n,
    )


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------


def collect_batches(
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    store_on_gpu: bool = False,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    """Collect batches from a DataLoader up to max_samples.

    Parameters
    ----------
    loader
        Source DataLoader to iterate.
    device
        Target device for inference.
    max_samples
        Stop after collecting this many samples.
    store_on_gpu
        When False (default), batches are stored on CPU and transferred to
        `device` per-forward during permutation. When True, batches are moved
        to `device` at collection time and kept resident.
    """
    storage_device = device if store_on_gpu else torch.device("cpu")
    batches = []
    total = 0
    for batch in loader:
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(storage_device, non_blocking=storage_device.type == "cuda")
            else:
                moved[k] = v
        labels = moved["label"].float()
        batches.append((moved, labels))
        total += labels.shape[0]
        if total >= max_samples:
            break
    LOG.info("  Collected %d samples in %d batches (on %s)", total, len(batches), storage_device)
    return batches


# ---------------------------------------------------------------------------
# Diagnostic code
# ---------------------------------------------------------------------------


class SageCode(DiagBase):
    """SAGE feature-group importance at training end.

    Runs permutation-based Shapley estimation on the validation set using
    v2 batch dicts and FeatureSchema. Emits during the ``done`` phase so
    results appear once at the end of the DIAG log.

    All hyperparameters come from ``diagnostics.code_config.sage`` in the
    YAML config.
    """

    code = "SAGE"
    config_key = "sage"
    emit = frozenset({"done"})
    accumulate = frozenset()
    init_params: ClassVar[tuple[str, ...]] = ("schema", "train_loader", "val_loader", "device")

    def __init__(
        self,
        *,
        schema: FeatureSchema = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        device: str = None,
        amp_dtype: torch.dtype = None,
        max_samples: int = 4096,
        n_permutations: int = 100,
        convergence_thresh: float = 0.025,
        seed: int = 42,
        groups: dict[str, dict[str, Any]] = None,
        batch_size: int,
        eval_train: bool = True,
        store_on_gpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("writer", "accumulate_freq", "warmup_steps")
            }
        )
        self._schema = schema
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device
        if isinstance(amp_dtype, str):
            self._amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype)
        else:
            self._amp_dtype = amp_dtype
        self._max_samples = max_samples
        self._n_permutations = n_permutations
        self._convergence_thresh = convergence_thresh
        self._seed = seed
        self._batch_size = batch_size
        self._eval_train = eval_train
        self._store_on_gpu = store_on_gpu
        self._groups_cfg = groups or {}

        # Resolved at first collect (lazy)
        self._group_defs: list[dict[str, Any]] = None
        self._group_names: list[str] = None
        self._train_batches: list[tuple[dict[str, Any], torch.Tensor]] = None
        self._val_batches: list[tuple[dict[str, Any], torch.Tensor]] = None

    def _ensure_ready(self) -> bool:
        """Resolve groups and collect batches on first use.

        Returns False if required dependencies are missing.
        """
        if self._schema is None or self._val_loader is None:
            LOG.warning("SAGE: schema or val_loader not provided, skipping")
            return False

        if self._group_defs is None:
            seq_domains = sorted(
                {
                    s.domain
                    for s in self._schema.query("scope = 'seq' and source != 'metadata'")
                    if s.domain
                }
            )
            self._group_defs = resolve_groups(self._schema, self._groups_cfg, seq_domains)
            self._group_names = [d["name"] for d in self._group_defs]

        device = torch.device(self._device) if self._device else torch.device("cpu")

        if self._train_batches is None and self._eval_train:
            loader = clone_loader(self._train_loader, batch_size=self._batch_size)
            LOG.info("SAGE: collecting training batches...")
            self._train_batches = collect_batches(
                loader, device, self._max_samples, store_on_gpu=self._store_on_gpu
            )

        if self._val_batches is None:
            loader = clone_loader(self._val_loader, batch_size=self._batch_size)
            LOG.info("SAGE: collecting validation batches...")
            self._val_batches = collect_batches(
                loader, device, self._max_samples, store_on_gpu=self._store_on_gpu
            )

        return True

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Run permutation importance at training end."""
        if phase != "done":
            return []

        if not self._ensure_ready():
            return []

        model = ctx.get("model") if isinstance(ctx, dict) else getattr(ctx, "model", None)
        if model is None:
            LOG.warning("SAGE: no model in context, skipping")
            return []

        was_training = model.training
        model.eval()

        parts = []
        with torch.no_grad():
            if self._train_batches:
                LOG.info("SAGE: running permutations on training set...")
                tr_loss, tr_loss_std, tr_auc, tr_auc_std, tr_n = estimate_sage(
                    model,
                    self._train_batches,
                    self._group_defs,
                    self._schema,
                    n_permutations=self._n_permutations,
                    convergence_thresh=self._convergence_thresh,
                    seed=self._seed,
                    amp_dtype=self._amp_dtype,
                )
                parts.append(f"train_n_perms={tr_n}")
                for i, name in enumerate(self._group_names):
                    parts.append(f"train_{name}_loss={tr_loss[i]:+.6f}")
                    parts.append(f"train_{name}_loss_std={tr_loss_std[i]:.6f}")
                    parts.append(f"train_{name}_auc={tr_auc[i]:+.6f}")
                    parts.append(f"train_{name}_auc_std={tr_auc_std[i]:.6f}")

            LOG.info("SAGE: running permutations on validation set...")
            val_loss, val_loss_std, val_auc, val_auc_std, val_n = estimate_sage(
                model,
                self._val_batches,
                self._group_defs,
                self._schema,
                n_permutations=self._n_permutations,
                convergence_thresh=self._convergence_thresh,
                seed=self._seed,
                amp_dtype=self._amp_dtype,
            )
            parts.append(f"val_n_perms={val_n}")
            for i, name in enumerate(self._group_names):
                parts.append(f"val_{name}_loss={val_loss[i]:+.6f}")
                parts.append(f"val_{name}_loss_std={val_loss_std[i]:.6f}")
                parts.append(f"val_{name}_auc={val_auc[i]:+.6f}")
                parts.append(f"val_{name}_auc_std={val_auc_std[i]:.6f}")

        if was_training:
            model.train()

        return [",".join(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse SAGE payload into structured dict."""
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    entry[k] = int(v)
                except ValueError:
                    try:
                        entry[k] = float(v)
                    except ValueError:
                        entry[k] = v
        accum.setdefault("done", {}).update(entry)
