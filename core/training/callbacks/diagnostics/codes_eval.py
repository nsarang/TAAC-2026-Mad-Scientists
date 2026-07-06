"""Evaluation codes: METRICS, PRED, LOGIT_DIST, LOSS_CONC, SAGE, and scoring helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext
from core.training.utils import ReservoirSampler


def compute_per_domain_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    domain_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """AUC restricted to samples within each domain.

    Domains with fewer than two label classes are skipped.
    """
    results: dict[str, float] = {}
    for domain, mask in domain_masks.items():
        sub_labels = labels[mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        results[domain] = float(roc_auc_score(sub_labels, probs[mask]))
    return results


def compute_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Return ``(mean_pred, actual_rate)``."""
    return float(np.mean(probs)), float(np.mean(labels))


class MetricsCode(DiagBase):
    """Validation metrics (epoch/step_eval) and rolling train AUC (step).

    Owns the reservoir sampler for train AUC: updates every step (cheap
    O(batch)), computes AUC only on emit steps (expensive O(n log n)).

    Also emits ``train_auc_adjusted``: an incremental correction for the
    cumulative-average lag in reservoir-sampled train AUC. The reservoir
    reports C(n) = (1/n) Σ A(i); the adjusted value recovers the average
    instantaneous AUC over the most recent diagnostic interval via
    Ā = n₂·C(n₂) - n₁·C(n₁), where n is the emission count (consecutive
    emissions always have n₂ - n₁ = 1, so the denominator is 1).

    Using emission count instead of step numbers avoids global-vs-local
    step confusion — C(n) is epoch-local (reservoir resets each epoch),
    and emission count is epoch-local by construction.
    """

    code = "METRICS"
    config_key = "metrics"
    emit = frozenset({"step", "epoch", "step_eval"})
    accumulate = frozenset({"always"})
    init_params: tuple[str, ...] = ()

    def __init__(self, train_auc_sample_size: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._epoch_losses: list[float] = []
        self._reservoir = (
            ReservoirSampler(train_auc_sample_size) if train_auc_sample_size > 0 else None
        )
        self._train_auc: float = None
        # Incremental AUC correction state. The reservoir reports cumulative
        # AUC C(n) after n emissions. We track the weighted sum n*C(n) at the
        # current and previous emission to recover instantaneous AUC via
        # Ā = n₂*C(n₂) - n₁*C(n₁).
        self._auc_emission_count: int = 0
        self._auc_weighted_prev: float = 0.0
        self._auc_weighted_curr: float = 0.0

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate step loss, update reservoir, compute AUC on emit."""
        self._epoch_losses.append(ctx.loss)
        if self._reservoir is not None:
            self._reservoir.update(
                ctx.logits.detach().cpu().float().numpy(),
                ctx.batch["label"].detach().cpu().float().numpy(),
            )
            if emit:
                self._train_auc = self._reservoir.compute_auc()
                if self._train_auc is not None:
                    self._auc_emission_count += 1
                    self._auc_weighted_prev = self._auc_weighted_curr
                    self._auc_weighted_curr = self._auc_emission_count * self._train_auc

    def flush(self) -> None:
        """No-op: loss accumulation is epoch-scoped, not emission-scoped."""

    def epoch_reset(self) -> None:
        """Clear epoch loss buffer, reservoir, and adjusted-AUC state."""
        self._epoch_losses.clear()
        self._train_auc = None
        if self._reservoir is not None:
            self._reservoir.reset()
        self._auc_emission_count = 0
        self._auc_weighted_prev = 0.0
        self._auc_weighted_curr = 0.0

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for the given phase."""
        if phase == "step":
            return self._collect_step(ctx)
        if phase == "step_eval":
            return self._collect_step_eval(ctx)
        return self._collect_epoch(ctx)

    def _adjusted_auc(self) -> float | None:
        """Incremental AUC from the snapshot advanced in ``step()``.

        With emission count n as index, C(n) is the cumulative AUC after
        n emissions. Since consecutive emissions have n₂ - n₁ = 1:

            Ā = n₂·C(n₂) - n₁·C(n₁)
        """
        if self._auc_emission_count < 2:
            return None
        return self._auc_weighted_curr - self._auc_weighted_prev

    def _collect_step(self, ctx: StepContext) -> list[str]:
        parts = [f"loss={ctx.loss:.4f}"]
        if self.writer:
            self.writer.add_scalar("Loss/train", ctx.loss, ctx.step)

        for k, v in ctx.aux_losses.items():
            parts.append(f"{k}={v:.4f}")
            if self.writer:
                self.writer.add_scalar(f"Loss/{k}", v, ctx.step)

        train_auc = self._train_auc
        if train_auc is not None:
            parts.append(f"train_auc={train_auc:.6f}")
            adjusted = self._adjusted_auc()
            if adjusted is not None:
                parts.append(f"train_auc_adjusted={adjusted:.6f}")
                if self.writer:
                    self.writer.add_scalar("AUC/train_adjusted_step", adjusted, ctx.step)
            if self.writer:
                self.writer.add_scalar("AUC/train_step", train_auc, ctx.step)

        return [",".join(parts)]

    def _collect_step_eval(self, ctx: EpochContext) -> list[str]:
        step = ctx.step
        val_auc = ctx.val_auc
        val_logloss = ctx.val_logloss
        if self.writer:
            self.writer.add_scalar("AUC/val_step", val_auc, step)
            self.writer.add_scalar("Loss/val_step", val_logloss, step)
        return [f"val_auc={val_auc:.6f},val_logloss={val_logloss:.6f}"]

    def _collect_epoch(self, ctx: EpochContext) -> list[str]:
        auc = ctx.val_auc
        ll = ctx.val_logloss
        t_train = ctx.train_time
        t_val = ctx.val_time
        epoch = ctx.epoch
        # Compute final epoch AUC from the reservoir before reset
        train_auc = self._reservoir.compute_auc() if self._reservoir is not None else None
        parts = [f"auc={auc:.6f},logloss={ll:.6f},train_sec={t_train:.0f},val_sec={t_val:.0f}"]

        if train_auc is not None:
            parts.append(f"train_auc={train_auc:.6f}")

        per_domain = ctx.per_domain_aucs
        if per_domain:
            for d, a in sorted(per_domain.items()):
                parts.append(f"auc_{d}={a:.6f}")
        cal = ctx.calibration
        if cal:
            parts.append(f"cal_pred={cal[0]:.6f},cal_actual={cal[1]:.6f}")

        eta = ctx.eta_sec
        parts.append(f"eta_sec={eta:.0f}")

        # Loss percentiles from self-accumulated epoch losses
        if self._epoch_losses:
            la = np.array(self._epoch_losses)
            for p in (5, 25, 50, 75, 95):
                parts.append(f"loss_p{p}={float(np.percentile(la, p)):.4f}")
            parts.append(f"loss_std={float(np.std(la)):.4f}")

        best_auc = ctx.best_val_auc
        delta = ctx.delta
        plateau = ctx.epochs_since_improvement
        parts.append(f"best_auc={best_auc:.6f},delta={delta:+.6f},plateau={plateau}")

        if self.writer:
            self.writer.add_scalar("AUC/val", auc, epoch)
            if train_auc is not None:
                self.writer.add_scalar("AUC/train", train_auc, epoch)
            self.writer.add_scalar("Loss/val", ll, epoch)
            self.writer.add_scalar("Timing/train_sec", t_train, epoch)
            self.writer.add_scalar("Timing/val_sec", t_val, epoch)
            self.writer.add_scalar("Timing/eta_sec", eta, epoch)
            self.writer.add_scalar("Convergence/best_val_auc", best_auc, epoch)
            self.writer.add_scalar("Convergence/delta", delta, epoch)
            self.writer.add_scalar("Convergence/plateau_counter", plateau, epoch)
            if per_domain:
                for d, a in per_domain.items():
                    self.writer.add_scalar(f"AUC/{d}", a, epoch)
            if cal:
                self.writer.add_scalar("Calibration/mean_pred", cal[0], epoch)
                self.writer.add_scalar("Calibration/actual_rate", cal[1], epoch)
        return [",".join(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        if "epoch" in ctx:
            accum.setdefault("epochs", {})[ctx["epoch"]] = entry
        else:
            accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


class PredCode(DiagBase):
    """Prediction confidence distribution."""

    code = "PRED"
    config_key = "pred_conf"
    emit = frozenset({"epoch", "step_eval"})
    accumulate = frozenset()

    def __init__(
        self,
        calibration_bins: list[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._calibration_bins = calibration_bins

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect prediction confidence percentiles and entropy."""
        if not isinstance(ctx, EpochContext):
            return []
        probs = ctx.val_probs
        if probs is None or len(probs) == 0:
            return []
        parts = [f"p{p}={float(np.percentile(probs, p)):.4f}" for p in (5, 25, 50, 75, 95)]
        eps = 1e-7
        clamped = np.clip(probs, eps, 1.0 - eps)
        entropy = -(clamped * np.log2(clamped) + (1 - clamped) * np.log2(1 - clamped))
        parts.append(f"entropy={np.mean(entropy):.4f}")
        high_conf = float(np.mean((probs < 0.05) | (probs > 0.95)))
        parts.append(f"high_conf={high_conf:.4f}")
        payloads = [",".join(parts)]
        payloads.extend(self._calibration_payloads(ctx))
        return payloads

    def _calibration_payloads(self, ctx: EpochContext) -> list[str]:
        """Emit binned calibration rows from validation probabilities and labels."""
        probs = ctx.val_probs
        labels = ctx.val_labels
        if probs is None or labels is None or self._calibration_bins is None:
            return []

        payloads: list[str] = []
        bins = self._calibration_bins
        for idx in range(len(bins) - 1):
            lo = bins[idx]
            hi = bins[idx + 1]
            if idx == len(bins) - 2:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            n = int(mask.sum())
            if n == 0:
                continue
            payloads.append(
                "calib:"
                f"bin={lo:g}-{hi:g},"
                f"n={n},"
                f"mean_pred={float(probs[mask].mean()):.6f},"
                f"actual_rate={float(labels[mask].mean()):.6f}"
            )
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        if payload.startswith("calib:"):
            row: dict[str, Any] = {}
            for kv in payload[len("calib:") :].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    row[k] = _try_numeric(v)
            if "epoch" in ctx:
                accum.setdefault("calibration", {}).setdefault(ctx["epoch"], []).append(row)
            else:
                accum.setdefault("calibration_steps", {}).setdefault(ctx.get("step", 0), []).append(
                    row
                )
            return

        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        if "epoch" in ctx:
            accum.setdefault("epochs", {})[ctx["epoch"]] = entry
        else:
            accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


class LogitDistCode(DiagBase):
    """Per-class logit percentiles, median gap, and overlap fraction."""

    code = "LOGIT_DIST"
    config_key = "logit_dist"
    emit = frozenset({"epoch", "step_eval"})
    accumulate = frozenset()

    def __init__(
        self,
        percentiles: list[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._percentiles = percentiles or [25, 50, 75]

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Emit per-class logit percentiles and separation metrics."""
        if not isinstance(ctx, EpochContext):
            return []
        logits = ctx.val_logits
        labels = ctx.val_labels
        if logits is None or labels is None:
            return []

        pos_mask = labels == 1
        neg_mask = ~pos_mask

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return []

        parts: list[str] = []
        for p in self._percentiles:
            parts.append(f"neg_p{p}={float(np.percentile(neg_logits, p)):.4f}")
        for p in self._percentiles:
            parts.append(f"pos_p{p}={float(np.percentile(pos_logits, p)):.4f}")

        neg_p50 = float(np.percentile(neg_logits, 50))
        pos_p50 = float(np.percentile(pos_logits, 50))
        pos_p25 = float(np.percentile(pos_logits, 25))
        median_gap = pos_p50 - neg_p50
        overlap = float(np.mean(neg_logits > pos_p25))

        parts.append(f"median_gap={median_gap:.4f}")
        parts.append(f"overlap={overlap:.4f}")

        epoch = ctx.epoch
        if self.writer:
            self.writer.add_scalar("LogitDist/neg_p50", neg_p50, epoch)
            self.writer.add_scalar("LogitDist/pos_p50", pos_p50, epoch)
            self.writer.add_scalar("LogitDist/median_gap", median_gap, epoch)
            self.writer.add_scalar("LogitDist/overlap", overlap, epoch)

        return [",".join(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        if "epoch" in ctx:
            accum.setdefault("epochs", {})[ctx["epoch"]] = entry
        else:
            accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


class LossConcCode(DiagBase):
    """Loss concentration: fraction of total validation loss from hardest samples.

    When ``profile_hard`` is enabled, emits a second payload line comparing
    the top-10% hardest samples against the bottom-90% across sequence metadata:
    raw sequence lengths, truncation severity, predicted probability, and
    positive label rate per split.
    """

    code = "LOSS_CONC"
    config_key = "loss_conc"
    emit = frozenset({"epoch", "step_eval"})
    accumulate = frozenset()

    def __init__(
        self,
        profile_hard: bool = False,
        rank_profile: bool = False,
        slice_profile: bool = False,
        temporal_profile: bool = False,
        max_pos_eval: int = 50_000,
        max_neg_eval: int = 50_000,
        max_pair_eval: int = 200_000,
        margin_eps: float = 0.05,
        slice_top_k: int = 5,
        rng_seed: int = 42,
        margin_percentiles: list[int] = None,
        near_zero_eps_values: list[float] = None,
        hub_top_fracs: list[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._profile_hard = profile_hard
        self._rank_profile = rank_profile
        self._slice_profile = slice_profile
        self._temporal_profile = temporal_profile
        self._max_pos_eval = max_pos_eval
        self._max_neg_eval = max_neg_eval
        self._max_pair_eval = max_pair_eval
        self._margin_eps = margin_eps
        self._slice_top_k = slice_top_k
        self._rng_seed = rng_seed
        self._margin_percentiles = (
            margin_percentiles if margin_percentiles is not None else [1, 5, 10, 25, 50, 75, 90]
        )
        self._near_zero_eps_values = (
            near_zero_eps_values
            if near_zero_eps_values is not None
            else [0.01, 0.03, 0.05, 0.10, 0.20]
        )
        self._hub_top_fracs = (
            hub_top_fracs if hub_top_fracs is not None else [0.001, 0.005, 0.01, 0.05, 0.10]
        )
        self._prev_hard_idx: np.ndarray = None
        self._prev_hub_neg_idx: np.ndarray = None

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect loss concentration fractions for hardest samples."""
        if not isinstance(ctx, EpochContext):
            return []

        per_sample_loss = ctx.val_losses
        if per_sample_loss is None:
            logits = ctx.val_logits
            labels = ctx.val_labels
            if logits is None or labels is None:
                return []
            import torch

            logits_t = torch.from_numpy(logits).float()
            labels_t = torch.from_numpy(labels).float()
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits_t, labels_t, reduction="none"
            ).numpy()

        sorted_indices = np.argsort(per_sample_loss)[::-1]
        sorted_losses = per_sample_loss[sorted_indices]
        total_loss = sorted_losses.sum()
        if total_loss == 0:
            return []

        n = len(sorted_losses)
        top20_n = max(1, n // 5)
        top5_n = max(1, n // 20)
        top10_n = max(1, n // 10)
        top1_n = max(1, n // 100)

        top20_frac = float(sorted_losses[:top20_n].sum() / total_loss)
        top5_frac = float(sorted_losses[:top5_n].sum() / total_loss)
        top10_frac = float(sorted_losses[:top10_n].sum() / total_loss)
        top1_frac = float(sorted_losses[:top1_n].sum() / total_loss)
        loss_hhi = float(np.square(sorted_losses / total_loss).sum())
        top1_over_top10 = float(top1_frac / max(top10_frac, 1e-12))
        top5_over_top20 = float(top5_frac / max(top20_frac, 1e-12))

        step = int(ctx.step or ctx.epoch)
        self._log_scalars(
            {
                "top20_frac": top20_frac,
                "top5_frac": top5_frac,
                "top10_frac": top10_frac,
                "top1_frac": top1_frac,
                "loss_hhi": loss_hhi,
                "top1_over_top10": top1_over_top10,
                "top5_over_top20": top5_over_top20,
            },
            step,
        )

        payloads = [
            (
                f"top20_frac={top20_frac:.4f},top5_frac={top5_frac:.4f},"
                f"top10_frac={top10_frac:.4f},top1_frac={top1_frac:.4f},"
                f"loss_hhi={loss_hhi:.6f},top1_over_top10={top1_over_top10:.4f},"
                f"top5_over_top20={top5_over_top20:.4f}"
            )
        ]

        if self._profile_hard:
            profile = self._profile_splits(
                sorted_indices[:top10_n],
                sorted_indices[top10_n:],
                ctx,
            )
            if profile:
                payloads.append(profile)
                self._log_profile_scalars(profile, step)

        labels = ctx.val_labels
        scores = self._resolve_scores(ctx)
        rank_data = None
        if (
            (self._rank_profile or self._slice_profile or self._temporal_profile)
            and labels is not None
            and scores is not None
        ):
            rng = np.random.default_rng(self._rng_seed + int(ctx.step or ctx.epoch or 0))
            rank_data = self._prepare_rank_eval(scores, labels, rng)

            if self._rank_profile:
                rank_payload = self._rank_payload(rank_data, rng, step)
                if rank_payload:
                    payloads.append(rank_payload)

            if self._slice_profile:
                slice_payloads = self._slice_payloads(
                    rank_data,
                    per_sample_loss,
                    scores,
                    labels,
                    ctx,
                    step,
                )
                payloads.extend(slice_payloads)

        if self._temporal_profile:
            temporal_payload = self._temporal_payload(
                sorted_indices[:top10_n],
                rank_data["hub_neg_idx"] if rank_data is not None else None,
                step,
            )
            if temporal_payload:
                payloads.append(temporal_payload)

        return payloads

    def _resolve_scores(self, ctx: EpochContext) -> np.ndarray:
        """Return ranking scores for inversion/margin diagnostics."""
        logits = ctx.val_logits
        if logits is not None:
            return logits.astype(np.float64, copy=False)
        probs = ctx.val_probs
        if probs is not None:
            return probs.astype(np.float64, copy=False)
        return None

    @staticmethod
    def _metric_label(value: float) -> str:
        """Convert threshold value into a metric-friendly label."""
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    @staticmethod
    def _parse_payload_values(payload: str, prefix: str = "") -> dict[str, Any]:
        """Parse comma-delimited key=value payload into typed dict."""
        text = payload[len(prefix) :] if prefix and payload.startswith(prefix) else payload
        out: dict[str, Any] = {}
        for kv in text.split(","):
            if "=" not in kv:
                continue
            key, value = kv.split("=", 1)
            out[key] = _try_numeric(value)
        return out

    @staticmethod
    def _parse_payload_numeric(payload: str, prefix: str = "") -> dict[str, float]:
        """Parse payload and keep numeric key/value pairs only."""
        parsed = LossConcCode._parse_payload_values(payload, prefix=prefix)
        out: dict[str, float] = {}
        for key, value in parsed.items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
        return out

    def _log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log a metric dictionary under the `LossConc/` namespace."""
        if not self.writer:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"LossConc/{key}", float(value), step)

    def _log_profile_scalars(self, payload: str, step: int) -> None:
        """Log selected hard/easy profile metrics to TensorBoard."""
        values = self._parse_payload_numeric(payload, prefix="profile:")
        keys = [
            "hard_pred_mean",
            "easy_pred_mean",
            "hard_pos_rate",
            "easy_pos_rate",
            "hard_loss_p90",
            "hard_loss_p99",
            "hard_n_domains",
            "easy_n_domains",
        ]
        self._log_scalars({key: values[key] for key in keys if key in values}, step)
        if "hard_pos_rate" in values and "easy_pos_rate" in values:
            self._log_scalars(
                {"hard_easy_pos_gap": values["hard_pos_rate"] - values["easy_pos_rate"]},
                step,
            )
        if "hard_pred_mean" in values and "easy_pred_mean" in values:
            self._log_scalars(
                {"hard_easy_pred_gap": values["hard_pred_mean"] - values["easy_pred_mean"]},
                step,
            )

    @staticmethod
    def _subsample_indices(indices: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
        """Subsample index vector without replacement when it exceeds a cap."""
        if max_n > 0 and len(indices) > max_n:
            return rng.choice(indices, size=max_n, replace=False)
        return indices.copy()

    def _prepare_rank_eval(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        """Prepare capped positive/negative subsets and inversion contributions."""
        pos_all = np.flatnonzero(labels > 0.5)
        neg_all = np.flatnonzero(labels <= 0.5)
        if len(pos_all) == 0 or len(neg_all) == 0:
            return None

        pos_idx = self._subsample_indices(pos_all, self._max_pos_eval, rng)
        neg_idx = self._subsample_indices(neg_all, self._max_neg_eval, rng)

        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        sorted_pos = np.sort(pos_scores)
        inv_per_neg = np.searchsorted(sorted_pos, neg_scores, side="right").astype(np.float64)
        pair_n = int(len(pos_scores) * len(neg_scores))
        inv_total = float(inv_per_neg.sum())

        neg_order = np.argsort(neg_scores)[::-1]
        top1_n = max(1, len(neg_scores) // 100)
        hub_neg_idx = neg_idx[neg_order[:top1_n]]

        return {
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "inv_per_neg": inv_per_neg,
            "pair_n": pair_n,
            "inv_total": inv_total,
            "hub_neg_idx": hub_neg_idx,
        }

    def _rank_payload(
        self,
        rank_data: dict[str, Any] | None,
        rng: np.random.Generator,
        step: int,
    ) -> str:
        """Build AUC-impact payload from sampled pairwise statistics."""
        if rank_data is None:
            return ""
        pair_n = rank_data["pair_n"]
        if pair_n <= 0:
            return ""

        inv_total = rank_data["inv_total"]
        inv_per_neg = rank_data["inv_per_neg"]
        neg_scores = rank_data["neg_scores"]
        pos_scores = rank_data["pos_scores"]

        inv_rate = float(inv_total / pair_n)
        neg_order = np.argsort(neg_scores)[::-1]
        top1_n = max(1, len(neg_scores) // 100)
        top10_n = max(1, len(neg_scores) // 10)
        top1_inv = float(inv_per_neg[neg_order[:top1_n]].sum())
        top10_inv = float(inv_per_neg[neg_order[:top10_n]].sum())
        inv_top1neg_frac = float(top1_inv / inv_total) if inv_total > 0 else 0.0
        inv_top10neg_frac = float(top10_inv / inv_total) if inv_total > 0 else 0.0

        hub_frac_metrics: dict[str, float] = {}
        for frac in self._hub_top_fracs:
            n_top = max(1, round(len(neg_scores) * float(frac)))
            top_inv = float(inv_per_neg[neg_order[:n_top]].sum())
            frac_name = self._metric_label(float(frac) * 100.0)
            key = f"inv_top{frac_name}neg_frac"
            hub_frac_metrics[key] = float(top_inv / inv_total) if inv_total > 0 else 0.0

        pair_eval_n = pair_n if self._max_pair_eval <= 0 else min(self._max_pair_eval, pair_n)
        if pair_eval_n <= 0:
            return ""
        pos_pick = rng.integers(0, len(pos_scores), size=pair_eval_n)
        neg_pick = rng.integers(0, len(neg_scores), size=pair_eval_n)
        margins = pos_scores[pos_pick] - neg_scores[neg_pick]
        margin_percentiles = {
            pct: float(np.percentile(margins, pct)) for pct in sorted(set(self._margin_percentiles))
        }
        margin_p10 = margin_percentiles.get(10, float(np.percentile(margins, 10)))
        margin_p25 = margin_percentiles.get(25, float(np.percentile(margins, 25)))
        near_zero_margin_rate = float((np.abs(margins) < self._margin_eps).mean())
        near_zero_by_eps = {
            eps: float((np.abs(margins) < float(eps)).mean())
            for eps in sorted(set(self._near_zero_eps_values))
        }
        inv_rate_se = float(np.sqrt(max(inv_rate * (1.0 - inv_rate), 0.0) / pair_eval_n))

        scalar_metrics = {
            "inv_rate": inv_rate,
            "inv_top1neg_frac": inv_top1neg_frac,
            "inv_top10neg_frac": inv_top10neg_frac,
            "inv_total": inv_total,
            "pair_n": float(pair_n),
            "inv_rate_se": inv_rate_se,
            "margin_p10": margin_p10,
            "margin_p25": margin_p25,
            "near_zero_margin_rate": near_zero_margin_rate,
        }
        scalar_metrics.update(hub_frac_metrics)
        scalar_metrics.update(
            {f"margin_p{pct}": value for pct, value in margin_percentiles.items()}
        )
        scalar_metrics.update(
            {
                f"near_zero_margin_rate_e{self._metric_label(float(eps))}": value
                for eps, value in near_zero_by_eps.items()
            }
        )
        self._log_scalars(scalar_metrics, step)

        parts = [
            f"inv_rate={inv_rate:.4f}",
            f"inv_top1neg_frac={inv_top1neg_frac:.4f}",
            f"inv_top10neg_frac={inv_top10neg_frac:.4f}",
            f"margin_p10={margin_p10:.4f}",
            f"margin_p25={margin_p25:.4f}",
            f"near_zero_margin_rate={near_zero_margin_rate:.4f}",
            f"pair_n={pair_n}",
            f"inv_rate_se={inv_rate_se:.4f}",
        ]
        for key, value in sorted(hub_frac_metrics.items()):
            parts.append(f"{key}={value:.4f}")
        for pct, value in sorted(margin_percentiles.items()):
            parts.append(f"margin_p{pct}={value:.4f}")
        for eps, value in sorted(near_zero_by_eps.items()):
            eps_label = self._metric_label(float(eps))
            parts.append(f"near_zero_margin_rate_e{eps_label}={value:.4f}")
        return "rank:" + ",".join(parts)

    @staticmethod
    def _bucket_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Digitize values into integer buckets from monotonic edges."""
        return np.digitize(values, edges, right=True).astype(np.int64)

    def _slice_payloads(
        self,
        rank_data: dict[str, Any] | None,
        per_sample_loss: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        ctx: EpochContext,
        step: int,
    ) -> list[str]:
        """Build top-k most distorted slices by inversion contribution."""
        if rank_data is None:
            return []
        pos_idx = rank_data["pos_idx"]
        neg_idx = rank_data["neg_idx"]
        eval_idx = np.concatenate([pos_idx, neg_idx])
        if len(eval_idx) == 0:
            return []

        eval_labels = labels[eval_idx]
        eval_scores = scores[eval_idx]
        eval_losses = per_sample_loss[eval_idx]
        total_eval_loss = float(eval_losses.sum())
        neg_inv = rank_data["inv_per_neg"]
        inv_total = rank_data["inv_total"]

        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

        decile_edges = np.quantile(eval_scores, np.linspace(0.1, 0.9, 9))
        decile_edges = np.unique(decile_edges)
        if len(decile_edges) > 0:
            candidates.append(
                (
                    "score_decile",
                    self._bucket_from_edges(eval_scores, decile_edges),
                    self._bucket_from_edges(rank_data["neg_scores"], decile_edges),
                )
            )

        meta = ctx.val_seq_metadata
        if meta:
            len_keys = sorted(
                key
                for key in meta
                if key.endswith("_len") and "_raw_len" not in key and "_recency_" not in key
            )
            if len_keys:
                eval_n_domains = sum((meta[key][eval_idx] > 0).astype(np.int64) for key in len_keys)
                neg_n_domains = sum((meta[key][neg_idx] > 0).astype(np.int64) for key in len_keys)
                candidates.append(
                    ("n_domains", eval_n_domains.astype(np.int64), neg_n_domains.astype(np.int64))
                )

            trunc_sources = []
            for raw_key in sorted(key for key in meta if key.endswith("_raw_len")):
                domain = raw_key.removesuffix("_raw_len")
                len_key = f"{domain}_len"
                if len_key in meta:
                    trunc_sources.append((raw_key, len_key))
            if trunc_sources:
                eval_trunc = np.zeros(len(eval_idx), dtype=np.float64)
                neg_trunc = np.zeros(len(neg_idx), dtype=np.float64)
                for raw_key, len_key in trunc_sources:
                    eval_raw = meta[raw_key][eval_idx].astype(np.float64)
                    eval_clamped = meta[len_key][eval_idx].astype(np.float64)
                    eval_trunc += 1.0 - eval_clamped / np.maximum(eval_raw, 1.0)

                    neg_raw = meta[raw_key][neg_idx].astype(np.float64)
                    neg_clamped = meta[len_key][neg_idx].astype(np.float64)
                    neg_trunc += 1.0 - neg_clamped / np.maximum(neg_raw, 1.0)

                eval_trunc /= len(trunc_sources)
                neg_trunc /= len(trunc_sources)
                trunc_edges = np.quantile(eval_trunc, np.linspace(0.2, 0.8, 4))
                trunc_edges = np.unique(trunc_edges)
                if len(trunc_edges) > 0:
                    candidates.append(
                        (
                            "trunc_quintile",
                            self._bucket_from_edges(eval_trunc, trunc_edges),
                            self._bucket_from_edges(neg_trunc, trunc_edges),
                        )
                    )

        rows: list[dict[str, Any]] = []
        for kind, eval_bucket, neg_bucket in candidates:
            for bucket in np.unique(eval_bucket):
                sample_mask = eval_bucket == bucket
                n_bucket = int(sample_mask.sum())
                if n_bucket == 0:
                    continue
                sample_share = float(n_bucket / len(eval_idx))
                loss_share = (
                    float(eval_losses[sample_mask].sum() / total_eval_loss)
                    if total_eval_loss > 0
                    else 0.0
                )
                neg_mask = neg_bucket == bucket
                inv_share = (
                    float(neg_inv[neg_mask].sum() / inv_total)
                    if inv_total > 0 and int(neg_mask.sum()) > 0
                    else 0.0
                )
                distortion = float(inv_share / max(sample_share, 1e-12))

                slice_auc = None
                bucket_labels = eval_labels[sample_mask]
                if len(np.unique(bucket_labels)) >= 2:
                    slice_auc = float(roc_auc_score(bucket_labels, eval_scores[sample_mask]))

                rows.append(
                    {
                        "kind": kind,
                        "bucket": int(bucket),
                        "sample_share": sample_share,
                        "loss_share": loss_share,
                        "inv_share": inv_share,
                        "distortion": distortion,
                        "slice_auc": slice_auc,
                    }
                )

        rows.sort(key=lambda item: item["distortion"], reverse=True)
        if rows:
            top_rows = rows[: max(self._slice_top_k, 1)]
            worst = top_rows[0]
            metrics = {
                "slice_max_distortion": worst["distortion"],
                "slice_mean_topk_distortion": float(
                    np.mean([row["distortion"] for row in top_rows])
                ),
                "slice_worst_sample_share": worst["sample_share"],
                "slice_worst_inv_share": worst["inv_share"],
                "slice_worst_loss_share": worst["loss_share"],
                "slice_count": float(len(rows)),
            }
            if worst["slice_auc"] is not None:
                metrics["slice_worst_auc"] = worst["slice_auc"]
            self._log_scalars(metrics, step)

        payloads = []
        for row in rows[: max(self._slice_top_k, 0)]:
            parts = [
                f"kind={row['kind']}",
                f"bucket={row['bucket']}",
                f"sample_share={row['sample_share']:.4f}",
                f"loss_share={row['loss_share']:.4f}",
                f"inv_share={row['inv_share']:.4f}",
                f"distortion={row['distortion']:.4f}",
            ]
            if row["slice_auc"] is not None:
                parts.append(f"slice_auc={row['slice_auc']:.4f}")
            payloads.append("slice:" + ",".join(parts))
        return payloads

    def _temporal_payload(
        self,
        hard_idx: np.ndarray,
        hub_neg_idx: np.ndarray = None,
        step: int = 0,
    ) -> str:
        """Track persistence/churn of hard examples and hub negatives."""
        curr_hard = hard_idx.astype(np.int64, copy=False)
        curr_hub = hub_neg_idx.astype(np.int64, copy=False) if hub_neg_idx is not None else None

        if self._prev_hard_idx is None or len(self._prev_hard_idx) == 0:
            self._prev_hard_idx = curr_hard.copy()
            self._prev_hub_neg_idx = curr_hub.copy() if curr_hub is not None else None
            return ""

        prev_hard_set = set(self._prev_hard_idx.tolist())
        curr_hard_set = set(curr_hard.tolist())
        hard_inter = len(prev_hard_set & curr_hard_set)
        hard_union = len(prev_hard_set | curr_hard_set)
        hard_jaccard = float(hard_inter / hard_union) if hard_union > 0 else 0.0
        hard_entry_rate = (
            float(len(curr_hard_set - prev_hard_set) / len(curr_hard_set))
            if len(curr_hard_set) > 0
            else 0.0
        )

        parts = [f"hard_jaccard={hard_jaccard:.4f}", f"hard_entry_rate={hard_entry_rate:.4f}"]

        if curr_hub is not None and self._prev_hub_neg_idx is not None:
            prev_hub_set = set(self._prev_hub_neg_idx.tolist())
            curr_hub_set = set(curr_hub.tolist())
            hub_inter = len(prev_hub_set & curr_hub_set)
            hub_union = len(prev_hub_set | curr_hub_set)
            hub_jaccard = float(hub_inter / hub_union) if hub_union > 0 else 0.0
            hub_entry_rate = (
                float(len(curr_hub_set - prev_hub_set) / len(curr_hub_set))
                if len(curr_hub_set) > 0
                else 0.0
            )
            parts.append(f"hub_jaccard={hub_jaccard:.4f}")
            parts.append(f"hub_entry_rate={hub_entry_rate:.4f}")

        metrics = {"hard_jaccard": hard_jaccard, "hard_entry_rate": hard_entry_rate}
        if curr_hub is not None and self._prev_hub_neg_idx is not None:
            metrics["hub_jaccard"] = hub_jaccard
            metrics["hub_entry_rate"] = hub_entry_rate
        self._log_scalars(metrics, step)

        self._prev_hard_idx = curr_hard.copy()
        self._prev_hub_neg_idx = curr_hub.copy() if curr_hub is not None else None
        return "temporal:" + ",".join(parts)

    def _profile_splits(
        self,
        hard_idx: np.ndarray,
        easy_idx: np.ndarray,
        ctx: EpochContext,
    ) -> str:
        """Compare hard vs easy splits across available metadata."""
        parts: list[str] = []
        probs = ctx.val_probs
        labels = ctx.val_labels
        losses = ctx.val_losses
        meta = ctx.val_seq_metadata

        # Predicted probability
        if probs is not None:
            parts.append(f"hard_pred_mean={probs[hard_idx].mean():.4f}")
            parts.append(f"easy_pred_mean={probs[easy_idx].mean():.4f}")

        # Positive rate per split
        if labels is not None:
            parts.append(f"hard_pos_rate={labels[hard_idx].mean():.4f}")
            parts.append(f"easy_pos_rate={labels[easy_idx].mean():.4f}")

        # Loss shape: how concentrated is loss within the hard set itself
        if losses is not None:
            hard_losses = losses[hard_idx]
            parts.append(f"hard_loss_p90={np.percentile(hard_losses, 90):.4f}")
            parts.append(f"hard_loss_p99={np.percentile(hard_losses, 99):.4f}")

        if meta:
            # Domain presence and cold-start
            domains_found = []
            for key, arr in sorted(meta.items()):
                if not key.endswith("_len"):
                    continue
                if "_raw_len" in key or "_recency_" in key:
                    continue
                domain = key.removesuffix("_len")
                domains_found.append(domain)
                hard_present = (arr[hard_idx] > 0).mean()
                easy_present = (arr[easy_idx] > 0).mean()
                parts.append(f"{domain}_present_hard={hard_present:.4f}")
                parts.append(f"{domain}_present_easy={easy_present:.4f}")

            # Domain count: how many domains are active per sample
            if domains_found:
                hard_n_domains = sum(
                    (meta[f"{d}_len"][hard_idx] > 0).astype(np.float64) for d in domains_found
                ).mean()
                easy_n_domains = sum(
                    (meta[f"{d}_len"][easy_idx] > 0).astype(np.float64) for d in domains_found
                ).mean()
                parts.append(f"hard_n_domains={hard_n_domains:.2f}")
                parts.append(f"easy_n_domains={easy_n_domains:.2f}")

            # Raw lengths and truncation
            for key, arr in sorted(meta.items()):
                if not key.endswith("_raw_len"):
                    continue
                domain = key.removesuffix("_raw_len")
                hard_raw = arr[hard_idx]
                easy_raw = arr[easy_idx]
                parts.append(f"{domain}_raw_len_hard_p50={np.median(hard_raw):.0f}")
                parts.append(f"{domain}_raw_len_easy_p50={np.median(easy_raw):.0f}")

                len_key = f"{domain}_len"
                if len_key in meta:
                    clamped = meta[len_key]
                    hard_trunc = 1.0 - clamped[hard_idx].astype(np.float64) / np.maximum(
                        hard_raw, 1
                    )
                    easy_trunc = 1.0 - clamped[easy_idx].astype(np.float64) / np.maximum(
                        easy_raw, 1
                    )
                    parts.append(f"{domain}_trunc_hard_mean={hard_trunc.mean():.4f}")
                    parts.append(f"{domain}_trunc_easy_mean={easy_trunc.mean():.4f}")

            # Recency: time bucket of most recent event per domain
            for key, arr in sorted(meta.items()):
                if not key.endswith("_recency_bucket"):
                    continue
                domain = key.removesuffix("_recency_bucket")
                hard_rec = arr[hard_idx].astype(np.float64)
                easy_rec = arr[easy_idx].astype(np.float64)
                # Higher bucket = more stale; 0 = no events
                parts.append(f"{domain}_recency_hard_mean={hard_rec.mean():.2f}")
                parts.append(f"{domain}_recency_easy_mean={easy_rec.mean():.2f}")

        if not parts:
            return ""
        return "profile:" + ",".join(parts)

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        epoch = ctx.get("epoch", 0)
        step = ctx.get("step", 0)
        key = step if step else epoch

        if payload.startswith("profile:"):
            entry = LossConcCode._parse_payload_values(payload, prefix="profile:")
            accum.setdefault("profile", {})[key] = entry
        elif payload.startswith("rank:"):
            entry = LossConcCode._parse_payload_values(payload, prefix="rank:")
            accum.setdefault("rank", {})[key] = entry
        elif payload.startswith("slice:"):
            entry = LossConcCode._parse_payload_values(payload, prefix="slice:")
            accum.setdefault("slices", {}).setdefault(key, []).append(entry)
        elif payload.startswith("temporal:"):
            entry = LossConcCode._parse_payload_values(payload, prefix="temporal:")
            accum.setdefault("temporal", {})[key] = entry
        else:
            entry = LossConcCode._parse_payload_values(payload)
            accum.setdefault("epochs", {})[key] = entry
