"""Loss functions: BCE variants, pairwise objectives, and hybrids."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Parameters
    ----------
    alpha
        Positive-class weight in (0, 1).
    gamma
        Focusing parameter. 0 degenerates to BCE; 2 is standard.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from raw logits and binary labels."""
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
        return (alpha_t * focal_weight * bce).mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits with an optional per-sample weight.

    Drop-in for ``torch.nn.BCEWithLogitsLoss`` on the pointwise path. The
    trainer supplies `sample_weight` from the batch (e.g. recency weighting)
    when present; with no weight this reduces to plain BCE. Reduction is a
    weighted mean.

    Parameters
    ----------
    pos_weight
        Positive-class weight for the underlying BCE. None disables.
    label_smoothing
        Smooth labels toward 0.5 before the loss.
    """

    def __init__(self, pos_weight: float = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
        self.register_buffer("pos_weight", pw)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute (optionally weighted) BCE from raw logits and binary labels."""
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction="none"
        )
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight.to(per_sample.dtype).reshape(per_sample.shape)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-8)


class _PairwiseAUCLoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return logits.new_tensor(0.0)
        positive_scores = logits[positive_mask]
        negative_scores = logits[negative_mask]
        margins = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
        return F.softplus(-margins).mean()


class SupervisedContrastiveLoss(nn.Module):
    """SupCon (Khosla et al. 2020) on L2-normalized embeddings.

    Parameters
    ----------
    temperature
        Softmax temperature. Lower = sharper distribution.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Parameters
        ----------
        z
            (B, proj_dim) L2-normalized embeddings.
        labels
            (B,) binary labels.
        """
        sim = z @ z.T / self.temperature  # (B, B)

        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
        pos_mask.fill_diagonal_(0)

        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - logits_max

        # Mask out self-similarity before exp to avoid in-place mutation
        self_mask = torch.ones_like(sim, dtype=torch.bool)
        self_mask.fill_diagonal_(False)
        exp_sim = torch.exp(sim) * self_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = sim - log_denom

        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return z.new_tensor(0.0)
        loss = -(pos_mask * log_prob).sum(dim=1)
        return (loss[valid] / n_pos[valid]).mean()


class RankingLoss(nn.Module):
    """Weighted combination of BCE and pairwise AUC loss."""

    def __init__(
        self,
        pairwise_weight: float,
        pos_weight: float = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.pairwise = _PairwiseAUCLoss()
        self.pairwise_weight = min(max(pairwise_weight, 0.0), 1.0)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the combined ranking loss."""
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = self.bce(logits, labels)
        if self.pairwise_weight == 0.0:
            return bce_loss
        pairwise_loss = self.pairwise(logits, labels)
        return (1.0 - self.pairwise_weight) * bce_loss + self.pairwise_weight * pairwise_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss: separate focusing parameters for positives and negatives.

    With gamma_pos=0, positives always receive full gradient. gamma_neg > 0
    suppresses easy negatives without ever downweighting hard positives.

    Parameters
    ----------
    gamma_neg
        Focusing parameter for negatives. Higher = stronger suppression of
        easy negatives.
    gamma_pos
        Focusing parameter for positives. 0 recommended for CVR (hard
        positives are the signal, never suppress them).
    label_smoothing
        Smooth labels toward 0.5 before loss computation.
    """

    def __init__(
        self, gamma_neg: float = 1.0, gamma_pos: float = 0.0, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric focal loss."""
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        # p_t: predicted prob of the true class
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        # Per-sample gamma: gamma_pos for positives, gamma_neg for negatives
        gamma = self.gamma_pos * labels + self.gamma_neg * (1.0 - labels)
        focal_weight = (1.0 - p_t) ** gamma
        return (focal_weight * bce).mean()


class GHMLoss(nn.Module):
    """Gradient Harmonizing Mechanism loss.

    Bins samples by gradient magnitude (|p - y|), computes per-bin density,
    and inversely weights each bin so no difficulty stratum dominates the
    gradient. Uses exponential moving average to smooth bin counts across
    steps.

    Parameters
    ----------
    bins
        Number of gradient magnitude bins in [0, 1].
    momentum
        EMA decay for bin count smoothing. 0 = no smoothing (pure per-batch).
    """

    def __init__(self, bins: int = 10, momentum: float = 0.75) -> None:
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.linspace(0.0, 1.0, bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("acc_sum", torch.zeros(bins))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute density-reweighted BCE via gradient magnitude bins."""
        p = torch.sigmoid(logits).detach()
        g = torch.abs(p - labels)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        n = logits.numel()
        weights = torch.zeros_like(logits)
        for i in range(self.bins):
            lo = self.edges[i]
            hi = self.edges[i + 1]
            if i == self.bins - 1:
                mask = (g >= lo) & (g <= hi)
            else:
                mask = (g >= lo) & (g < hi)
            count = mask.sum().float()
            if count > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * count
                    density = self.acc_sum[i]
                else:
                    density = count
                weights[mask] = n / density

        return (weights * bce).sum() / n


class OHEMLoss(nn.Module):
    """Online Hard Example Mining: only backprop through the hardest samples.

    Computes per-sample BCE, keeps all positives unconditionally, then retains
    the top `keep_ratio` fraction of negatives ranked by loss. The rest get
    zero gradient.

    Parameters
    ----------
    keep_ratio
        Fraction of the batch to retain (0.5-0.8). Applies to negatives only;
        all positives are always kept.
    """

    def __init__(self, keep_ratio: float = 0.7) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE over all positives + hardest `keep_ratio` negatives."""
        per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pos_mask = labels > 0.5
        neg_losses = per_sample[~pos_mask]
        if neg_losses.numel() == 0:
            return per_sample.mean()
        n_keep = max(1, int(neg_losses.numel() * self.keep_ratio))
        threshold = neg_losses.topk(n_keep).values[-1]
        neg_keep_mask = per_sample >= threshold
        keep = pos_mask | neg_keep_mask
        return per_sample[keep].mean()


class AsymmetricRankingLoss(nn.Module):
    """ASL + pairwise BPR auxiliary: asymmetric focal pointwise loss combined
    with pairwise ranking gradient signal.

    Parameters
    ----------
    gamma_neg
        ASL focusing on negatives.
    gamma_pos
        ASL focusing on positives (0 recommended).
    pairwise_weight
        Weight of the pairwise BPR term relative to ASL.
    label_smoothing
        Smooth labels toward 0.5 before the pointwise (ASL) term.
    """

    def __init__(
        self,
        gamma_neg: float = 1.0,
        gamma_pos: float = 0.0,
        pairwise_weight: float = 0.15,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg, gamma_pos, label_smoothing)
        self.pairwise = _PairwiseAUCLoss()
        self.pairwise_weight = min(max(pairwise_weight, 0.0), 1.0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of ASL pointwise and BPR pairwise losses."""
        asl_loss = self.asl(logits, labels)
        if self.pairwise_weight == 0.0:
            return asl_loss
        pairwise_loss = self.pairwise(logits, labels)
        return (1.0 - self.pairwise_weight) * asl_loss + self.pairwise_weight * pairwise_loss


class FrontierPairwiseBudgetLoss(nn.Module):
    """Pointwise anchor + frontier pairwise auxiliary with dynamic budget.

    The anchor term is BCE with optional negative focusing and positive
    upweighting. The pairwise term only compares positives against the highest
    scoring negatives in-batch and emphasizes near-boundary margins. A dynamic
    lambda keeps pairwise pressure bounded relative to the anchor term.

    Parameters
    ----------
    lambda_max
        Upper bound on dynamic pairwise weight.
    pairwise_budget_ratio
        Target pairwise-to-anchor budget used to compute dynamic lambda.
    neg_top_frac
        Fraction of negatives to keep (highest-score tail) for pairwise terms.
    neg_top_k
        Hard cap on negative count used in pairwise terms.
    margin_center
        Margin center for frontier weighting.
    margin_temp
        Temperature of frontier weighting. Lower sharpens boundary focus.
    fp_guard_weight
        Weight for high-scoring-negative guard penalty.
    fp_guard_quantile
        Negative-score quantile used to define guard tail.
    fp_guard_margin
        Margin for guard penalty ``softplus(neg_logit - fp_guard_margin)``.
    gamma_neg
        Negative focusing exponent in the anchor term.
    pos_weight
        Positive-class multiplier for the anchor term.
    label_smoothing
        Smooth labels toward 0.5 before loss computation.
    """

    def __init__(
        self,
        lambda_max: float = 0.12,
        pairwise_budget_ratio: float = 0.35,
        neg_top_frac: float = 0.2,
        neg_top_k: int = 128,
        margin_center: float = 0.0,
        margin_temp: float = 0.5,
        fp_guard_weight: float = 0.05,
        fp_guard_quantile: float = 0.95,
        fp_guard_margin: float = 0.0,
        gamma_neg: float = 1.0,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_max = lambda_max
        self.pairwise_budget_ratio = pairwise_budget_ratio
        self.neg_top_frac = neg_top_frac
        self.neg_top_k = neg_top_k
        self.margin_center = margin_center
        self.margin_temp = margin_temp
        self.fp_guard_weight = fp_guard_weight
        self.fp_guard_quantile = fp_guard_quantile
        self.fp_guard_margin = fp_guard_margin
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.last_metrics: dict[str, float] = {}

    def _anchor_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hard_pos_mask = labels > 0.5
        if self.label_smoothing > 0.0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        neg_weight = probs.pow(self.gamma_neg)
        sample_weight = torch.where(hard_pos_mask, torch.ones_like(labels), neg_weight)
        if self.pos_weight != 1.0:
            sample_weight = torch.where(
                hard_pos_mask,
                sample_weight * self.pos_weight,
                sample_weight,
            )

        return (sample_weight * bce).mean()

    def _frontier_pairwise(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return logits.new_tensor(0.0)

        positive_scores = logits[positive_mask].float()
        negative_scores = logits[negative_mask].float()

        n_negative = int(negative_scores.numel())
        n_top = max(1, round(self.neg_top_frac * n_negative))
        if self.neg_top_k > 0:
            n_top = min(n_top, self.neg_top_k)
        n_top = min(n_top, n_negative)

        top_negative_scores = torch.topk(negative_scores, k=n_top).values
        margins = positive_scores.unsqueeze(1) - top_negative_scores.unsqueeze(0)
        pairwise = F.softplus(-margins)

        frontier_weight = torch.sigmoid((self.margin_center - margins) / self.margin_temp)
        frontier_weight = frontier_weight / (frontier_weight.sum(dim=1, keepdim=True) + 1e-8)
        return (frontier_weight * pairwise).sum(dim=1).mean()

    def _false_positive_guard(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        negative_scores = logits[labels <= 0.5].float()
        if negative_scores.numel() == 0:
            return logits.new_tensor(0.0)

        threshold = torch.quantile(negative_scores.detach(), self.fp_guard_quantile)
        tail = negative_scores[negative_scores >= threshold]
        if tail.numel() == 0:
            return logits.new_tensor(0.0)
        return F.softplus(tail - self.fp_guard_margin).mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute dynamic-budget frontier pairwise loss."""
        anchor_loss = self._anchor_loss(logits, labels)
        pairwise_loss = self._frontier_pairwise(logits, labels)
        fp_guard = self._false_positive_guard(logits, labels)

        lambda_budget = (
            self.pairwise_budget_ratio * anchor_loss.detach() / (pairwise_loss.detach() + 1e-8)
        )
        lambda_dynamic = torch.clamp(lambda_budget, max=self.lambda_max)
        total = anchor_loss + lambda_dynamic * pairwise_loss + self.fp_guard_weight * fp_guard

        self.last_metrics = {
            "anchor": float(anchor_loss.detach().item()),
            "pairwise": float(pairwise_loss.detach().item()),
            "fp_guard": float(fp_guard.detach().item()),
            "lambda_dynamic": float(lambda_dynamic.detach().item()),
        }
        return total


class DisagreementGatedBCELoss(nn.Module):
    """BCE with positive reliability gating from stochastic disagreement.

    Uses two logit views when provided. For positives, disagreement between
    views downweights unreliable updates. Negatives use focal-like downweighting
    to suppress easy cases.

    Parameters
    ----------
    disagreement_temperature
        Temperature for reliability gate ``exp(-|d| / T)``.
    pos_reliability_floor
        Minimum positive reliability to avoid fully zeroing gradients.
    unstable_target_mix
        Blend factor for soft target on positives using detached view-average
        probability.
    gamma_neg
        Negative focusing exponent.
    label_smoothing
        Smooth labels toward 0.5 before loss computation.
    view_noise_std
        Std of Gaussian noise for synthetic second view in `forward()`.
    """

    def __init__(
        self,
        disagreement_temperature: float = 0.75,
        pos_reliability_floor: float = 0.1,
        unstable_target_mix: float = 0.15,
        gamma_neg: float = 1.0,
        label_smoothing: float = 0.0,
        view_noise_std: float = 0.15,
    ) -> None:
        super().__init__()
        self.disagreement_temperature = disagreement_temperature
        self.pos_reliability_floor = pos_reliability_floor
        self.unstable_target_mix = unstable_target_mix
        self.gamma_neg = gamma_neg
        self.label_smoothing = label_smoothing
        self.view_noise_std = view_noise_std
        self.last_metrics: dict[str, float] = {}

    def _build_targets(
        self,
        labels: torch.Tensor,
        mean_prob: torch.Tensor,
    ) -> torch.Tensor:
        targets = labels
        if self.unstable_target_mix > 0.0:
            positive_mask = labels > 0.5
            pos_target = (
                1.0 - self.unstable_target_mix
            ) + self.unstable_target_mix * mean_prob.detach().to(labels.dtype)
            targets = torch.where(positive_mask, pos_target, labels)
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return targets

    def _build_weights(
        self,
        labels: torch.Tensor,
        logits_view_a: torch.Tensor,
        logits_view_b: torch.Tensor,
        mean_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_mask = labels > 0.5
        disagreement = torch.abs(logits_view_a - logits_view_b)
        reliability = torch.exp(-disagreement / self.disagreement_temperature)
        reliability = torch.clamp(reliability, min=self.pos_reliability_floor, max=1.0)

        negative_weight = mean_prob.pow(self.gamma_neg)
        sample_weight = torch.where(positive_mask, reliability, negative_weight)
        return sample_weight.to(labels.dtype), reliability

    def forward_with_views(
        self,
        logits_view_a: torch.Tensor,
        logits_view_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disagreement-gated BCE from two logit views."""
        mean_prob = torch.sigmoid(0.5 * (logits_view_a + logits_view_b))
        targets = self._build_targets(labels, mean_prob)
        sample_weight, reliability = self._build_weights(
            labels,
            logits_view_a,
            logits_view_b,
            mean_prob,
        )

        loss_view_a = F.binary_cross_entropy_with_logits(logits_view_a, targets, reduction="none")
        loss_view_b = F.binary_cross_entropy_with_logits(logits_view_b, targets, reduction="none")
        loss = 0.5 * (loss_view_a + loss_view_b)
        total = (sample_weight * loss).mean()

        positive_mask = labels > 0.5
        if positive_mask.any():
            mean_pos_reliability = float(reliability[positive_mask].mean().detach().item())
        else:
            mean_pos_reliability = 1.0
        self.last_metrics = {
            "mean_pos_reliability": mean_pos_reliability,
            "mean_weight": float(sample_weight.mean().detach().item()),
            "mean_loss": float(total.detach().item()),
        }
        return total

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss using synthetic perturbation as second stochastic view."""
        logits_view_b = logits + self.view_noise_std * torch.randn_like(logits)
        return self.forward_with_views(logits, logits_view_b, labels)
