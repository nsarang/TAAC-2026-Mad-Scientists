"""Batch-level feature masking for SAGE importance and training-time dropout."""

from __future__ import annotations

import math
from typing import Any

import torch

from core.data.schema import FeatureSchema, FeatureSpec


def mask_batch(
    batch: dict[str, Any],
    specs: list[FeatureSpec],
    schema: FeatureSchema,
) -> dict[str, Any]:
    """Zero specified features in a batch dict.

    Returns a new dict. Unaffected keys are shared references; affected
    tensors are cloned before zeroing so the original batch is not mutated.

    Parameters
    ----------
    batch
        Collated batch dict (torch tensors).
    specs
        Features to zero out.
    schema
        FeatureSchema for resolving domain membership.
    """
    out = dict(batch)
    cloned_keys: set[str] = set()

    for spec in specs:
        key = spec.batch_key
        if key not in out:
            continue

        if key not in cloned_keys:
            out[key] = out[key].clone()
            cloned_keys.add(key)

        if spec.col_range is not None:
            s, e = spec.col_range
            out[key][..., s:e] = 0
        else:
            out[key].zero_()

    return out


def mask_domain(
    batch: dict[str, Any],
    domain: str,
    schema: FeatureSchema,
) -> dict[str, Any]:
    """Zero all features in a sequence domain plus its length key.

    Query: ``domain = '{d}' or name = '{d}_len'`` finds all SEQ content
    (sideinfo, timestamps, time_bucket) plus the post-truncation length.
    Does NOT zero ``{domain}_raw_len`` — that's observational truth for
    diagnostics.

    Parameters
    ----------
    batch
        Collated batch dict.
    domain
        Sequence domain name to mask.
    schema
        FeatureSchema for resolving domain specs.
    """
    specs = schema.query(f"domain = '{domain}' or name = '{domain}_len'")
    return mask_batch(batch, specs, schema)


class FeatureMaskingScheduler:
    """Scheduled per-group masking with decay for training-time feature dropout.

    Groups are defined as name/expression pairs resolved against the schema.
    Two masking modes:

    - ``"input"`` — zeros features in the batch dict before embedding.
      Applied via `apply(batch)`.
    - ``"token"`` — returns per-sample keep tensors ``(B, 1, 1)`` for the
      model to multiply on embedded token slices. Retrieved via `sample()`.

    Parameters
    ----------
    schema
        FeatureSchema for resolving expressions to specs.
    groups
        Per-group config. Keys are group names, values are dicts with:
        - ``type``: ``"input"`` or ``"token"`` (required)
        - ``expr``: DSL expression string (for input groups)
        - ``domain_name``: sequence domain name (for domain-level input groups)
        - ``initial_prob``, ``final_prob``, ``decay_end_epoch``,
          ``decay_type``, ``fixed_prob``: schedule overrides
    seq_domains
        Sequence domain names (for domain-level masking).
    total_steps
        Total training steps across all epochs.
    epochs
        Total training epochs.
    initial_prob
        Default masking probability at step 0.
    final_prob
        Default masking probability after decay completes.
    decay_end_epoch
        Epoch by which decay reaches `final_prob`.
    decay_type
        ``"linear"`` or ``"cosine"``.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        groups: dict[str, dict[str, Any]],
        seq_domains: list[str],
        total_steps: int,
        epochs: int,
        initial_prob: float = 0.99,
        final_prob: float = 0.0,
        decay_end_epoch: int = None,
        decay_type: str = "linear",
    ) -> None:
        self._schema = schema
        self._seq_domains = seq_domains
        self._total_steps = max(1, total_steps)
        self._epochs = max(1, epochs)
        steps_per_epoch = self._total_steps / self._epochs

        if decay_end_epoch is None:
            decay_end_epoch = epochs
        default_decay_steps = max(1, int(decay_end_epoch * steps_per_epoch))

        self._default_schedule = {
            "initial_prob": initial_prob,
            "final_prob": final_prob,
            "decay_steps": default_decay_steps,
            "decay_type": decay_type,
        }

        self._groups: dict[str, dict[str, Any]] = {}
        self._schedules: dict[str, dict[str, Any]] = {}
        self._current_probs: dict[str, float] = {}

        # Resolved specs for input-type groups
        self._input_specs: dict[str, list[FeatureSpec]] = {}
        self._input_domain_names: dict[str, str] = {}

        for name, cfg in groups.items():
            self._groups[name] = cfg
            sched = self._make_schedule(cfg, default_decay_steps, steps_per_epoch)
            self._schedules[name] = sched
            self._current_probs[name] = sched["prob"]

            if cfg.get("type") == "input":
                if "domain_name" in cfg:
                    self._input_domain_names[name] = cfg["domain_name"]
                else:
                    self._input_specs[name] = schema.query(cfg["expr"])

        self._current_step = 0

    def _make_schedule(
        self,
        cfg: dict[str, Any],
        default_decay_steps: int,
        steps_per_epoch: float,
    ) -> dict[str, Any]:
        if "fixed_prob" in cfg:
            return {"fixed": True, "prob": cfg["fixed_prob"]}
        sched = dict(self._default_schedule)
        if "initial_prob" in cfg:
            sched["initial_prob"] = cfg["initial_prob"]
        if "final_prob" in cfg:
            sched["final_prob"] = cfg["final_prob"]
        if "decay_type" in cfg:
            sched["decay_type"] = cfg["decay_type"]
        if "decay_end_epoch" in cfg:
            sched["decay_steps"] = max(1, int(cfg["decay_end_epoch"] * steps_per_epoch))
        sched["prob"] = sched["initial_prob"]
        return sched

    def tick(self) -> None:
        """Advance the schedule by one training step."""
        self._current_step += 1
        for name, sched in self._schedules.items():
            if sched.get("fixed"):
                continue
            progress = min(1.0, self._current_step / sched["decay_steps"])
            if sched["decay_type"] == "cosine":
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                decay = max(0.0, 1.0 - progress)
            self._current_probs[name] = (
                sched["final_prob"] + (sched["initial_prob"] - sched["final_prob"]) * decay
            )

    def apply(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply input-level masking to a batch dict.

        Samples each input-type group independently and zeros masked groups.
        Returns a new batch dict (originals not mutated).

        Parameters
        ----------
        batch
            Collated batch dict.
        """
        out = batch
        for name, specs in self._input_specs.items():
            prob = self._current_probs.get(name, 0.0)
            if prob > 0 and torch.rand(1).item() < prob:
                out = mask_batch(out, specs, self._schema)

        for name, domain in self._input_domain_names.items():
            prob = self._current_probs.get(name, 0.0)
            if prob > 0 and torch.rand(1).item() < prob:
                out = mask_domain(out, domain, self._schema)

        return out

    def sample(
        self,
        group_names: list[str],
        B: int,
        device: torch.device = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-group bernoulli keep masks for token-level masking.

        Parameters
        ----------
        group_names
            Token-type group names to sample.
        B
            Batch size.
        device
            Torch device.

        Returns
        -------
        Dict of ``{name: (B, 1, 1)}`` keep tensors (1 = keep, 0 = masked).
        """
        result: dict[str, torch.Tensor] = {}
        for name in group_names:
            prob = self._current_probs.get(name, 0.0)
            if prob <= 0:
                result[name] = torch.ones(B, 1, 1, device=device)
            elif prob >= 1.0:
                result[name] = torch.zeros(B, 1, 1, device=device)
            else:
                result[name] = torch.bernoulli(torch.full((B, 1, 1), 1.0 - prob, device=device))
        return result
