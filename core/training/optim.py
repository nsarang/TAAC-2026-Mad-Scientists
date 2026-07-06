"""Custom optimizers: sparse-embedding (AdagradFreqDecay, FTRL),
dense wrappers (Lookahead), and spectral (Muon).

AdagradFreqDecay subclasses ``torch.optim.Adagrad`` so the native C++/CUDA kernel
handles the gradient step, then applies frequency-scaled decay in a thin Python
post-step on accessed rows only.

FTRL implements FTRL-Proximal (McMahan et al., 2013) — Adagrad with built-in L1
regularization that drives rare/noisy feature weights to exactly zero.

FTRLTriton is a Triton-accelerated variant that fuses the z/n accumulation, L1
thresholding, and weight update into a single GPU kernel over accessed rows only.

Lookahead wraps any base optimizer with slow-weight interpolation every k steps.

Muon applies Newton-Schulz orthogonalization for spectrally-normalized updates
on 2D weight matrices (ported from torch.optim.Muon, PyTorch 2.12+).
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor
from torch.optim import Adagrad, Optimizer

LOG = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ModuleNotFoundError:
    _HAS_TRITON = False

LOG.info(
    f"triton import {'succeeded' if _HAS_TRITON else 'failed'};"
    f" FTRLTriton CUDA kernel {'available' if _HAS_TRITON else 'unavailable'}"
)


class Lookahead(Optimizer):
    """Lookahead optimizer wrapper (Zhang et al., 2019).

    Maintains slow weights that interpolate toward the fast weights every `k`
    steps. Wraps any base optimizer.

    Paper: "Lookahead Optimizer: k steps forward, 1 step back" (arXiv:1907.08610)

    Parameters
    ----------
    base_optimizer
        Any ``torch.optim.Optimizer`` instance (Adam, AdamW, etc.).
    alpha
        Interpolation rate for slow weight update. ``slow += alpha * (fast - slow)``.
    k
        Number of fast-weight steps between each slow-weight sync.
    """

    def __init__(self, base_optimizer: Optimizer, alpha: float = 0.5, k: int = 6) -> None:
        # NOTE: intentionally not calling super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self._base_optimizer = base_optimizer
        self._alpha = alpha
        self._k = k
        self._step_count = 0
        self.defaults = base_optimizer.defaults

    @property
    def state(self):
        """Proxy to base optimizer state dict."""
        return self._base_optimizer.state

    @state.setter
    def state(self, value):
        self._base_optimizer.state = value

    @property
    def param_groups(self):
        """Proxy to base optimizer param groups."""
        return self._base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._base_optimizer.param_groups = value

    @torch.no_grad()
    def _update_slow(self) -> None:
        for group in self._base_optimizer.param_groups:
            for fast_p in group["params"]:
                if fast_p.grad is None:
                    continue
                state = self._base_optimizer.state[fast_p]
                if "lookahead_slow" not in state:
                    state["lookahead_slow"] = fast_p.data.clone()
                slow = state["lookahead_slow"]
                slow.add_(fast_p.data - slow, alpha=self._alpha)
                fast_p.data.copy_(slow)

    def sync_lookahead(self) -> None:
        """Force slow-weight sync (call before evaluation)."""
        self._update_slow()

    @torch.no_grad()
    def step(self, closure=None):
        """Base optimizer step, then slow-weight sync every `k` steps."""
        loss = self._base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self._k == 0:
            self._update_slow()
        return loss

    def state_dict(self):
        """Return base state dict augmented with lookahead step count."""
        d = self._base_optimizer.state_dict()
        d["lookahead_step_count"] = self._step_count
        return d

    def load_state_dict(self, state_dict):
        """Restore base state and lookahead step count."""
        self._step_count = state_dict.pop("lookahead_step_count", 0)
        self._base_optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients on the base optimizer."""
        self._base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        """Add a param group to the base optimizer."""
        self._base_optimizer.add_param_group(param_group)


# ---------------------------------------------------------------------------
# Muon optimizer — Newton-Schulz orthogonalized momentum (Zhang et al., 2024)
# Ported from pytorch/pytorch (torch/optim/_muon.py, 2.12+).
# ---------------------------------------------------------------------------

_MUON_NS_A = 3.4445
_MUON_NS_B = -4.7750
_MUON_NS_C = 2.0315
_MUON_EPS = 1e-7


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Newton-Schulz orthogonalization of `grad`.

    Produces an approximate UV^T (zeroth power of the SVD) using a quintic
    polynomial iteration tuned for fast convergence.
    """
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if transposed:
        ortho_grad = ortho_grad.T
    return ortho_grad


class Muon(Optimizer):
    """Muon optimizer for 2D parameters (hidden-layer weight matrices).

    Applies Newton-Schulz orthogonalization to the momentum buffer so updates
    have uniform singular values, acting as implicit spectral normalization.

    Non-2D parameters (biases, norms, embeddings) should use a separate
    optimizer such as AdamW.

    Reference: "Muon: An optimizer for hidden layers in neural networks"
    (Keller Jordan, 2024) — https://kellerjordan.github.io/posts/muon/

    Parameters
    ----------
    params
        Iterable of 2D parameters.
    lr
        Learning rate.
    weight_decay
        Decoupled weight decay coefficient.
    momentum
        Momentum factor for the buffer.
    nesterov
        Whether to use Nesterov-style lookahead on the buffer.
    ns_steps
        Number of Newton-Schulz iterations (higher = more accurate orthogonalization).
    ns_coefficients
        Polynomial coefficients ``(a, b, c)`` for the quintic NS iteration.
    eps
        Numerical stability term for the initial normalization.
    adjust_lr_fn
        LR scaling mode: ``"original"`` scales by ``sqrt(max(1, rows/cols))``,
        ``"match_rms_adamw"`` scales by ``0.2 * sqrt(max(rows, cols))``.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = (_MUON_NS_A, _MUON_NS_B, _MUON_NS_C),
        eps: float = _MUON_EPS,
        adjust_lr_fn: str = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters, got shape {p.shape}")

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

                adjusted_lr = self._adjust_lr(lr, adjust_lr_fn, p.shape)
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-adjusted_lr)

        return loss

    @staticmethod
    def _adjust_lr(lr: float, adjust_lr_fn: str, shape: torch.Size) -> float:
        """Scale LR based on parameter matrix aspect ratio."""
        A, B = shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            return lr * math.sqrt(max(1, A / B))
        elif adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(A, B))
        return lr


class AdagradFreqDecay(Adagrad):
    """Adagrad with post-step frequency-scaled weight decay.

    The native Adagrad kernel handles the gradient step. After each
    ``step()``, applies decoupled decay to accessed rows scaled by a
    running estimate of inverse frequency.

    Each row maintains an EMA hit count that updates on every access:
    ``ema_count = (1 - beta) * ema_count + beta``. Rows that rarely
    get gradients converge to a low count and receive heavier decay.

    Parameters
    ----------
    params
        Iterable of parameters.
    lr
        Learning rate.
    lr_decay
        Adagrad LR decay coefficient.
    eps
        Denominator stability term.
    freq_decay
        Base weight decay before frequency scaling.
    freq_beta
        EMA smoothing factor for the per-row hit counter.
    freq_floor
        Minimum value of the frequency factor (prevents complete
        zeroing of the rarest rows).
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        lr_decay: float = 0.0,
        eps: float = 1e-10,
        freq_decay: float = 0.01,
        freq_beta: float = 0.001,
        freq_floor: float = 0.1,
        weight_decay: float = 0.0,
    ) -> None:
        self._freq_decay = freq_decay
        self._freq_beta = freq_beta
        self._freq_floor = freq_floor
        self._decay_state: dict[int, torch.Tensor] = {}
        super().__init__(params, lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay)

    def _get_ema(self, p: torch.nn.Parameter) -> torch.Tensor:
        pid = p.data_ptr()
        if pid not in self._decay_state:
            self._decay_state[pid] = torch.zeros(p.data.shape[0], device=p.data.device)
        return self._decay_state[pid]

    @torch.no_grad()
    def step(self, closure=None):
        """Native Adagrad step, then frequency-scaled decay on accessed rows."""
        loss = super().step(closure)

        if self._freq_decay > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        indices = grad._indices().squeeze(0)
                    elif p.data.dim() > 1:
                        indices = torch.nonzero(grad.norm(dim=-1) > 0, as_tuple=True)[0]
                    else:
                        continue
                    if indices.numel() == 0:
                        continue
                    ema = self._get_ema(p)
                    ema_vals = (1.0 - self._freq_beta) * ema[indices] + self._freq_beta
                    ema[indices] = ema_vals
                    row_scale = 1.0 - self._freq_decay * (
                        (1.0 - ema_vals.clamp_(max=1.0)).clamp_(min=self._freq_floor)
                    )
                    p.data[indices] *= row_scale.unsqueeze(-1)

        return loss

    def _param_order(self) -> list[torch.nn.Parameter]:
        return [p for group in self.param_groups for p in group["params"]]

    def state_dict(self):
        """Serialize optimizer + frequency EMA state."""
        d = super().state_dict()
        ema_list = []
        for p in self._param_order():
            pid = p.data_ptr()
            if pid in self._decay_state:
                ema_list.append(self._decay_state[pid].cpu())
            else:
                ema_list.append(None)
        d["freq_ema"] = ema_list
        return d

    def load_state_dict(self, state_dict):
        """Restore optimizer + frequency EMA state."""
        ema_list = state_dict.pop("freq_ema", [])
        super().load_state_dict(state_dict)
        for p, ema in zip(self._param_order(), ema_list):
            if ema is not None:
                self._decay_state[p.data_ptr()] = ema.to(p.device)


class FTRL(Optimizer):
    """FTRL-Proximal optimizer (McMahan et al., 2013).

    Adagrad-family optimizer with built-in L1 regularization that drives
    small/noisy weights to exactly zero. Designed for sparse features where
    rare IDs should be killed rather than memorized.

    Parameters
    ----------
    params
        Iterable of parameters (typically embedding weights).
    lr
        Learning rate numerator. Effective LR = lr / (beta + n_t^{-lr_power}).
    beta
        Smoothing parameter for LR denominator.
    l1
        L1 regularization strength. Higher = more weights zeroed.
    l2
        L2 regularization strength added to denominator.
    lr_power
        Power for scaling the accumulator in the LR denominator. Must be <= 0.
        -0.5 (default) recovers standard Adagrad; 0 gives a fixed learning rate.
    l2_shrinkage
        L2 shrinkage regularization (magnitude penalty). Added to the gradient
        as ``2 * l2_shrinkage * w`` before updating ``z``, only on rows that
        received a gradient. Different from `l2` which stabilizes the denominator.
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        beta: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0,
        lr_power: float = -0.5,
        l2_shrinkage: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        if l1 < 0.0:
            raise ValueError(f"Invalid l1: {l1}")
        if l2 < 0.0:
            raise ValueError(f"Invalid l2: {l2}")
        if lr_power > 0.0:
            raise ValueError(f"lr_power must be <= 0, got {lr_power}")
        if l2_shrinkage < 0.0:
            raise ValueError(f"Invalid l2_shrinkage: {l2_shrinkage}")
        defaults = dict(
            lr=lr,
            beta=beta,
            l1=l1,
            l2=l2,
            lr_power=lr_power,
            l2_shrinkage=l2_shrinkage,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["lr"]
            beta = group["beta"]
            l1 = group["l1"]
            l2 = group["l2"]
            lr_power = group["lr_power"]
            l2_shrinkage = group["l2_shrinkage"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)

                z, n = state["z"], state["n"]

                if grad.is_sparse:
                    grad = grad.coalesce()
                    indices = grad.indices().squeeze(0)
                    values = grad.values()
                else:
                    if p.data.dim() > 1:
                        row_nz = grad.any(dim=-1)
                        indices = torch.where(row_nz)[0]
                    else:
                        indices = torch.where(grad != 0)[0]
                    if indices.numel() == 0:
                        continue
                    values = grad[indices]

                w_rows = p.data[indices]
                grad_to_use = values
                if l2_shrinkage > 0:
                    grad_to_use = values + 2.0 * l2_shrinkage * w_rows

                n_rows = n[indices]
                n_new = n_rows + values**2
                sigma = (n_new.pow(-lr_power) - n_rows.pow(-lr_power)) / alpha
                z[indices] += grad_to_use - sigma * w_rows
                n[indices] = n_new

                z_rows = z[indices]
                mask = z_rows.abs() <= l1
                w_new = -(z_rows - l1 * z_rows.sign()) / (
                    l2 + (beta + n_new.pow(-lr_power)) / alpha
                )
                w_new[mask] = 0
                p.data[indices] = w_new

        return loss


# ---------------------------------------------------------------------------
# Triton-fused FTRL kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _ftrl_fused_kernel(
        # Full table pointers (shape [V, D], row-major)
        z_ptr,
        n_ptr,
        w_ptr,
        # Gradient rows (shape [num_idx, D], contiguous)
        grad_ptr,
        # Index array (shape [num_idx])
        idx_ptr,
        # Dims
        D: tl.constexpr,
        alpha,
        beta,
        l1,
        l2,
        lr_power,
        l2_shrinkage,
        num_idx,
        BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        if row_id >= num_idx:
            return

        table_row = tl.load(idx_ptr + row_id)
        col_offs = tl.arange(0, BLOCK_D)
        col_mask = col_offs < D

        grad_base = row_id * D
        table_base = table_row * D

        g = tl.load(grad_ptr + grad_base + col_offs, mask=col_mask, other=0.0)
        z = tl.load(z_ptr + table_base + col_offs, mask=col_mask, other=0.0)
        n = tl.load(n_ptr + table_base + col_offs, mask=col_mask, other=0.0)
        w = tl.load(w_ptr + table_base + col_offs, mask=col_mask, other=0.0)

        g_use = g + 2.0 * l2_shrinkage * w

        n_new = n + g * g
        neg_lr_power = -lr_power
        sigma = (tl.math.pow(n_new, neg_lr_power) - tl.math.pow(n, neg_lr_power)) / alpha
        z_new = z + g_use - sigma * w

        z_sign = tl.where(z_new > 0, 1.0, -1.0)
        z_abs = tl.abs(z_new)
        denom = l2 + (beta + tl.math.pow(n_new, neg_lr_power)) / alpha
        w_new = tl.where(z_abs <= l1, 0.0, -(z_new - l1 * z_sign) / denom)

        tl.store(z_ptr + table_base + col_offs, z_new, mask=col_mask)
        tl.store(n_ptr + table_base + col_offs, n_new, mask=col_mask)
        tl.store(w_ptr + table_base + col_offs, w_new, mask=col_mask)


class FTRLTriton(Optimizer):
    """Triton-accelerated FTRL-Proximal.

    Fuses the z/n accumulation, L1 thresholding, and weight update into a
    single Triton kernel launched over accessed rows only (for sparse grads)
    or the full tensor (for dense grads).

    Falls back to the pure-PyTorch FTRL path on CPU or when Triton is
    unavailable.

    Parameters
    ----------
    params
        Iterable of parameters (typically embedding weights).
    lr
        Learning rate numerator. Effective LR = lr / (beta + n_t^{-lr_power}).
    beta
        Smoothing parameter for LR denominator.
    l1
        L1 regularization strength. Higher = more weights zeroed.
    l2
        L2 regularization strength added to denominator.
    lr_power
        Power for scaling the accumulator in the LR denominator. Must be <= 0.
        -0.5 (default) recovers standard Adagrad; 0 gives a fixed learning rate.
    l2_shrinkage
        L2 shrinkage regularization (magnitude penalty). Added to the gradient
        as ``2 * l2_shrinkage * w`` before updating ``z``, only on rows that
        received a gradient. Different from `l2` which stabilizes the denominator.
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        beta: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0,
        lr_power: float = -0.5,
        l2_shrinkage: float = 0.0,
    ) -> None:
        if not _HAS_TRITON:
            raise RuntimeError("FTRLTriton requires triton. Install it or use FTRL instead.")
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        if l1 < 0.0:
            raise ValueError(f"Invalid l1: {l1}")
        if l2 < 0.0:
            raise ValueError(f"Invalid l2: {l2}")
        if lr_power > 0.0:
            raise ValueError(f"lr_power must be <= 0, got {lr_power}")
        if l2_shrinkage < 0.0:
            raise ValueError(f"Invalid l2_shrinkage: {l2_shrinkage}")
        defaults = dict(
            lr=lr,
            beta=beta,
            l1=l1,
            l2=l2,
            lr_power=lr_power,
            l2_shrinkage=l2_shrinkage,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)

                z, n = state["z"], state["n"]

                if grad.is_sparse:
                    grad = grad.coalesce()
                    indices = grad.indices().squeeze(0)
                    values = grad.values()
                else:
                    if p.data.dim() > 1:
                        row_nz = grad.any(dim=-1)
                        indices = torch.where(row_nz)[0]
                    else:
                        indices = torch.where(grad != 0)[0]
                    if indices.numel() == 0:
                        continue
                    values = grad[indices]

                if not p.data.is_cuda:
                    alpha = group["lr"]
                    lr_power = group["lr_power"]
                    l2_shrinkage = group["l2_shrinkage"]
                    w_rows = p.data[indices]
                    grad_to_use = values
                    if l2_shrinkage > 0:
                        grad_to_use = values + 2.0 * l2_shrinkage * w_rows
                    n_rows = n[indices]
                    n_new = n_rows + values**2
                    sigma = (n_new.pow(-lr_power) - n_rows.pow(-lr_power)) / alpha
                    z[indices] += grad_to_use - sigma * w_rows
                    n[indices] = n_new
                    z_rows = z[indices]
                    mask = z_rows.abs() <= group["l1"]
                    w_new = -(z_rows - group["l1"] * z_rows.sign()) / (
                        group["l2"] + (group["beta"] + n_new.pow(-lr_power)) / alpha
                    )
                    w_new[mask] = 0
                    p.data[indices] = w_new
                    continue

                D = p.data.shape[1] if p.data.dim() > 1 else p.data.shape[0]
                num_idx = indices.shape[0]
                BLOCK_D = triton.next_power_of_2(D)
                indices_i64 = indices.to(torch.int64).contiguous()
                values_c = values.contiguous()

                _ftrl_fused_kernel[(num_idx,)](
                    z,
                    n,
                    p.data,
                    values_c,
                    indices_i64,
                    D,
                    group["lr"],
                    group["beta"],
                    group["l1"],
                    group["l2"],
                    group["lr_power"],
                    group["l2_shrinkage"],
                    num_idx,
                    BLOCK_D=BLOCK_D,
                )

        return loss
