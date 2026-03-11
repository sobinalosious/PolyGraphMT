# utils.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Literal

import math
import numpy as np
import torch
import torch.nn as nn

# Re-exported conveniences from data_builder
from .data_builder import TargetScaler, grouped_split_by_smiles  # noqa: F401


# ---------------------------------------------------------
# Seeding and device helpers
# ---------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Deterministically seed Python, NumPy, and PyTorch (CPU/CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device: torch.device):
    """Move a PyG Batch or simple dict of tensors to device."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


# ---------------------------------------------------------
# Masked metrics (canonical)
# ---------------------------------------------------------

def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    den = torch.clamp(den, min=1e-12)
    return num / den


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    """
    pred/target: [B, T]; mask: [B, T] bool
    """
    pred, target = pred.float(), target.float()
    mask = mask.bool()
    se = ((pred - target) ** 2) * mask
    if reduction == "sum":
        return se.sum()
    return _safe_div(se.sum(), mask.sum().float())


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    ae = (pred - target).abs() * mask
    if reduction == "sum":
        return ae.sum()
    return _safe_div(ae.sum(), mask.sum().float())


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask, reduction="mean"))


def masked_r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked coefficient of determination across all elements jointly.
    """
    pred, target = pred.float(), target.float()
    mask = mask.bool()
    count = mask.sum().float().clamp(min=1.0)
    mean = _safe_div((target * mask).sum(), count)
    sst = (((target - mean) ** 2) * mask).sum()
    sse = (((target - pred) ** 2) * mask).sum()
    return 1.0 - _safe_div(sse, sst.clamp(min=1e-12))


def masked_metrics_overall(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    return {
        "rmse": float(masked_rmse(pred, target, mask).detach().cpu()),
        "mae": float(masked_mae(pred, target, mask).detach().cpu()),
        "r2": float(masked_r2(pred, target, mask).detach().cpu()),
    }


def masked_metrics_per_task(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    task_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Per-task metrics using the same masked formulations.
    """
    out: Dict[str, Dict[str, float]] = {}
    for t, name in enumerate(task_names):
        m = mask[:, t]
        if m.any():
            rmse = float(masked_rmse(pred[:, t:t+1], target[:, t:t+1], m.unsqueeze(1)).detach().cpu())
            mae  = float(masked_mae(pred[:, t:t+1], target[:, t:t+1], m.unsqueeze(1)).detach().cpu())
            r2   = float(masked_r2(pred[:, t:t+1], target[:, t:t+1], m.unsqueeze(1)).detach().cpu())
        else:
            rmse = mae = r2 = float("nan")
        out[name] = {"rmse": rmse, "mae": mae, "r2": r2}
    return out


def masked_metrics_by_fidelity(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fid_idx: torch.Tensor,
    fid_names: Sequence[str],
    task_names: Sequence[str],  # kept for API parity; not used in overall-by-fid
) -> Dict[str, Dict[str, float]]:
    """
    Overall metrics per fidelity (aggregated across tasks).
    """
    out: Dict[str, Dict[str, float]] = {}
    fid_idx = fid_idx.view(-1).long()
    for i, fname in enumerate(fid_names):
        sel = (fid_idx == i)
        if sel.any():
            p = pred[sel]
            y = target[sel]
            m = mask[sel]
            out[fname] = masked_metrics_overall(p, y, m)
        else:
            out[fname] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return out


# ---------------------------------------------------------
# Multitask, multi-fidelity loss (canonical)
# ---------------------------------------------------------

def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Element-wise Gaussian NLL (no reduction).
    Shapes: mu, logvar, target -> [B, T] (or broadcastable).
    """
    logvar = torch.as_tensor(logvar, device=mu.device, dtype=mu.dtype)
    logvar = logvar.clamp(min=-20.0, max=20.0)  # numerical guard
    var = torch.exp(logvar)
    err2_over_var = (target - mu) ** 2 / var
    nll = 0.5 * (err2_over_var + logvar + math.log(2.0 * math.pi))  # [B, T]
    return nll


def loss_multitask_fidelity(
    *,
    pred: torch.Tensor,            # [B, T] (or means if heteroscedastic)
    target: torch.Tensor,          # [B, T]
    mask: torch.Tensor,            # [B, T] bool
    fid_idx: torch.Tensor,         # [B] long (per-row fidelity index)
    fid_loss_w: Sequence[float] | torch.Tensor | None,     # [F] weights per fidelity
    task_weights: Optional[Sequence[float] | torch.Tensor] = None,  # [T]
    hetero_logvar: Optional[torch.Tensor] = None,   # [B, T] if heteroscedastic head
    reduction: Literal["mean", "sum"] = "mean",
    task_log_sigma2: Optional[torch.Tensor] = None, # [T] learned homoscedastic uncertainty
    balanced: bool = True,
) -> torch.Tensor:
    """
    Multi-task, multi-fidelity loss with *balanced per-task reduction* by default.

    - If `hetero_logvar` is given: uses Gaussian NLL per element.
    - Applies per-fidelity weights via `fid_idx`.
    - Balanced reduction: compute mean loss per task first, then average across tasks
      (optionally weight by `task_weights` or learned uncertainty `task_log_sigma2`).
    - If `balanced=False`, uses legacy global reduction.
    """
    B, T = pred.shape
    pred = pred.float()
    target = target.float()
    mask = mask.bool()
    fid_idx = fid_idx.view(-1).long()

    # Task weights (optional)
    if task_weights is None:
        tw = pred.new_ones(T)  # [T]
    else:
        tw = torch.as_tensor(task_weights, dtype=pred.dtype, device=pred.device)
        assert tw.numel() == T, f"task_weights len {tw.numel()} != T {T}"
        s = tw.sum().clamp(min=1e-12)
        tw = tw * (T / s)      # normalize to sum=T for stable scale

    # Fidelity weights
    if fid_loss_w is None:
        fw = pred.new_ones(int(fid_idx.max().item()) + 1)
    else:
        fw = torch.as_tensor(fid_loss_w, dtype=pred.dtype, device=pred.device)
    w_fid = fw[fid_idx].unsqueeze(1).expand(-1, T)  # [B, T]

    # Elementwise loss
    if hetero_logvar is not None:
        elem_loss = gaussian_nll(pred, hetero_logvar.float(), target)  # [B, T]
    else:
        elem_loss = (pred - target) ** 2                                # [B, T]

    if not balanced:
        # Legacy global reduction (label-count biased)
        w_task = tw.view(1, T).expand(B, -1)
        weighted = elem_loss * mask * w_task * w_fid
        if reduction == "sum":
            return weighted.sum()
        denom = (mask * w_task * w_fid).sum().float().clamp(min=1e-12)
        return weighted.sum() / denom

    # -------- Balanced per-task reduction --------
    # First compute a per-task average (exclude tw here)
    num = (elem_loss * mask * w_fid).sum(dim=0)              # [T]
    den = (mask * w_fid).sum(dim=0).float().clamp(min=1e-12) # [T]
    per_task_loss = num / den                                # [T]

    # Optional manual task weights AFTER per-task averaging
    if task_weights is not None:
        per_task_loss = per_task_loss * tw

    # Optional homoscedastic task-uncertainty weighting (Kendall & Gal)
    if task_log_sigma2 is not None:
        assert task_log_sigma2.numel() == T, f"task_log_sigma2 must be [T], got {task_log_sigma2.shape}"
        sigma2 = torch.exp(task_log_sigma2)  # [T]
        per_task_loss = per_task_loss / (2.0 * sigma2) + 0.5 * torch.log(sigma2)

    if reduction == "sum":
        return per_task_loss.sum()
    return per_task_loss.mean()


# ---------------------------------------------------------
# Curriculum scheduler for EXP fidelity
# ---------------------------------------------------------

def exp_weight_at_epoch(
    epoch: int,
    total_epochs: int,
    schedule: Literal["none", "linear", "cosine"] = "none",
    start: float = 0.6,
    end: float = 1.0,
) -> float:
    """
    Returns the EXP loss weight for a given epoch under the chosen schedule.
    """
    if schedule == "none":
        return float(end)
    epoch = max(0, min(epoch, total_epochs))
    if total_epochs <= 0:
        return float(end)
    t = epoch / float(total_epochs)
    if schedule == "linear":
        return float(start + (end - start) * t)
    if schedule == "cosine":
        cos_t = 0.5 - 0.5 * math.cos(math.pi * t)  # 0->1 smoothly
        return float(start + (end - start) * cos_t)
    raise ValueError(f"Unknown schedule: {schedule}")


def make_fid_loss_weights(
    fids: Sequence[str],
    base_weights: Optional[Sequence[float]] = None,
    exp_weight: Optional[float] = None,
) -> List[float]:
    """
    Builds a per-fidelity weight vector aligned with dataset.fids order.
    If exp_weight is provided, it overrides the weight for the 'exp' fidelity.
    If base_weights is provided, it must match len(fids) and is used as a template.
    """
    fids_lc = [f.lower() for f in fids]
    F = len(fids_lc)
    if base_weights is None:
        w = [1.0] * F
    else:
        assert len(base_weights) == F, f"base_weights len {len(base_weights)} != {F}"
        w = [float(x) for x in base_weights]
    if exp_weight is not None and "exp" in fids_lc:
        idx = fids_lc.index("exp")
        w[idx] = float(exp_weight)
    return w


# ---------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------

def apply_inverse_transform(pred: torch.Tensor, scaler):
    """
    Apply inverse target scaling safely on the same device as pred.
    Works for CPU/GPU and legacy scalers.
    """
    dev = pred.device

    # Move scaler tensors to pred device if needed
    if hasattr(scaler, "mean") and scaler.mean.device != dev:
        scaler.mean = scaler.mean.to(dev)
    if hasattr(scaler, "std") and scaler.std.device != dev:
        scaler.std = scaler.std.to(dev)
    if hasattr(scaler, "eps") and scaler.eps is not None and scaler.eps.device != dev:
        scaler.eps = scaler.eps.to(dev)

    return scaler.inverse(pred)



def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Utility to guarantee [B, T] shape for single-task or squeezed outputs."""
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x


# ---------------------------------------------------------
# Simple test harness (optional)
# ---------------------------------------------------------

if __name__ == "__main__":
    # Minimal sanity checks
    torch.manual_seed(0)
    B, T = 5, 3
    pred = torch.randn(B, T)
    targ = torch.randn(B, T)
    mask = torch.rand(B, T) > 0.3
    fid_idx = torch.randint(0, 4, (B,))
    fid_w = [1.0, 0.8, 0.6, 0.5]
    task_w = [1.0, 2.0, 1.0]

    l1 = loss_multitask_fidelity(pred=pred, target=targ, mask=mask, fid_idx=fid_idx, fid_loss_w=fid_w, task_weights=task_w)
    l2 = loss_multitask_fidelity(pred=pred, target=targ, mask=mask, fid_idx=fid_idx, fid_loss_w=fid_w, task_weights=None)
    print("Loss with task weights:", float(l1))
    print("Loss without task weights:", float(l2))

    m_all = masked_metrics_overall(pred, targ, mask)
    print("Overall metrics:", m_all)
