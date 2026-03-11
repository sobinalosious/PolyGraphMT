"""Training entry point for the PolyGraphMT package."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader as PYGDataLoader
from tqdm import tqdm

import optuna
from optuna.pruners import MedianPruner

from .data_builder import TargetScaler, build_dataset_from_dir
from .model import build_model
from .plot_utils import generate_all_plots
from .utils import (
    seed_everything,
    to_device,
    loss_multitask_fidelity,
    masked_metrics_overall,
    masked_metrics_per_task,
    masked_metrics_by_fidelity,
    exp_weight_at_epoch,
    make_fid_loss_weights,
    apply_inverse_transform,
)


def _pick_selection_score(metrics: Dict[str, float], args: argparse.Namespace) -> tuple[float, str]:
    """
    Choose the validation selection score safely.
    Priority:
      1) select_task_<metric>  (if provided and present/finite)
      2) select_fidelity_<metric> (if provided and present/finite)
      3) overall_<metric>  (fallback)
    Returns (score, key_used).
    """
    import numpy as _np  # local import to avoid header churn
    base_key = f"overall_{args.metric}"

    # Prefer task if requested and available
    if getattr(args, "select_task", None):
        k = f"{args.select_task}_{args.metric}"
        if k in metrics and _np.isfinite(metrics[k]):
            return metrics[k], k

    # Next prefer fidelity if requested and available
    if getattr(args, "select_fidelity", None):
        k = f"{args.select_fidelity}_{args.metric}"
        if k in metrics and _np.isfinite(metrics[k]):
            return metrics[k], k

    # Fallback
    return metrics.get(base_key, float("nan")), base_key

def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None or s == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Multitask Multi-fidelity GNN with Optuna")

    # ---------------- Data ----------------
    p.add_argument("--root_dir", type=str, required=True, help="Data root containing target/fidelity CSVs")
    p.add_argument("--targets", type=_csv_list, required=True, help="Comma separated targets, e.g. cp,tg,tc,rho")
    p.add_argument("--fidelities", type=_csv_list, default="exp,dft,md,gc", help="Comma separated fidelities")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--split_file", type=str, default=None, help="Path to persist or reuse SMILES splits JSON")

    # ---------------- Model ----------------
    p.add_argument("--gnn_type", type=str, default="gine", choices=["gine", "gin", "gcn"])
    p.add_argument("--gnn_emb_dim", type=int, default=256)
    p.add_argument("--gnn_layers", type=int, default=5)
    p.add_argument("--gnn_norm", type=str, default="batch", choices=["batch", "layer", "none"])
    p.add_argument("--gnn_readout", type=str, default="mean", choices=["mean", "sum", "max"])
    p.add_argument("--gnn_act", type=str, default="relu", choices=["relu", "gelu", "silu", "leaky_relu"])
    p.add_argument("--gnn_dropout", type=float, default=0.0)
    p.add_argument("--gnn_residual", action="store_true", default=True)
    p.add_argument("--no_gnn_residual", action="store_true")

    # ---------------- Conditioning ----------------
    p.add_argument("--fid_emb_dim", type=int, default=64)
    p.add_argument("--use_film", action="store_true", default=True)
    p.add_argument("--no_film", action="store_true")
    p.add_argument("--use_task_embed", action="store_true", default=True)
    p.add_argument("--no_task_embed", action="store_true")
    p.add_argument("--task_emb_dim", type=int, default=32)

    # ---------------- Heads ----------------
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--head_depth", type=int, default=2)
    p.add_argument("--head_act", type=str, default="relu", choices=["relu", "gelu", "silu", "leaky_relu"])
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--heteroscedastic", action="store_true", default=False)

    # ---------------- Regularization ----------------
    p.add_argument("--fid_emb_l2", type=float, default=0.0)
    p.add_argument("--task_emb_l2", type=float, default=0.0)

    # ---------------- Train ----------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # ---------------- Loss weights (equal by default) ----------------
    p.add_argument("--task_weights", type=_csv_floats, default=None, help="Comma separated weights per target (default equal)")
    p.add_argument("--fid_loss_w", type=_csv_floats, default=None, help="Comma separated base weights per fidelity order in dataset")

    # ---------------- Curriculum for EXP ----------------
    p.add_argument("--exp_weight_schedule", type=str, default="none", choices=["none", "linear", "cosine"])
    p.add_argument("--exp_weight_start", type=float, default=0.6)
    p.add_argument("--exp_weight_end", type=float, default=1.0)

    # ---------------- Checkpointing / metric ----------------
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae", "r2"], help="Primary val metric to select best")

    # ---------------- Selection focus ----------------
    p.add_argument("--select_task", type=str, default=None, help="If set, choose best by <task>_<metric>")
    p.add_argument("--select_fidelity", type=str, default=None, help="If set, choose best by <fid>_<metric>")

    # ---------------- Sampler ----------------
    p.add_argument("--sampler", type=str, default="uniform", choices=["uniform", "balance_fidelity"], help="Train loader sampler")

    # ---------------- Task-uncertainty weighting ----------------
    p.add_argument("--task_uncertainty", action="store_true", help="Enable learned homoscedastic task-uncertainty weighting")

    # ---------------- Optuna ----------------
    p.add_argument("--n_trials", type=int, default=0, help="0 to skip HPO and run a single config")
    p.add_argument("--pruner", type=str, default="median", choices=["none", "median"])
    p.add_argument("--study_name", type=str, default=None)

    # ---- Subsample TRAIN for a specific (task,fidelity) ----
    p.add_argument("--subsample_target", type=str, default=None, help="Property to subsample in TRAIN only (e.g., cp)")
    p.add_argument("--subsample_fidelity", type=str, default=None, help="Fidelity to subsample in TRAIN only (e.g., gc)")
    p.add_argument("--subsample_pct", type=float, default=1.0, help="Fraction (0<..<=1) of TRAIN to keep for the specified (task,fidelity)")
    p.add_argument("--subsample_seed", type=int, default=137, help="Seed for deterministic subsampling by SMILES")
    return p


# --------------- Training / eval helpers ---------------

@dataclass
class TrainState:
    model: nn.Module
    optimizer: optim.Optimizer
    scaler: Optional[torch.cuda.amp.GradScaler]
    device: torch.device


def make_dataloaders(args: argparse.Namespace, train_ds, val_ds, test_ds):
    from torch.utils.data import WeightedRandomSampler

    if args.sampler == "balance_fidelity":
        # Inverse-frequency weights per fidelity (on train split)
        fid_counts = train_ds.rows["fid_idx_local"].value_counts().to_dict()
        weights = train_ds.rows["fid_idx_local"].map(lambda k: 1.0 / max(1, fid_counts.get(k, 1))).to_numpy().astype("float64")
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = PYGDataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, pin_memory=True)
    else:
        train_loader = PYGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_loader = PYGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = PYGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model_and_optim(args, train_ds):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_residual = args.gnn_residual and not args.no_gnn_residual
    use_film = args.use_film and not args.no_film
    use_task_embed = args.use_task_embed and not args.no_task_embed

    model = build_model(
        in_dim_node=train_ds.in_dim_node,
        in_dim_edge=train_ds.in_dim_edge,
        task_names=args.targets,
        num_fids=len(train_ds.fids),
        gnn_type=args.gnn_type,
        gnn_emb_dim=args.gnn_emb_dim,
        gnn_layers=args.gnn_layers,
        gnn_norm=args.gnn_norm,
        gnn_readout=args.gnn_readout,
        gnn_act=args.gnn_act,
        gnn_dropout=args.gnn_dropout,
        gnn_residual=use_residual,
        fid_emb_dim=args.fid_emb_dim,
        use_film=use_film,
        use_task_embed=use_task_embed,
        task_emb_dim=args.task_emb_dim,
        head_hidden=args.head_hidden,
        head_depth=args.head_depth,
        head_act=args.head_act,
        head_dropout=args.head_dropout,
        heteroscedastic=args.heteroscedastic,
        fid_emb_l2=args.fid_emb_l2,
        task_emb_l2=args.task_emb_l2,
        use_task_uncertainty=args.task_uncertainty,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and not args.no_amp and device.type == "cuda"))
    return TrainState(model=model, optimizer=optimizer, scaler=scaler, device=device)


def run_epoch(state: TrainState, loader, args, epoch: int, total_epochs: int, fids: Sequence[str]):
    model = state.model
    model.train()
    running_loss = 0.0
    n_batches = 0

    # EXP curriculum
    exp_w = exp_weight_at_epoch(epoch, total_epochs, args.exp_weight_schedule, args.exp_weight_start, args.exp_weight_end)
    # If schedule is used but no base weights provided, reduce non-EXP fidelities by default
    base_w = args.fid_loss_w
    if (base_w is None) and (args.exp_weight_schedule != "none"):
        base_w = [(1.0 if f.lower() == "exp" else 0.4) for f in fids]
    fid_loss_w = make_fid_loss_weights(fids=fids, base_weights=base_w, exp_weight=exp_w)

    pbar = tqdm(loader, desc=f"Train e{epoch+1}/{total_epochs}", leave=False)
    for batch in pbar:
        batch = to_device(batch, state.device)
        state.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(state.scaler is not None and state.scaler.is_enabled())):
            out = model(batch)
            pred = out["pred"]                  # [B, T]
            logvar = out.get("logvar", None)

            # --- shape guard (important) ---
            y = batch.y
            m = batch.y_mask
            if y.dim() == 1:
                y = y.view(pred.size(0), pred.size(1))
            if m.dim() == 1:
                m = m.view(pred.size(0), pred.size(1))
            # --------------------------------

            loss = loss_multitask_fidelity(
                pred=pred,
                target=y,
                mask=m,
                fid_idx=batch.fid_idx.view(-1),
                fid_loss_w=fid_loss_w,
                task_weights=args.task_weights,  # None => equal importance (default)
                hetero_logvar=logvar,
                task_log_sigma2=(getattr(model, "task_log_sigma2", None) if getattr(args, "task_uncertainty", False) else None),
            )

            loss = loss + model.regularization_loss()

        if state.scaler is not None and state.scaler.is_enabled():
            state.scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                state.scaler.unscale_(state.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            state.scaler.step(state.optimizer)
            state.scaler.update()
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            state.optimizer.step()

        running_loss += float(loss.detach().cpu())
        n_batches += 1
        pbar.set_postfix({"loss": running_loss / max(n_batches, 1), "exp_w": exp_w})

    return running_loss / max(n_batches, 1), {"exp_weight": exp_w}


@torch.no_grad()
def evaluate(state: TrainState, loader, args, scaler: TargetScaler,
             fid_names: Sequence[str], task_names: Sequence[str]) -> Dict[str, float]:
    model = state.model
    model.eval()

    all_pred, all_y, all_mask, all_fid, all_logvar = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, state.device)
        out = model(batch)
        all_pred.append(out["pred"].cpu())          # [b, T] (normalized / transformed space)
        all_y.append(batch.y.cpu())                 # [b, T] or [b*T]
        all_mask.append(batch.y_mask.cpu())         # [b, T] or [b*T]
        all_fid.append(batch.fid_idx.cpu())         # [b, 1] (collated)
        lv = out.get("logvar", None)
        if lv is not None:
            all_logvar.append(lv.detach().cpu())

    pred = torch.cat(all_pred, dim=0)              # [B, T]
    y    = torch.cat(all_y, dim=0)                 # [B, T] or [B*T]
    mask = torch.cat(all_mask, dim=0)              # [B, T] or [B*T]
    fid_idx = torch.cat(all_fid, dim=0).view(-1)   # [B]

    # --- shape guard (ensure [B, T]) ---
    if y.dim() == 1:
        y = y.view(pred.size(0), pred.size(1))
    if mask.dim() == 1:
        mask = mask.view(pred.size(0), pred.size(1))
    # -----------------------------------

    # -------- Metrics in normalized / transformed space (as before) --------
    overall = masked_metrics_overall(pred, y, mask)
    per_task = masked_metrics_per_task(pred, y, mask, task_names)
    by_fid   = masked_metrics_by_fidelity(pred, y, mask, fid_idx, fid_names, task_names)

    metrics: Dict[str, float] = {}
    for k, v in overall.items():
        metrics[f"overall_{k}"] = v
    for t, mv in per_task.items():
        for k, v in mv.items():
            metrics[f"{t}_{k}"] = v
    for f, mv in by_fid.items():
        for k, v in mv.items():
            metrics[f"{f}_{k}"] = v

    # -------- Metrics in ORIGINAL physical units (after inverse transform) --------
    # apply_inverse_transform handles per-task identity/log based on scaler
    pred_orig = apply_inverse_transform(pred, scaler)
    y_orig    = apply_inverse_transform(y, scaler)

    overall_o = masked_metrics_overall(pred_orig, y_orig, mask)
    per_task_o = masked_metrics_per_task(pred_orig, y_orig, mask, task_names)
    by_fid_o   = masked_metrics_by_fidelity(pred_orig, y_orig, mask, fid_idx, fid_names, task_names)

    for k, v in overall_o.items():
        metrics[f"overall_{k}_orig"] = v
    for t, mv in per_task_o.items():
        for k, v in mv.items():
            metrics[f"{t}_{k}_orig"] = v
    for f, mv in by_fid_o.items():
        for k, v in mv.items():
            metrics[f"{f}_{k}_orig"] = v

    # -------- Uncertainty diagnostics (kept in normalized space; PI width in original) --------
    if len(all_logvar) > 0:
        logvar = torch.cat(all_logvar, dim=0)  # [B, T]
        if logvar.dim() == 1:
            logvar = logvar.view(pred.size(0), pred.size(1))
        var_n = torch.exp(logvar).clamp_min(1e-12)
        std_n = torch.sqrt(var_n)

        # Normalized-space NLL
        err_n = pred - y
        nll_mat = 0.5 * ( (err_n**2) / var_n + logvar + math.log(2*math.pi) )  # [B, T]

        # 95% coverage; PI width in original units via linearized mapping
        z95 = 1.959963984540054
        std_orig = std_n * scaler.std.view(1, -1)  # NOTE: exact only for identity transform
        width95 = 2.0 * z95 * std_orig

        m_all = mask
        n_all = m_all.sum().item()
        if n_all > 0:
            metrics["overall_nll"] = float(nll_mat[m_all].mean().item())
            metrics["overall_picp95"] = ((err_n.abs() <= z95 * std_n) & m_all).sum().item() / n_all
            metrics["overall_pi95w"] = float(width95[m_all].mean().item())

        # Per-task
        T = pred.size(1)
        for t_idx, tname in enumerate(task_names):
            mt = m_all[:, t_idx]
            n_t = mt.sum().item()
            if n_t > 0:
                metrics[f"{tname}_nll"] = float(nll_mat[:, t_idx][mt].mean().item())
                metrics[f"{tname}_picp95"] = ((err_n[:, t_idx].abs() <= z95 * std_n[:, t_idx]) & mt).sum().item() / n_t
                metrics[f"{tname}_pi95w"] = float(width95[:, t_idx][mt].mean().item())

        # By fidelity (across tasks)
        fid_mat = fid_idx.view(-1, 1).expand(pred.size(0), pred.size(1))  # [B,T]
        for f_id, f_name in enumerate(fid_names):
            mf = (fid_mat == f_id) & m_all
            n_f = mf.sum().item()
            if n_f > 0:
                metrics[f"{f_name}_nll"] = float(nll_mat[mf].mean().item())
                metrics[f"{f_name}_picp95"] = ((err_n.abs() <= z95 * std_n) & mf).sum().item() / n_f
                metrics[f"{f_name}_pi95w"] = float(width95[mf].mean().item())

    return metrics


@torch.no_grad()
def predict_and_save(state: TrainState, loader, args, scaler: TargetScaler,
                     fid_names: Sequence[str], task_names: Sequence[str],
                     split: str, out_dir: Path):
    model = state.model
    model.eval()

    rows = []
    for batch in loader:
        batch = to_device(batch, state.device)
        out = model(batch)

        pred_n = out["pred"].detach().cpu()     # [b, T]
        y_n    = batch.y.detach().cpu()         # [b, T] or [b*T]
        mask   = batch.y_mask.detach().cpu()    # [b, T] or [b*T]
        fid_idx_vec = batch.fid_idx.view(-1).detach().cpu().numpy()

        # shape guard
        if y_n.dim() == 1:
            y_n = y_n.view(pred_n.size(0), pred_n.size(1))
        if mask.dim() == 1:
            mask = mask.view(pred_n.size(0), pred_n.size(1))

        # back to original units
        pred = apply_inverse_transform(pred_n, scaler).cpu()
        y    = apply_inverse_transform(y_n, scaler).cpu()

        # optional std/PI if heteroscedastic
        logvar = out.get("logvar", None)
        if logvar is not None:
            logvar = logvar.detach().cpu()
            if logvar.dim() == 1:
                logvar = logvar.view(pred_n.size(0), pred_n.size(1))
            std_n = torch.sqrt(torch.exp(logvar))
            std_orig = std_n * scaler.std.view(1, -1)
            z95 = 1.959963984540054
            lo95 = pred - z95 * std_orig
            hi95 = pred + z95 * std_orig
        else:
            std_orig = lo95 = hi95 = None

        smiles_list = getattr(batch, "smiles", None)
        fid_str_list = getattr(batch, "fid_str", None)
        if isinstance(smiles_list, tuple):
            smiles_list = list(smiles_list)
        if isinstance(fid_str_list, tuple):
            fid_str_list = list(fid_str_list)

        for i in range(pred.shape[0]):
            fid_i = int(fid_idx_vec[i])
            row = {
                "split": split,
                "smiles": smiles_list[i] if isinstance(smiles_list, list) else None,
                "fid": (fid_str_list[i] if isinstance(fid_str_list, list) and i < len(fid_str_list) else fid_names[fid_i]),
                "fid_idx": fid_i,
            }
            for t, name in enumerate(task_names):
                row[f"y_{name}"] = float(y[i, t]) if bool(mask[i, t]) else np.nan
                row[f"pred_{name}"] = float(pred[i, t])
                row[f"mask_{name}"] = bool(mask[i, t])
                if std_orig is not None:
                    row[f"std_{name}"] = float(std_orig[i, t])
                    row[f"pi95_lo_{name}"] = float(lo95[i, t])
                    row[f"pi95_hi_{name}"] = float(hi95[i, t])
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = out_dir / f"predictions_{split}.csv"
    df.to_csv(out_path, index=False)


# --------------- Optuna objective (minimal, strong priors) ---------------

def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    # Copy CLI args and override only the tuned subset
    args = argparse.Namespace(**vars(base_args))

    # ====== High-impact search space (TOP-10 knobs) ======
    # 1) Capacity
    args.gnn_emb_dim  = trial.suggest_categorical("gnn_emb_dim", [256, 384, 512, 640, 768])

    # 2) Depth
    args.gnn_layers   = trial.suggest_int("gnn_layers", 3, 7)

    # 3) Regularization
    args.gnn_dropout  = trial.suggest_float("gnn_dropout", 0.0, 0.40)

    # 4) Head capacity
    args.head_hidden  = trial.suggest_categorical("head_hidden", [256, 384, 512, 768])
    args.head_depth   = trial.suggest_int("head_depth", 1, 3)

    # 5) Fidelity conditioning (FiLM)
    args.use_film     = trial.suggest_categorical("use_film", [True, False])
    # If FiLM is off, force fid_emb_dim=0 so the model doesn't depend on it
    if args.use_film:
        args.fid_emb_dim = trial.suggest_categorical("fid_emb_dim", [32, 64, 96])
    else:
        args.fid_emb_dim = 0

    # 6) Normalization
    args.gnn_norm     = trial.suggest_categorical("gnn_norm", ["batch", "layer"])

    # 7) Activation (GNN)
    args.gnn_act      = trial.suggest_categorical("gnn_act", ["relu", "gelu", "silu"])

    # 8) Convolution type
    args.gnn_type     = trial.suggest_categorical("gnn_type", ["gine", "gin"])

    # 9) Optimizer knobs
    args.lr           = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)

    # 10) Readout
    #args.gnn_readout  = trial.suggest_categorical("gnn_readout", ["mean", "sum", "max"])
    args.gnn_readout ="mean"
    # ====== Fixed, sensible defaults during HPO ======
    args.gnn_residual    = True
    args.use_task_embed  = True
    args.task_emb_dim    = 32
    args.head_act        = "relu"
    args.head_dropout    = 0.0
    args.heteroscedastic = False  # focus on point accuracy during HPO

    # Mild EXP curriculum
    args.exp_weight_schedule = "cosine"
    args.exp_weight_start    = 0.6
    args.exp_weight_end      = 1.0

    # ====== Build data/model and run ======
    seed_everything(args.seed)
    train_ds, val_ds, _, scaler = build_dataset_from_dir(
        args.root_dir,
        targets=args.targets,
        fidelities=args.fidelities,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        save_splits_path=args.split_file,
        subsample_target=args.subsample_target,
        subsample_fidelity=args.subsample_fidelity,
        subsample_pct=args.subsample_pct,
        subsample_seed=args.subsample_seed,
    )

    train_loader, val_loader, _ = make_dataloaders(args, train_ds, val_ds, val_ds)
    state = build_model_and_optim(args, train_ds)

    best_val = math.inf if args.metric in ("rmse", "mae") else -math.inf
    direction_min = args.metric in ("rmse", "mae")

    pruner_step = 0
    for epoch in range(args.epochs):
        run_epoch(state, train_loader, args, epoch, args.epochs, fids=train_ds.fids)
        # Use the *training* fidelity list so keys exist even if VAL misses a fidelity
        val_metrics = evaluate(state, val_loader, args, scaler,
                            fid_names=train_ds.fids, task_names=args.targets)

        # Safe selection (falls back to overall if requested key missing/NaN)
        score, used_key = _pick_selection_score(val_metrics, args)
        # Optional: print which key was used
#        print(f"[HPO] selecting by {used_key}: {score:.4f}")


        # Report to Optuna (minimize; flip sign if maximizing)
        trial.report(score if direction_min else -score, step=pruner_step)
        pruner_step += 1
        if trial.should_prune():
            raise optuna.TrialPruned()

        improved = (score < best_val) if direction_min else (score > best_val)
        if improved:
            best_val = score

    return best_val if direction_min else -best_val

# --------------- Main training path ---------------

def train_single_run(args: argparse.Namespace):
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds, val_ds, test_ds, scaler = build_dataset_from_dir(
        root_dir=args.root_dir,
        targets=args.targets,
        fidelities=args.fidelities,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        save_splits_path=args.split_file,
        subsample_target=args.subsample_target,
        subsample_fidelity=args.subsample_fidelity,
        subsample_pct=args.subsample_pct,
        subsample_seed=args.subsample_seed,
    )

    train_loader, val_loader, test_loader = make_dataloaders(args, train_ds, val_ds, test_ds)
    state = build_model_and_optim(args, train_ds)
    device = state.device

    best_metric = math.inf if args.metric in ("rmse", "mae") else -math.inf
    direction_min = args.metric in ("rmse", "mae")
    best_ckpt = out_dir / "best.pt"

    history = []
    for epoch in range(args.epochs):
        train_loss, extra = run_epoch(state, train_loader, args, epoch, args.epochs, fids=train_ds.fids)

        # Use training fidelity names to keep keys stable even if VAL has no rows for a fidelity
        val_metrics = evaluate(state, val_loader, args, scaler,
                       fid_names=train_ds.fids, task_names=args.targets)

        # Safe selection (falls back to overall if requested key missing/NaN)
        val_score, used_key = _pick_selection_score(val_metrics, args)
        # Optional: print which key was used
        # print(f"[train] selecting by {used_key}: {val_score:.4f}")


        history.append({"epoch": epoch, "train_loss": float(train_loss), **{k: float(v) for k, v in val_metrics.items()}, **extra})

        improved = (val_score < best_metric) if direction_min else (val_score > best_metric)
        if improved:
            best_metric = val_score
            torch.save({"model": state.model.state_dict(), "args": vars(args)}, best_ckpt)

        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_{args.metric}={val_score:.4f} exp_w={extra['exp_weight']:.3f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best for final reporting/saves
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        state.model.load_state_dict(ckpt["model"])

    # Save scaler for inference
    torch.save({
        "mean": scaler.mean.cpu(),
        "std": scaler.std.cpu(),
        "targets": args.targets,
        "transforms": getattr(scaler, "transforms", None),
        "eps": getattr(scaler, "eps", None),
    }, out_dir / "target_scaler.pt")


    # Save metrics + predictions per split
    for split_name, loader, ds in [("train", train_loader, train_ds), ("val", val_loader, val_ds), ("test", test_loader, test_ds)]:
        metrics = evaluate(state, loader, args, scaler, fid_names=ds.fids, task_names=args.targets)
        with open(out_dir / f"metrics_{split_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        predict_and_save(state, loader, args, scaler, fid_names=ds.fids, task_names=args.targets, split=split_name, out_dir=out_dir)

    # >>> parity plots & metric grids
    generate_all_plots(out_dir=out_dir, task_names=args.targets, heatmaps=False)


def main():
    args = build_argparser().parse_args()

    # Ensure lists regardless of how CLI passed them
    if isinstance(args.targets, str):
        args.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        args.targets = [str(t).strip() for t in args.targets]

    if isinstance(args.fidelities, str):
        args.fidelities = [f.strip() for f in args.fidelities.split(",") if f.strip()]
    else:
        args.fidelities = [str(f).strip() for f in args.fidelities]

    # Normalize booleans from paired flags
    if args.no_amp:
        args.amp = False
    if args.no_gnn_residual:
        args.gnn_residual = False
    if args.no_film:
        args.use_film = False
    if args.no_task_embed:
        args.use_task_embed = False

    # Run HPO or single run
    if args.n_trials and args.n_trials > 0:
        pruner = None if args.pruner == "none" else MedianPruner(n_warmup_steps=max(5, args.epochs // 10))
        study = optuna.create_study(
            direction=("minimize" if args.metric in ("rmse", "mae") else "maximize"),
            study_name=args.study_name,
            pruner=pruner,
        )
        study.optimize(lambda tr: objective(tr, args), n_trials=args.n_trials, show_progress_bar=True)

        results = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        }
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out_dir) / "optuna_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Retrain best config
        for k, v in study.best_params.items():
            setattr(args, k, v)
        args.out_dir = str(Path(args.out_dir) / "best_run")
        train_single_run(args)
    else:
        train_single_run(args)


if __name__ == "__main__":
    main()
