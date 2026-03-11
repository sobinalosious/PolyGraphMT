# plot_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import erfinv


# =========================
# Publication styling
# =========================
def _set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
    })


# =========================
# Helpers
# =========================
SPLIT_COLORS = {
    "train": "#1b9e77",  # green
    "val":   "#d95f02",  # orange
    "test":  "#7570b3",  # purple/blue
}

def _finite_mask(*arrs) -> np.ndarray:
    m = np.ones_like(np.asarray(arrs[0], dtype=float), dtype=bool)
    for a in arrs:
        aa = np.asarray(a)
        m &= np.isfinite(aa)
    return m

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mask = _finite_mask(y_true, y_pred)
    y = y_true[mask]
    yhat = y_pred[mask]
    if y.size == 0:
        return np.nan, np.nan, np.nan
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae  = float(np.mean(np.abs(yhat - y)))
    ybar = float(np.mean(y))
    sst  = float(np.sum((y - ybar) ** 2))
    sse  = float(np.sum((y - yhat) ** 2))
    r2   = float(1.0 - (sse / sst if sst > 0 else np.nan))
    return r2, rmse, mae

def _safe_scatter(ax, x, y, **kw):
    # Consistent markers/alpha/size for publication look
    kw.setdefault("s", 18)
    kw.setdefault("alpha", 0.9)
    kw.setdefault("linewidths", 0.4)
    kw.setdefault("edgecolors", "black")
    return ax.scatter(x, y, **kw)

def _nominal_grid() -> np.ndarray:
    return np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.98], dtype=float)

def _z_from_nominal(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.sqrt(2.0) * erfinv(a)


# =========================
# Individual calibration/ parity (kept)
# =========================
def _draw_calibration(df: pd.DataFrame, task: str, out_dir: Path, split: str):
    std_col = f"std_{task}"
    if std_col not in df.columns:
        return
    y = df.get(f"y_{task}", pd.Series([np.nan] * len(df))).to_numpy()
    yhat = df.get(f"pred_{task}", pd.Series([np.nan] * len(df))).to_numpy()
    std = df.get(std_col, pd.Series([np.nan] * len(df))).to_numpy()
    mask = df.get(f"mask_{task}", pd.Series([False] * len(df))).astype(bool).to_numpy()

    good = mask & _finite_mask(y, yhat, std) & (std > 0)
    if good.sum() == 0:
        return

    err = np.abs(yhat - y) / std
    alphas = _nominal_grid()
    z = _z_from_nominal(alphas)
    emp = [(err[good] <= zi).mean() for zi in z]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="black")
    ax.plot(alphas, emp, marker="o")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Calibration ({task}, {split})")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{split}_{task}.png")
    plt.close(fig)


def _draw_parity_single(df: pd.DataFrame, task: str, out_dir: Path, split: str):
    """Backward-compatible single-split parity with metrics."""
    y = df.get(f"y_{task}", pd.Series([np.nan] * len(df))).to_numpy()
    yhat = df.get(f"pred_{task}", pd.Series([np.nan] * len(df))).to_numpy()
    std = df.get(f"std_{task}", None)
    if std is not None:
        std = df[f"std_{task}"].to_numpy()
    mask = df.get(f"mask_{task}", pd.Series([False] * len(df))).astype(bool).to_numpy()

    good = mask & _finite_mask(y, yhat)
    if std is not None:
        good &= np.isfinite(std)
    if good.sum() == 0:
        return

    r2, rmse, mae = _metrics(y[good], yhat[good])

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ymin = np.nanmin(np.concatenate([y[good], yhat[good]]))
    ymax = np.nanmax(np.concatenate([y[good], yhat[good]]))
    pad = 0.02 * (ymax - ymin + 1e-8)
    lims = [ymin - pad, ymax + pad]

    ax.plot(lims, lims, ls="--", lw=1.2, color="black", alpha=0.9)
    if std is None:
        _safe_scatter(ax, y[good], yhat[good], color=SPLIT_COLORS.get(split, "gray"))
    else:
        eb = ax.errorbar(y[good], yhat[good], yerr=std[good], fmt="none", ecolor="0.5", elinewidth=0.6, alpha=0.6)
        _safe_scatter(ax, y[good], yhat[good], color=SPLIT_COLORS.get(split, "gray"))

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity ({task}, {split})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    for s in ax.spines.values():
        s.set_linewidth(1.2)

    txt = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", alpha=0.95))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"parity_{split}_{task}.png")
    plt.close(fig)


# =========================
# NEW: Combined parity (train+val+test)
# =========================
def _draw_parity_all_splits(dfs_by_split: Dict[str, pd.DataFrame], task: str, out_dir: Path):
    """
    One publication-quality parity plot overlaying train/val/test with three colors.
    Adds a metrics textbox listing R² / RMSE / MAE for each split.
    Shows vertical error bars if std_<task> exists.
    """
    # Gather arrays per split
    data = {}
    for split, df in dfs_by_split.items():
        if df is None:
            continue
        y = df.get(f"y_{task}", pd.Series([np.nan] * len(df))).to_numpy()
        yhat = df.get(f"pred_{task}", pd.Series([np.nan] * len(df))).to_numpy()
        std = df.get(f"std_{task}", None)
        std = df[f"std_{task}"].to_numpy() if std is not None else None
        mask = df.get(f"mask_{task}", pd.Series([False] * len(df))).astype(bool).to_numpy()

        good = mask & _finite_mask(y, yhat)
        if std is not None:
            good &= np.isfinite(std)
        if good.sum() == 0:
            continue
        data[split] = dict(y=y[good], yhat=yhat[good], std=(std[good] if std is not None else None))

    if not data:
        return

    # Global limits
    all_y = np.concatenate([v["y"] for v in data.values()])
    all_yhat = np.concatenate([v["yhat"] for v in data.values()])
    ymin = float(np.nanmin(np.concatenate([all_y, all_yhat])))
    ymax = float(np.nanmax(np.concatenate([all_y, all_yhat])))
    pad = 0.02 * (ymax - ymin + 1e-8)
    lims = [ymin - pad, ymax + pad]

    # Plot
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(lims, lims, ls="--", lw=1.2, color="black", alpha=0.9, label="y = x")

    # Draw per split
    legend_elems = []
    lines = []
    for split in ["train", "val", "test"]:
        if split not in data:
            continue
        col = SPLIT_COLORS[split]
        y = data[split]["y"]
        yhat = data[split]["yhat"]
        std = data[split]["std"]

        if std is not None:
            ax.errorbar(y, yhat, yerr=std, fmt="none", ecolor=col, elinewidth=0.6, alpha=0.4, capsize=0)

        _safe_scatter(ax, y, yhat, color=col, label=split.capitalize())

        r2, rmse, mae = _metrics(y, yhat)
        lines.append(f"{split.capitalize()}:  $R^2$={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    # Cosmetics
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity (All splits, {task})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax.legend(loc="lower right")

    # Metrics textbox (multi-line, one line per split)
    txt = "\n".join(lines)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3", alpha=0.95))

    fig.tight_layout()
    out_path = out_dir / f"parity_all_{task}.png"
    fig.savefig(out_path)
    plt.close(fig)


# =========================
# Public entry point
# =========================
def generate_all_plots(out_dir: str | Path, task_names: List[str], heatmaps: bool = False):
    """
    - Saves individual parity & (if available) calibration plots for val/test.
    - Additionally saves a **combined parity** plot per task overlaying train+val+test
      with distinct colors and per-split metrics.
    """
    _set_pub_style()
    out_dir = Path(out_dir)

    # Load what exists
    dfs: Dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        csv = out_dir / f"predictions_{split}.csv"
        dfs[split] = pd.read_csv(csv) if csv.exists() else None

    # Individual (keep previous behavior)
    for split in ["val", "test"]:
        df = dfs.get(split)
        if df is None:
            continue
        for t in task_names:
            if f"std_{t}" in df.columns:
                _draw_calibration(df, t, out_dir, split)
            _draw_parity_single(df, t, out_dir, split)

    # Combined overlay: train+val+test
    for t in task_names:
        _draw_parity_all_splits(dfs, t, out_dir)
