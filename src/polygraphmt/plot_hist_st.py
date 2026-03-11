#!/usr/bin/env python3
"""
Plot histogram(s) of predicted ensemble mean values from SINGLE-TASK output.

Expected input CSV (single-task ensemble output):
  - typically: smiles, mean_<prop>, std_<prop>
  - or: smiles, mean, std
  - or: smiles, pred_<prop> / pred (fallback)

This script auto-detects which column to plot and also writes:
  <out_dir>/summary_stats.csv

Usage:
  python scripts/plot_hist_st.py \
    --csv RESULTS_Single/cp/ensemble_cp_POLYINFO.csv \
    --prop cp \
    --out_dir results_hist_st/POLYINFO/cp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# Global font-size standard (IDENTICAL everywhere)
# -------------------------------------------------
def set_font_sizes():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 24,
        "axes.titlesize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "figure.titlesize": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# -------------------------------------------------
# Property label + unit map (EXACT keys you listed)
# -------------------------------------------------
def property_label_map() -> Dict[str, str]:
    return {
        # Thermal
        "tm":   r"Predicted $T_m$ (K)",
        "tg":   r"Predicted $T_g$ (K)",
        "td":   r"Predicted $\alpha_{\rm T}$ (m$^2$/s)",
        "tc":   r"Predicted $\kappa$ (W/m$\cdot$K)",
        "cp":   r"Predicted $C_p$ (J/kg$\cdot$K)",

        # Mechanical
        "young":    r"Predicted $E$ (GPa)",
        "shear":    r"Predicted $G$ (GPa)",
        "bulk":     r"Predicted $K$ (GPa)",
        "poisson":  r"Predicted $\nu$",

        # Transport
        "visc": r"Predicted $\eta$ (Pa$\cdot$s)",
        "dif":  r"Predicted $D$ (cm$^2$/s)",

        # Gas Permeability
        "phe":  r"Predicted $P_{\mathrm{He}}$ (Barrer)",
        "ph2":  r"Predicted $P_{\mathrm{H_2}}$ (Barrer)",
        "pco2": r"Predicted $P_{\mathrm{CO_2}}$ (Barrer)",
        "pn2":  r"Predicted $P_{\mathrm{N_2}}$ (Barrer)",
        "po2":  r"Predicted $P_{\mathrm{O_2}}$ (Barrer)",
        "pch4": r"Predicted $P_{\mathrm{CH_4}}$ (Barrer)",

        # Electronic / Optical
        "alpha":   r"Predicted $\alpha$ (a.u.)",
        "homo":    r"Predicted $E_{\mathrm{HOMO}}$ (eV)",
        "lumo":    r"Predicted $E_{\mathrm{LUMO}}$ (eV)",
        "bandgap": r"Predicted $E_g$ (eV)",
        "mu":      r"Predicted $\mu$ (Debye)",
        "etotal":  r"Predicted $E_{\mathrm{total}}$ (eV)",
        "ri":      r"Predicted $n$",
        "dc":      r"Predicted $\varepsilon$",
        "pe":      r"Predicted $\epsilon_r$",

        # Structural / Physical
        "rg":  r"Predicted $R_g$ (\AA)",
        "rho": r"Predicted $\rho$ (g/cm$^3$)",
    }


def _safe_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _choose_value_column(df: pd.DataFrame, prop: str) -> str:
    """
    Pick the best available column to histogram for a single-task ensemble file.

    Priority:
      1) mean_<prop>
      2) mean
      3) pred_<prop>
      4) pred
    """
    candidates = [f"mean_{prop}", "mean", f"pred_{prop}", "pred"]
    cols_lc = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want.lower() in cols_lc:
            return cols_lc[want.lower()]

    raise RuntimeError(
        f"Could not find a suitable prediction column. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _numeric_values(df: pd.DataFrame, col: str) -> np.ndarray:
    s = pd.to_numeric(df[col], errors="coerce")
    arr = s.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def plot_hist(values: np.ndarray, xlabel: str, out_png: Path, bins: int = 100, title: Optional[str] = None):
    if values.size == 0:
        raise ValueError("No finite numeric values to plot.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(values, bins=bins, edgecolor="black", linewidth=0.3, alpha=0.6)

    ymax = ax.get_ylim()[1] * 0.9
    ax.text(vmin, ymax, f"Min: {vmin:.3g}", color="red", ha="left", weight="bold")
    ax.text(vmax, ymax, f"Max: {vmax:.3g}", color="green", ha="right", weight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if title:
        ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.grid(True, linestyle="--", alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Plot histogram + stats for single-task ensemble predictions.")
    p.add_argument("--csv", required=True, help="Single-task ensemble CSV")
    p.add_argument("--prop", required=True, help="Property key (cp, tg, tc, ...)")
    p.add_argument("--out_dir", required=True, help="Output dir for PNG + stats CSV")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--title", default=None, help="Optional plot title")
    p.add_argument(
        "--stats_csv",
        default=None,
        help="Optional stats CSV path (default: <out_dir>/summary_stats.csv)",
    )
    return p


def main():
    args = build_argparser().parse_args()
    set_font_sizes()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)

    prop = _safe_key(args.prop)
    col = _choose_value_column(df, prop)

    label_map = property_label_map()
    xlabel = label_map.get(prop, f"Predicted {prop} (units)")

    values = _numeric_values(df, col)
    if values.size == 0:
        raise RuntimeError(f"No valid numeric values found in column '{col}' of {csv_path}")

    out_png = out_dir / f"hist_{prop}.png"
    plot_hist(values, xlabel=xlabel, out_png=out_png, bins=args.bins, title=args.title)

    # stats CSV (single row, but we keep a consistent schema)
    stats = pd.DataFrame([{
        "property": prop,
        "value_col": col,
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }])

    stats_csv = Path(args.stats_csv) if args.stats_csv else (out_dir / "summary_stats.csv")
    stats.to_csv(stats_csv, index=False)


if __name__ == "__main__":
    main()
