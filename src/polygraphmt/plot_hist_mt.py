#!/usr/bin/env python3
"""
Plot histogram(s) of predicted ensemble mean values from multi-task output.

Expected input CSV (ensemble output):
  - columns like: smiles, mean_<prop>, std_<prop>, ...
We plot histograms of mean_<prop>.

Also writes a summary CSV with min/max/median for each plotted property:
  <out_dir>/summary_stats.csv

Usage examples:
  # Plot all properties (all mean_* columns)
  python scripts/plot_hist_mt.py --csv RESULTS_G4/multitask_overall/ensemble_predictions_POLYINFO.csv --out_dir results_hist

  # Plot one property
  python scripts/plot_hist_mt.py --csv ... --out_dir results_hist --prop cp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
# Property label + unit map (edit anytime)
# Keys should match your column suffix in mean_<key>
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


def _safe_prop_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _find_mean_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("mean_")]


def _extract_prop_from_mean_col(mean_col: str) -> str:
    return mean_col.replace("mean_", "", 1)


def _numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Robust numeric conversion:
    - coerces strings to NaN
    - returns finite float array
    """
    s = pd.to_numeric(df[col], errors="coerce")
    arr = s.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def plot_histogram(
    values: np.ndarray,
    xlabel: str,
    out_png: Path,
    bins: int = 100,
    title: Optional[str] = None,
):
    if values.size == 0:
        raise ValueError("No finite numeric values to plot.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.hist(
        values,
        bins=bins,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
    )

    # annotate min/max
    ymax = ax.get_ylim()[1] * 0.9
    ax.text(vmin, ymax, f"Min: {vmin:.3g}", color="red", ha="left", weight="bold")
    ax.text(vmax, ymax, f"Max: {vmax:.3g}", color="green", ha="right", weight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    if title:
        ax.set_title(title)

    # axis styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.grid(True, linestyle="--", alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Plot histograms of ensemble mean predictions (multi-task).")
    p.add_argument("--csv", required=True, help="Ensemble CSV with mean_* columns.")
    p.add_argument("--out_dir", required=True, help="Output directory for PNGs + stats CSV.")
    p.add_argument("--prop", default=None, help="Plot only this property key (suffix of mean_<prop>).")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--title_prefix", default="", help="Optional prefix for plot titles.")
    p.add_argument(
        "--stats_csv",
        default=None,
        help="Optional output stats CSV path (default: <out_dir>/summary_stats.csv)",
    )
    return p


def main():
    args = build_argparser().parse_args()

    set_font_sizes()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # low_memory=False avoids mixed-type chunk inference warnings
    df = pd.read_csv(csv_path, low_memory=False)

    mean_cols = _find_mean_columns(df)
    if not mean_cols:
        raise RuntimeError(f"No mean_* columns found in: {csv_path}")

    label_map = property_label_map()

    # Decide which columns to plot
    if args.prop:
        prop = _safe_prop_key(args.prop)
        col = f"mean_{prop}"
        if col not in df.columns:
            # try case-insensitive match
            candidates = {c.lower(): c for c in mean_cols}
            if col.lower() in candidates:
                col = candidates[col.lower()]
            else:
                raise RuntimeError(f"Requested prop '{args.prop}' not found. Available: {mean_cols}")
        cols_to_plot = [col]
    else:
        cols_to_plot = mean_cols

    stats_rows = []

    # Plot each + collect stats
    for mean_col in cols_to_plot:
        prop = _extract_prop_from_mean_col(mean_col)
        prop_key = _safe_prop_key(prop)

        xlabel = label_map.get(prop_key, f"Predicted {prop} (units)")
        title = f"{args.title_prefix}{prop}" if args.title_prefix else None

        values = _numeric_series(df, mean_col)
        if values.size == 0:
            # skip silently (or you can print a warning)
            continue

        out_png = out_dir / f"hist_{prop_key}.png"

        plot_histogram(
            values=values,
            xlabel=xlabel,
            out_png=out_png,
            bins=args.bins,
            title=title,
        )

        stats_rows.append({
            "property": prop_key,
            "mean_col": mean_col,
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        })

    # Write stats CSV
    stats_csv = Path(args.stats_csv) if args.stats_csv else (out_dir / "summary_stats.csv")
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)


if __name__ == "__main__":
    main()
