from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER_PREFIX = "RESULTS_"
METRICS = ["r2", "rmse", "mae"]
VARIANTS = [(metric, False) for metric in METRICS] + [(metric, True) for metric in METRICS]
PATTERN_STD = re.compile(r"^([A-Za-z0-9]+)_(r2|rmse|mae)$")
PATTERN_ORIG = re.compile(r"^([A-Za-z0-9]+)_(r2|rmse|mae)_orig$")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Summarize RESULTS_* folders into CSVs and bar charts.")
    parser.add_argument("--root", default=".", help="Directory containing RESULTS_* folders.")
    parser.add_argument(
        "--folder_prefix",
        default=FOLDER_PREFIX,
        help="Folder prefix to scan for result groups.",
    )
    parser.add_argument(
        "--out_dir",
        default="RESULTS_SUMMARY",
        help="Output directory for summary CSVs and figures.",
    )
    return parser


def load_seed_summary(folder: Path) -> Optional[List[dict]]:
    for filename in ("seed_summary.json", "seed_summary.jason"):
        path = folder / filename
        if path.exists():
            with path.open("r") as handle:
                return json.load(handle)
    print(f"[warn] Skipping {folder.name}: no seed_summary.(json|jason)")
    return None


def extract(records: List[dict]) -> Dict[Tuple[str, str, bool], Tuple[float, float]]:
    out: Dict[Tuple[str, str, bool], Tuple[float, float]] = {}
    for record in records:
        metric_name = record.get("metric", "")
        mean = record.get("mean")
        std = record.get("std")
        if mean is None:
            continue

        match = PATTERN_ORIG.match(metric_name)
        if match:
            prop, metric = match.group(1), match.group(2)
            out[(prop, metric, True)] = (float(mean), 0.0 if std is None else float(std))
            continue

        match = PATTERN_STD.match(metric_name)
        if match:
            prop, metric = match.group(1), match.group(2)
            out[(prop, metric, False)] = (float(mean), 0.0 if std is None else float(std))
    return out


def build_wide(
    *,
    metric: str,
    is_orig: bool,
    kind: str,
    all_props: List[str],
    groups_full: List[str],
    groups_short: List[str],
    per_folder: Dict[str, Dict[Tuple[str, str, bool], Tuple[float, float]]],
) -> pd.DataFrame:
    values = []
    for prop in all_props:
        row = []
        for folder_name in groups_full:
            item = per_folder[folder_name].get((prop, metric, is_orig))
            if item is None:
                row.append(np.nan)
                continue
            mean_value, std_value = item
            row.append(float(mean_value if kind == "mean" else std_value))
        values.append(row)
    return pd.DataFrame(values, index=all_props, columns=groups_short, dtype=float).round(2)


def nice_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_bar_labels(ax, rects) -> None:
    for rect in rects:
        height = rect.get_height()
        if np.isfinite(height):
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def main() -> None:
    args = build_argparser().parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dirs = {
        ("r2", False): out_dir / "BAR_R2",
        ("r2", True): out_dir / "BAR_R2_ORIG",
        ("rmse", False): out_dir / "BAR_RMSE",
        ("rmse", True): out_dir / "BAR_RMSE_ORIG",
        ("mae", False): out_dir / "BAR_MAE",
        ("mae", True): out_dir / "BAR_MAE_ORIG",
    }
    for path in plot_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    csv_mean_paths = {
        ("r2", False): out_dir / "r2.csv",
        ("r2", True): out_dir / "r2_orig.csv",
        ("rmse", False): out_dir / "rmse.csv",
        ("rmse", True): out_dir / "rmse_orig.csv",
        ("mae", False): out_dir / "mae.csv",
        ("mae", True): out_dir / "mae_orig.csv",
    }
    csv_std_paths = {
        ("r2", False): out_dir / "r2_std.csv",
        ("r2", True): out_dir / "r2_orig_std.csv",
        ("rmse", False): out_dir / "rmse_std.csv",
        ("rmse", True): out_dir / "rmse_orig_std.csv",
        ("mae", False): out_dir / "mae_std.csv",
        ("mae", True): out_dir / "mae_orig_std.csv",
    }

    folders = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith(args.folder_prefix)
    )
    if not folders:
        raise SystemExit(
            f"No folders starting with '{args.folder_prefix}' found under {root}"
        )

    per_folder: Dict[str, Dict[Tuple[str, str, bool], Tuple[float, float]]] = {}
    for folder in folders:
        records = load_seed_summary(folder)
        if not records:
            continue
        extracted = extract(records)
        if not extracted:
            print(f"[warn] {folder.name}: no r2/rmse/mae metrics found")
            continue
        per_folder[folder.name] = extracted

    if not per_folder:
        raise SystemExit("No usable metrics found in any results folder.")

    groups_full = list(per_folder.keys())
    groups_short = [
        group[len(args.folder_prefix):] if group.startswith(args.folder_prefix) else group
        for group in groups_full
    ]
    all_props = sorted(
        {
            prop
            for data in per_folder.values()
            for (prop, _, _) in data.keys()
        }
    )

    for key, path in csv_mean_paths.items():
        df = build_wide(
            metric=key[0],
            is_orig=key[1],
            kind="mean",
            all_props=all_props,
            groups_full=groups_full,
            groups_short=groups_short,
            per_folder=per_folder,
        )
        df.to_csv(path, index_label="property")
        print(f"[csv] wrote {path}")

    for key, path in csv_std_paths.items():
        df = build_wide(
            metric=key[0],
            is_orig=key[1],
            kind="std",
            all_props=all_props,
            groups_full=groups_full,
            groups_short=groups_short,
            per_folder=per_folder,
        )
        df.to_csv(path, index_label="property")
        print(f"[csv] wrote {path}")

    for prop in all_props:
        for metric, is_orig in VARIANTS:
            labels: List[str] = []
            means: List[float] = []
            stds: List[float] = []

            for folder_name, short_name in zip(groups_full, groups_short):
                item = per_folder[folder_name].get((prop, metric, is_orig))
                if item is None:
                    continue
                mean_value, std_value = item
                if not np.isfinite(mean_value):
                    continue
                labels.append(short_name)
                means.append(float(mean_value))
                stds.append(float(std_value) if np.isfinite(std_value) else 0.0)

            if not labels:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            x = np.arange(len(labels))
            bars = ax.bar(x, means, width=0.6, yerr=stds, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)

            ylabel = metric.upper() if metric != "r2" else "R²"
            suffix = " original unit" if is_orig else ""
            nice_ax(ax, f"{prop} - {ylabel}{suffix}", ylabel=ylabel)
            add_bar_labels(ax, bars)

            out_path = plot_dirs[(metric, is_orig)] / f"{prop}_{metric}{'_orig' if is_orig else ''}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[fig] wrote {out_path}")

    print(f"[done] summaries written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
