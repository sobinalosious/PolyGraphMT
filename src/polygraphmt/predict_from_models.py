"""Resolve released checkpoints from models/ and run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict import predict_from_artifacts


def _norm_property(value: str) -> str:
    return value.strip().lower()


def _load_model_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r") as handle:
        data = json.load(handle)
    return {str(key).lower(): value for key, value in data.items()}


def _expect_single_match(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for '{pattern}' in {directory}, found {len(matches)}."
        )
    return matches[0]


def resolve_model_artifacts(
    models_dir: Path,
    property_name: str,
) -> tuple[Path, Path, dict[str, str]]:
    model_map = _load_model_map(models_dir / "best_model_map.json")
    prop = _norm_property(property_name)

    entry = model_map.get(prop, {"family": "single", "task": prop})
    family = entry["family"].strip().lower()
    task = entry["task"].strip().lower()

    if family == "single":
        model_dir = models_dir / "single_models"
        ckpt = _expect_single_match(f"{task}_single_model_*.pt", model_dir)
        scaler = _expect_single_match(f"{task}_single_scalar_*.pt", model_dir)
    elif family == "multitask":
        model_dir = models_dir / "multitask_models"
        ckpt = _expect_single_match(f"{task}_model_*.pt", model_dir)
        scaler = _expect_single_match(f"{task}_scalar_*.pt", model_dir)
    else:
        raise RuntimeError(f"Unsupported family '{family}' for property '{prop}'.")

    return ckpt, scaler, entry


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Predict from released model artifacts under models/")
    parser.add_argument("--property", required=True, help="Property key to predict, e.g. cp or rho")
    parser.add_argument("--models_dir", default="models", help="Directory containing model artifacts")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--fidelity", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--keep_all_outputs",
        action="store_true",
        help="Keep all outputs from the resolved model family instead of only the requested property.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    prop = _norm_property(args.property)
    models_dir = Path(args.models_dir)
    ckpt_path, scaler_path, entry = resolve_model_artifacts(models_dir, prop)

    print(
        f"[info] property={prop} uses family={entry['family']} task={entry['task']} "
        f"checkpoint={ckpt_path.name}"
    )

    df = predict_from_artifacts(
        ckpt_path=ckpt_path,
        scaler_path=scaler_path,
        input_csv=args.input_csv,
        smiles=args.smiles,
        fidelity=args.fidelity,
        batch_size=args.batch_size,
        device_name=args.device,
    )

    if not args.keep_all_outputs:
        pred_col = f"pred_{prop}"
        if pred_col not in df.columns:
            raise RuntimeError(
                f"Resolved model does not produce '{pred_col}'. Available columns: {list(df.columns)}"
            )
        df = df[["smiles", "fid", pred_col]]

    out_csv = Path(args.out_csv) if args.out_csv else Path.cwd() / f"predictions_{prop}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {len(df)} rows → {out_csv}")


if __name__ == "__main__":
    main()
