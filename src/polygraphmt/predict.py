"""Inference entry point for the PolyGraphMT package."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PYGDataLoader

from .data_builder import TargetScaler, featurize_smiles
from .model import build_model
from .utils import apply_inverse_transform, to_device


# =================================================
# Utilities
# =================================================
def _read_smiles_from_args(input_csv: Optional[str], smiles: Optional[str]) -> List[str]:
    if input_csv:
        df = pd.read_csv(input_csv)
        smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
        if smiles_col is None:
            raise ValueError(f"{input_csv} must have a 'smiles' column.")
        return [s for s in df[smiles_col].astype(str).tolist() if s.strip()]
    if smiles:
        return [s.strip() for s in smiles.split(",") if s.strip()]
    raise ValueError("Provide either --input_csv or --smiles.")


def _resolve_fid_index(fid_names: Sequence[str], arg_fid: Optional[str]) -> Tuple[int, str]:
    if not fid_names:
        return 0, "exp"

    if arg_fid is None or arg_fid == "":
        return 0, fid_names[0]

    arg = arg_fid.strip().lower()
    if arg.isdigit():
        i = int(arg)
        if not (0 <= i < len(fid_names)):
            raise ValueError(f"--fidelity index {i} out of range")
        return i, fid_names[i]

    fid_lc = [f.lower() for f in fid_names]
    if arg not in fid_lc:
        raise ValueError(f"--fidelity '{arg_fid}' not in trained fidelities: {fid_names}")
    i = fid_lc.index(arg)
    return i, fid_names[i]


def _make_inference_dataset(smiles_list: List[str], T: int, fid_idx: int, fid_name: str):
    data_list = []
    for s in smiles_list:
        try:
            x, edge_index, edge_attr = featurize_smiles(s)
        except Exception as e:
            print(f"[warn] RDKit failed for {s}: {e}")
            continue

        d = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.zeros(1, T),
            y_mask=torch.zeros(1, T, dtype=torch.bool),
            fid_idx=torch.tensor([fid_idx], dtype=torch.long),
        )
        d.smiles = s
        d.fid_str = fid_name
        data_list.append(d)
    return data_list


def _load_scaler_compat(path: Path) -> TargetScaler:
    blob = torch.load(path, map_location="cpu")

    if "mean" not in blob or "std" not in blob:
        raise RuntimeError("Unrecognized target_scaler format.")

    ts = TargetScaler(
        transforms=blob.get("transforms", None),
        eps=blob.get("eps", None),
    )
    ts.load_state_dict({
        "mean": blob["mean"].float(),
        "std": blob["std"].float(),
        "transforms": blob.get("transforms", ts.transforms),
        "eps": blob.get("eps", ts.eps),
    })
    ts.targets = [str(t) for t in blob.get("targets", [])]
    return ts


# =================================================
# Argparser
# =================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Unified predictor (single + multitask)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--input_csv", default=None)
    p.add_argument("--smiles", default=None)
    p.add_argument("--fidelity", default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_csv", default=None)
    return p


# =================================================
# Main
# =================================================
def main():
    args = build_argparser().parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    scaler_path = run_dir / "target_scaler.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------
    # Load checkpoint & scaler
    # -------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]
    train_args = ckpt.get("args", {})

    scaler = _load_scaler_compat(scaler_path)

    # -------------------------------------------------
    # Infer TASKS (from scaler only)
    # -------------------------------------------------
    task_names = list(getattr(scaler, "targets", []))
    if not task_names:
        raise RuntimeError("Could not infer task names from scaler.")
    num_tasks = len(task_names)

    # -------------------------------------------------
    # Infer FIDELITIES (checkpoint is authority)
    # -------------------------------------------------
    if "fid_embed.weight" in state_dict:
        ckpt_num_fids = state_dict["fid_embed.weight"].shape[0]
    else:
        ckpt_num_fids = 1

    num_fids = ckpt_num_fids

    if ckpt_num_fids == 1:
        fid_names = ["exp"]
        fid_idx = 0
        fid_name = "exp"
    else:
        raw_names = list(train_args.get("fidelities", []))
        if len(raw_names) >= ckpt_num_fids:
            fid_names = raw_names[:ckpt_num_fids]
        else:
            fid_names = [f"fid{i}" for i in range(ckpt_num_fids)]
        fid_idx, fid_name = _resolve_fid_index(fid_names, args.fidelity)

    # -------------------------------------------------
    # Read SMILES
    # -------------------------------------------------
    smiles_list = _read_smiles_from_args(args.input_csv, args.smiles)
    if not smiles_list:
        raise RuntimeError("No valid SMILES provided.")

    # -------------------------------------------------
    # Infer feature dimensions
    # -------------------------------------------------
    x0, _, e0 = featurize_smiles(smiles_list[0])
    in_dim_node = x0.shape[1]
    in_dim_edge = e0.shape[1]

    # -------------------------------------------------
    # Rebuild model EXACTLY as trained
    # -------------------------------------------------
    model = build_model(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        task_names=task_names,
        num_fids=num_fids,
        gnn_type=train_args.get("gnn_type", "gine"),
        gnn_emb_dim=train_args.get("gnn_emb_dim", 256),
        gnn_layers=train_args.get("gnn_layers", 5),
        gnn_norm=train_args.get("gnn_norm", "batch"),
        gnn_readout=train_args.get("gnn_readout", "mean"),
        gnn_act=train_args.get("gnn_act", "relu"),
        gnn_dropout=train_args.get("gnn_dropout", 0.0),
        gnn_residual=train_args.get("gnn_residual", True),
        fid_emb_dim=train_args.get("fid_emb_dim", 64),
        use_film=train_args.get("use_film", True),
        use_task_embed=train_args.get("use_task_embed", True),
        task_emb_dim=train_args.get("task_emb_dim", 32),
        head_hidden=train_args.get("head_hidden", 512),
        head_depth=train_args.get("head_depth", 2),
        head_act=train_args.get("head_act", "relu"),
        head_dropout=train_args.get("head_dropout", 0.0),
        heteroscedastic=train_args.get("heteroscedastic", False),
        fid_emb_l2=0.0,
        task_emb_l2=0.0,
        use_task_uncertainty=train_args.get("task_uncertainty", False),
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -------------------------------------------------
    # Dataset / loader
    # -------------------------------------------------
    data_list = _make_inference_dataset(
        smiles_list, T=num_tasks, fid_idx=fid_idx, fid_name=fid_name
    )
    loader = PYGDataLoader(
        data_list,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            pred_n = out["pred"]
            pred = apply_inverse_transform(pred_n, scaler).cpu()

            smiles_b = list(batch.smiles)
            for i in range(pred.shape[0]):
                row = {"smiles": smiles_b[i], "fid": fid_name}
                for t, name in enumerate(task_names):
                    row[f"pred_{name}"] = float(pred[i, t])
                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "predictions_new.csv")
    df.to_csv(out_csv, index=False)

    print(f"[done] wrote {len(df)} rows → {out_csv}")


if __name__ == "__main__":
    main()
