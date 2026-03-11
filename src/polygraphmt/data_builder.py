# data_builder.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
import json
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# RDKit is required
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType, BondStereo

# ---------------------------------------------------------
# Fidelity handling
# ---------------------------------------------------------

FID_PRIORITY = ["exp", "dft", "md", "gc"]  # internal lower-case canonical order


def _norm_fid(fid: str) -> str:
    return fid.strip().lower()


def _ensure_targets_order(requested: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for t in requested:
        key = t.strip()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


# ---------------------------------------------------------
# RDKit featurization
# ---------------------------------------------------------

_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
_ATOM2IDX = {s: i for i, s in enumerate(_ATOMS)}
_HYBS = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]
_HYB2IDX = {h: i for i, h in enumerate(_HYBS)}
_BOND_STEREOS = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
]
_STEREO2IDX = {s: i for i, s in enumerate(_BOND_STEREOS)}


def _one_hot(index: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= index < size:
        v[index] = 1.0
    return v


def atom_features(atom: Chem.Atom) -> List[float]:
    # Element one-hot with "other"
    elem_idx = _ATOM2IDX.get(atom.GetSymbol(), None)
    elem_oh = _one_hot(elem_idx if elem_idx is not None else len(_ATOMS), len(_ATOMS) + 1)

    # Degree one-hot up to 5 (bucket 5+)
    deg = min(int(atom.GetDegree()), 5)
    deg_oh = _one_hot(deg, 6)

    # Formal charge one-hot in [-2,-1,0,+1,+2]
    fc = max(-2, min(2, int(atom.GetFormalCharge())))
    fc_oh = _one_hot(fc + 2, 5)

    # Aromatic, in ring flags
    aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
    in_ring = [1.0 if atom.IsInRing() else 0.0]

    # Hybridization one-hot with "other"
    hyb_idx = _HYB2IDX.get(atom.GetHybridization(), None)
    hyb_oh = _one_hot(hyb_idx if hyb_idx is not None else len(_HYBS), len(_HYBS) + 1)

    # Implicit H count capped at 4
    imp_h = min(int(atom.GetTotalNumHs(includeNeighbors=True)), 4)
    imp_h_oh = _one_hot(imp_h, 5)

    # length: 11+6+5+1+1+6+5 = 35 (element has 11 buckets incl. "other")
    feats = elem_oh + deg_oh + fc_oh + aromatic + in_ring + hyb_oh + imp_h_oh
    return feats


def bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    single = 1.0 if bt == BondType.SINGLE else 0.0
    double = 1.0 if bt == BondType.DOUBLE else 0.0
    triple = 1.0 if bt == BondType.TRIPLE else 0.0
    aromatic = 1.0 if bt == BondType.AROMATIC else 0.0
    conj = 1.0 if bond.GetIsConjugated() else 0.0
    in_ring = 1.0 if bond.IsInRing() else 0.0
    stereo_oh = _one_hot(_STEREO2IDX.get(bond.GetStereo(), 0), len(_BOND_STEREOS))
    # length: 4 + 1 + 1 + 6 = 12
    return [single, double, triple, aromatic, conj, in_ring] + stereo_oh


def featurize_smiles(smiles: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    # Nodes
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    # Edges (bidirectional)
    rows, cols, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        rows.extend([i, j])
        cols.extend([j, i])
        eattr.extend([bf, bf])

    if not rows:
        # single-atom molecules, add a dummy self-loop edge
        rows, cols = [0], [0]
        eattr = [[0.0] * 12]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32)
    return x, edge_index, edge_attr


# ---------------------------------------------------------
# CSV discovery and reading
# ---------------------------------------------------------

def discover_target_fid_csvs(
    root: Path,
    targets: Sequence[str],
    fidelities: Sequence[str],
) -> Dict[tuple[str, str], Path]:
    """
    Discover CSV files for (target, fidelity) pairs.

    Supported layouts (case-insensitive):

      1) {root}/{fid}/{target}.csv
         e.g. data/raw/MD/SHEAR.csv, data/raw/exp/cp.csv

      2) {root}/{target}_{fid}.csv
         e.g. data/raw/SHEAR_MD.csv, data/raw/cp_exp.csv

    Matching is STRICT:
      - target and fid must appear as full '_' tokens in the stem
      - no substring matching, so 'he' will NOT match 'shear_md.csv'
    """
    root = Path(root)
    targets = _ensure_targets_order(targets)
    fids_lc = [_norm_fid(f) for f in fidelities]

    # Collect all CSVs under root
    all_paths = list(root.rglob("*.csv"))

    # Pre-index: (parent_name_lower, stem_lower, tokens_lower)
    indexed = []
    for p in all_paths:
        parent = p.parent.name.lower()
        stem = p.stem.lower()  # filename without extension
        tokens = stem.split("_")
        tokens_l = [t.lower() for t in tokens]
        indexed.append((p, parent, stem, tokens_l))

    mapping: Dict[tuple[str, str], Path] = {}

    for fid in fids_lc:
        fid_l = fid.strip().lower()

        for tgt in targets:
            tgt_l = tgt.strip().lower()

            # ---- 1) Prefer explicit folder layout: {root}/{fid}/{target}.csv ----
            #      parent == fid AND stem == target  (case-insensitive)
            folder_matches = [
                p for (p, parent, stem, tokens_l) in indexed
                if parent == fid_l and stem == tgt_l
            ]
            if folder_matches:
                # If you ever get more than one, it’s a config problem
                if len(folder_matches) > 1:
                    warnings.warn(
                        f"[discover_target_fid_csvs] Multiple matches for "
                        f"target='{tgt}' fid='{fid}' under folder layout: "
                        + ", ".join(str(p) for p in folder_matches)
                    )
                mapping[(tgt, fid)] = folder_matches[0]
                continue

            # ---- 2) Fallback: {target}_{fid}.csv anywhere under root ----
            #      require BOTH tgt and fid as full '_' tokens
            token_matches = [
                p for (p, parent, stem, tokens_l) in indexed
                if (tgt_l in tokens_l) and (fid_l in tokens_l)
            ]

            if token_matches:
                if len(token_matches) > 1:
                    warnings.warn(
                        f"[discover_target_fid_csvs] Multiple token matches for "
                        f"target='{tgt}' fid='{fid}': "
                        + ", ".join(str(p) for p in token_matches)
                    )
                mapping[(tgt, fid)] = token_matches[0]
                continue

            # If neither layout exists, we simply do not add (tgt, fid) to mapping.
            # build_long_table will just skip that combination.
            # You can enable a warning if you want:
            # warnings.warn(f"[discover_target_fid_csvs] No CSV for target='{tgt}', fid='{fid}'")

    return mapping


def read_target_csv(path: Path, target: str) -> pd.DataFrame:
    """
    Accepts:
      - 'smiles' column (case-insensitive)
      - value column named '{target}' or one of ['value','y' or lower-case target]
    Deduplicates by SMILES with mean.
    """
    df = pd.read_csv(path)

    # smiles column
    smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError(f"{path} must contain a 'smiles' column.")
    df = df.rename(columns={smiles_col: "smiles"})

    # value column
    val_col = None
    if target in df.columns:
        val_col = target
    else:
        for c in df.columns:
            if c.lower() in ("value", "y", target.lower()):
                val_col = c
                break
    if val_col is None:
        raise ValueError(f"{path} must contain a '{target}' column or one of ['value','y'].")

    df = df[["smiles", val_col]].copy()
    df = df.dropna(subset=[val_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    # Deduplicate SMILES by mean
    if df.duplicated(subset=["smiles"]).any():
        warnings.warn(f"[data_builder] Duplicates by SMILES in {path}. Averaging duplicates.")
        df = df.groupby("smiles", as_index=False)[val_col].mean()

    return df.rename(columns={val_col: target})


def build_long_table(root: Path, targets: Sequence[str], fidelities: Sequence[str]) -> pd.DataFrame:
    """
    Returns long-form table with columns: [smiles, fid, fid_idx, target, value]
    """
    targets = _ensure_targets_order(targets)
    fids_lc = [_norm_fid(f) for f in fidelities]

    mapping = discover_target_fid_csvs(root, targets, fidelities)
    if not mapping:
        raise FileNotFoundError(f"No CSVs found under {root} for the given targets and fidelities.")

    long_rows = []
    for (tgt, fid), path in mapping.items():
        df = read_target_csv(path, tgt)
        df["fid"] = _norm_fid(fid)
        df["target"] = tgt
        df = df.rename(columns={tgt: "value"})
        long_rows.append(df[["smiles", "fid", "target", "value"]])

    long = pd.concat(long_rows, axis=0, ignore_index=True)

    # attach fid index by priority
    fid2idx = {f: i for i, f in enumerate(FID_PRIORITY)}
    long["fid"] = long["fid"].str.lower()
    unknown = sorted(set(long["fid"]) - set(fid2idx.keys()))
    if unknown:
        warnings.warn(f"[data_builder] Unknown fidelities found: {unknown}. Appending after known ones.")
        start = len(fid2idx)
        for i, f in enumerate(unknown):
            fid2idx[f] = start + i

    long["fid_idx"] = long["fid"].map(fid2idx)
    return long


def pivot_to_rows_by_smiles_fid(long: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    """
    Input: long table [smiles, fid, fid_idx, target, value]
    Output: row-per-(smiles,fid) with wide columns for each target
    """
    targets = _ensure_targets_order(targets)
    wide = long.pivot_table(index=["smiles", "fid", "fid_idx"], columns="target", values="value", aggfunc="mean")
    wide = wide.reset_index()

    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan

    cols = ["smiles", "fid", "fid_idx"] + list(targets)
    return wide[cols]


# ---------------------------------------------------------
# Grouped split by SMILES and transforms/normalization
# ---------------------------------------------------------

def grouped_split_by_smiles(
    df_rows: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = df_rows["smiles"].drop_duplicates().values
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(uniq)

    n = len(uniq)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test_smiles = set(uniq[:n_test])
    val_smiles = set(uniq[n_test:n_test + n_val])
    train_smiles = set(uniq[n_test + n_val:])

    train_idx = df_rows.index[df_rows["smiles"].isin(train_smiles)].to_numpy()
    val_idx = df_rows.index[df_rows["smiles"].isin(val_smiles)].to_numpy()
    test_idx = df_rows.index[df_rows["smiles"].isin(test_smiles)].to_numpy()
    return train_idx, val_idx, test_idx


# ---------------- Enhanced TargetScaler with per-task transforms ----------------

class TargetScaler:
    """
    Per-task transform + standardization fitted on the training split only.

    - transforms[t] in {"identity","log10"}
    - eps[t] is added before log for numerical safety (only used if transforms[t]=="log10")
    - mean/std are computed in the *transformed* domain
    """
    def __init__(self, transforms: Optional[Sequence[str]] = None, eps: Optional[Sequence[float] | torch.Tensor] = None):
        self.mean: Optional[torch.Tensor] = None   # [T] (transformed domain)
        self.std: Optional[torch.Tensor] = None    # [T] (transformed domain)
        self.transforms: List[str] = [str(t).lower() for t in transforms] if transforms is not None else []
        if eps is None:
            self.eps: Optional[torch.Tensor] = None
        else:
            self.eps = torch.as_tensor(eps, dtype=torch.float32)
        self._tiny = 1e-12

    def _ensure_cfg(self, T: int):
        if not self.transforms or len(self.transforms) != T:
            self.transforms = ["identity"] * T
        if self.eps is None or self.eps.numel() != T:
            self.eps = torch.zeros(T, dtype=torch.float32)

    def _forward_transform_only(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply per-task transforms *before* standardization.
        y: [N, T] in original units. Returns transformed y_tf in same shape.
        """
        out = y.clone()
        T = out.size(1)
        self._ensure_cfg(T)
        for t in range(T):
            if self.transforms[t] == "log10":
                out[:, t] = torch.log10(torch.clamp(out[:, t] + self.eps[t], min=self._tiny))
        return out

    def _inverse_transform_only(self, y_tf: torch.Tensor) -> torch.Tensor:
        """
        Inverse the per-task transform (no standardization here).
        y_tf: [N, T] in transformed units.
        """
        out = y_tf.clone()
        T = out.size(1)
        self._ensure_cfg(T)
        for t in range(T):
            if self.transforms[t] == "log10":
                out[:, t] = (10.0 ** out[:, t]) - self.eps[t]
        return out

    def fit(self, y: torch.Tensor, mask: torch.Tensor):
        """
        y: [N, T] original units; mask: [N, T] bool
        Chooses eps automatically if not provided; mean/std computed in transformed space.
        """
        T = y.size(1)
        self._ensure_cfg(T)

        if self.eps is None or self.eps.numel() != T:
            # Auto epsilon: 0.1 * min positive per task (robust)
            eps_vals: List[float] = []
            y_np = y.detach().cpu().numpy()
            m_np = mask.detach().cpu().numpy().astype(bool)
            for t in range(T):
                if self.transforms[t] != "log10":
                    eps_vals.append(0.0)
                    continue
                vals = y_np[m_np[:, t], t]
                pos = vals[vals > 0]
                if pos.size == 0:
                    eps_vals.append(1e-8)
                else:
                    eps_vals.append(0.1 * float(max(np.min(pos), 1e-8)))
            self.eps = torch.tensor(eps_vals, dtype=torch.float32)

        y_tf = self._forward_transform_only(y)
        eps = 1e-8
        y_masked = torch.where(mask, y_tf, torch.zeros_like(y_tf))
        counts = mask.sum(dim=0).clamp_min(1)
        mean = y_masked.sum(dim=0) / counts
        var = ((torch.where(mask, y_tf - mean, torch.zeros_like(y_tf))) ** 2).sum(dim=0) / counts
        std = torch.sqrt(var + eps)
        self.mean, self.std = mean, std

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y_tf = self._forward_transform_only(y)
        return (y_tf - self.mean) / self.std

    def inverse(self, y_std: torch.Tensor) -> torch.Tensor:
        """
        Inverse standardization + inverse transform → original units.
        y_std: [N, T] in standardized-transformed space
        """
        y_tf = y_std * self.std + self.mean
        return self._inverse_transform_only(y_tf)

    def state_dict(self) -> Dict[str, torch.Tensor | List[str]]:
        return {
            "mean": self.mean,
            "std": self.std,
            "transforms": self.transforms,
            "eps": self.eps,
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor | List[str]]):
        self.mean = state["mean"]
        self.std = state["std"]
        self.transforms = [str(t) for t in state.get("transforms", [])]
        eps = state.get("eps", None)
        self.eps = torch.as_tensor(eps, dtype=torch.float32) if eps is not None else None


def auto_select_task_transforms(
    y_train: torch.Tensor,          # [N, T] original units (train split only)
    mask_train: torch.Tensor,       # [N, T] bool
    task_names: Sequence[str],
    *,
    min_pos_frac: float = 0.95,     # ≥95% of labels positive
    orders_threshold: float = 2.0,  # ≥2 orders of magnitude between p95 and p5
    tiny: float = 1e-12,
) -> tuple[List[str], torch.Tensor]:
    """
    Decide per-task transform: "log10" if (mostly-positive AND large dynamic range), else "identity".
    Returns (transforms, eps_vector) where eps is only used for log tasks.
    """
    Y = y_train.detach().cpu().numpy()
    M = mask_train.detach().cpu().numpy().astype(bool)

    transforms: List[str] = []
    eps_vals: List[float] = []

    for t in range(Y.shape[1]):
        yt = Y[M[:, t], t]
        if yt.size == 0:
            transforms.append("identity")
            eps_vals.append(0.0)
            continue

        pos_frac = (yt > 0).mean()
        p5 = float(np.percentile(yt, 5))
        p95 = float(np.percentile(yt, 95))
        denom = max(p5, tiny)
        dyn_orders = float(np.log10(max(p95 / denom, 1.0)))
        use_log = (pos_frac >= min_pos_frac) and (dyn_orders >= orders_threshold)

        if use_log:
            pos_vals = yt[yt > 0]
            if pos_vals.size == 0:
                eps_vals.append(1e-8)
            else:
                eps_vals.append(0.1 * float(max(np.min(pos_vals), 1e-8)))
            transforms.append("log10")
        else:
            transforms.append("identity")
            eps_vals.append(0.0)

    return transforms, torch.tensor(eps_vals, dtype=torch.float32)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class MultiFidelityMoleculeDataset(Dataset):
    """
    Each item is a PyG Data with:
      - x: [N_nodes, F_node]
      - edge_index: [2, N_edges]
      - edge_attr: [N_edges, F_edge]
      - y: [T] normalized targets (zeros where missing)
      - y_mask: [T] bool mask of present targets
      - fid_idx: [1] long
      - .smiles and .fid_str added for debugging

    Targets are kept in the exact order provided by the user.
    """
    def __init__(
        self,
        rows: pd.DataFrame,
        targets: Sequence[str],
        scaler: Optional[TargetScaler],
        smiles_graph_cache: Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.rows = rows.reset_index(drop=True).copy()
        self.targets = _ensure_targets_order(targets)
        self.scaler = scaler
        self.smiles_graph_cache = smiles_graph_cache

        # Build y and mask tensors
        ys, masks = [], []
        for _, r in self.rows.iterrows():
            yv, mv = [], []
            for t in self.targets:
                v = r[t]
                if pd.isna(v):
                    yv.append(np.nan)
                    mv.append(False)
                else:
                    yv.append(float(v))
                    mv.append(True)
            ys.append(yv)
            masks.append(mv)

        y = torch.tensor(np.array(ys, dtype=np.float32))   # [N, T]
        mask = torch.tensor(np.array(masks, dtype=np.bool_))

        if scaler is not None and scaler.mean is not None:
            y_norm = torch.where(mask, scaler.transform(y), torch.zeros_like(y))
        else:
            y_norm = y

        self.y = y_norm
        self.mask = mask

        # Input dims
        any_smiles = self.rows.iloc[0]["smiles"]
        x0, _, e0 = smiles_graph_cache[any_smiles]
        self.in_dim_node = x0.shape[1]
        self.in_dim_edge = e0.shape[1]

        # Fidelity metadata for reference (local indexing in this dataset)
        self.fids = sorted(
            self.rows["fid"].str.lower().unique().tolist(),
            key=lambda f: (FID_PRIORITY + [f]).index(f) if f in FID_PRIORITY else len(FID_PRIORITY),
        )
        self.fid2idx = {f: i for i, f in enumerate(self.fids)}
        self.rows["fid_idx_local"] = self.rows["fid"].str.lower().map(self.fid2idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Data:
        idx = int(idx)
        r = self.rows.iloc[idx]
        smi = r["smiles"]

        x, edge_index, edge_attr = self.smiles_graph_cache[smi]
        # Ensure [1, T] so batches become [B, T]
        y_i = self.y[idx].clone().unsqueeze(0)      # [1, T]
        m_i = self.mask[idx].clone().unsqueeze(0)   # [1, T]
        fid_idx = int(r["fid_idx_local"])

        d = Data(
            x=x.clone(),
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone(),
            y=y_i,
            y_mask=m_i,
            fid_idx=torch.tensor([fid_idx], dtype=torch.long),
        )
        d.smiles = smi
        d.fid_str = r["fid"]
        return d


def subsample_train_indices(
    rows: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    target: Optional[str],
    fidelity: Optional[str],
    pct: float = 1.0,
    seed: int = 137,
) -> np.ndarray:
    """
    Return a filtered train_idx that keeps only a 'pct' fraction (0<pct<=1)
    of TRAIN rows for the specified (target, fidelity) block. Selection is
    deterministic by unique SMILES. Rows outside the block are untouched.

    rows: wide table with columns ["smiles","fid","fid_idx", <targets...>]
    """
    if target is None or fidelity is None or pct >= 0.999:
        return train_idx

    if target not in rows.columns:
        return train_idx

    fid_lc = fidelity.strip().lower()

    # Identify TRAIN rows in the specified block: matching fid and having a label for 'target'
    train_rows = rows.iloc[train_idx]
    block_mask = (train_rows["fid"].str.lower() == fid_lc) & (~train_rows[target].isna())
    if not bool(block_mask.any()):
        return train_idx  # nothing to subsample

    # Sample by unique SMILES (stable & grouped)
    smiles_all = pd.Index(train_rows.loc[block_mask, "smiles"].unique())
    n_all = len(smiles_all)
    if n_all == 0:
        return train_idx

    if pct <= 0.0:
        pct = 0.0001
    n_keep = max(1, int(round(pct * n_all)))

    rng = np.random.RandomState(int(seed))
    smiles_sorted = np.array(sorted(smiles_all.tolist()))
    keep_smiles = set(rng.choice(smiles_sorted, size=n_keep, replace=False).tolist())

    # Keep all non-block rows; within block keep selected SMILES
    keep_mask_local = (~block_mask) | (train_rows["smiles"].isin(keep_smiles))
    kept_train_idx = train_rows.index[keep_mask_local].to_numpy()
    return kept_train_idx


# ---------------------------------------------------------
# High-level builder
# ---------------------------------------------------------

def build_dataset_from_dir(
    root_dir: str | Path,
    targets: Sequence[str],
    fidelities: Sequence[str] = ("exp", "dft", "md", "gc"),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    save_splits_path: Optional[str | Path] = None,
    # Optional subsampling of a (target, fidelity) block in TRAIN
    subsample_target: Optional[str] = None,
    subsample_fidelity: Optional[str] = None,
    subsample_pct: float = 1.0,
    subsample_seed: int = 137,
    # -------- NEW: auto/explicit log transforms --------
    auto_log: bool = True,
    log_orders_threshold: float = 2.0,
    log_min_pos_frac: float = 0.95,
    explicit_log_targets: Optional[Sequence[str]] = None,  # e.g. ["permeability"]
) -> tuple[MultiFidelityMoleculeDataset, MultiFidelityMoleculeDataset, MultiFidelityMoleculeDataset, TargetScaler]:
    """
    Returns train_ds, val_ds, test_ds, scaler.

    - Discovers CSVs for requested targets and fidelities
    - Builds a row-per-(smiles,fid) table with columns for each target
    - Splits by unique SMILES to avoid leakage across fidelity or targets
    - Fits transform+normalization on the training split only, applies to val/test
    - Builds RDKit graphs once per unique SMILES and reuses them

    NEW:
      - Auto per-task transform selection ("log10" vs "identity") by criteria
      - Optional explicit override via explicit_log_targets
    """
    root = Path(root_dir)
    targets = _ensure_targets_order(targets)
    fids_lc = [_norm_fid(f) for f in fidelities]

    # Build long and pivot to rows
    long = build_long_table(root, targets, fids_lc)
    rows = pivot_to_rows_by_smiles_fid(long, targets)

    # Deterministic grouped split by SMILES
    if save_splits_path is not None and Path(save_splits_path).exists():
        with open(save_splits_path, "r") as f:
            split_obj = json.load(f)
        train_smiles = set(split_obj["train_smiles"])
        val_smiles = set(split_obj["val_smiles"])
        test_smiles = set(split_obj["test_smiles"])
        train_idx = rows.index[rows["smiles"].isin(train_smiles)].to_numpy()
        val_idx = rows.index[rows["smiles"].isin(val_smiles)].to_numpy()
        test_idx = rows.index[rows["smiles"].isin(test_smiles)].to_numpy()
    else:
        train_idx, val_idx, test_idx = grouped_split_by_smiles(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        if save_splits_path is not None:
            split_obj = {
                "train_smiles": rows.iloc[train_idx]["smiles"].drop_duplicates().tolist(),
                "val_smiles": rows.iloc[val_idx]["smiles"].drop_duplicates().tolist(),
                "test_smiles": rows.iloc[test_idx]["smiles"].drop_duplicates().tolist(),
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
            }
            Path(save_splits_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_splits_path, "w") as f:
                json.dump(split_obj, f, indent=2)

    # Build RDKit graphs once per unique SMILES
    uniq_smiles = rows["smiles"].drop_duplicates().tolist()
    smiles_graph_cache: Dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for smi in uniq_smiles:
        try:
            x, edge_index, edge_attr = featurize_smiles(smi)
            smiles_graph_cache[smi] = (x, edge_index, edge_attr)
        except Exception as e:
            warnings.warn(f"[data_builder] Dropping SMILES due to RDKit parse error: {smi} ({e})")

    # Filter rows to those that featurized successfully
    rows = rows[rows["smiles"].isin(smiles_graph_cache.keys())].reset_index(drop=True)

    # Re-map indices after filtering using smiles membership
    train_idx = rows.index[rows["smiles"].isin(set(rows.iloc[train_idx]["smiles"]))].to_numpy()
    val_idx = rows.index[rows["smiles"].isin(set(rows.iloc[val_idx]["smiles"]))].to_numpy()
    test_idx = rows.index[rows["smiles"].isin(set(rows.iloc[test_idx]["smiles"]))].to_numpy()

    # Optional subsampling (train only) for a specific (target, fidelity) block
    train_idx = subsample_train_indices(
        rows,
        train_idx,
        target=subsample_target,
        fidelity=subsample_fidelity,
        pct=float(subsample_pct),
        seed=int(subsample_seed),
    )

    # Fit scaler on training split only
    def build_y_mask(df_slice: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        ys, ms = [], []
        for _, r in df_slice.iterrows():
            yv, mv = [], []
            for t in targets:
                v = r[t]
                if pd.isna(v):
                    yv.append(np.nan)
                    mv.append(False)
                else:
                    yv.append(float(v))
                    mv.append(True)
            ys.append(yv)
            ms.append(mv)
        y = torch.tensor(np.array(ys, dtype=np.float32))
        mask = torch.tensor(np.array(ms, dtype=np.bool_))
        return y, mask

    y_train, mask_train = build_y_mask(rows.iloc[train_idx])

    # Decide transforms per task
    if explicit_log_targets:
        explicit_set = set(explicit_log_targets)
        transforms = [("log10" if t in explicit_set else "identity") for t in targets]
        eps_vec = None  # will be auto-chosen in scaler.fit if not provided
    elif auto_log:
        transforms, eps_vec = auto_select_task_transforms(
            y_train,
            mask_train,
            targets,
            min_pos_frac=float(log_min_pos_frac),
            orders_threshold=float(log_orders_threshold),
        )
    else:
        transforms, eps_vec = (["identity"] * len(targets), None)

    scaler = TargetScaler(transforms=transforms, eps=eps_vec)
    scaler.fit(y_train, mask_train)

    # Build datasets
    train_rows = rows.iloc[train_idx].reset_index(drop=True)
    val_rows = rows.iloc[val_idx].reset_index(drop=True)
    test_rows = rows.iloc[test_idx].reset_index(drop=True)

    train_ds = MultiFidelityMoleculeDataset(train_rows, targets, scaler, smiles_graph_cache)
    val_ds = MultiFidelityMoleculeDataset(val_rows, targets, scaler, smiles_graph_cache)
    test_ds = MultiFidelityMoleculeDataset(test_rows, targets, scaler, smiles_graph_cache)

    return train_ds, val_ds, test_ds, scaler


__all__ = [
    "build_dataset_from_dir",
    "discover_target_fid_csvs",
    "read_target_csv",
    "build_long_table",
    "pivot_to_rows_by_smiles_fid",
    "grouped_split_by_smiles",
    "TargetScaler",
    "MultiFidelityMoleculeDataset",
    "atom_features",
    "bond_features",
    "featurize_smiles",
    "auto_select_task_transforms",
]
