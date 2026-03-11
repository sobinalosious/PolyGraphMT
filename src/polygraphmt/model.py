# model.py
from __future__ import annotations

from typing import List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .conv import GNNEncoder, build_gnn_encoder


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


class FiLM(nn.Module):
    """
    Simple FiLM: gamma, beta from condition vector; apply to features as (1+gamma)*h + beta
    """
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond)
        b = self.beta(cond)
        return (1.0 + g) * h + b


class TaskHead(nn.Module):
    """
    Per-task MLP head. Input is concatenation of [graph_embed, optional task_embed].
    Outputs either a mean only (scalar) or mean+logvar (heteroscedastic).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        depth: int = 2,
        act: str = "relu",
        dropout: float = 0.0,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(get_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        out_dim = 2 if heteroscedastic else 1
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.hetero = heteroscedastic

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # returns [B, 1] or [B, 2] where [...,0] is mean and [...,1] is logvar if heteroscedastic
        return self.net(z)


class MultiTaskMultiFidelityModel(nn.Module):
    """
    General multi-task, multi-fidelity GNN.

    - Any number of tasks (properties) via T = len(task_names)
    - Any number of fidelities via num_fids
    - Fidelity conditioning with an embedding and FiLM on the graph embedding
    - Optional task embeddings concatenated into each task head input
    - Single forward returning predictions [B, T] (means); if heteroscedastic, also returns log-variances

    Expected input Batch fields (PyG):
      - x          : [N_nodes, F_node]
      - edge_index : [2, N_edges]
      - edge_attr  : [N_edges, F_edge] (required if gnn_type="gine")
      - batch      : [N_nodes]
      - fid_idx    : [B] or [B, 1] long; integer fidelity per graph

    Notes:
      - Targets should already be normalized outside the model; apply inverse transform for plots.
      - Loss weighting/equal-importance and curriculum happen in the trainer, not here.
    """

    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        task_names: List[str],
        num_fids: int,
        gnn_type: Literal["gine", "gin", "gcn"] = "gine",
        gnn_emb_dim: int = 256,
        gnn_layers: int = 5,
        gnn_norm: Literal["batch", "layer", "none"] = "batch",
        gnn_readout: Literal["mean", "sum", "max"] = "mean",
        gnn_act: str = "relu",
        gnn_dropout: float = 0.0,
        gnn_residual: bool = True,
        # Fidelity conditioning
        fid_emb_dim: int = 64,
        use_film: bool = True,
        # Task conditioning
        use_task_embed: bool = True,
        task_emb_dim: int = 32,
        # Heads
        head_hidden: int = 512,
        head_depth: int = 2,
        head_act: str = "relu",
        head_dropout: float = 0.0,
        heteroscedastic: bool = False,
        # Optional homoscedastic task uncertainty (used in loss, kept here for checkpoint parity)
        use_task_uncertainty: bool = False,
        # Embedding regularization (used via regularization_loss)
        fid_emb_l2: float = 0.0,
        task_emb_l2: float = 0.0,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.num_tasks = len(task_names)
        self.num_fids = int(num_fids)
        self.hetero = heteroscedastic
        self.fid_emb_l2 = float(fid_emb_l2)
        self.task_emb_l2 = float(task_emb_l2)
        self.use_film = use_film
        self.use_task_embed = use_task_embed

        # Optional learned homoscedastic uncertainty per task (trainer may use it)
        self.use_task_uncertainty = bool(use_task_uncertainty)
        if self.use_task_uncertainty:
            self.task_log_sigma2 = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.task_log_sigma2 = None

        # Encoder
        self.encoder: GNNEncoder = build_gnn_encoder(
            in_dim_node=in_dim_node,
            emb_dim=gnn_emb_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            in_dim_edge=in_dim_edge,
            act=gnn_act,
            dropout=gnn_dropout,
            residual=gnn_residual,
            norm=gnn_norm,
            readout=gnn_readout,
        )

        # Fidelity embedding + FiLM
        self.fid_embed = nn.Embedding(self.num_fids, fid_emb_dim) if fid_emb_dim > 0 else None
        self.film = FiLM(gnn_emb_dim, fid_emb_dim) if (use_film and fid_emb_dim > 0) else None

        # --- Compute the true feature dim sent to heads ---
        # If FiLM is ON: g stays [B, gnn_emb_dim]
        # If FiLM is OFF but fid_embed exists: we CONCAT c → g becomes [B, gnn_emb_dim + fid_emb_dim]
        self.gnn_out_dim = gnn_emb_dim + (fid_emb_dim if (self.fid_embed is not None and self.film is None) else 0)

        # Task embeddings
        self.task_embed = nn.Embedding(self.num_tasks, task_emb_dim) if (use_task_embed and task_emb_dim > 0) else None

        # Per-task heads
        head_in_dim = self.gnn_out_dim + (task_emb_dim if self.task_embed is not None else 0)
        self.heads = nn.ModuleList([
            TaskHead(
                in_dim=head_in_dim,
                hidden_dim=head_hidden,
                depth=head_depth,
                act=head_act,
                dropout=head_dropout,
                heteroscedastic=heteroscedastic,
            ) for _ in range(self.num_tasks)
        ])


    def reset_parameters(self):
        if self.fid_embed is not None:
            nn.init.normal_(self.fid_embed.weight, mean=0.0, std=0.02)
        if self.task_embed is not None:
            nn.init.normal_(self.task_embed.weight, mean=0.0, std=0.02)
        # Encoder/heads rely on their internal initializations.

    def forward(self, data: Batch) -> dict:
        """
        Returns:
          {
            "pred":   [B, T] means,
            "logvar": [B, T] optional if heteroscedastic,
            "h":      [B, D] graph embedding after FiLM (useful for diagnostics).
          }
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = data.batch
        if edge_attr is None and hasattr(self.encoder, "gnn_type") and self.encoder.gnn_type == "gine":
            raise ValueError("GINE encoder requires edge_attr, but Batch.edge_attr is None.")

        # Graph embedding
        g = self.encoder(x, edge_index, edge_attr, batch)  # [B, D]

        # Fidelity conditioning
        fid_idx = data.fid_idx.view(-1).long()  # [B]
        if self.fid_embed is not None:
            c = self.fid_embed(fid_idx)  # [B, C]
            if self.film is not None:
                g = self.film(g, c)  # [B, D]
            else:
                g = torch.cat([g, c], dim=-1)

        # Per-task heads
        preds: List[torch.Tensor] = []
        logvars: Optional[List[torch.Tensor]] = [] if self.hetero else None
        for t_idx, head in enumerate(self.heads):
            if self.task_embed is not None:
                tvec = self.task_embed.weight[t_idx].unsqueeze(0).expand(g.size(0), -1)
                z = torch.cat([g, tvec], dim=-1)
            else:
                z = g
            out = head(z)  # [B, 1] or [B, 2]
            if self.hetero:
                mu = out[..., 0:1]
                lv = out[..., 1:2]
                preds.append(mu)
                logvars.append(lv)  # type: ignore[arg-type]
            else:
                preds.append(out)

        pred = torch.cat(preds, dim=-1)  # [B, T]
        result = {"pred": pred, "h": g}
        if self.hetero and logvars is not None:
            result["logvar"] = torch.cat(logvars, dim=-1)  # [B, T]
        return result

    def regularization_loss(self) -> torch.Tensor:
        """
        Optional small L2 on embeddings to keep them bounded.
        """
        device = next(self.parameters()).device
        reg = torch.zeros([], device=device)
        if self.fid_embed is not None and self.fid_emb_l2 > 0:
            reg = reg + self.fid_emb_l2 * (self.fid_embed.weight.pow(2).mean())
        if self.task_embed is not None and self.task_emb_l2 > 0:
            reg = reg + self.task_emb_l2 * (self.task_embed.weight.pow(2).mean())
        return reg


def build_model(
    *,
    in_dim_node: int,
    in_dim_edge: int,
    task_names: List[str],
    num_fids: int,
    gnn_type: Literal["gine", "gin", "gcn"] = "gine",
    gnn_emb_dim: int = 256,
    gnn_layers: int = 5,
    gnn_norm: Literal["batch", "layer", "none"] = "batch",
    gnn_readout: Literal["mean", "sum", "max"] = "mean",
    gnn_act: str = "relu",
    gnn_dropout: float = 0.0,
    gnn_residual: bool = True,
    fid_emb_dim: int = 64,
    use_film: bool = True,
    use_task_embed: bool = True,
    task_emb_dim: int = 32,
    head_hidden: int = 512,
    use_task_uncertainty: bool = False,
    head_depth: int = 2,
    head_act: str = "relu",
    head_dropout: float = 0.0,
    heteroscedastic: bool = False,
    fid_emb_l2: float = 0.0,
    task_emb_l2: float = 0.0,
) -> MultiTaskMultiFidelityModel:
    """
    Factory to construct the multi-task, multi-fidelity model with a consistent API.
    """
    return MultiTaskMultiFidelityModel(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        task_names=task_names,
        num_fids=num_fids,
        gnn_type=gnn_type,
        gnn_emb_dim=gnn_emb_dim,
        gnn_layers=gnn_layers,
        gnn_norm=gnn_norm,
        gnn_readout=gnn_readout,
        gnn_act=gnn_act,
        gnn_dropout=gnn_dropout,
        gnn_residual=gnn_residual,
        fid_emb_dim=fid_emb_dim,
        use_film=use_film,
        use_task_embed=use_task_embed,
        task_emb_dim=task_emb_dim,
        head_hidden=head_hidden,
        head_depth=head_depth,
        head_act=head_act,
        head_dropout=head_dropout,
        heteroscedastic=heteroscedastic,
        fid_emb_l2=fid_emb_l2,
        task_emb_l2=task_emb_l2,
        use_task_uncertainty=use_task_uncertainty,
    )
