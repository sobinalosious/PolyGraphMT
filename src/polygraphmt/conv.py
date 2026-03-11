# conv.py
# Clean, dependency-light graph encoder blocks for molecular GNNs.
# - Single source of truth for convolution choices: "gine", "gin", "gcn"
# - Edge attributes are supported for "gine" (recommended for chemistry)
# - No duplication with PyG built-ins; everything wraps torch_geometric.nn
# - Consistent encoder API: GNNEncoder(...).forward(x, edge_index, edge_attr, batch) -> graph embedding [B, emb_dim]

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    GINConv,
    GCNConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


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


class MLP(nn.Module):
    """Small MLP used inside GNN layers and projections."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(get_activation(act))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeProjector(nn.Module):
    """Projects raw node features to model embedding size."""
    def __init__(self, in_dim_node: int, emb_dim: int, act: str = "relu"):
        super().__init__()
        if in_dim_node == emb_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim_node, emb_dim),
                get_activation(act),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EdgeProjector(nn.Module):
    """Projects raw edge attributes to model embedding size for GINE."""
    def __init__(self, in_dim_edge: int, emb_dim: int, act: str = "relu"):
        super().__init__()
        if in_dim_edge <= 0:
            raise ValueError("in_dim_edge must be > 0 when using edge attributes")
        self.proj = nn.Sequential(
            nn.Linear(in_dim_edge, emb_dim),
            get_activation(act),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.proj(e)


class GNNEncoder(nn.Module):
    """
    Backbone GNN with selectable conv type.

    gnn_type:
        - "gine": chemistry-ready, uses edge_attr (recommended)
        - "gin" : ignores edge_attr, strong node MPNN
        - "gcn" : ignores edge_attr, fast spectral conv
    norm: "batch" | "layer" | "none"
    readout: "mean" | "sum" | "max"
    """

    def __init__(
        self,
        in_dim_node: int,
        emb_dim: int,
        num_layers: int = 5,
        gnn_type: Literal["gine", "gin", "gcn"] = "gine",
        in_dim_edge: int = 0,
        act: str = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        norm: Literal["batch", "layer", "none"] = "batch",
        readout: Literal["mean", "sum", "max"] = "mean",
    ):
        super().__init__()
        assert num_layers >= 1

        self.gnn_type = gnn_type.lower()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.dropout_p = float(dropout)
        self.readout = readout.lower()

        self.node_proj = NodeProjector(in_dim_node, emb_dim, act=act)
        self.edge_proj: Optional[EdgeProjector] = None

        if self.gnn_type == "gine":
            if in_dim_edge <= 0:
                raise ValueError(
                    "gine selected but in_dim_edge <= 0. Provide edge attributes or switch gnn_type."
                )
            self.edge_proj = EdgeProjector(in_dim_edge, emb_dim, act=act)

        # Build conv stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if self.gnn_type == "gine":
                # edge_attr must be projected to emb_dim
                nn_mlp = MLP(emb_dim, emb_dim, emb_dim, num_layers=2, act=act, dropout=0.0)
                conv = GINEConv(nn_mlp)
            elif self.gnn_type == "gin":
                nn_mlp = MLP(emb_dim, emb_dim, emb_dim, num_layers=2, act=act, dropout=0.0)
                conv = GINConv(nn_mlp)
            elif self.gnn_type == "gcn":
                conv = GCNConv(emb_dim, emb_dim, add_self_loops=True, normalize=True)
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.convs.append(conv)

            if norm == "batch":
                self.norms.append(nn.BatchNorm1d(emb_dim))
            elif norm == "layer":
                self.norms.append(nn.LayerNorm(emb_dim))
            elif norm == "none":
                self.norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown norm: {norm}")

        self.act = get_activation(act)

    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        if self.readout == "sum":
            return global_add_pool(x, batch)
        if self.readout == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Unknown readout: {self.readout}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns a graph-level embedding of shape [B, emb_dim].
        If batch is None, assumes a single graph and creates a zero batch vector.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Project features (ensure float dtype)
        x = x.float()
        x = self.node_proj(x)

        e = None
        if self.gnn_type == "gine":
            if edge_attr is None:
                raise ValueError("GINE requires edge_attr, but got None.")
            e = self.edge_proj(edge_attr.float())

        # Message passing
        h = x
        for conv, norm in zip(self.convs, self.norms):
            if self.gnn_type == "gcn":
                h_next = conv(h, edge_index)  # GCNConv ignores edge_attr
            elif self.gnn_type == "gin":
                h_next = conv(h, edge_index)  # GINConv ignores edge_attr
            else:  # gine
                h_next = conv(h, edge_index, e)

            h_next = norm(h_next)
            h_next = self.act(h_next)

            if self.residual and h_next.shape == h.shape:
                h = h + h_next
            else:
                h = h_next

            if self.dropout_p > 0:
                h = F.dropout(h, p=self.dropout_p, training=self.training)

        g = self._readout(h, batch)
        return g  # [B, emb_dim]


def build_gnn_encoder(
    in_dim_node: int,
    emb_dim: int,
    num_layers: int = 5,
    gnn_type: Literal["gine", "gin", "gcn"] = "gine",
    in_dim_edge: int = 0,
    act: str = "relu",
    dropout: float = 0.0,
    residual: bool = True,
    norm: Literal["batch", "layer", "none"] = "batch",
    readout: Literal["mean", "sum", "max"] = "mean",
) -> GNNEncoder:
    """
    Factory to create a GNNEncoder with a consistent, minimal API.
    Prefer calling this from model.py so encoder construction is centralized.
    """
    return GNNEncoder(
        in_dim_node=in_dim_node,
        emb_dim=emb_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        in_dim_edge=in_dim_edge,
        act=act,
        dropout=dropout,
        residual=residual,
        norm=norm,
        readout=readout,
    )


__all__ = ["GNNEncoder", "build_gnn_encoder"]
