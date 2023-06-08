from __future__ import annotations

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGPooling
from torch_geometric.nn.conv import GCNConv

from matdeeplearn.models.routines.attention import MetaPathImportance


class SelfAttentionRVPooling(nn.Module):
    def __init__(self, pool: str, **kwargs) -> None:
        """
        Concatenate a bit to node embedding (1 for real, 0 for virtual) and then perform self-attention pooling.
        """
        super().__init__()
        in_channels = kwargs.get("in_channels")
        ratio = kwargs.get("ratio")
        nonlinearity = kwargs.get("nonlinearity", "tanh")
        self.sag_pool = SAGPooling(
            in_channels + 1, ratio=ratio, nonlinearity=nonlinearity, GNN=GCNConv
        )
        self.pooling = getattr(torch_geometric.nn, pool)

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        real_mask = (data.z != 100).unsqueeze(-1).to(out.dtype)
        # shape (N, F + 1)
        out = torch.cat((out, real_mask), dim=-1).to(torch.float)
        edge_indices = torch.cat(
            [
                getattr(data, edge_label)
                for edge_label in data.__dict__.get("_store").keys()
                if edge_label.startswith("edge_index_")
            ],
            dim=-1,
        )
        edge_attrs = torch.cat(
            [
                getattr(data, edge_label)
                for edge_label in data.__dict__.get("_store").keys()
                if edge_label.startswith("edge_attr_")
            ],
            dim=0,
        )
        out = self.sag_pool(out, edge_indices, edge_attr=edge_attrs, batch=data.batch)
        return self.pooling(out[0], data.batch)


class RealVirtualAttention(nn.Module):
    def __init__(self, pool: str, **kwargs) -> None:
        """
        Attention mechanism to pool real and virtual nodes separately.
        """
        super().__init__()
        self.pool_choice = kwargs.get("pool_choice", "both")
        self.pooling = getattr(torch_geometric.nn, pool)

        # attention mechanism parameters for each class
        self.embed_dim = kwargs.get("embed_dim", 150)
        self.attn_size = kwargs.get("attn_size", 128)
        self.importance = MetaPathImportance(self.embed_dim, self.attn_size)

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        real_mask = torch.argwhere(data.z != 100).squeeze(1)
        virtual_mask = torch.argwhere(data.z == 100).squeeze(1)

        out_real = self.pooling(
            torch.index_select(out, 0, real_mask),
            torch.index_select(data.batch, 0, real_mask),
        )
        out_virtual = self.pooling(
            torch.index_select(out, 0, virtual_mask),
            torch.index_select(data.batch, 0, virtual_mask),
        )

        # attention computation to determine which class is most important
        out = torch.stack((out_real, out_virtual), dim=1)
        scores = self.importance(out)
        weighted_res = torch.sum(out * scores, dim=1)

        return weighted_res


class RealVirtualPooling(nn.Module):
    def __init__(self, pool: str, **kwargs) -> None:
        """
        Pool real and virtual nodes separately then concatenate them.
        """
        super().__init__()
        self.pooling = getattr(torch_geometric.nn, pool)
        self.pool_choice = kwargs.get("pool_choice", "both")

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        real_mask = torch.argwhere(data.z != 100).squeeze(1)
        virtual_mask = torch.argwhere(data.z == 100).squeeze(1)

        if self.pool_choice == "both":
            out_real = self.pooling(
                torch.index_select(out, 0, real_mask),
                torch.index_select(data.batch, 0, real_mask),
            )
            out_virtual = self.pooling(
                torch.index_select(out, 0, virtual_mask),
                torch.index_select(data.batch, 0, virtual_mask),
            )

            out = torch.cat((out_real, out_virtual), dim=1)

        elif self.pool_choice == "real":
            out = self.pooling(
                torch.index_select(out, 0, real_mask),
                torch.index_select(data.batch, 0, real_mask),
            )

        elif self.pool_choice == "virtual":
            out = self.pooling(
                torch.index_select(out, 0, virtual_mask),
                torch.index_select(data.batch, 0, virtual_mask),
            )

        return out


class AtomicNumberPooling(nn.Module):
    def __init__(self, pool: str, **kwargs) -> None:
        """
        Expands the node embedding from length N to length N*100, where a node of a specific atomic number
        is indexed to the appropriate location in the N*100 tensor. If atomic number = 1, then the embedding is found in 0:99,
        if atomic_number = 2, then it is found in 100:199, etc.
        """
        super().__init__()
        del kwargs
        self.pooling = getattr(torch_geometric.nn, pool)

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        # pool as before, but now each element within a graph is pooled separately
        elem_pool = torch.zeros((out.shape[0], out.shape[1] * 100), device=out.device)
        indices = torch.arange(
            start=0, end=out.shape[1], step=1, device=out.device
        ).repeat(out.shape[0], 1)
        indices = indices + ((data.z - 1) * out.shape[1]).unsqueeze(dim=1).repeat(
            1, out.shape[1]
        )
        elem_pool.scatter_(dim=1, index=indices, src=out)

        out = self.pooling(elem_pool, data.batch)
        return out
