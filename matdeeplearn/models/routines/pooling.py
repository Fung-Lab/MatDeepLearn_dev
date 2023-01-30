# TODO: add pooling routines to the registry
from __future__ import annotations

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data


class RealVirtualPooling(nn.Module):
    def __init__(self, pool: str, pool_choice: str) -> None:
        """
        Pool real and virtual nodes separately then concatenate them.
        """
        super().__init__()
        self.pooling = getattr(torch_geometric.nn, pool)
        self.pool_choice = pool_choice

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        real_mask = torch.argwhere(data.z_rv != 100).squeeze(1)
        virtual_mask = torch.argwhere(data.z_rv == 100).squeeze(1)

        if self.pool_choice == "both":
            out_real = self.pooling(
                torch.index_select(out, 0, real_mask),
                torch.index_select(data.x_rv_batch, 0, real_mask),
            )
            out_virtual = self.pooling(
                torch.index_select(out, 0, virtual_mask),
                torch.index_select(data.x_rv_batch, 0, virtual_mask),
            )

            out = torch.cat((out_real, out_virtual), dim=1)

        elif self.pool_choice == "real":
            out = self.pooling(
                torch.index_select(out, 0, real_mask),
                torch.index_select(data.x_rv_batch, 0, real_mask),
            )

        elif self.pool_choice == "virtual":
            out = self.pooling(
                torch.index_select(out, 0, virtual_mask),
                torch.index_select(data.x_rv_batch, 0, virtual_mask),
            )

        return out


class AtomicNumberPooling(nn.Module):
    def __init__(self, pool, *args) -> None:
        """
        Expands the node embedding from length N to length N*100, where a node of a specific atomic number
        is indexed to the appropriate location in the N*100 tensor. If atomic number = 1, then the embedding is found in 0:99,
        if atomic_number = 2, then it is found in 100:199, etc.
        """
        super().__init__()
        self.pooling = getattr(torch_geometric.nn, pool)
        del args

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        elem_pool = torch.zeros((out.shape[0], out.shape[1] * 100), device=out.device)
        indices = torch.arange(
            start=0, end=out.shape[1], step=1, device=out.device
        ).repeat(out.shape[0], 1)
        indices = indices + ((data.z_rv - 1) * out.shape[1]).unsqueeze(dim=1).repeat(
            1, out.shape[1]
        )
        elem_pool.scatter_(dim=1, index=indices, src=out)

        # pool as before, but now each element within a graph is pooled separately
        out = self.pooling(elem_pool, data.x_rv_batch)
        return out
