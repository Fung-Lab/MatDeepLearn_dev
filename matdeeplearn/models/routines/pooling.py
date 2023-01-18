from __future__ import annotations

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data


class RealVirtualPooling(nn.Module):
    def __init__(self, pool) -> None:
        """
        Pool real and virtual nodes separately then concatenate them.
        """
        super().__init__()
        self.pooling = getattr(torch_geometric.nn, pool)

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        # obtain mask for all real nodes
        mask = torch.argwhere(data.z != 100).squeeze(1)
        out_masked = torch.index_select(out, 0, mask)
        batch_masked = torch.index_select(data.batch, 0, mask)    
        out_1 = self.pooling(out_masked, batch_masked)

        # obtain mask for all virtual nodes
        mask2 = torch.argwhere(data.z != 100).squeeze(1)
        out_masked2 = torch.index_select(out, 0, mask2)
        batch_masked2 = torch.index_select(data.batch, 0, mask2)
        out_2 = self.pooling(out_masked2, batch_masked2)

        # concatenate pooled embedding from real and virtual nodes
        out = torch.cat((out_1, out_2), dim=1)
        return out


class AtomicNumberPooling(nn.Module):
    def __init__(self, pool) -> None:
        """
        Expands the node embedding from length N to length N*100, where a node of a specific atomic number
        is indexed to the appropriate location in the N*100 tensor. If atomic number = 1, then the embedding is found in 0:99,
        if atomic_number = 2, then it is found in 100:199, etc.
        """
        super().__init__()
        self.pooling = getattr(torch_geometric.nn, pool)

    def forward(self, data: Data, out: torch.Tensor) -> torch.Tensor:
        elem_pool = torch.zeros((out.shape[0], out.shape[1] * 100), device=out.device)
        indices = torch.arange(
            start=0, end=out.shape[1], step=1, device=out.device
        ).repeat(out.shape[0], 1)
        indices = indices + ((data.z - 1) * out.shape[1]).unsqueeze(dim=1).repeat(
            1, out.shape[1]
        )
        elem_pool.scatter_(dim=1, index=indices, src=out)

        # pool as before, but now each element within a graph is pooled separately
        out = self.pooling(elem_pool, data.batch)
        return out
