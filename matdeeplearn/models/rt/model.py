from typing import List

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_dense_adj, unbatch, unbatch_edge_index

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


@registry.register_model("RTModel")
class RTModel(nn.Module):
    def ____init__(
        self,
        node_dim,
        node_hidden,
        edge_dim,
        edge_hidden_1,
        edge_hidden_2,
        heads,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.node_hidden = node_hidden
        self.edge_dim = edge_dim
        self.edge_hidden_1 = edge_hidden_1
        self.edge_hidden_2 = edge_hidden_2
        self.heads = heads

    @staticmethod
    def unbatch_edge_attr(
        edge_attr: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Modify from PyG implementation.
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/unbatch.html
        Split edge_attr according to node assignment batch vector.
        """
        deg = degree(batch, dtype=torch.int64)
        ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

        edge_batch = batch[edge_index[0]]
        edge_index = edge_index - ptr[edge_batch]
        sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
        return edge_attr.split(sizes, dim=0)

    def forward(self, data: Data) -> None:
        # get edges into form (B, N, N, Ed)
        dense_adj = torch.stack(
            [
                to_dense_adj(dense, edge_attr=attr).squeeze(0)
                for dense, attr in zip(
                    unbatch_edge_index(data.edge_index, data.batch),
                    self.unbatch_edge_attr(data.edge_attr, data.edge_index, data.batch),
                )
            ],
            dim=0,
        )
        x = torch.stack(unbatch(data.x, data.batch), dim=0)
