"""
Implementation of Relational Transformer (ICLR 2023).
Modified to support PyG and PyTorch
https://openreview.net/pdf?id=cFuMmbWiN6
https://github.com/CameronDiao/relational-transformer
"""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_dense_adj, unbatch, unbatch_edge_index

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.models.rt.layers import RTLayer


@registry.register_model("RTModel")
class RTModel(BaseModel):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        output_dim: int,
        node_hidden: int = None,
        edge_hidden_1: int = None,
        edge_hidden_2: int = None,
        heads: int = 1,
        layers: int = 1,
        dropout: float = None,
        disable_edge_updates: bool = False,
        node_level_output: bool = False,
        edge_level_output: bool = False,
        **kwargs,
    ) -> None:
        super(RTModel, self).__init__(edge_dim)
        del kwargs
        self.node_dim = node_dim
        self.node_hidden = node_hidden
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.edge_hidden_1 = edge_hidden_1
        self.edge_hidden_2 = edge_hidden_2
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.disable_edge_updates = disable_edge_updates
        self.node_level_output = node_level_output
        self.edge_level_output = edge_level_output

        # define layers
        self.rt_blocks = nn.ModuleList(
            [
                RTLayer(
                    node_dim=node_dim,
                    node_hidden=node_hidden,
                    edge_dim=edge_dim,
                    edge_hidden_1=edge_hidden_1,
                    edge_hidden_2=edge_hidden_2,
                    heads=heads,
                    dropout=dropout,
                    disable_edge_updates=disable_edge_updates,
                )
                for _ in range(layers)
            ]
        )

        self.out_proj = nn.Linear(node_dim, output_dim)

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

        for i in range(self.layers):
            x, dense_adj = self.rt_blocks[i](x, dense_adj)

        # global readout from graph token
        out_graph = x[:, 0, :]

        # project to output dim
        out_graph = self.out_proj(out_graph)

        if self.node_level_output or self.edge_level_output:
            raise NotImplementedError(
                "Node and edge level outputs not implemented for RTModel"
            )

        return out_graph.view(-1, 1, self.output_dim)
