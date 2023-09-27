from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


@registry.register_model("DOSPredict")
class DOSPredict(BaseModel):
    def __init__(
        self,
        edge_steps,
        self_loop,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        batch_norm=True,
        batch_track_stats=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(DOSPredict, self).__init__(edge_steps, self_loop)
        self.dim1 = dim1
        self.dim2 = dim2
        self.pre_fc_count = pre_fc_count
        self.gc_count = gc_count
        self.num_features = data.num_features
        self.num_edge_features = data.num_edge_features
        self.batch_norm = batch_norm
        self.batch_track_stats = batch_track_stats
        self.dropout_rate = dropout_rate

        # Determine gc dimension and post_fc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = data.num_features, data.num_features
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        # Determine output dimension length
        if data[0][self.target_attr].ndim == 0:
            self.output_dim = 1
        else:
            self.output_dim = len(data[0][self.target_attr])

        # setup layers
        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()

        self.dos_mlp = Sequential(
            Linear(self.post_fc_dim, self.dim2),
            torch.nn.PReLU(),
            Linear(self.dim2, self.output_dim),
            torch.nn.PReLU(),
        )

        self.scaling_mlp = Sequential(
            Linear(self.post_fc_dim, self.dim2),
            torch.nn.PReLU(),
            Linear(self.dim2, 1),
        )

    @property
    def target_attr(self):
        return "scaled"

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.num_features, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)

                pre_lin_list.append(Sequential(lin, torch.nn.PReLU()))

        return pre_lin_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):
            conv = GCBlock(self.gc_dim, self.num_edge_features, aggr="mean")
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats, affine=True
                )
                bn_list.append(bn)

        return conv_list, bn_list

    def forward(self, data):

        # Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x.float())
            else:
                out = self.pre_lin_list[i](out)

        # GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                out = self.conv_list[i](data.x, data.edge_index, data.edge_attr.float())
            else:
                out = self.conv_list[i](out, data.edge_index, data.edge_attr.float())
            if self.batch_norm:
                out = self.bn_list[i](out)

        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        # Post-GNN dense layers
        dos_out = self.dos_mlp(out)
        scaling = self.scaling_mlp(out)

        if dos_out.shape[1] == 1:
            return dos_out.view(-1), scaling.view(-1)
        else:
            return dos_out, scaling.view(-1)


class GCBlock(MessagePassing):
    def __init__(
        self,
        channels: int | tuple[int, int],
        dim: int = 0,
        aggr: str = "mean",
        **kwargs,
    ):
        super(GCBlock, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)

        self.mlp = Sequential(
            Linear(sum(channels) + dim, channels[1]),
            torch.nn.PReLU(),
        )
        self.mlp2 = Sequential(
            Linear(dim, dim),
            torch.nn.PReLU(),
        )

    def forward(
        self,
        x: Tensor | PairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, self.mlp2(edge_attr)], dim=-1)
        z = self.mlp(z)
        return z
