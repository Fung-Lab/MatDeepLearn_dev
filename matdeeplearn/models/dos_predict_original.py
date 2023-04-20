from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


# GNN model
@registry.register_model("DOSPredict_Original")
class DOSPredict_Original(BaseModel):
    def __init__(
        self,
        edge_steps,
        self_loop,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        batch_norm="True",
        batch_track_stats="True",
        dropout_rate=0.0,
        **kwargs
    ):
        super(DOSPredict_Original, self).__init__(edge_steps, self_loop)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        # Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim = data.num_features
            post_fc_dim = data.num_features
        else:
            self.gc_dim = dim1
            post_fc_dim = dim1

        # Determine output dimension length
        if data[0][self.target_attr].ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0][self.target_attr][0])

        # Set up pre-GNN dense layers
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = Sequential(
                        torch.nn.Linear(data.num_features, dim1), torch.nn.PReLU()
                    )
                    self.pre_lin_list.append(lin)
                else:
                    lin = Sequential(torch.nn.Linear(dim1, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        # Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GC_block(self.gc_dim, data.num_edge_features, aggr="mean")
            # conv = CGConv(self.gc_dim, data.num_edge_features, aggr="mean", batch_norm=False)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats, affine=True
                )
                self.bn_list.append(bn)

        self.dos_mlp = Sequential(
            Linear(post_fc_dim, dim2),
            torch.nn.PReLU(),
            Linear(dim2, output_dim),
            torch.nn.PReLU(),
        )

        self.scaling_mlp = Sequential(
            Linear(post_fc_dim, dim2),
            torch.nn.PReLU(),
            Linear(dim2, 1),
        )

    @property
    def target_attr(self):
        return "scaled"

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
                if self.batch_norm == "True":
                    out = self.conv_list[i](
                        data.x, data.edge_index, data.edge_attr.float()
                    )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        data.x, data.edge_index, data.edge_attr.float()
                    )
            else:
                if self.batch_norm == "True":
                    out = self.conv_list[i](
                        out, data.edge_index, data.edge_attr.float()
                    )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        out, data.edge_index, data.edge_attr.float()
                    )

        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        # Post-GNN dense layers
        dos_out = self.dos_mlp(out)
        scaling = self.scaling_mlp(out)

        if dos_out.shape[1] == 1:
            return dos_out.view(-1), scaling.view(-1)
        else:
            return dos_out, scaling.view(-1)


class GC_block(MessagePassing):
    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        dim: int = 0,
        aggr: str = "mean",
        **kwargs
    ):
        super(GC_block, self).__init__(aggr=aggr, **kwargs)
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
        x: Union[Tensor, PairTensor],
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
