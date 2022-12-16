import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import CGConv, Set2Set

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


@registry.register_model("CGCNN")
class CGCNN(BaseModel):
    def __init__(
        self,
        edge_steps,
        self_loop,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(CGCNN, self).__init__(edge_steps, self_loop)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.gc_count = gc_count
        self.post_fc_count = post_fc_count
        self.num_features = data.num_features
        self.num_edge_features = data.num_edge_features

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
            self.output_dim = len(data[0][self.target_attr][0])

        # setup layers
        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()

        # Should processing_steps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(self.post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not recommended to use set2set
            self.lin_out_2 = Linear(self.output_dim * 2, self.output_dim)

    @property
    def target_attr(self):
        return "y"

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.num_features, self.dim1)
                else:
                    lin = Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):
            conv = CGConv(
                self.gc_dim, self.num_edge_features, aggr="mean", batch_norm=False
            )
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats
                )
                bn_list.append(bn)

        return conv_list, bn_list

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = Linear(self.post_fc_dim * 2, self.dim2)
                    else:
                        lin = Linear(self.post_fc_dim, self.dim2)
                else:
                    lin = Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out

    def forward(self, data):
        if len(self.pre_lin_list) == 0:
            out = data.x
            # if data.x can be a float, then don't need this if/else statement
        else:
            out = self._forward_pre_gnn_layers(data.x.float())

        out = self._forward_gnn_layers(data, out)
        out = self._forward_post_gnn_layers(data, out)

        out = out.view(-1) if out.shape[1] == 1 else out
        return out

    def _forward_pre_gnn_layers(self, out):
        """Pre-GNN dense layers"""
        for i in range(len(self.pre_lin_list)):
            out = self.pre_lin_list[i](out)
            out = getattr(F, self.act)(out)
        return out

    def _forward_gnn_layers(self, data, out):
        """GNN layers"""
        for i in range(len(self.conv_list)):
            out = self.conv_list[i](out, data.edge_index, data.edge_attr.float())
            if self.batch_norm:
                out = self.bn_list[i](out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        return out

    def _forward_post_gnn_layers(self, data, out):
        """Post-GNN dense layers"""
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        for i in range(0, len(self.post_lin_list)):
            out = self.post_lin_list[i](out)
            out = getattr(F, self.act)(out)
        out = self.lin_out(out)

        if self.pool_order == "late":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
        return out
