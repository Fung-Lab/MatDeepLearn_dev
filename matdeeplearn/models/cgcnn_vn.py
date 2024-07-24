import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
from torch_geometric.nn import (
    CGConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot


@registry.register_model("CGCNN_VN")
class CGCNN(BaseModel):
    def __init__(
            self,
            node_dim,
            edge_dim,
            output_dim,
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
            cutoff_radius_rn_vn=8,
            cutoff_radius_vn_vn=4,
            **kwargs
    ):
        super(CGCNN, self).__init__(**kwargs)

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
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.cutoff_radius_rn_vn = cutoff_radius_rn_vn
        self.cutoff_radius_vn_vn = cutoff_radius_vn_vn
        print("rn_vn radius:", self.cutoff_radius_rn_vn)

        if isinstance(self.cutoff_radius, dict):
            self.rn_rn_radius = self.cutoff_radius['rn-rn']
            self.rn_vn_radius = self.cutoff_radius['rn-vn']
            self.cutoff_radius = max(self.cutoff_radius.values())
        else:
            self.rn_rn_radius = self.cutoff_radius
            self.rn_vn_radius = self.cutoff_radius

        self.distance_expansion = GaussianSmearing(0.0, self.cutoff_radius, self.edge_dim, 0.2)

        # Determine gc dimension and post_fc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.node_dim, self.node_dim
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        # setup layers
        self.pre_lin_list, self.pre_lin_list_vn = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list, self.rn_to_vn_conv_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()
        self.vn_conv = self._setup_vn_gnn_layer()

        # set up output layer
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(self.post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not recommended to use set2set
            self.lin_out_2 = torch.nn.Linear(self.output_dim * 2, self.output_dim)

    @property
    def target_attr(self):
        return "y"

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()
        pre_lin_list_vn = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            pre_lin_list_vn = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.node_dim, self.dim1)
                    lin_vn = torch.nn.Linear(self.node_dim, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                    lin_vn = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)
                pre_lin_list_vn.append(lin_vn)

        return pre_lin_list, pre_lin_list_vn

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        rn_to_vn_conv_list = torch.nn.ModuleList()
        # vn_to_vn_conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):
            conv = CGConv(
                self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
            )
            rn_vn_conv = CGConv(
                self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
            )
            # vn_vn_conv = CGConv(
            #     self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
            # )
            conv_list.append(conv)
            rn_to_vn_conv_list.append(rn_vn_conv)
            # vn_to_vn_conv_list.append(vn_vn_conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats
                )
                bn_list.append(bn)

        return conv_list, bn_list, rn_to_vn_conv_list

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(self.post_fc_dim * 2, self.dim2)
                    else:
                        lin = torch.nn.Linear(self.post_fc_dim, self.dim2)
                else:
                    lin = torch.nn.Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = torch.nn.Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = torch.nn.Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = torch.nn.Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out

    def _setup_vn_gnn_layer(self):
        vn_gnn_conv = CGConv(
            self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
        )
        return vn_gnn_conv

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        if self.otf_edge_index == True:
            # data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)
            data.edge_index, data.edge_weight, _, _, _, _ = self.generate_graph(data, self.cutoff_radius,
                                                                                self.n_neighbors)
            edge_mask = torch.zeros_like(data.edge_index[0])
            edge_mask[(data.z[data.edge_index[0]] == 100) & (
                        data.z[data.edge_index[1]] == 100)] = 0  # virtual node to virtual node
            edge_mask[(data.z[data.edge_index[0]] != 100) & (
                        data.z[data.edge_index[1]] == 100)] = 1  # regular node to virtual node
            edge_mask[(data.z[data.edge_index[0]] == 100) & (
                        data.z[data.edge_index[1]] != 100)] = 2  # virtual node to regular node
            edge_mask[(data.z[data.edge_index[0]] != 100) & (
                        data.z[data.edge_index[1]] != 100)] = 3  # regular node to regular node

            indices_rn_to_rn = (edge_mask == 3) & (data.edge_weight <= self.cutoff_radius)
            indices_rn_to_vn = (edge_mask == 1) & (data.edge_weight <= self.cutoff_radius_rn_vn)
            # indices_vn_to_vn = (edge_mask == 0) & (edge_weights <= 4)
            indices_to_keep = indices_rn_to_rn | indices_rn_to_vn  # | indices_vn_to_vn
            indices_rn_to_rn = indices_rn_to_rn[indices_to_keep]
            indices_rn_to_vn = indices_rn_to_vn[indices_to_keep]
            # indices_vn_to_vn = indices_vn_to_vn[indices_to_keep]

            edge_indices = data.edge_index[:, indices_to_keep]
            edge_weights = data.edge_weight[indices_to_keep]

            data.edge_index, data.edge_weight = edge_indices, edge_weights
            data.indices_rn_to_rn = indices_rn_to_rn
            data.indices_rn_to_vn = indices_rn_to_vn
            # data.indices_vn_to_vn = indices_vn_to_vn

            data.edge_mask = edge_mask
            if self.otf_edge_attr == True:
                data.edge_attr = self.distance_expansion(data.edge_weight)
            else:
                logging.warning("Edge attributes should be re-computed for otf edge indices.")

        if self.otf_edge_index == False:
            if self.otf_edge_attr == True:
                data.edge_attr = self.distance_expansion(data.edge_weight)

        if self.otf_node_attr == True:
            data.x = node_rep_one_hot(data.z).float()

        # Pre-GNN dense layers
        rn_mask = torch.argwhere(data.z != 100).squeeze(1)
        vn_mask = torch.argwhere(data.z == 100).squeeze(1)
        rn = data.x[rn_mask]
        vn = data.x[vn_mask]
        rn_out = torch.zeros_like(rn, dtype=data.x.dtype)
        vn_out = torch.zeros_like(vn, dtype=data.x.dtype)
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                rn_out = self.pre_lin_list[i](rn)
                rn_out = getattr(F, self.act)(rn_out)
            else:
                rn_out = self.pre_lin_list[i](rn_out)
                rn_out = getattr(F, self.act)(rn_out)

        for i in range(0, len(self.pre_lin_list_vn)):
            if i == 0:
                vn_out = self.pre_lin_list_vn[i](vn)
                vn_out = getattr(F, self.act)(vn_out)
            else:
                vn_out = self.pre_lin_list_vn[i](vn_out)
                vn_out = getattr(F, self.act)(vn_out)

        out = torch.zeros_like(data.x)  # Create a tensor of zeros with the same shape as data.x
        out[rn_mask] = rn_out.float()
        out[vn_mask] = vn_out.float()

        # GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm:
                    out = self.conv_list[i](
                        data.x, data.edge_index[:, data.indices_rn_to_rn], data.edge_attr[data.indices_rn_to_rn, :]
                    )
                    out = self.rn_to_vn_conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_vn], data.edge_attr[data.indices_rn_to_vn, :]
                    )
                    # out = self.vn_to_vn_conv_list[i](
                    #     out, data.edge_index[:, data.indices_vn_to_vn], data.edge_attr[data.indices_vn_to_vn, :]
                    # )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        data.x, data.edge_index[:, data.indices_rn_to_rn], data.edge_attr[data.indices_rn_to_rn, :]
                    )
                    out = self.rn_to_vn_conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_vn], data.edge_attr[data.indices_rn_to_vn, :]
                    )
                    # out = self.vn_to_vn_conv_list[i](
                    #     out, data.edge_index[:, data.indices_vn_to_vn], data.edge_attr[data.indices_vn_to_vn, :]
                    # )
            else:
                if self.batch_norm:
                    out = self.conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_rn], data.edge_attr[data.indices_rn_to_rn, :]
                    )
                    out = self.rn_to_vn_conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_vn], data.edge_attr[data.indices_rn_to_vn, :]
                    )
                    # out = self.vn_to_vn_conv_list[i](
                    #     out, data.edge_index[:, data.indices_vn_to_vn], data.edge_attr[data.indices_vn_to_vn, :]
                    # )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_rn], data.edge_attr[data.indices_rn_to_rn, :]
                    )
                    out = self.rn_to_vn_conv_list[i](
                        out, data.edge_index[:, data.indices_rn_to_vn], data.edge_attr[data.indices_rn_to_vn, :]
                    )
                    # out = self.vn_to_vn_conv_list[i](
                    #     out, data.edge_index[:, data.indices_vn_to_vn], data.edge_attr[data.indices_vn_to_vn, :]
                    # )
                    # out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Last Layer
        out = self.vn_conv(out, data.edge_index[:, data.indices_rn_to_vn], data.edge_attr[data.indices_rn_to_vn, :])

        virtual_mask = torch.argwhere(data.z == 100).squeeze(1)
        out = torch.index_select(out, 0, virtual_mask)

        for i in range(0, len(self.post_lin_list)):
            out = self.post_lin_list[i](out)
            out = getattr(F, self.act)(out)
        out = self.lin_out(out)

        '''
        # Post-GNN dense layers
        if self.prediction_level == "graph":
            if self.pool_order == "early":
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)

            elif self.pool_order == "late":
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                    out = self.lin_out_2(out)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)                
        '''
        return out

    def forward(self, data):

        output = {}
        out = self._forward(data)
        output["output"] = out

        if self.gradient == True and out.requires_grad == True:
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :],
                                  torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)
            grad = torch.autograd.grad(
                out,
                [data.pos, data.displacement],
                grad_outputs=torch.ones_like(out),
                create_graph=self.training)
            forces = -1 * grad[0]
            stress = grad[1]
            stress = stress / volume.view(-1, 1, 1)

            output["pos_grad"] = forces
            output["cell_grad"] = stress
        else:
            output["pos_grad"] = None
            output["cell_grad"] = None

        return output