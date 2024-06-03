# TODO: make sure that parameters are same for both checkpointed and non-checkpointed runs

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

import logging

@registry.register_model("CGCNN")
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
        
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff_radius, self.edge_dim, 0.2)

        # Determine gc dimension and post_fc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.node_dim, self.node_dim
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        # setup layers
        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()
        
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
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.node_dim, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):
            conv = CGConv(
                self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
            )
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                if self.gradient_checkpointing:
                    bn = BatchNorm1d(
                        self.gc_dim, track_running_stats=self.batch_track_stats
                    )
                else:
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

    # Pre-GNN dense layers
    def pre_run_function(self, module):
        def custom_forward(*inputs):
            out = module(*inputs)
            out = getattr(F, self.act)(out)
            return out
        return custom_forward

    # GNN layers
    def run_function(self, module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    # Post-GNN dense layers
    def post_run_function(self, module):
        def custom_forward(*inputs):
            out = module(*inputs)
            out = getattr(F, self.act)(out)
            return out
        return custom_forward

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):    
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, _, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
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
        for i in range(0, len(self.pre_lin_list)):
            if self.gradient_checkpointing:
                if i == 0:
                    out = torch.utils.checkpoint.checkpoint(self.pre_run_function(self.pre_lin_list[i]), data.x, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                else:
                    out = torch.utils.checkpoint.checkpoint(self.pre_run_function(self.pre_lin_list[i]), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
            else:
                if i == 0:
                    out = self.pre_lin_list[i](data.x)
                    out = getattr(F, self.act)(out)
                else:
                    out = self.pre_lin_list[i](out)
                    out = getattr(F, self.act)(out)

        # GNN layers
        for i in range(0, len(self.conv_list)):
            if self.gradient_checkpointing:
                if len(self.pre_lin_list) == 0 and i == 0:
                    if self.batch_norm:
                        out = torch.utils.checkpoint.checkpoint(self.run_function(self.conv_list[i]), data.x, data.edge_index, data.edge_attr, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                        out = torch.utils.checkpoint.checkpoint(self.bn_list[i], out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                    else:
                        out = torch.utils.checkpoint.checkpoint(self.run_function(self.conv_list[i]), data.x, data.edge_index, data.edge_attr, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                else:
                    if self.batch_norm:  
                        out = torch.utils.checkpoint.checkpoint(self.run_function(self.conv_list[i]), out, data.edge_index, data.edge_attr, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                        out = torch.utils.checkpoint.checkpoint(self.bn_list[i], out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                    else:
                        out = torch.utils.checkpoint.checkpoint(self.run_function(self.conv_list[i]), out, data.edge_index, data.edge_attr, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
            else:
                if len(self.pre_lin_list) == 0 and i == 0:
                    if self.batch_norm:
                        out = self.conv_list[i](
                            data.x, data.edge_index, data.edge_attr
                        )
                        out = self.bn_list[i](out)
                    else:
                        out = self.conv_list[i](
                            data.x, data.edge_index, data.edge_attr
                        )
                else:
                    if self.batch_norm:  
                        out = self.conv_list[i](
                            out, data.edge_index, data.edge_attr
                        )
                        out = self.bn_list[i](out)
                    else:
                        out = self.conv_list[i](
                            out, data.edge_index, data.edge_attr
                        )
                        # out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Post-GNN dense layers
        if self.prediction_level == "graph":
            if self.pool_order == "early":
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                for i in range(0, len(self.post_lin_list)):
                    if self.gradient_checkpointing:
                        out = torch.utils.checkpoint.checkpoint(self.post_run_function(self.post_lin_list[i]), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                    else:
                        out = self.post_lin_list[i](out)
                        out = getattr(F, self.act)(out)
                out = torch.utils.checkpoint.checkpoint(self.run_function(self.lin_out), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
    
            elif self.pool_order == "late":
                for i in range(0, len(self.post_lin_list)):
                    if self.gradient_checkpointing:
                        out = torch.utils.checkpoint.checkpoint(self.post_run_function(self.post_lin_list[i]), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                    else:
                        out = self.post_lin_list[i](out)
                        out = getattr(F, self.act)(out)
                out = torch.utils.checkpoint.checkpoint(self.run_function(self.lin_out), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                    out = torch.utils.checkpoint.checkpoint(self.run_function(self.lin_out_2), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                    
        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list)):
                if self.gradient_checkpointing:
                    out = torch.utils.checkpoint.checkpoint(self.post_run_function(self.post_lin_list[i]), out, preserve_rng_state=self.preserve_rng_state, use_reentrant=self.use_reentrant)
                else:
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
            out = self.lin_out(out) # gc here?             
                     
        return out   
        
        
    def forward(self, data):
        
        output = {}
        out = self._forward(data)
        output["output"] =  out

        if self.gradient == True and out.requires_grad == True:         
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :], torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)                      
            grad = torch.autograd.grad(
                    out,
                    [data.pos, data.displacement],
                    grad_outputs=torch.ones_like(out),
                    create_graph=self.training) 
            forces = -1 * grad[0]
            stress = grad[1]
            stress = stress / volume.view(-1, 1, 1)             

            output["pos_grad"] =  forces
            output["cell_grad"] =  stress
        else:
            output["pos_grad"] =  None
            output["cell_grad"] =  None    
                  
        return output