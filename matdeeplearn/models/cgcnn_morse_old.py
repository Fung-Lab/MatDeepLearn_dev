import logging
from time import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
from torch.nn import Parameter, ParameterList
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

@registry.register_model("CGCNN_Morse_Old")
class CGCNN_Morse_Old(BaseModel):
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
        super(CGCNN_Morse_Old, self).__init__(**kwargs)

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

        self.combination_method = kwargs.get('combination_method', 'average')
        self.with_coefs = kwargs.get("with_coefs", False)
        self.ratio = kwargs.get("gnn_potential_ratio", [1., 1.])

        self.rm = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0') 
        self.sigmas = ParameterList([Parameter(1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.D = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.base_atomic_energy = ParameterList([Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        
        self.return_comb = kwargs.get("return_comb", True)
        self.train_first = kwargs.get("train_first", "gnn")
        
        if self.with_coefs:
            self.coef_e = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
            self.coef_2e = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        
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
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # GNN layers
        for i in range(0, len(self.conv_list)):
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

        morse = self.morse_potential(data)
        if not self.return_comb:
            return self.ratio[0] * out if self.train_first == 'gnn' else self.ratio[1] * morse
        return self.ratio[0] * out + self.ratio[1] * morse
    
    def forward(self, data):
        
        output = {}
        out = self._forward(data)
        output["output"] = out

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
    
    def cutoff_function(self, r, rc, ro):
        """
        Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.
        Ported from JuLIP.jl, https://github.com/JuliaMolSim/JuLIP.jl

        Parameters
        ----------
        rc - inner cutoff radius
        ro - outder cutoff radius
        """""
        s = 1.0 - (r - rc) / (ro - rc)
        return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) *
                            (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))
    
    def morse_potential(self, data):
        atoms = data.z[data.edge_index] - 1
        
        atomic_rm = torch.zeros((len(self.rm), 1)).to('cuda:0')
        atomic_D = torch.zeros((len(self.D), 1)).to('cuda:0')
        atomic_sigmas = torch.zeros((len(self.sigmas), 1)).to('cuda:0')
        base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        
        if self.with_coefs:
            coef_e = torch.zeros((len(self.coef_e), 1)).to('cuda:0')
            coef_2e = torch.zeros((len(self.coef_2e), 1)).to('cuda:0')
    
        for z in np.unique(data.z.cpu()):
            atomic_sigmas[z - 1] = self.sigmas[z - 1]
            atomic_rm[z - 1] = self.rm[z - 1]
            atomic_D[z - 1] = self.D[z - 1]
            
            if self.with_coefs:
                coef_e[z - 1] = self.coef_e[z - 1]
                coef_2e[z - 1] = self.coef_2e[z - 1]
            
            base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
            
        if self.with_coefs:
            coef_e = (coef_e[atoms[0]] + coef_e[atoms[1]]).squeeze() / 2
            coef_2e = (coef_2e[atoms[0]] + coef_2e[atoms[1]]).squeeze() / 2
        
        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight
        fc = self.cutoff_function(d, ro, rc)
        
        rm_i, rm_j = atomic_rm[atoms[0]], atomic_rm[atoms[1]]
        sigma_i, sigma_j = atomic_sigmas[atoms[0]], atomic_sigmas[atoms[1]]
        D_i, D_j = atomic_D[atoms[0]], atomic_D[atoms[1]]

        rm = (rm_i + rm_j).squeeze() / 2
        sigma = (sigma_i + sigma_j).squeeze() / 2
        D = (D_i + D_j).squeeze() / 2
        
        if self.with_coefs:
            E = D * (coef_e - coef_2e * torch.exp(-sigma * (d - rm))) ** 2 - D
        else:
            E = D * (1 - torch.exp(-sigma * (d - rm))) ** 2 - D
        
        pairwise_energies = 0.5 * (E * fc)

        edge_idx_to_graph = data.batch[data.edge_index[0]]
        morse_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
    
        base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        return morse_out.reshape(-1, 1) + base_atomic_energy.reshape(-1, 1)