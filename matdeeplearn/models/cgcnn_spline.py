from math import prod

import numpy as np
import logging
from time import time

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

@registry.register_model("CGCNN_Spline")
class CGCNN_Spline(BaseModel):
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
        super(CGCNN_Spline, self).__init__(**kwargs)

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
            
        subinterval_size = kwargs.get('subinterval_size', 5)
        n_intervals = kwargs.get('n_intervals', 25)
        self.n_trail = kwargs.get('n_trail', 1)
        self.n_lead = kwargs.get('n_lead', 1)
        self.n_repeats = kwargs.get('n_repeats', 3)
        
        init_knots = torch.linspace(0, self.cutoff_radius, n_intervals + 1)
        init_knots = torch.cat([torch.tensor([init_knots[0] for _ in range(self.n_repeats, 0, -1)]),
                                init_knots,
                                torch.tensor([init_knots[-1] for _ in range(self.n_repeats)])])
        self.subintervals = torch.stack([init_knots[i:i+subinterval_size] for i in range(len(init_knots)-subinterval_size+1)])
        self.coefs = ParameterList([Parameter(torch.ones(self.subintervals.shape[0] - self.n_trail - self.n_lead), requires_grad=True) for _ in range(100)]).to('cuda:0')

        self.use_atomic_energy = kwargs.get('use_atomic_energy', True)
        if self.use_atomic_energy:
            self.base_atomic_energy = ParameterList([Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        

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

        spline = self.compute_b_spline(data)
        return 0.1 * out + spline
    
    def forward(self, data):
        
        output = {}
        out = self._forward(data)
        output["c"] = self.coefs
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
    
    def vectorized_deboor(self, t, x, k, interval, result):
        hh = result[:, k+1:]
        h = result
        h[:, 0] = 1.
            
        for j in range(1, k + 1):
            hh[:, :j] = h[:, :j]
            h[:, 0] = 0
            for n in range(1, j + 1):
                ind = interval + n
                xb = t[ind]
                xa = t[ind - j]
                zero_mask = xa == xb
                h[zero_mask, n] = 1.
                w = hh[:, n - 1] / (xb - xa)
                h[:, n - 1] += w * (xb - x)
                h[:, n] = w * (x - xa)
                h[zero_mask, n] = 0.
            
        return result
    
    def vec_find_intervals(self, knots: torch.tensor, x: torch.tensor, k: int):
        n = knots.shape[0] - k - 1

        pos_idxs = torch.empty(len(x), device='cuda:0')
        invalid_mask = (x < knots[k]) | (x > knots[n])
        pos_idxs[invalid_mask] = -1

        x_masked = x[~invalid_mask]
        knots = knots[k:n+1]

        t = knots - x_masked.view(-1, 1)
        pos_idxs[~invalid_mask] = torch.argmax((t >= 0).int(), dim=1).to(torch.float) + k - 1
        pos_idxs[x == knots[0]] = k
        return pos_idxs.to(torch.long)
    
    def evaluate_spline(self, t, c, k, xp, out):
        if out.shape[0] != xp.shape[0]:
            raise ValueError("out and xp have incompatible shapes")
        if out.shape[1] != c.shape[1]:
            raise ValueError("out and c have incompatible shapes")

        # work: (N, 2k + 2)
        work = torch.empty(xp.shape[0], 2 * k + 2, dtype=torch.float, device='cuda:0')

        # intervals: (N, )
        intervals = self.vec_find_intervals(t, xp, k)
        invalid_mask = intervals < 0
        out[invalid_mask, :] = np.nan
        out[~invalid_mask, :] = 0.
        
        if invalid_mask.all():
            return out
        
        work[~invalid_mask] = self.vectorized_deboor(t, xp[~invalid_mask], k, intervals[~invalid_mask], work[~invalid_mask])
        
        # c = torch.stack([c[i-k:i+1] for i in intervals[~invalid_mask]]).squeeze(dim=2)
        c = c[intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].squeeze(dim=2)
        out[~invalid_mask, :] = torch.sum(work[~invalid_mask, :k+1] * c, dim=1).unsqueeze(dim=-1)
        return out

    def B(self, x, k, i, t):
        if k == 0:
            return torch.where((t[i] <= x) & (x < t[i+1]), 1.0, 0.0)
        
        c1 = torch.where(t[i+k] == t[i], 0.0, (x - t[i]) / (t[i+k] - t[i]) * self.B(x, k-1, i, t))
        c2 = torch.where(t[i+k+1] == t[i+1], 0.0, (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * self.B(x, k-1, i+1, t))
        
        return c1 + c2
        
    def compute_b_spline(self, data):
        coefs = torch.zeros((len(self.coefs), self.subintervals.shape[0] - self.n_trail - self.n_lead)).to('cuda:0')
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        for z in np.unique(data.z.cpu()):
            if self.use_atomic_energy:
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
            coefs[z - 1] = self.coefs[z - 1]
        
        atoms = data.z[data.edge_index] - 1
        coef_i, coef_j = coefs[atoms[0]], coefs[atoms[1]]
        coef = (coef_i + coef_j) / 2
        
        x = data.edge_weight
        # coef[:, :self.n_lead] = 
        
        out_stack = None
        for i in range(self.n_lead, len(self.subintervals) - self.n_trail):
            t = self.subintervals[i]
            k = len(t) - 2
            t = torch.cat([torch.tensor((t[0]-1,) * k), t, torch.tensor((t[-1]+1,) * k)]).to('cuda:0')
            c = torch.zeros_like(t, device='cuda:0')
            c[k] = 1.
            
            out = torch.empty((len(x), prod(c.shape[1:])), dtype=c.dtype, device='cuda:0')
            res = torch.nan_to_num(self.evaluate_spline(t, c.reshape(c.shape[0], -1), k, x, out), nan=0)
            
            # print(res.shape, coef.shape, coefs.shape)
            interval_res = res * coef[:, i-self.n_lead].view(-1, 1)
    
            out_stack = interval_res if out_stack is None else torch.cat([out_stack, interval_res], dim=-1)

        # print(out_stack.shape)
        spline_res = torch.sum(out_stack, dim=1)
        
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        spline_out = 0.5 * scatter_add(spline_res, index=edge_idx_to_graph, dim_size=len(data))
        
        if self.use_atomic_energy:
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        res = -spline_out.reshape(-1, 1)
        if self.use_atomic_energy:
            res += base_atomic_energy.reshape(-1, 1)
        return res
        