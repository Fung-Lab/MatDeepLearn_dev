from math import prod
from time import time
import operator

import numpy as np
from scipy import interpolate
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
import logging

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot

@registry.register_model("Spline_New_Si")
class Spline_New_Si(BaseModel):
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
        super(Spline_New_Si, self).__init__(**kwargs)

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
        
        n_intervals = kwargs.get('n_intervals', 25)
        self.degree = kwargs.get('degree', 3)
        
        self.n_trail = kwargs.get('n_trail', 1)
        self.n_lead = kwargs.get('n_lead', 0)
        
        init_knots = torch.linspace(0, self.cutoff_radius, n_intervals + 1, device='cuda:0')
        knot_dist = init_knots[1] - init_knots[0]
        self.t = init_knots
        self.t = torch.cat([torch.tensor([init_knots[0] - i * knot_dist for i in range(3, 0, -1)], device='cuda:0'),
                            init_knots,
                            torch.tensor([init_knots[-1] + i * knot_dist for i in range(1, 4, 1)], device='cuda:0')])
        
        self.n = self.t.shape[0] - self.degree - 1
        
        self.coefs_o_o = ParameterList([Parameter(torch.ones(self.n), requires_grad=True)]).to('cuda:0')
        self.coefs_si_si = ParameterList([Parameter(torch.ones(self.n), requires_grad=True)]).to('cuda:0')
        self.coefs_si_o = ParameterList([Parameter(torch.ones(self.n), requires_grad=True)]).to('cuda:0')

        self.use_atomic_energy = kwargs.get('use_atomic_energy', True)
        if self.use_atomic_energy:
            self.base_atomic_energy = ParameterList([Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff_radius, self.edge_dim, 0.2)

    @property
    def target_attr(self):
        return "y"

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

        return self.compute_b_spline(data)
        
    def forward(self, data):
        
        output = {}
        out = self._forward(data)
        # output["c"] = self.coefs
        output["output"] = out

        if self.gradient == True and out.requires_grad == True:         
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :], torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)                      
            grad = torch.autograd.grad(
                    out,
                    [data.pos, data.displacement],
                    grad_outputs=torch.ones_like(out),
                    create_graph=self.training,
                    )
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
    
    def vec_find_intervals(self, t: torch.tensor, x: torch.tensor, k: int):
        n = t.shape[0] - k - 1
        tb = t[k]
        te = t[n]

        result = torch.zeros_like(x, dtype=int, device='cuda:0')

        nan_indices = torch.isnan(x)
        out_of_bounds_indices = torch.logical_or(x < tb, x > te)
        result[out_of_bounds_indices | nan_indices] = -1

        l = torch.full_like(x, k, dtype=int)
        while torch.any(torch.logical_and(x < t[l], l != k)):
            l = torch.where(torch.logical_and(x < t[l], l != k), l - 1, l)

        l = torch.where(l != n, l + 1, l)

        while torch.any(torch.logical_and(x >= t[l], l != n)):
            l = torch.where(torch.logical_and(x >= t[l], l != n), l + 1, l)

        result = torch.where(result != -1, l - 1, result)
        return result
    
    def evaluate_spline(self, t, c, k, xp, out):
        if out.shape[0] != xp.shape[0]:
            raise ValueError("out and xp have incompatible shapes")

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
    
    def pair_b_spline(self, data, c, mask):
        x = data.edge_weight[mask]
        out = torch.empty((len(x), 1), dtype=c.dtype, device='cuda:0')
        spline_res = torch.nan_to_num(self.evaluate_spline(self.t, c.reshape(c.shape[0], -1), self.degree, x, out), nan=0).squeeze()
        # print(spline_res.shape)
        edge_idx_to_graph = data.batch[data.edge_index[0, mask]]
        return 0.5 * scatter_add(spline_res, index=edge_idx_to_graph, dim_size=len(data))
        
        
    def compute_b_spline(self, data):
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        
        if self.use_atomic_energy:
            for z in np.unique(data.z.cpu()):
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
        
        spline_res = torch.empty(len(data.edge_weight), device='cuda:0')
        
        atoms = data.z[data.edge_index] - 1
        print(torch.min(data.edge_weight))
        
        oxygen_oxygen_mask = (atoms[0] == 7) & (atoms[1] == 7)
        si_si_mask = (atoms[0] == 13) & (atoms[1] == 13)
        si_oxygen_mask = (~oxygen_oxygen_mask) & (~si_si_mask)

        spline_out = self.pair_b_spline(data, self.coefs_o_o[0], oxygen_oxygen_mask)\
            + self.pair_b_spline(data, self.coefs_si_si[0], si_si_mask)\
            + self.pair_b_spline(data, self.coefs_si_o[0], si_oxygen_mask)
        
        if self.use_atomic_energy:
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        res = spline_out.reshape(-1, 1)
        if self.use_atomic_energy:
            res += base_atomic_energy.reshape(-1, 1)
        return res
        