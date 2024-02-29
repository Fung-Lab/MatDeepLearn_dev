from math import prod
from time import time

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

@registry.register_model("Spline_Si")
class Spline_Si(BaseModel):
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
        super(Spline_Si, self).__init__(**kwargs)

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
        
        subinterval_size = kwargs.get('subinterval_size', 5)
        n_intervals = kwargs.get('n_intervals', 25)
        self.n_trail = kwargs.get('n_trail', 1)
        self.n_lead = kwargs.get('n_lead', 0)
        self.n_repeats = kwargs.get('n_repeats', 3)
        
        init_knots = torch.linspace(0, self.cutoff_radius, n_intervals + 1)
        init_knots = torch.cat([torch.tensor([init_knots[0] for _ in range(self.n_repeats, 0, -1)]),
                                init_knots,
                                torch.tensor([init_knots[-1] for _ in range(self.n_repeats)])])
        self.subintervals = torch.stack([init_knots[i:i+subinterval_size] for i in range(len(init_knots)-subinterval_size+1)])
        
        self.coefs_o_o = ParameterList([Parameter(torch.ones(self.subintervals.shape[0] - self.n_trail - self.n_lead), requires_grad=True)]).to('cuda:0')
        self.coefs_si_si = ParameterList([Parameter(torch.ones(self.subintervals.shape[0] - self.n_trail - self.n_lead), requires_grad=True)]).to('cuda:0')
        self.coefs_si_o = ParameterList([Parameter(torch.ones(self.subintervals.shape[0] - self.n_trail - self.n_lead), requires_grad=True)]).to('cuda:0')

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
        c = c[intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].squeeze(dim=2)
        out[~invalid_mask, :] = torch.sum(work[~invalid_mask, :k+1] * c, dim=1).unsqueeze(dim=-1)
        return out
    
    def pair_b_spline(self, atoms, data, coef, mask):
        
        x = data.edge_weight[mask]
        out_stack = None
        
        for i in range(self.n_lead, len(self.subintervals) - self.n_trail):
            t = self.subintervals[i]
            k = len(t) - 2
            t = torch.cat([torch.tensor((t[0]-1,) * k), t, torch.tensor((t[-1]+1,) * k)]).to('cuda:0')
            c = torch.zeros_like(t, device='cuda:0')
            c[k] = 1.
            
            out = torch.empty((len(x), prod(c.shape[1:])), dtype=c.dtype, device='cuda:0')
            res = torch.nan_to_num(self.evaluate_spline(t, c.reshape(c.shape[0], -1), k, x, out), nan=0)
            interval_res = res * coef[i-self.n_lead].view(-1, 1)
            out_stack = interval_res if out_stack is None else torch.cat([out_stack, interval_res], dim=-1)

        spline_res = torch.sum(out_stack, dim=1)
        edge_idx_to_graph = data.batch[data.edge_index[0, mask]]
        spline_out = 0.5 * scatter_add(spline_res, index=edge_idx_to_graph, dim_size=len(data))
        return spline_out
        
    def compute_b_spline(self, data):
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        for z in np.unique(data.z.cpu()):
            if self.use_atomic_energy:
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
        
        atoms = data.z[data.edge_index] - 1
        
        oxygen_oxygen_mask = (atoms[0] == 7) & (atoms[1] == 7)
        si_si_mask = (atoms[0] == 13) & (atoms[1] == 13)
        si_oxygen_mask = (~oxygen_oxygen_mask) & (~si_si_mask)

        spline_out = self.pair_b_spline(atoms, data, self.coefs_o_o[0], oxygen_oxygen_mask)\
            + self.pair_b_spline(atoms, data, self.coefs_si_si[0], si_si_mask)\
            + self.pair_b_spline(atoms, data, self.coefs_si_o[0], si_oxygen_mask)
        
        if self.use_atomic_energy:
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        res = -spline_out.reshape(-1, 1)
        if self.use_atomic_energy:
            res += base_atomic_energy.reshape(-1, 1)
        return res
        