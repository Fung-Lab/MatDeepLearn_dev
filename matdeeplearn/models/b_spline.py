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
import logging

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot

@registry.register_model("Spline")
class Spline(BaseModel):
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
        super(Spline, self).__init__(**kwargs)

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
        
        self.b_splines = None
        self.degree = kwargs.get("degree", 2)
        knots_interval = kwargs.get("knots_interval", 1)
        self.knots = torch.arange(0, self.cutoff_radius, knots_interval)
        self.num_splines = len(self.knots) - self.degree - 1
        self.coefs = ParameterList([Parameter(torch.ones(1, self.num_splines), requires_grad=True) for _ in range(100)]).to('cuda:0')
  
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
    
    def B(self, x, k, i, t):
        if k == 0:
            return torch.where((t[i] <= x) & (x < t[i+1]), 1.0, 0.0)
        
        c1 = torch.where(t[i+k] == t[i], 0.0, (x - t[i]) / (t[i+k] - t[i]) * self.B(x, k-1, i, t))
        c2 = torch.where(t[i+k+1] == t[i+1], 0.0, (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * self.B(x, k-1, i+1, t))
        
        return c1 + c2
    
    def compute_b_spline(self, data):
        coefs = torch.zeros((len(self.coefs), self.num_splines)).to('cuda:0')
        base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        for z in np.unique(data.z.cpu()):
            base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
            coefs[z - 1] = self.coefs[z - 1]
        
        atoms = data.z[data.edge_index] - 1
        coef_i, coef_j = coefs[atoms[0]], coefs[atoms[1]]
        coef = (coef_i + coef_j) / 2
        
        x = data.edge_weight
        raw_splines = torch.stack([coef[:, i] * self.B(x, self.degree, i, self.knots) for i in range(self.num_splines)], dim=1)
        spline_res = torch.sum(coef * raw_splines, dim=1)
        # print(coef.shape, raw_splines.shape, spline_res.shape)
        
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        spline_out = 0.5 * scatter_add(spline_res, index=edge_idx_to_graph, dim_size=len(data))
        base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        return spline_out.reshape(-1, 1) + base_atomic_energy.reshape(-1, 1)
        