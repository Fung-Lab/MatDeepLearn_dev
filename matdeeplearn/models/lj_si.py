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

@registry.register_model("LJ_Si")
class LJ_Si(BaseModel):
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
        super(LJ_Si, self).__init__(**kwargs)

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
        self.with_exp_coefs = kwargs.get("with_exp_coefs", False)

        self.ep_sig_o_o = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True)]).to('cuda:0')
        self.ep_sig_si_si = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True)]).to('cuda:0')
        self.ep_sig_si_o = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True)]).to('cuda:0')
        
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

        return self.lj_potential(data)
    
    def cutoff_function(self, r, rc, ro):
        """
        
        Smooth cutoff function.

        Goes from 1 to 0 between ro and rc, ensuring
        that u(r) = lj(r) * cutoff_function(r) is C^1.

        Defined as 1 below ro, 0 above rc.

        Note that r, rc, ro are all expected to be squared,
        i.e. `r = r_ij^2`, etc.

        Taken from https://github.com/google/jax-md.

        """

        return torch.where(
            r < ro,
            1.0,
            torch.where(r < rc, (rc - r) ** 2 * (rc + 2 *
                    r - 3 * ro) / (rc - ro) ** 3, 0.0),
        )
        
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
    
    def pairwise_lj(self, data, epsilon, sigma, mask):
        x = data.edge_weight[mask]
        rc = self.cutoff_radius
        ro = 0.66 * rc
        r2 = x ** 2
        cutoff_fn = self.cutoff_function(r2, rc**2, ro**2)
        
        c6 = (sigma ** 2 / r2) ** 3
        c6[r2 > rc ** 2] = 0.0
        c12 = c6 ** 2
                
        pairwise_energies = 4 * epsilon * (c12 - c6)
        
        pairwise_energies *= cutoff_fn

        edge_idx_to_graph = data.batch[data.edge_index[0, mask]]
        return 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
        
    
    def lj_potential(self, data):
        atoms = data.z[data.edge_index] - 1
        base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
    
        for z in np.unique(data.z.cpu()):
            base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
        
        oxygen_oxygen_mask = (atoms[0] == 7) & (atoms[1] == 7)
        si_si_mask = (atoms[0] == 13) & (atoms[1] == 13)
        si_oxygen_mask = (~oxygen_oxygen_mask) & (~si_si_mask)
        
        lennard_jones_out = self.pairwise_lj(data, self.ep_sig_o_o[0], self.ep_sig_o_o[1], oxygen_oxygen_mask) +\
            self.pairwise_lj(data, self.ep_sig_si_si[0], self.ep_sig_si_si[1], si_si_mask) +\
            self.pairwise_lj(data, self.ep_sig_si_o[0], self.ep_sig_si_o[1], si_oxygen_mask)
    
        base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
               
        return lennard_jones_out.reshape(-1, 1) + base_atomic_energy.reshape(-1, 1)
    