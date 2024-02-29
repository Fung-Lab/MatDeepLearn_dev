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

@registry.register_model("Morse_Old_Si")
class Morse_Old_Si(BaseModel):
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
        super(Morse_Old_Si, self).__init__(**kwargs)

        self.combination_method = kwargs.get('combination_method', 'average')
        self.with_coefs = kwargs.get("with_coefs", False)

        self.rm_sig_d_o_o = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True),
                                         Parameter(torch.ones(1), requires_grad=True)]).to('cuda:0')
        self.rm_sig_d_si_si = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True),
                                         Parameter(torch.ones(1), requires_grad=True)]).to('cuda:0')
        self.rm_sig_d_si_o = ParameterList([Parameter(torch.ones(1), requires_grad=True),
                                         Parameter(1.5 * torch.ones(1), requires_grad=True),
                                         Parameter(torch.ones(1), requires_grad=True)]).to('cuda:0')
        
        self.base_atomic_energy = ParameterList([Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        
        if self.with_coefs:
            self.coef_e = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
            self.coef_2e = ParameterList([Parameter(torch.ones(1, 1), requires_grad=True) for _ in range(100)]).to('cuda:0')
        
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

        return self.morse_potential(data)
    
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
        
    def pairwise_morse(self, data, rm, sigma, D, mask):
        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight[mask]
        fc = self.cutoff_function(d, ro, rc)
        E = D * (1 - torch.exp(-sigma * (d - rm))) ** 2 - D
        
        pairwise_energies = 0.5 * (E * fc)

        edge_idx_to_graph = data.batch[data.edge_index[0, mask]]
        return 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
    
    def morse_potential(self, data):
        atoms = data.z[data.edge_index] - 1
        
        base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
        
        for z in np.unique(data.z.cpu()):
            base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
            
        oxygen_oxygen_mask = (atoms[0] == 7) & (atoms[1] == 7)
        si_si_mask = (atoms[0] == 13) & (atoms[1] == 13)
        si_oxygen_mask = (~oxygen_oxygen_mask) & (~si_si_mask)
            
        morse_out = self.pairwise_morse(data, self.rm_sig_d_o_o[0], self.rm_sig_d_o_o[1], self.rm_sig_d_o_o[2], oxygen_oxygen_mask) +\
            self.pairwise_morse(data, self.rm_sig_d_si_si[0], self.rm_sig_d_si_si[1], self.rm_sig_d_si_si[2], si_si_mask) +\
            self.pairwise_morse(data, self.rm_sig_d_si_o[0], self.rm_sig_d_si_o[1], self.rm_sig_d_si_o[2], si_oxygen_mask)
    
        base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
    
        base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        return morse_out.reshape(-1, 1) + base_atomic_energy.reshape(-1, 1)