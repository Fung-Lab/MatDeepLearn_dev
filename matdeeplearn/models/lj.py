import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
from torch.nn import Parameter
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

@registry.register_model("LJ")
class LJ(BaseModel):
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
        super(LJ, self).__init__(**kwargs)

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

        self.sigmas = Parameter(5 * torch.ones(100, 100), requires_grad=True)
        self.epsilons = Parameter(5 * torch.ones(100, 100), requires_grad=True)
        
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
            
            
        out = torch.zeros((len(data), 1)).to('cuda:0')

        num_edges = len(data.edge_index[0])
        atoms = torch.zeros(2, num_edges, dtype=torch.int64)

        atoms[0] = data.z[data.edge_index[0]] - 1
        atoms[1] = data.z[data.edge_index[1]] - 1

        flat_index = atoms[0] * 100 + atoms[1]
        sigma = self.sigmas.view(-1)[flat_index].view(num_edges, 1).squeeze()
        epsilon = self.epsilons.view(-1)[flat_index].view(num_edges, 1).squeeze()
        
        mask = torch.zeros(100, 100).to('cuda:0')
        mask.view(-1, 1)[flat_index] = 1
        
        rc = self.cutoff_radius
        ro = 0.66 * rc
        r2 = data.edge_weight ** 2

        c6 = (sigma ** 2 / r2) ** 3
        c6[r2 > rc ** 2] = 0.0
        c12 = c6 ** 2
        cutoff_fn = self.cutoff_function(r2, rc**2, ro**2)
        pairwise_energies = 4 * epsilon * (c12 - c6)
        pairwise_energies *= cutoff_fn
        #lennard_jones_out = 0.5 * pairwise_energies.sum()

        edge_idx_to_graph = data.batch[data.edge_index[0]]
        lennard_jones_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
        
        return out + lennard_jones_out.reshape(-1, 1), mask
    
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
        out, mask = self._forward(data)
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

        return output, mask