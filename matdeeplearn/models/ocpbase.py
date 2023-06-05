
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from matdeeplearn.models.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None, prediction_level="graph",
        otf_edge=False,
        graph_method="ocp",
        gradient=False,
        gradient_method="nequip",
        cutoff_radius=8,
        n_neighbors=None,
        edge_steps=50,        
        num_offsets=1,        
        **kwargs):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets
        self.prediction_level = prediction_level
        self.otf_edge = otf_edge
        self.gradient = gradient
        self.cutoff_radius = cutoff_radius
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.graph_method = graph_method
        self.num_offsets = num_offsets
        self.gradient_method = gradient_method

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(self, data, cutoff_radius, n_neighbors):
        """
        generates the graph on-the-fly.

        Parameters
        ----------
            data: torch_geometric.data.data.Data
                data for which graph is to be generated

            r: float
                cutoff radius

            n_neighbors: int
                max number of neighbors

            otf: bool
                otf == on-the-fly
                if True, this function will be called
        """
        #if not otf:
        #    warnings.warn("On-the-fly graph generation is called but otf is False")
        #    return
        if self.gradient_method == "conventional":
            data.pos.requires_grad_(True)
            data.cell.requires_grad_(True)
        elif self.gradient_method == "nequip":
            data.pos.requires_grad_(True)
            data.displacement = torch.zeros_like(data.cell)
            data.displacement.requires_grad_(True)
            symmetric_displacement = 0.5 * (data.displacement + data.displacement.transpose(-1, -2))
            data.cell = data.cell + torch.bmm(data.cell, symmetric_displacement) 
                    
        if self.graph_method == "ocp":
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                cutoff_radius,
                n_neighbors,
                data.pos,
                data.cell,
                data.n_atoms,
                [True, True, True],
            )
                                  
            edge_gen_out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )
            edge_index = edge_gen_out["edge_index"]
            edge_weights = edge_gen_out["distances"]
            offset_distance = edge_gen_out["offsets"]                
            edge_vec = edge_gen_out["distance_vec"]
            if(edge_vec.dim() > 2):
                edge_vec = edge_vec[edge_indices[0], edge_indices[1]]      
                              
        elif self.graph_method == "mdl":
            edge_index_list = []
            edge_weights_list = []
            edge_vec_list = []
            cell_offsets_list = []
            for i in range(0, len(data)):
                
                cutoff_distance_matrix, cell_offsets, edge_vec = get_cutoff_distance_matrix(
                    data[i].pos,
                    data[i].cell,
                    cutoff_radius,
                    n_neighbors,
                    self.num_offsets,
                )
        
                edge_index, edge_weights = dense_to_sparse(cutoff_distance_matrix)
        
                # get into correct shape for model stage
                edge_vec = edge_vec[edge_index[0], edge_index[1]]
                
                edge_index_list.append(edge_index)                
                edge_weights_list.append(edge_weights)
                edge_vec_list.append(edge_vec)
                cell_offsets_list.append(cell_offsets)
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_weights = torch.cat(edge_weights_list)
            edge_vec = torch.cat(edge_vec_list)
            cell_offsets = torch.cat(cell_offsets_list)
            neighbors = None
            offset_distance = None
        #print(edge_index.shape, edge_weights.shape, edge_vec.shape, cell_offsets.shape, neighbors.shape, offset_distance.shape)
        
        '''
        # get cutoff distance matrix
        cd_matrix, cell_offsets = get_cutoff_distance_matrix(
            data.pos, data.cell, r, n_neighbors
        )

        n_atoms = data.n_atoms.item()

        edge_indices, edge_weights, cd_matrix_masked = add_selfloop(
            n_atoms, *dense_to_sparse(cd_matrix), cd_matrix, self_loop=self.self_loop
        )

        data.edge_index, data.edge_weight = edge_indices, edge_weights
        data.cell_offsets = cell_offsets

        # generate node features
        generate_node_features(data, n_neighbors)
        # TODO
        # check if edge features that is normalized over the entire dataset can be skipped
        generate_edge_features(data, self.edge_steps)
        '''
        return (
            edge_index,
            edge_weights,
            edge_vec,
            cell_offsets,
            offset_distance,
            neighbors,
        )
    def conditional_grad(dec):
        "Decorator to enable/disable grad depending on whether force/energy predictions are being made"
        # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
        def decorator(func):
            @wraps(func)
            def cls_method(self, *args, **kwargs):
                f = func
                if self.gradient == True:
                    f = dec(func)
                return f(self, *args, **kwargs)

            return cls_method

        return decorator

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    @property
    def target_attr(self):
        return "y"