import warnings
from abc import ABCMeta, abstractmethod
from functools import wraps

import numpy as np 
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from torch_scatter import segment_coo, segment_csr

from matdeeplearn.preprocessor.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
    calculate_edges_master,
    get_pbc_distances,
    radius_graph_pbc,
)


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self,        
        prediction_level="graph",
        otf_edge_index=False,
        otf_edge_attr=False,
        otf_node_attr=False,
        graph_method="ocp",
        gradient=False,
        cutoff_radius=8,
        n_neighbors=None,
        edge_dim=50,        
        num_offsets=1,        
        **kwargs
        ) -> None:
        super(BaseModel, self).__init__()
        
        self.prediction_level = prediction_level
        self.otf_edge_index = otf_edge_index
        self.otf_edge_attr = otf_edge_attr
        self.otf_node_attr = otf_node_attr
        self.gradient = gradient
        self.cutoff_radius = cutoff_radius
        self.n_neighbors = n_neighbors
        self.edge_dim = edge_dim
        self.graph_method = graph_method
        self.num_offsets = num_offsets
        
    @property
    @abstractmethod
    def target_attr(self):
        """Specifies the target attribute property for writing output to file"""

    def __str__(self):
        # Prints model summary
        str_representation = "\n"
        model_params_list = list(self.named_parameters())
        separator = (
            "--------------------------------------------------------------------------"
        )
        str_representation += separator + "\n"
        line_new = "{:>30}  {:>20} {:>20}".format(
            "Layer.Parameter", "Param Tensor Shape", "Param #"
        )
        str_representation += line_new + "\n" + separator + "\n"
        for elem in model_params_list:
            p_name = elem[0]
            p_shape = list(elem[1].size())
            p_count = torch.tensor(elem[1].size()).prod().item()
            line_new = "{:>30}  {:>20} {:>20}".format(
                p_name, str(p_shape), str(p_count)
            )
            str_representation += line_new + "\n"
        str_representation += separator + "\n"
        total_params = sum([param.nelement() for param in self.parameters()])
        str_representation += f"Total params: {total_params}" + "\n"
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        str_representation += f"Trainable params: {num_trainable_params}" + "\n"
        str_representation += (
            f"Non-trainable params: {total_params - num_trainable_params}"
        )

        return str_representation

    @abstractmethod
    def forward(self):
        """The forward method for the model."""

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
        """

        #For calculation of stress, see https://github.com/mir-group/nequip/blob/main/nequip/nn/_grad_output.py
        #Originally from: https://github.com/atomistic-machine-learning/schnetpack/issues/165                 
        if self.gradient:
            data.pos.requires_grad_(True)
            data.displacement = torch.zeros((len(data), 3, 3), dtype=data.pos.dtype, device=data.pos.device)            
            data.displacement.requires_grad_(True)
            symmetric_displacement = 0.5 * (data.displacement + data.displacement.transpose(-1, -2))
            data.pos = data.pos + torch.bmm(data.pos.unsqueeze(-2), symmetric_displacement[data.batch]).squeeze(-2)            
            data.cell = data.cell + torch.bmm(data.cell, symmetric_displacement) 

        if torch.sum(data.cell) == 0:
            self.graph_method = "mdl"

        #Can differ from non-otf if amp=True for a very small percentage of edges ~0.01%                    
        if self.graph_method == "ocp":
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                cutoff_radius,
                n_neighbors,
                data.pos,
                data.cell,
                data.n_atoms,
                [True, True, True],
                self.num_offsets,
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
            count = 0
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

                edge_index = edge_index + count
                count = count + data[i].pos.shape[0]
                                
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
        generate_edge_features(data, self.edge_dim)
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