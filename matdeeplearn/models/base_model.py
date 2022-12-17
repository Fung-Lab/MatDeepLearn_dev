import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from matdeeplearn.preprocessor.helpers import (
    add_selfloop,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
)


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, edge_steps: int = 50, self_loop: bool = True) -> None:
        super(BaseModel, self).__init__()
        self.edge_steps = edge_steps
        self.self_loop = self_loop

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

    def generate_graph(self, data, r, n_neighbors, otf: bool = False):
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
        if not otf:
            warnings.warn("On-the-fly graph generation is called but otf is False")
            return

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

        return data
