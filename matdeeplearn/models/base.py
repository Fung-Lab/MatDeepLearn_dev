import torch
import warnings
import torch.nn as nn
from torch_geometric.nn import radius_graph

from matdeeplearn.process.helpers import *

class BaseModel(nn.Module):
    def __init__(
        self,
        edge_steps: int = 50,
        self_loop: bool = True
    ) -> None:
        super(BaseModel, self).__init__()
        self.edge_steps = edge_steps
        self.self_loop = self_loop

    def generate_graph(
        self,
        data,
        r,
        n_neighbors,
        otf: bool = False
    ):
        '''
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
        '''
        if not otf:
            warnings.warn('On-the-fly graph generation is called but otf is False')
            return

        # get cutoff distance matrix
        cd_matrix = get_cutoff_distance_matrix(
            data.pos, data.cell, r, n_neighbors
        )

        n_atoms = data.n_atoms.item()

        edge_indices, edge_weights, cd_matrix_masked = add_selfloop(
            n_atoms, *dense_to_sparse(cd_matrix), cd_matrix, self_loop=self.self_loop
        )

        data.edge_index, data.edge_weight = edge_indices, edge_weights

        # generate node features
        generate_node_features(data, n_neighbors)
        # TODO
        # check if edge features that is normalized over the entire dataset can be skipped
        generate_edge_features(data, self.edge_steps)

        return data