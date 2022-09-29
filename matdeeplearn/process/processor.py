from tabnanny import verbose
from typing_extensions import dataclass_transform
import numpy as np
from pathlib import Path
import warnings, yaml, ase, os, torch, json
from ase import io

from .helpers import *

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops


def process_data(dataset_config):
    root_path = dataset_config['src']
    target_path = dataset_config['target_path']
    cutoff_radius = dataset_config['cutoff_radius']
    n_neighbors = dataset_config['n_neighbors']
    edge_steps = dataset_config['edge_steps']
    data_format = dataset_config.get('data_format', 'json')
    self_loop = dataset_config.get("self_loop", True)
    node_representation = dataset_config.get('node_representation', 'onehot') ,
    verbose: bool = dataset_config.get('verbose', True)

    processor = DataProcessor(root_path, target_path, cutoff_radius, n_neighbors, edge_steps, data_format, self_loop, node_representation, verbose)
    processor.process()

class DataProcessor():
    def __init__(
        self,
        root_path: str,
        target_file_path: str,
        r: float,
        n_neighbors: int,
        edge_steps: int,
        data_format: str = 'json',
        self_loop: bool = True,
        node_representation: str = 'onehot',
        verbose: bool = True
    ) -> None:
        '''
        create a DataProcessor that processes the raw data and save into data.pt file.

        Parameters
        ----------
            root_path: str
                a path to the root folder for all data

            target_file_path: str
                a path to a CSV file containing target y values

            r: float
                cutoff radius

            n_neighbors: int
                max number of neighbors to be considered
                => closest n neighbors will be kept

            edge_steps: int
                step size for creating Gaussian basis for edges
                used in torch.linspace

            data_format: str
                format of the raw data file

            self_loop: bool
                if True, every node in a graph will have a self loop

            node_representation: str
                a path to a JSON file containing vectorized representations of elements
                of atomic numbers from 1 to 100 inclusive.

                default: one-hot representation
        '''

        self.root_path = root_path
        self.target_file_path = target_file_path
        self.r = r
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.data_format = data_format
        self.self_loop = self_loop
        self.node_representation = node_representation
        self.verbose = verbose

        self.y = self.load_target()

    def load_target(self):
        '''
        load target values as numpy.ndarray
        '''
        y = np.genfromtxt(self.target_file_path, delimiter=',')
        return y

    def process(self, save=True):
        n_samples = len(self.y)

        data_list = self.get_data_list()

        # TODO: for future work, need to consider larger-than-memory dataset
        # slices is a dictionary that stores a compressed index representation
        # of each attribute and is needed to re-construct individual elements
        # from mini-batches.

        data, slices = InMemoryDataset.collate(data_list)
        if save:
            save_path = os.path.join(self.root_path, "processed/data.pt")
            torch.save((data, slices), save_path)
            if self.verbose:
                print('Dataset saved successfully to {save_path}')

        return data, slices

    def get_data_list(self):
        n_samples = len(self.y)
        data_list = [Data() for _ in range(n_samples)]

        for i, row in enumerate(self.y):
            # TODO: need to better way to handle different datatypes in one CSV (see self.load_target()).
            #       currently structure_id has to be int and target_val is float
            structure_id, target_val = row[0], row[1:]
            structure_id = str(int(structure_id))
            # type(data) == torch_geometric.data.Data()
            data = data_list[i]

            # read structure data into ase Atoms object
            ase_crystal = ase.io.read(
                # TODO: generalize to other data formats
                os.path.join(self.root_path, 'raw/' + structure_id + '.' + self.data_format)
            )

            # get cutoff_distance matrix
            pos = ase_crystal.get_positions()
            cell = np.array(ase_crystal.get_cell())
            cd_matrix = get_cutoff_distance_matrix(pos, cell, self.r, self.n_neighbors)
            edge_indices, edge_weights, cd_matrix_masked = add_selfloop(
                len(ase_crystal),
                *dense_to_sparse(cd_matrix),
                cd_matrix,
                self_loop=self.self_loop
            )

            data.n_atoms = len(ase_crystal)
            data.ase = ase_crystal
            data.pos = pos
            data.cell = cell
            data.y = torch.Tensor(np.array([target_val], dtype=np.float32))
            data.z = torch.LongTensor(ase_crystal.get_atomic_numbers())
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights

            data.edge_descriptor= {}
            data.edge_descriptor['mask'] = cd_matrix_masked
            data.edge_descriptor['distance'] = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]

        # add node features
        generate_node_features(data_list, self.n_neighbors)

        # add edge features
        generate_edge_features(data_list, self.edge_steps)

        # clean up
        clean_up(data_list, ['ase'])

        return data_list
