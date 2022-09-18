import numpy as np
from pathlib import Path
import warnings, yaml, ase, os, torch, json
from ase import io

from .utils import *

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops

class DataProcessor():
    '''
    Converts structural data to graph data

    1) load dictionary (part of 4)
    2) load target
    3) create structure graphs
    4) generate node features
    5) vornooi connectivity (X)
    6) SOAP & SM features
    7) generate edge features
    8) save as pt file
    '''

    def __init__(
        self,
        root_path: str,
        target_file_path: str,
        r: float,
        n_neighbors: int,
        edge_steps: int,
        self_loop: bool = True,
        node_representation: str = 'onehot'
    ) -> None:
        '''
        initialize

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
        self.self_loop = self_loop
        self.node_representation = node_representation

        self.y = self.load_target()

    def load_target(self):
        '''
        load target values as numpy.ndarray
        '''
        y = np.genfromtxt(self.target_file_path, delimiter=',')
        return y

    def process(self):
        n_samples = len(self.y)

        data_list = self.get_data_list()

        # TODO: for future work, need to consider larger-than-memory dataset
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(self.root_path, "data.pt"))

        return data_list

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
                os.path.join(self.root_path, structure_id + '.json')
            )

            # get cutoff_distance matrix
            cd_matrix = self.get_cutoff_distance_matrix(ase_crystal)
            edge_indices, edge_weights, cd_matrix_masked = self.add_selfloop(
                len(ase_crystal), 
                *dense_to_sparse(cd_matrix),
                cd_matrix
            )

            data.ase = ase_crystal
            data.y = torch.Tensor(np.array([target_val], dtype=np.float32))
            data.z = torch.LongTensor(ase_crystal.get_atomic_numbers())
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights

            data.edge_descriptor= {}
            data.edge_descriptor['mask'] = cd_matrix_masked
            data.edge_descriptor['distance'] = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]
        
        # add node features
        self.generate_node_features(data_list)

        # add edge features
        self.generate_edge_features(data_list)

        # clean up
        clean_up(data_list, ['ase', 'edge_descriptor'])

        return data_list
    
    def get_cutoff_distance_matrix(self, ase_crystal):
        '''
        get the distance matrix

        Parameters
        ----------
            ase_crystal: ase.atoms.Atoms
                ase Atoms object of an input structure
        '''
        # ase_distance_matrix = torch.tensor(ase_crystal.get_all_distances(mic=True), dtype=torch.float)

        # unit cell offset number
        # TODO: need to tune this for elongated structures
        offset_number = 1
        offsets = get_pbc_offsets(np.array(ase_crystal.get_cell()), offset_number)
        distance_matrix, _ = get_distances(ase_crystal.get_positions(), offsets)

        cutoff_distance_matrix = threshold_sort(distance_matrix, self.r, self.n_neighbors, adj=False)

        return torch.Tensor(cutoff_distance_matrix)

    def add_selfloop(self, num_nodes, edge_indices, edge_weights, cutoff_distance_matrix):
        if not self.self_loop:
            return edge_indices, edge_weights, (cutoff_distance_matrix != 0).int()
        
        edge_indices, edge_weights = add_self_loops(
            edge_indices, edge_weights, num_nodes=num_nodes, fill_value=0
        )

        distance_matrix_masked = (cutoff_distance_matrix.fill_diagonal_(1) != 0).int()
        return edge_indices, edge_weights, distance_matrix_masked
    
    def generate_node_features(self, data_list):
        node_reps = self.load_node_representation()

        for i, data in enumerate(data_list):
            atomic_numbers = data.ase.get_atomic_numbers()
            # minus 1 as the reps are 0-indexed but atomic number starts from 1
            data.x = torch.Tensor(node_reps[atomic_numbers-1])

        for i, data in enumerate(data_list):
            data_list[i] = one_hot_degree(data, self.n_neighbors+1)

    def generate_edge_features(self, data_list):
        distance_gaussian = GaussianSmearing(0, 1, self.edge_steps, 0.2)
        normalize_edge(data_list, 'distance')

        for i, data in enumerate(data_list):
            data_list[i].edge_attr = distance_gaussian(data_list[i].edge_descriptor['distance'])
            
    def load_node_representation(self):
        node_rep_path = Path(__file__).parent
        default_reps = {
            'onehot': str(node_rep_path / './node_representations/onehot.csv')
        }

        # print(default_reps['onehot'])

        rep_file_path = self.node_representation
        if self.node_representation in default_reps:
            rep_file_path = default_reps[self.node_representation]
        
        file_type = rep_file_path.split('.')[-1]
        loaded_rep = None

        if file_type == 'csv':
            loaded_rep = np.genfromtxt(rep_file_path, delimiter=',')
            # TODO: need to check if typecasting to integer is needed
            loaded_rep = loaded_rep.astype(int)

        elif file_type == 'json':
            # TODO
            pass

        return loaded_rep