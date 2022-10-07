import os
import numpy as np
from scipy.stats import rankdata
import ase
from ase import io
import torch
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
from torch_geometric.data.data import Data

def threshold_sort(all_distances, r, n_neighbors):
    A = all_distances.clone().detach()

    # keep n_neighbors only
    N = len(A) - n_neighbors - 1
    if N > 0:
        _, indices = torch.topk(A, N)
        A.scatter_(1, indices, torch.zeros(len(A), len(A)))

    A[A > r] = 0
    return A

def one_hot_degree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


class GaussianSmearing(torch.nn.Module):
    '''
    slightly edited version from pytorch geometric to create edge from gaussian basis
    '''
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def normalize_edge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = get_ranges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)

def get_ranges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max

def clean_up(data_list, attr_list):
    if not attr_list:
        return data_list
    
    for data in data_list:
        for attr in attr_list:
            try:
                delattr(data, attr)
            except:
                continue

def get_distances(
    positions: np.ndarray,
    offsets: torch.Tensor,
    device: str = 'cpu',
    mic: bool = True
):
    '''
    Get pairwise atomic distances

    Parameters
        positions:  np.ndarray
                    positions of atoms in a unit cell
        
        offsets:    torch.Tensor
                    offsets for the unit cell
        
        device:     str
                    torch device type
        
        mic:        bool
                    minimum image convention
    '''
    
    # convert numpy array to torch tensors
    n_atoms = len(positions)
    n_cells = len(offsets)

    positions = torch.tensor(positions, device=device, dtype=torch.float)

    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)
    offsets = offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + offsets

    # calculate pairwise distances
    atomic_distances = torch.linalg.norm(pos1 - pos2, dim=-1)
    # get minimum
    min_atomic_distances, min_indices = torch.min(atomic_distances, dim=-1)

    atom_rij = pos1 - pos2
    min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()

    return min_atomic_distances, atom_rij


def get_pbc_offsets(cell: np.ndarray, offset_number: int, device: str = 'cpu'):
    '''
    Get the periodic boundary condition (PBC) offsets for a unit cell
    
    Parameters
        cell:       np.ndarray
                    unit cell vectors of ase.cell.Cell

        offset_number:  int
                    the number of offsets for the unit cell
                    if == 0: no PBC
                    if == 1: 27-cell offsets (3x3x3)
    '''
    cell = torch.tensor(cell, device=device, dtype=torch.float)

    _range = np.arange(-offset_number, offset_number+1)
    offsets = [list(x) for x in itertools.product(_range, _range, _range)]
    offsets = torch.tensor(offsets, device=device, dtype=torch.float)
    return offsets @ cell

def get_cutoff_distance_matrix(pos, cell, r, n_neighbors, offset_number=1):
    '''
    get the distance matrix
    TODO: need to tune this for elongated structures

    Parameters
    ----------
        pos: np.ndarray 
            positions of atoms in a unit cell
            get from crystal.get_positions()
        
        cell: np.ndarray
            unit cell of a ase Atoms object

        r: float
            cutoff radius

        n_neighbors: int
            max number of neighbors to be considered
    '''

    offsets = get_pbc_offsets(cell, offset_number)
    distance_matrix, _ = get_distances(pos, offsets)

    cutoff_distance_matrix = threshold_sort(distance_matrix, r, n_neighbors)

    return torch.Tensor(cutoff_distance_matrix)

def add_selfloop(num_nodes, edge_indices, edge_weights, cutoff_distance_matrix, self_loop=True):
    '''
    add self loop to graph structure

    Parameters
    ----------
        n_nodes: int
            number of nodes
    '''

    if not self_loop:
        return edge_indices, edge_weights, (cutoff_distance_matrix != 0).int()
    
    edge_indices, edge_weights = add_self_loops(
        edge_indices, edge_weights, num_nodes=num_nodes, fill_value=0
    )

    distance_matrix_masked = (cutoff_distance_matrix.fill_diagonal_(1) != 0).int()
    return edge_indices, edge_weights, distance_matrix_masked

def load_node_representation(node_representation='onehot'):
    node_rep_path = Path(__file__).parent
    default_reps = {
        'onehot': str(node_rep_path / './node_representations/onehot.csv')
    }

    # print(default_reps['onehot'])

    rep_file_path = node_representation
    if node_representation in default_reps:
        rep_file_path = default_reps[node_representation]
    
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

def generate_node_features(input_data, n_neighbors):
    node_reps = load_node_representation()
    
    if isinstance(input_data, Data):
        input_data.x = torch.Tensor(node_reps[input_data.z-1])
        return one_hot_degree(input_data, n_neighbors+1)

    for i, data in enumerate(input_data):
        # minus 1 as the reps are 0-indexed but atomic number starts from 1
        data.x = torch.Tensor(node_reps[data.z-1])

    for i, data in enumerate(input_data):
        input_data[i] = one_hot_degree(data, n_neighbors+1)

def generate_edge_features(input_data, edge_steps):
    distance_gaussian = GaussianSmearing(0, 1, edge_steps, 0.2)

    if isinstance(input_data, Data):
        input_data.edge_attr = distance_gaussian(input_data.edge_descriptor['distance'])
        return

    normalize_edge(input_data, 'distance')
    for i, data in enumerate(input_data):
        input_data[i].edge_attr = distance_gaussian(input_data[i].edge_descriptor['distance'])