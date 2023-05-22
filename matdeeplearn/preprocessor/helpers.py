import itertools
import logging
from pathlib import Path
from typing import Literal, Union, Optional, Tuple

import ase
import numpy as np
from time import perf_counter
import torch
import torch.nn.functional as F
from ase import Atoms
from torch_geometric.data.data import Data
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_min, segment_coo, segment_csr
from torch import scatter
from torch_sparse import SparseTensor
from matdeeplearn.common.graph_data import CustomData, CustomBatchingData
from functools import wraps


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def scatter_det(*args, **kwargs):
    from matdeeplearn.common.registry import registry

    if registry.get("set_deterministic_scatter", no_warning=True):
        torch.use_deterministic_algorithms(mode=True)

    out = scatter(*args, **kwargs)

    if registry.get("set_deterministic_scatter", no_warning=True):
        torch.use_deterministic_algorithms(mode=False)

    return out


def calculate_edges_master(
    method: Literal["ase", "ocp", "mdl"],
    all_neighbors: bool,
    data: Union[CustomData, CustomBatchingData],
    r: float,
    n_neighbors: int,
    offset_number: int,
    remove_virtual_edges: bool = False,
    experimental_distance: bool = False,
    batching: bool = False,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Generates edges using one of three methods (ASE, OCP, or MDL implementations) due to limitations of each method.

    Args:
        all_neighbors (bool): Whether or not to use all neighbors (ASE method)
                or only the n_neighbors closest neighbors.
                OCP based on all_neighbors and MDL based on original without considering all.
        r (float): cutoff radius
        n_neighbors (int): number of neighbors to consider
        structure_id (str): structure id
        cell (torch.Tensor): unit cell
        pos (torch.Tensor): positions of atom in unit cell
    """

    if method == "ase" or method == "ocp":
        assert (method == "ase" and all_neighbors) or (
            method == "ocp" and all_neighbors
        ), "OCP and ASE methods only support all_neighbors=True"

    out = dict()
    neighbors = torch.empty(0)
    cell_offset_distances = torch.empty(0)

    pos = data.pos
    cell = data.cell
    z = data.z
    structure_id = data.structure_id

    if method == "mdl":
        if batching:
            raise NotImplementedError("Batching not implemented for MDL method")

        cutoff_distance_matrix, cell_offsets, edge_vec = get_cutoff_distance_matrix(
            pos,
            cell,
            r,
            n_neighbors,
            device,
            experimental=experimental_distance,
            offset_number=offset_number,
            remove_virtual_edges=remove_virtual_edges,
            vn=z,
        )

        edge_index, edge_weights = dense_to_sparse(cutoff_distance_matrix)
        # get into correct shape for model stage
        edge_vec = edge_vec[edge_index[0], edge_index[1]]

    elif method == "ase":
        if batching:
            raise NotImplementedError("Batching not implemented for ASE method")

        edge_index, cell_offsets, edge_weights, edge_vec = calculate_edges_ase(
            all_neighbors, r, n_neighbors, structure_id, cell, pos, z
        )

    elif method == "ocp":  # batching compatible
        # OCP requires a different format for the cell
        cell = cell.view(-1, 3, 3)

        # Calculate neighbors to allow compatibility with models like GemNet_OC
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            r, n_neighbors, pos, cell, data.n_atoms, use_thresh=True
        )

        ocp_out = get_pbc_distances(
            data.pos,
            edge_index,
            cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = ocp_out["edge_index"]
        edge_weights = ocp_out["distances"]
        cell_offset_distances = ocp_out["offsets"]
        edge_vec = ocp_out["distance_vec"]

    out["edge_index"] = edge_index
    out["edge_weights"] = edge_weights
    out["cell_offsets"] = cell_offsets
    out["offsets"] = cell_offset_distances
    out["edge_vec"] = edge_vec
    out["neighbors"] = neighbors

    return out


class PerfTimer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = perf_counter() - self.start


def get_mask(
    pattern: str, data: Data, src: torch.Tensor, dest: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    patterns = {
        "rv": (data.z[src] != 100, data.z[dest] == 100),
        "rr": (data.z[src] != 100, data.z[dest] != 100),
        "vr": (data.z[src] == 100, data.z[dest] != 100),
        "vv": (data.z[src] == 100, data.z[dest] == 100),
    }

    assert pattern in patterns.keys(), f"pattern {pattern} not found"

    # get specific binary MP conditions
    cond1, cond2 = patterns.get(pattern)

    edge_mask = torch.argwhere(cond1 & cond2).squeeze(1)
    return edge_mask


def threshold_sort(all_distances: torch.Tensor, r: float, n_neighbors: int):
    # A = all_distances.clone().detach()
    A = all_distances

    # set diagonal to zero to exclude self-loop distance
    # A.fill_diagonal_(0)

    # keep n_neighbors only
    N = len(A) - n_neighbors - 1
    if N > 0:
        if all_distances.dim() > 2:
            # TODO WIP experimental method with 3D distance tensor
            A = A.reshape(len(A), -1)
            _, indices = torch.topk(A, k=N, dim=1)
            A = torch.scatter(
                A,
                1,
                indices,
                torch.zeros_like(
                    A,
                    device=all_distances.device,
                    dtype=torch.float,
                ),
            )
            # return A to original shape
            A = A.reshape(len(A), len(A), -1)
        else:
            _, indices = torch.topk(A, k=N, dim=1)
            A = torch.scatter(
                A,
                1,
                indices,
                torch.zeros(
                    len(A),
                    len(A),
                    device=all_distances.device,
                    dtype=torch.float,
                ),
            )
        _, indices = torch.topk(A, N)
        A = torch.scatter(
            A,
            1,
            indices,
            torch.zeros(len(A), len(A), device=all_distances.device, dtype=torch.float),
        )

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
    """
    slightly edited version from pytorch geometric to create edge from gaussian basis
    """

    def __init__(
        self, start=0.0, stop=5.0, resolution=50, width=0.05, device="cpu", **kwargs
    ):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution, device=device)
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


def normalize_edge_cutoff(dataset, descriptor_label, r):
    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] / r
        )


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


def clean_up(data_list: Union[DataLoader, list], attr_list):
    if not attr_list:
        return
    # check which attributes in the list are removable
    removable_attrs = [t for t in attr_list if t in data_list[0].to_dict()]
    # remove the attributes
    for data in data_list:
        for attr in removable_attrs:
            delattr(data, attr)


def get_pbc_cells(cell: torch.Tensor, offset_number: int, device: str = "cpu"):
    """
    Get the periodic boundary condition (PBC) offsets for a unit cell

    Parameters
        cell:       torch.Tensor
                    unit cell vectors of ase.cell.Cell

        offset_number:  int
                    the number of offsets for the unit cell
                    if == 0: no PBC
                    if == 1: 27-cell offsets (3x3x3)
                    if == 2: 125-cell offsets (5x5x5)
                    etc.
    """

    _range = np.arange(-offset_number, offset_number + 1)
    offsets = [list(x) for x in itertools.product(_range, _range, _range)]
    # offsets = torch.cartesian_prod([_range, _range, _range]).to(device).type(torch.float)
    offsets = torch.tensor(offsets, device=device, dtype=torch.float)
    return offsets @ cell, offsets


def control_virtual_edges(distance_matrix, an):
    """
    Removes virtual edges from the structure

    Args:
        distance_matrix (torch.Tensor): interatomic distances
        an (torch.Tensor): corresponding atomic numbers in the structure

    Returns:
        torch.Tensor: modified distance matrix
    """

    indices = np.argwhere(an == 100).squeeze(1)
    distance_matrix[indices, :] = torch.zeros(distance_matrix.shape[2])
    return distance_matrix


def get_distances(
    positions: torch.Tensor,
    offsets: torch.Tensor,
    device: str = "cpu",
):
    """
    Get pairwise atomic distances

    Parameters
        positions:  torch.Tensor
                    positions of atoms in a unit cell


        offsets:    torch.Tensor
                    offsets for the unit cell

        all_neighbors:  bool
                        whether or not to use MIC, which does not account for neighboring cells

        device:     str
                    torch device type

    """

    # convert numpy array to torch tensors
    n_atoms = len(positions)
    n_cells = len(offsets)

    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)

    offsets = offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + offsets

    # calculate pairwise distances
    atomic_distances = torch.linalg.norm(pos1 - pos2, dim=-1)

    # set diagonal of the (0,0,0) unit cell to infinity
    # this allows us to get the minimum self-loop distance
    # of an atom to itself in all other images
    # origin_unit_cell_idx = 13
    # origin_unit_cell_idx = 13
    # atomic_distances[:,:,origin_unit_cell_idx].fill_diagonal_(float("inf"))
    atom_rij = (pos1 - pos2).squeeze(2)

    min_atomic_distances, min_indices = torch.min(atomic_distances, dim=-1)
    expanded_min_indices = min_indices.clone().detach()

    expanded_min_indices = expanded_min_indices[..., None, None].expand(
        -1, -1, 1, atom_rij.size(3)
    )

    atom_rij = torch.gather(atom_rij, dim=2, index=expanded_min_indices)

    return min_atomic_distances, min_indices, atom_rij


def get_distances_experimental(
    positions: torch.Tensor,
    offsets: torch.Tensor,
    all_neighbors: bool = False,
    device: str = "cpu",
):
    """
    Get the periodic boundary condition (PBC) offsets for a unit cell

    Parameters
        positions:  torch.Tensor
                    positions of atoms in a unit cell

        offsets:    torch.Tensor
                    offsets for the unit cell

        all_neighbors:  bool
                        whether or not to use MIC, which does not account for neighboring cells

        device:     str
                    torch device type

    """

    _range = np.arange(-offset_number, offset_number + 1)
    offsets = [list(x) for x in itertools.product(_range, _range, _range)]
    offsets = torch.tensor(offsets, device=device, dtype=torch.float)
    return offsets @ cell, offsets


def get_cutoff_distance_matrix(
    pos, cell, r, n_neighbors, device, image_selfloop, offset_number=1
):
    """
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

        device: str
            torch device type

        image_selfloop: bool
            whether or not to include self loops in the distance matrix

        experimental: bool
            whether or not to use experimental method for calculating distances

        offset_number: int
            number of unit cells to consider in each direction

        remove_virtual_edges: bool
            whether or not to remove virtual edges from the distance matrix

        vn : torch.Tensor
            virtual node atomic indices
    """

    cells, cell_coors = get_pbc_cells(cell, offset_number, device=device)

    # can calculate distances using WIP experimental method or current implementation (MIC)
    distance_matrix, min_indices, atom_rij = (
        get_distances(pos, cells, device=device)
        if not experimental
        else get_distances_experimental(pos, cells, all_neighbors=True, device=device)
    )

    if remove_virtual_edges:
        distance_matrix = control_virtual_edges(distance_matrix, vn)

    cutoff_distance_matrix = threshold_sort(distance_matrix, r, n_neighbors)

    # if image_selfloop:
    #     # output of threshold sort has diagonal == 0
    #     # fill in the original values
    #     self_loop_diag = distance_matrix.diagonal()
    #     cutoff_distance_matrix.diagonal().copy_(self_loop_diag)

    all_cell_offsets = cell_coors[torch.flatten(min_indices)]
    all_cell_offsets = all_cell_offsets.view(len(pos), -1, 3)
    # cell_offsets = all_cell_offsets[cutoff_distance_matrix != 0]

    # self loops will always have cell of (0,0,0)
    # N: no of selfloops; M: no of non selfloop edges
    # self loops are the last N edge_index pairs
    # thus initialize a zero matrix of (M+N, 3) for cell offsets
    n_edges = torch.count_nonzero(cutoff_distance_matrix).item()
    cell_offsets = torch.zeros(n_edges + len(pos), 3, dtype=torch.float)

    # get cells for edges except for self loops
    cell_offsets[:n_edges, :] = all_cell_offsets[cutoff_distance_matrix != 0]

    return cutoff_distance_matrix, cell_offsets, atom_rij


def add_selfloop(
    num_nodes, edge_indices, edge_weights, cutoff_distance_matrix, self_loop=True
):
    """
    add self loop (i, i) to graph structure

    Parameters
    ----------
        n_nodes: int
            number of nodes
    """

    if not self_loop:
        return edge_indices, edge_weights, (cutoff_distance_matrix != 0).int()

    edge_indices, edge_weights = add_self_loops(
        edge_indices, edge_weights, num_nodes=num_nodes, fill_value=0
    )

    distance_matrix_masked = (cutoff_distance_matrix.fill_diagonal_(1) != 0).int()
    return edge_indices, edge_weights, distance_matrix_masked


def load_node_representation(node_representation="onehot"):
    node_rep_path = Path(__file__).parent
    default_reps = {"onehot": str(node_rep_path / "./node_representations/onehot.csv")}

    rep_file_path = node_representation
    if node_representation in default_reps:
        rep_file_path = default_reps[node_representation]

    file_type = rep_file_path.split(".")[-1]
    loaded_rep = None

    if file_type == "csv":
        loaded_rep = np.genfromtxt(rep_file_path, delimiter=",")
        # TODO: need to check if typecasting to integer is needed
        loaded_rep = loaded_rep.astype(int)

    elif file_type == "json":
        # TODO
        pass

    return loaded_rep


def generate_node_features(input_data, n_neighbors, device):
    node_reps = load_node_representation()
    node_reps = torch.from_numpy(node_reps).to(device)
    n_elements, n_features = node_reps.shape

    if isinstance(input_data, Data):
        input_data.x = node_reps[input_data.z - 1].view(-1, n_features)
        return one_hot_degree(input_data, n_neighbors + 1)

    for i, data in enumerate(input_data):
        # minus 1 as the reps are 0-indexed but atomic number starts from 1
        data.x = node_reps[data.z - 1].view(-1, n_features)

    for i, data in enumerate(input_data):
        input_data[i] = one_hot_degree(data, n_neighbors + 1)


def generate_edge_features(input_data, edge_steps, r, device):
    distance_gaussian = GaussianSmearing(0, 1, edge_steps, 0.2, device=device)

    if isinstance(input_data, Data):
        input_data = [input_data]

    normalize_edge_cutoff(input_data, "distance", r)
    for i, data in enumerate(input_data):
        input_data[i].edge_attr = distance_gaussian(
            input_data[i].edge_descriptor["distance"]
        )
