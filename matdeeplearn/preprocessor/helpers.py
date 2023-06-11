import itertools
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
from matdeeplearn.common.graph_data import VirtualNodeData, CustomBatchingData
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
    data: Union[VirtualNodeData, CustomBatchingData],
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


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(
        np.sort(np.abs(np.real(EigVal)))
    ).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    dense_adj = dense_adj.detach().float().numpy()
    in_degree = in_degree.detach().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


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
    # atomic_distances[:,:,origin_unit_cell_idx].fill_diagonal_(float("inf"))
    atom_rij = pos1 - pos2

    if all_neighbors:
        all_indices = torch.arange(
            0, np.prod(atomic_distances.shape), device=device
        ).view_as(atomic_distances)

        return atomic_distances, all_indices, atom_rij
    else:
        min_atomic_distances, min_indices = torch.min(atomic_distances, dim=-1)
        expanded_min_indices = min_indices.clone().detach()

        expanded_min_indices = expanded_min_indices[..., None, None].expand(
            -1, -1, 1, atom_rij.size(3)
        )

        atom_rij = torch.gather(atom_rij, dim=2, index=expanded_min_indices)

        min_atomic_distances = min_atomic_distances.reshape(n_atoms, n_atoms, 1)
        min_indices = min_indices.view_as(min_atomic_distances)

        return min_atomic_distances, min_indices, atom_rij


def get_cutoff_distance_matrix(
    pos,
    cell,
    r,
    n_neighbors,
    device,
    image_selfloop=False,
    experimental=False,
    offset_number=1,
    remove_virtual_edges=False,
    vn: torch.Tensor = None,
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


def generate_node_features(input_data, n_neighbors, use_degree, device):
    node_reps = load_node_representation()
    node_reps = torch.from_numpy(node_reps).to(device)
    n_elements, n_features = node_reps.shape

    if isinstance(input_data, Data):
        input_data.x = node_reps[input_data.z - 1].view(-1, n_features)
        if use_degree:
            return one_hot_degree(input_data, n_neighbors)
        return input_data

    for i, data in enumerate(input_data):
        # minus 1 as the reps are 0-indexed but atomic number starts from 1
        data.x = node_reps[data.z - 1].view(-1, n_features)

    if use_degree:
        for i, data in enumerate(input_data):
            input_data[i] = one_hot_degree(data, n_neighbors)


def generate_edge_features(input_data, edge_steps, r, device):
    distance_gaussian = GaussianSmearing(0, 1, edge_steps, 0.2, device=device)

    if isinstance(input_data, Data):
        input_data = [input_data]

    normalize_edge_cutoff(input_data, "distance", r)
    for i, data in enumerate(input_data):
        input_data[i].edge_attr = distance_gaussian(
            input_data[i].edge_descriptor["distance"]
        )


def custom_node_edge_feats(
    atomic_numbers,
    num_nodes,
    n_neighbors,
    edge_descriptor,
    edge_index,
    edge_steps,
    r,
    device,
    cat=True,
    in_degree=False,
):
    # generate node_features
    node_reps = torch.from_numpy(
        load_node_representation(),
    ).to(device)

    x = node_reps[atomic_numbers - 1].view(-1, node_reps.shape[1])

    idx = edge_index[1 if in_degree else 0]
    deg = degree(idx, num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=n_neighbors + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        x = deg

    # generate edge_features
    distance_gaussian = GaussianSmearing(0, 1, edge_steps, 0.2, device=device)
    # perform normalization and feature generation in one step
    edge_attr = distance_gaussian(edge_descriptor / r)

    return x, edge_attr


def custom_node_feats(
    atomic_numbers,
    edge_index,
    num_nodes,
    n_neighbors,
    device,
    cat=True,
    in_degree=False,
    use_degree: bool = False,
):
    # generate node_features
    node_reps = torch.from_numpy(
        load_node_representation(),
    ).to(device)

    x = node_reps[atomic_numbers - 1].view(-1, node_reps.shape[1])

    # only add degree if needed
    if use_degree:
        idx = edge_index[1 if in_degree else 0]
        deg = degree(idx, num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=n_neighbors + 1).to(torch.float)

        if x is not None and cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            x = deg

    return x


def custom_edge_feats(
    edge_descriptor,
    edge_steps,
    r,
    device,
):
    # generate edge_features
    distance_gaussian = GaussianSmearing(0, 1, edge_steps, 0.2, device=device)
    # perform normalization and feature generation in one step
    edge_attr = distance_gaussian(edge_descriptor / r)

    return edge_attr


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.
    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    From https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_matrix_to_params(matrix: torch.Tensor):
    """From https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py"""
    lengths = torch.sqrt(torch.sum(matrix**2, dim=1)).tolist()

    angles = torch.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(torch.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = torch.arccos(angles) * 180.0 / torch.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    """From https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py"""
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    """From https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py"""
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # cart coords

    return pos


def generate_virtual_nodes(
    cell,  # TODO: add types
    increment: int,
    device: torch.device = torch.device("cpu"),
):
    """_summary_

    Args:
        cell (torch.Tensor): _description_
        device (torch.device): _description_
        increment (int, optional): increment specifies the spacing between virtual nodes in cartesian
        space (units in Angstroms); increment is a hyperparameter. Defaults to 3.
    """

    # get lengths and angles for unit parallelpiped
    a, b, c, alpha, beta, gamma = (
        cell if isinstance(cell, list) else torch.split(cell, 1)
    )

    # obtain fractional spacings from 0 to 1 of the virtual atoms
    ar1 = torch.arange(0, 1, increment / a)
    ar2 = torch.arange(0, 1, increment / b)
    ar3 = torch.arange(0, 1, increment / c)

    # use meshgrid to obtain x,y,z, coordinates for the virtual atoms
    xx, yy, zz = torch.meshgrid([ar1[:], ar2[:], ar3[:]], indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

    """
    if increment is larger than size of the unit cell, create a minimum
    of one virtual atom at the origin
    """
    if coords.shape[0] == 0:
        coords = np.array([[0, 0, 0]])

    # obtain cartesian coordinates of virtual atoms
    lengths = torch.tensor([[a, b, c]], device=device)
    angles = torch.tensor([[alpha, beta, gamma]], device=device)

    # TODO: fix this calculation
    virtual_pos = frac_to_cart_coords(coords, lengths, angles, len(coords))

    # virtual positions and atomic numbers
    return virtual_pos, torch.tensor([100] * len(coords), device=device)


def generate_virtual_nodes_ase(
    structure, increment: float, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    increment specifies the spacing between virtual nodes in cartesian
    space (units in Angstroms); increment is a hyperparameter.
    obtain the lengths of the sides of the unit cell; s is an ASE atoms
    object, atoms.cell.cellpar() returns l1,l2,3,a1,a2,a3
    """

    l_and_a = structure.cell.cellpar()

    # obtain fractional spacings from 0 to 1 of the virtual atoms
    ar1 = np.arange(0, 1, increment / l_and_a[0])
    ar2 = np.arange(0, 1, increment / l_and_a[1])
    ar3 = np.arange(0, 1, increment / l_and_a[2])

    # use meshgrid to obtain x,y,z, coordinates for the virtual atoms
    xx, yy, zz = np.meshgrid(ar1[:], ar2[:], ar3[:])
    coords = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

    """
    if increment is larger than size of the unit cell, create a minimum
    of one virtual atom at the origin
    """
    if coords.shape[0] == 0:
        coords = np.array([[0, 0, 0]])

    """
    create a new ASE Atoms object and input fractional coordinates; we
    so this so we can use the method get_positions()
    to obtain the non-fractional coordinates
    """
    temp = Atoms(
        [100] * coords.shape[0],
        scaled_positions=coords,
        cell=l_and_a,
        pbc=[1, 1, 1],
    )

    # set atomic numbers of the virtual atoms to be 100
    atomic_numbers = torch.LongTensor([100] * coords.shape[0])
    return (
        torch.tensor(temp.get_positions(), device=device, dtype=torch.float),
        atomic_numbers,
    )


def calculate_edges_ase(
    all_neighbors: bool,
    r: float,
    n_neighbors: int,
    structure_id: str,
    cell: torch.Tensor,
    pos: torch.Tensor,
):
    # Compute graph using ASE method
    (
        first_idex,
        second_idex,
        rij,
        rij_vec,
        shifts,
    ) = ase.neighborlist.primitive_neighbor_list(
        "ijdDS",
        (True, True, True),
        ase.geometry.complete_cell(cell),
        pos.numpy(),
        cutoff=r,
        self_interaction=True,
        use_scaled_positions=False,
    )

    if not all_neighbors:
        raise NotImplementedError("Only all_neighbors=True is supported for now.")

    # Eliminate true self-edges that don't cross periodic boundaries
    # (https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py)
    bad_edge = first_idex == second_idex
    bad_edge &= np.all(shifts == 0, axis=1)
    keep_edge = ~bad_edge
    first_idex = first_idex[keep_edge]
    second_idex = second_idex[keep_edge]
    rij = rij[keep_edge]
    rij_vec = rij_vec[keep_edge]
    shifts = shifts[keep_edge]

    first_idex = torch.tensor(first_idex).long()
    second_idex = torch.tensor(second_idex).long()
    edge_index = torch.stack([first_idex, second_idex], dim=0)
    edge_weights = torch.tensor(rij).float()
    edge_vec = torch.tensor(rij_vec).float()
    cell_offsets = torch.tensor(shifts).int()

    if not all_neighbors:
        # select minimum distances from full ASE neighborlist
        num_cols = pos.shape[0]
        indices_1d = edge_index[0] * num_cols + edge_index[1]
        out, argmin = scatter_min(edge_weights, indices_1d)
        # remove placeholder values from scatter_min
        empty = argmin == edge_index.shape[1]
        argmin = argmin[~empty]
        out = out[~empty]

        edge_index = edge_index[:, argmin]
        edge_weights = edge_weights[argmin]
        edge_vec = edge_vec[argmin, :]
        cell_offsets = cell_offsets[argmin, :]

        # get closest n neighbors
        if len(edge_weights) > n_neighbors:
            _, topk_indices = torch.topk(
                edge_weights, n_neighbors, largest=False, sorted=False
            )

            # if len(topk_indices) < len(first_idex):
            #     logging.warning(
            #         f"Atoms in structure {structure_id} have more neighbors than n_neighbors. Consider increasing the number to avoid missing neighbors."
            #     )

            # TODO convert back to sparse representation

            first_idex = first_idex[topk_indices]
            second_idex = second_idex[topk_indices]
            edge_index = edge_index[:, topk_indices]
            edge_weights = edge_weights[topk_indices]
            edge_vec = edge_vec[topk_indices]
            cell_offsets = cell_offsets[topk_indices]

    edge_index = torch.stack([first_idex, second_idex], dim=0)

    return edge_index, cell_offsets, edge_weights, edge_vec


def radius_graph_pbc(
    radius: float,
    max_num_neighbors_threshold: int,
    pos: torch.Tensor,
    cell: torch.Tensor,
    n_atoms: torch.Tensor,
    pbc: list[bool] = [True, True, True],
    use_thresh: bool = True,
):
    """
    Calculate the radius graph for a given structure with periodic boundary conditions, including all neighbors for each atom
    From https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py
    Args:
        radius (float): _description_
        max_num_neighbors_threshold (int): _description_
        pos (torch.Tensor): _description_
        cell (torch.Tensor): _description_
        n_atoms (torch.Tensor): _description_
        pbc (list[bool], optional): _description_. Defaults to [True, True, True].

    Returns:
        _type_: _description_
    """

    device = pos.device
    batch_size = len(n_atoms)

    # position of the atoms
    atom_pos = pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = n_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)

    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    if use_thresh:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=n_atoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
    else:
        num_neighbors_image = None

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    From https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # sort index
    index = index.sort()[0]

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def compute_neighbors(data: VirtualNodeData, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])

    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.n_atoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.n_atoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )

    image_indptr[1:] = torch.cumsum(data.n_atoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    """From https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py

    Args:
        pos (_type_): _description_
        edge_index (_type_): _description_
        cell (_type_): _description_
        cell_offsets (_type_): _description_
        neighbors (_type_): _description_
        return_offsets (bool, optional): _description_. Defaults to False.
        return_distance_vec (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    row, col = edge_index

    # this is r_ij or edge_vec
    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def triplets(edge_index, cell_offsets, num_nodes):
    """
    Taken from the DimeNet implementation on OCP
    """

    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Edge indices (k->j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()

    # Remove self-loop triplets d->b->d
    # Check atom as well as cell offset
    cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
    mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1).to(
        device=idx_i.device
    )

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

    return idx_i, idx_j, idx_k, idx_kj, idx_ji


def tripletsOld(
    edge_index,
    num_nodes,
):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def compute_bond_angles(
    pos: torch.Tensor, offsets: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Compute angle between bonds to compute node embeddings for L(g)
    Taken from the DimeNet implementation on OCP
    """

    # Calculate triplets
    idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
        edge_index, offsets.to(device=edge_index.device), num_nodes
    )

    # Calculate angles.
    pos_i = pos[idx_i]
    pos_j = pos[idx_j]

    offsets = offsets.to(pos.device)

    pos_ji, pos_kj = (
        pos[idx_j] - pos_i + offsets[idx_ji],
        pos[idx_k] - pos_j + offsets[idx_kj],
    )

    a = (pos_ji * pos_kj).sum(dim=-1)
    b = torch.cross(pos_ji, pos_kj).norm(dim=-1)

    angle = torch.atan2(b, a)

    return angle, idx_kj, idx_ji


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def masked_select_sparsetensor_flat(src, mask):
    row, col, value = src.coo()
    row = row[mask]
    col = col[mask]
    value = value[mask]
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=src.sparse_sizes())


def calculate_interatomic_vectors(R, id_s, id_t, offsets_st):
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Arguments
    ---------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    # ReLU prevents negative numbers in sqrt
    if offsets_st is None:
        V_st = Rt - Rs  # s -> t
    else:
        V_st = Rt - Rs + offsets_st  # s -> t
    D_st = torch.sqrt(torch.sum(V_st**2, dim=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_clamped(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def get_angle(R_ac, R_ab):
    """Calculate angles between atoms c -> a <- b.

    Arguments
    ---------
        R_ac: Tensor, shape = (N, 3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=-1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab, dim=-1).norm(dim=-1)  # shape = (N,)
    y = y.clamp(min=1e-9)  # Avoid NaN gradient for y = (0,0,0)

    angle = torch.atan2(y, x)
    return angle


def vector_rejection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N, 3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def get_projected_angle(R_ab, P_n, eps=1e-4):
    """
    Project the vector R_ab onto a plane with normal vector P_n,
    then calculate the angle w.r.t. the (x [cross] P_n),
    or (y [cross] P_n) if the former would be ill-defined/numerically unstable.

    Arguments
    ---------
        R_ab: Tensor, shape = (N, 3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N, 3)
            Normal vector of a plane onto which to project R_ab.
        eps: float
            Norm of projection below which to use the y-axis instead of x.

    Returns
    -------
        angle_ab: Tensor, shape = (N)
            Angle on plane w.r.t. x- or y-axis.
    """
    R_ab_proj = torch.cross(R_ab, P_n, dim=-1)

    # Obtain axis defining the angle=0
    x = P_n.new_tensor([[1, 0, 0]]).expand_as(P_n)
    zero_angle = torch.cross(x, P_n, dim=-1)

    use_y = torch.norm(zero_angle, dim=-1) < eps
    P_n_y = P_n[use_y]
    y = P_n_y.new_tensor([[0, 1, 0]]).expand_as(P_n_y)
    y_cross = torch.cross(y, P_n_y, dim=-1)
    zero_angle[use_y] = y_cross

    angle = get_angle(zero_angle, R_ab_proj)

    # Flip sign of angle if necessary to obtain clock-wise angles
    cross = torch.cross(zero_angle, R_ab_proj, dim=-1)
    flip_sign = torch.sum(cross * P_n, dim=-1) < 0
    angle[flip_sign] = -angle[flip_sign]

    return angle


def mask_neighbors(neighbors, edge_mask):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
    return neighbors


def get_neighbor_order(num_atoms, index, atom_distance):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    """
    device = index.device

    # Get sorted index and inverse sorting
    # Necessary for index_sort_map
    index_sorted, index_order = torch.sort(index)
    index_order_inverse = torch.argsort(index_order)

    # Get number of neighbors
    ones = index_sorted.new_ones(1).expand_as(index_sorted)
    num_neighbors = segment_coo(ones, index_sorted, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index_sorted * max_num_neighbors
        + torch.arange(len(index_sorted), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Offset index_sort so that it indexes into index_sorted
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # Create indices specifying the order in index_sort
    order_peratom = torch.arange(max_num_neighbors, device=device)[None, :].expand_as(
        mask_finite
    )
    order_peratom = torch.masked_select(order_peratom, mask_finite)

    # Re-index to obtain order value of each neighbor in index_sorted
    order = torch.zeros(len(index), device=device, dtype=torch.long)
    order[index_sort] = order_peratom

    return order[index_order_inverse]


def get_inner_idx(idx, dim_size):
    """
    Assign an inner index to each element (neighbor) with the same index.
    For example, with idx=[0 0 0 1 1 1 1 2 2] this returns [0 1 2 0 1 2 3 0 1].
    These indices allow reshape neighbor indices into a dense matrix.
    idx has to be sorted for this to work.
    """
    ones = idx.new_ones(1).expand_as(idx)
    num_neighbors = segment_coo(ones, idx, dim_size=dim_size)
    inner_idx = ragged_range(num_neighbors)
    return inner_idx


def get_edge_id(edge_idx, cell_offsets, num_atoms):
    cell_basis = cell_offsets.max() - cell_offsets.min() + 1
    cell_id = (
        (cell_offsets * cell_offsets.new_tensor([[1, cell_basis, cell_basis**2]]))
        .sum(-1)
        .long()
    )
    edge_id = edge_idx[0] + edge_idx[1] * num_atoms + cell_id * num_atoms**2
    return edge_id


def get_triplets(graph, num_atoms):
    """
    Get all input edges b->a for each output edge c->a.
    It is possible that b=c, as long as the edges are distinct
    (i.e. atoms b and c stem from different unit cells).

    Arguments
    ---------
    graph: dict of torch.Tensor
        Contains the graph's edge_index.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
        in: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        out: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
    """
    idx_s, idx_t = graph["edge_index"]  # c->a (source=c, target=a)
    num_edges = idx_s.size(0)

    value = torch.arange(num_edges, device=idx_s.device, dtype=idx_s.dtype)
    # Possibly contains multiple copies of the same edge (for periodic interactions)
    adj = SparseTensor(
        row=idx_t,
        col=idx_s,
        value=value,
        sparse_sizes=(num_atoms, num_atoms),
    )
    adj_edges = adj[idx_t]

    # Edge indices (b->a, c->a) for triplets.
    idx = {}
    idx["in"] = adj_edges.storage.value()
    idx["out"] = adj_edges.storage.row()

    # Remove self-loop triplets
    # Compare edge indices, not atom indices to correctly handle periodic interactions
    mask = idx["in"] != idx["out"]
    idx["in"] = idx["in"][mask]
    idx["out"] = idx["out"][mask]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_mixed_triplets(
    graph_in,
    graph_out,
    num_atoms,
    to_outedge=False,
    return_adj=False,
    return_agg_idx=False,
):
    """
    Get all output edges (ingoing or outgoing) for each incoming edge.
    It is possible that in atom=out atom, as long as the edges are distinct
    (i.e. they stem from different unit cells). In edges and out edges stem
    from separate graphs (hence "mixed") with shared atoms.

    Arguments
    ---------
    graph_in: dict of torch.Tensor
        Contains the input graph's edge_index and cell_offset.
    graph_out: dict of torch.Tensor
        Contains the output graph's edge_index and cell_offset.
        Input and output graphs use the same atoms, but different edges.
    num_atoms: int
        Total number of atoms.
    to_outedge: bool
        Whether to map the output to the atom's outgoing edges a->c
        instead of the ingoing edges c->a.
    return_adj: bool
        Whether to output the adjacency (incidence) matrix between output
        edges and atoms adj_edges.
    return_agg_idx: bool
        Whether to output the indices enumerating the intermediate edges
        of each output edge.

    Returns
    -------
    Dictionary containing the entries:
        in: torch.Tensor, shape (num_triplets,)
            Indices of input edges
        out: torch.Tensor, shape (num_triplets,)
            Indices of output edges
        adj_edges: SparseTensor, shape (num_edges, num_atoms)
            Adjacency (incidence) matrix between output edges and atoms,
            with values specifying the input edges.
            Only returned if return_adj is True.
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
            Only returned if return_agg_idx is True.
    """
    idx_out_s, idx_out_t = graph_out["edge_index"]
    # c->a (source=c, target=a)
    idx_in_s, idx_in_t = graph_in["edge_index"]
    num_edges = idx_out_s.size(0)

    value_in = torch.arange(
        idx_in_s.size(0), device=idx_in_s.device, dtype=idx_in_s.dtype
    )
    # This exploits that SparseTensor can have multiple copies of the same edge!
    adj_in = SparseTensor(
        row=idx_in_t,
        col=idx_in_s,
        value=value_in,
        sparse_sizes=(num_atoms, num_atoms),
    )
    if to_outedge:
        adj_edges = adj_in[idx_out_s]
    else:
        adj_edges = adj_in[idx_out_t]

    # Edge indices (b->a, c->a) for triplets.
    idx_in = adj_edges.storage.value()
    idx_out = adj_edges.storage.row()

    # Remove self-loop triplets c->a<-c or c<-a<-c
    # Check atom as well as cell offset
    if to_outedge:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_t[idx_out]
        cell_offsets_sum = (
            graph_out["cell_offset"][idx_out] + graph_in["cell_offset"][idx_in]
        )
    else:
        idx_atom_in = idx_in_s[idx_in]
        idx_atom_out = idx_out_s[idx_out]
        cell_offsets_sum = (
            graph_out["cell_offset"][idx_out] - graph_in["cell_offset"][idx_in]
        )
    mask = (idx_atom_in != idx_atom_out) | torch.any(cell_offsets_sum != 0, dim=-1)

    idx = {}
    if return_adj:
        idx["adj_edges"] = masked_select_sparsetensor_flat(adj_edges, mask)
        idx["in"] = idx["adj_edges"].storage.value().clone()
        idx["out"] = idx["adj_edges"].storage.row()
    else:
        idx["in"] = idx_in[mask]
        idx["out"] = idx_out[mask]

    if return_agg_idx:
        # idx['out'] has to be sorted
        idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx


def get_quadruplets(
    main_graph,
    qint_graph,
    num_atoms,
):
    """
    Get all d->b for each edge c->a and connection b->a
    Careful about periodic images!
    Separate interaction cutoff not supported.

    Arguments
    ---------
    main_graph: dict of torch.Tensor
        Contains the main graph's edge_index and cell_offset.
        The main graph defines which edges are embedded.
    qint_graph: dict of torch.Tensor
        Contains the quadruplet interaction graph's edge_index and
        cell_offset. main_graph and qint_graph use the same atoms,
        but different edges.
    num_atoms: int
        Total number of atoms.

    Returns
    -------
    Dictionary containing the entries:
        triplet_in['in']: torch.Tensor, shape (nTriplets,)
            Indices of input edge d->b in triplet d->b->a.
        triplet_in['out']: torch.Tensor, shape (nTriplets,)
            Interaction indices of output edge b->a in triplet d->b->a.
        triplet_out['in']: torch.Tensor, shape (nTriplets,)
            Interaction indices of input edge b->a in triplet c->a<-b.
        triplet_out['out']: torch.Tensor, shape (nTriplets,)
            Indices of output edge c->a in triplet c->a<-b.
        out: torch.Tensor, shape (nQuadruplets,)
            Indices of output edge c->a in quadruplet
        trip_in_to_quad: torch.Tensor, shape (nQuadruplets,)
            Indices to map from input triplet d->b->a
            to quadruplet d->b->a<-c.
        trip_out_to_quad: torch.Tensor, shape (nQuadruplets,)
            Indices to map from output triplet c->a<-b
            to quadruplet d->b->a<-c.
        out_agg: torch.Tensor, shape (num_triplets,)
            Indices enumerating the intermediate edges of each output edge.
            Used for creating a padded matrix and aggregating via matmul.
    """
    idx_s, _ = main_graph["edge_index"]
    idx_qint_s, _ = qint_graph["edge_index"]
    # c->a (source=c, target=a)
    num_edges = idx_s.size(0)
    idx = {}

    idx["triplet_in"] = get_mixed_triplets(
        main_graph,
        qint_graph,
        num_atoms,
        to_outedge=True,
        return_adj=True,
    )
    # Input triplets d->b->a

    idx["triplet_out"] = get_mixed_triplets(
        qint_graph,
        main_graph,
        num_atoms,
        to_outedge=False,
    )
    # Output triplets c->a<-b

    # ---------------- Quadruplets -----------------
    # Repeat indices by counting the number of input triplets per
    # intermediate edge ba. segment_coo assumes sorted idx['triplet_in']['out']
    ones = idx["triplet_in"]["out"].new_ones(1).expand_as(idx["triplet_in"]["out"])
    num_trip_in_per_inter = segment_coo(
        ones, idx["triplet_in"]["out"], dim_size=idx_qint_s.size(0)
    )

    num_trip_out_per_inter = num_trip_in_per_inter[idx["triplet_out"]["in"]]
    idx["out"] = torch.repeat_interleave(
        idx["triplet_out"]["out"], num_trip_out_per_inter
    )
    idx_inter = torch.repeat_interleave(
        idx["triplet_out"]["in"], num_trip_out_per_inter
    )
    idx["trip_out_to_quad"] = torch.repeat_interleave(
        torch.arange(
            len(idx["triplet_out"]["out"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        num_trip_out_per_inter,
    )

    # Generate input indices by using the adjacency
    # matrix idx['triplet_in']['adj_edges']
    idx["triplet_in"]["adj_edges"].set_value_(
        torch.arange(
            len(idx["triplet_in"]["in"]),
            device=idx_s.device,
            dtype=idx_s.dtype,
        ),
        layout="coo",
    )
    adj_trip_in_per_trip_out = idx["triplet_in"]["adj_edges"][idx["triplet_out"]["in"]]
    # Rows in adj_trip_in_per_trip_out are intermediate edges ba
    idx["trip_in_to_quad"] = adj_trip_in_per_trip_out.storage.value()
    idx_in = idx["triplet_in"]["in"][idx["trip_in_to_quad"]]

    # Remove quadruplets with c == d
    # Triplets should already ensure that a != d and b != c
    # Compare atom indices and cell offsets
    idx_atom_c = idx_s[idx["out"]]
    idx_atom_d = idx_s[idx_in]

    cell_offset_cd = (
        main_graph["cell_offset"][idx_in]
        + qint_graph["cell_offset"][idx_inter]
        - main_graph["cell_offset"][idx["out"]]
    )
    mask_cd = (idx_atom_c != idx_atom_d) | torch.any(cell_offset_cd != 0, dim=-1)

    idx["out"] = idx["out"][mask_cd]
    idx["trip_out_to_quad"] = idx["trip_out_to_quad"][mask_cd]
    idx["trip_in_to_quad"] = idx["trip_in_to_quad"][mask_cd]

    # idx['out'] has to be sorted for this
    idx["out_agg"] = get_inner_idx(idx["out"], dim_size=num_edges)

    return idx
