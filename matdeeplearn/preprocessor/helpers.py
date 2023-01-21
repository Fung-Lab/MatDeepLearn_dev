import contextlib
import itertools
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch_geometric.data.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor


@contextlib.contextmanager
def prof_ctx():
    """Primitive debug tool which allows profiling of PyTorch code"""
    with profile(
        activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:

        yield

    logging.debug(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


def threshold_sort(all_distances, r, n_neighbors):
    # A = all_distances.clone().detach()
    A = all_distances

    # set diagonal to zero to exclude self-loop distance
    # A.fill_diagonal_(0)

    # keep n_neighbors only
    N = len(A) - n_neighbors - 1
    if N > 0:
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


def clean_up(data_list, attr_list):
    if not attr_list:
        return

    # check which attributes in the list are removable
    removable_attrs = [t for t in attr_list if t in data_list[0].to_dict()]
    for data in data_list:
        for attr in removable_attrs:
            delattr(data, attr)


def get_distances(
    positions: torch.Tensor,
    offsets: torch.Tensor,
    device: str = "cpu",
    mic: bool = True,
):
    """
    Get pairwise atomic distances

    Parameters
        positions:  torch.Tensor
                    positions of atoms in a unit cell

        offsets:    torch.Tensor
                    offsets for the unit cell

        device:     str
                    torch device type

        mic:        bool
                    minimum image convention
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

    # get minimum
    min_atomic_distances, min_indices = torch.min(atomic_distances, dim=-1)
    expanded_min_indices = min_indices.clone().detach()

    atom_rij = pos1 - pos2
    expanded_min_indices = expanded_min_indices[..., None, None].expand(
        -1, -1, 1, atom_rij.size(3)
    )
    atom_rij = torch.gather(atom_rij, dim=2, index=expanded_min_indices).squeeze()

    return min_atomic_distances, min_indices


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
    """

    _range = np.arange(-offset_number, offset_number + 1)
    offsets = [list(x) for x in itertools.product(_range, _range, _range)]
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
    distance_matrix[indices, :] = 0
    return distance_matrix


def get_cutoff_distance_matrix(
    pos,
    cell,
    r,
    n_neighbors,
    device,
    offset_number=1,
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

        atomic_numbers: np.ndarray
            atomic numbers of the atoms in the structure (used only for virtual nodes)
    """

    cells, cell_coors = get_pbc_cells(cell, offset_number, device=device)
    distance_matrix, min_indices = get_distances(pos, cells, device=device)

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

    return cutoff_distance_matrix, cell_offsets


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
    cell: torch.Tensor,
    pos: torch.Tensor,
    atomic_numbers: torch.Tensor,
    device: torch.device,
    increment: int = 3,
):
    """_summary_

    Args:
        cell (torch.Tensor): _description_
        pos (torch.Tensor): _description_
        atomic_numbers (torch.Tensor): _description_
        device (torch.device): _description_
        increment (int, optional): increment specifies the spacing between virtual nodes in cartesian
        space (units in Angstroms); increment is a hyperparameter. Defaults to 3.
    """

    # get lengths and angles for unit parallelpiped
    a, b, c, alpha, beta, gamma = lattice_matrix_to_params(cell)

    # obtain fractional spacings from 0 to 1 of the virtual atoms
    ar1 = torch.arange(0, 1, increment / a)
    ar2 = torch.arange(0, 1, increment / b)
    ar3 = torch.arange(0, 1, increment / c)

    # use meshgrid to obtain x,y,z, coordinates for the virtual atoms
    xx, yy, zz = torch.meshgrid(ar1[:], ar2[:], ar3[:])
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

    # obtain cartesian coordinates of virtual atoms
    lengths = torch.tensor([[a, b, c]], device=device)
    angles = torch.tensor([[alpha, beta, gamma]], device=device)    
    virtual_pos = frac_to_cart_coords(coords, lengths, angles, len(coords))

    # add virtual atoms to the original atomic positions
    pos = torch.cat([pos, virtual_pos], dim=0)
    atomic_numbers = torch.LongTensor(list(atomic_numbers) + [100] * len(coords))

    return pos, atomic_numbers


def generate_virtual_nodes_ase(structure, device: torch.device):
    """
    increment specifies the spacing between virtual nodes in cartesian
    space (units in Angstroms); increment is a hyperparameter
    """
    increment = 3

    """
    obtain the lengths of the sides of the unit cell; s is an ASE atoms
    object, get_cell_lengths_and_angles() returns l1,l2,3,a1,a2,a3
    """

    l_and_a = structure.get_cell_lengths_and_angles()

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
    
    # obtain non-fractional coordinates and append to real atoms
    pos = torch.tensor(
        np.vstack((structure.get_positions(), temp.get_positions())),
        device=device,
        dtype=torch.float,
    )
    # set atomic numbers of the virtual atoms to be 100 and append
    atomic_numbers = torch.LongTensor(
        list(structure.get_atomic_numbers()) + [100] * coords.shape[0]
    )

    return pos, atomic_numbers


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
