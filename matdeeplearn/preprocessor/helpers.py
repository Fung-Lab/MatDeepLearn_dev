import itertools
import logging
from pathlib import Path

import ase
import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from torch_geometric.data.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_min, segment_coo, segment_csr
from torch_sparse import SparseTensor


def argmax(arr: list[dict], key: str) -> int:
    """List of Dict argmax utility function

    Args:
        arr (list[dict]): _description_
        key (str): _description_

    Returns:
        _type_: _description_
    """
    return max(enumerate(arr), key=lambda x: x.get(key))[0]


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

    mask = torch.argwhere(cond1 & cond2).squeeze(1)
    return mask


def threshold_sort(all_distances, r, n_neighbors):
    # A = all_distances.clone().detach()
    A = all_distances

    # set diagonal to zero to exclude self-loop distance
    # A.fill_diagonal_(0)

    # keep n_neighbors only
    N = len(A) - n_neighbors - 1
    if N > 0:
        _, indices = torch.topk(A, N)
        print(indices.shape)
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
    use_atom_rij: bool = False,
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

    if use_atom_rij:
        atom_rij = pos1 - pos2
        expanded_min_indices = expanded_min_indices[..., None, None].expand(
            -1, -1, 1, atom_rij.size(3)
        )
        atom_rij = torch.gather(atom_rij, dim=2, index=expanded_min_indices).squeeze()

    return min_atomic_distances, min_indices, atom_rij if use_atom_rij else None


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
    image_selfloop=False,
    offset_number=1,  # TODO make this a parameter
    remove_virtual_edges=False,
    vn: torch.Tensor = None,
    use_atom_rij=False,
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
    """
    cells, cell_coors = get_pbc_cells(cell, offset_number, device=device)
    distance_matrix, min_indices, atom_rij = get_distances(
        pos, cells, device=device, use_atom_rij=use_atom_rij
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
        return one_hot_degree(input_data, n_neighbors)

    for i, data in enumerate(input_data):
        # minus 1 as the reps are 0-indexed but atomic number starts from 1
        data.x = node_reps[data.z - 1].view(-1, n_features)

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
    device: torch.device,
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
    xx, yy, zz = torch.meshgrid(ar1[:], ar2[:], ar3[:])
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
    virtual_pos = frac_to_cart_coords(coords, lengths, angles, len(coords))

    # virtual positions and atomic numbers
    return virtual_pos, torch.tensor([100] * len(coords), device=device)


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


def calculate_edges_all_neighbors(
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
            if len(topk_indices) < len(first_idex):
                logging.warning(
                    f"Atoms in structure {structure_id} have more neighbors than n_neighbors. Consider increasing the number to avoid missing neighbors."
                )

            first_idex = first_idex[topk_indices]
            second_idex = second_idex[topk_indices]
            edge_index = edge_index[:, topk_indices]
            edge_weights = edge_weights[topk_indices]
            edge_vec = edge_vec[topk_indices]
            cell_offsets = cell_offsets[topk_indices]

    elif all_neighbors:
        pass

    edge_index = torch.stack([first_idex, second_idex], dim=0)

    return edge_index, cell_offsets, edge_weights, edge_vec


def radius_graph_pbc(
    radius: float,
    max_num_neighbors_threshold: int,
    pos: torch.Tensor,
    cell: torch.Tensor,
    n_atoms: torch.Tensor,
    pbc: list[bool] = [True, True, True],
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
