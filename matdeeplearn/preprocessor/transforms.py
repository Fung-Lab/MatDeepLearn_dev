import torch
from torch_geometric.data import Data
from typing import List
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    compute_bond_angles,
    custom_node_edge_feats,
    generate_virtual_nodes,
    get_cutoff_distance_matrix,
)

"""
here resides the transform classes needed for data processing

From PyG:
    Transform: A function/transform that takes in an torch_geometric.data.Data
    object and returns a transformed version.
    The data object will be transformed before every access.
"""


@registry.register_transform("GetY")
class GetY(object):
    """Get the target from the data object."""

    def __init__(self, index=0):
        self.index = index

    def __call__(self, data: Data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data


@registry.register_transform("VirtualNodes")
class VirtualNodes(object):
    """Generate virtual nodes along the unit cell."""

    def __init__(
        self,
        virtual_box_increment: float = 3.0,
        vv_cutoff: float = 3,
        vv_neighbors: int = 50,
        rv_cutoff: float = 3,
        rv_neighbors: int = 50,
        n_neighbors: int = 50,
        cutoff: float = 5,
        edge_steps: int = 50,
        mp_attrs: List[str] = ["rv", "rr", "vv"],
    ):
        self.virtual_box_increment = virtual_box_increment
        self.vv_cutoff = vv_cutoff
        self.vv_neighbors = vv_neighbors
        self.rv_cutoff = rv_cutoff
        self.rv_neighbors = rv_neighbors
        self.n_neighbors = n_neighbors
        self.cutoff = cutoff
        self.edge_steps = edge_steps
        self.mp_attrs = mp_attrs
        # processing is done on cpu
        self.device = torch.device("cpu")
        # attrs that remain the same for all cases
        self.keep_attrs = [
            "x",
            "edge_index",
            "edge_attr",
            "pos",
            "z",
            "cell",
            "cell2",
            "structure_id",
            "distances",
            "u",
            "cell_offsets",
            "y",
        ]

    def __call__(self, data: Data) -> Data:
        # make sure n_neigbhors for node embeddings were specified equally for all nodes
        assert (
            self.vv_neighbors == self.rv_neighbors == self.n_neighbors
        ), "n_neighbors must be the same for all node types"

        # NOTE use cell2 instead of cell for correct VN generation
        vpos, virtual_z = generate_virtual_nodes(
            data.cell2, self.virtual_box_increment, self.device
        )

        data.rv_pos = torch.cat((data.o_pos, vpos), dim=0)
        data.z = torch.cat((data.o_z, virtual_z), dim=0)

        cd_matrix, cell_offsets = get_cutoff_distance_matrix(
            data.rv_pos,
            data.cell,
            self.cutoff,
            self.n_neighbors,
            self.device,
        )

        edge_index, edge_weight = dense_to_sparse(cd_matrix)

        data.x, data.edge_attr = custom_node_edge_feats(
            data.z,
            len(data.z),
            self.n_neighbors,
            edge_weight,
            edge_index,
            self.edge_steps,
            self.cutoff,
            self.device,
        )

        data.cell_offsets = cell_offsets

        # create edge indices for different MP routines
        rv_edge_mask = torch.argwhere(
            (data.z[edge_index[0]] != 100) & (data.z[edge_index[1]] == 100)
        )
        vr_edge_mask = torch.argwhere(
            (data.z[edge_index[0]] == 100) & (data.z[edge_index[1]] != 100)
        )
        rr_edge_mask = torch.argwhere(
            (data.z[edge_index[0]] != 100) & (data.z[edge_index[1]] != 100)
        )
        vv_edge_mask = torch.argwhere(
            (data.z[edge_index[0]] == 100) & (data.z[edge_index[1]] == 100)
        )

        if "vr" in self.mp_attrs:
            data.edge_index_vr = torch.index_select(
                edge_index, 1, vr_edge_mask.squeeze(1)
            )
            data.edge_attr_vr = torch.index_select(
                data.edge_attr, 0, vr_edge_mask.squeeze(1)
            )

        if "rv" in self.mp_attrs:
            data.edge_index_rv = torch.index_select(
                edge_index, 1, rv_edge_mask.squeeze(1)
            )
            data.edge_attr_rv = torch.index_select(
                data.edge_attr, 0, rv_edge_mask.squeeze(1)
            )

        if "rr" in self.mp_attrs:
            data.edge_index_rr = torch.index_select(
                edge_index, 1, rr_edge_mask.squeeze(1)
            )
            data.edge_attr_rr = torch.index_select(
                data.edge_attr, 0, rr_edge_mask.squeeze(1)
            )

        if "vv" in self.mp_attrs:
            data.edge_index_vv = torch.index_select(
                edge_index, 1, vv_edge_mask.squeeze(1)
            )
            data.edge_attr_vv = torch.index_select(
                data.edge_attr, 0, vv_edge_mask.squeeze(1)
            )

        # remove unnecessary attributes to reduce memory overhead
        edge_index_attrs = [f"edge_index_{s}" for s in self.mp_attrs]
        edge_attr_attrs = [f"edge_attr_{s}" for s in self.mp_attrs]

        for attr in list(data.__dict__.get("_store").keys()):
            if (
                attr not in edge_index_attrs
                and attr not in edge_attr_attrs
                and attr not in self.keep_attrs
            ):
                data.__dict__.get("_store")[attr] = None

        return data


@registry.register_transform("NumNodeTransform")
class NumNodeTransform(object):
    """
    Adds the number of nodes to the data object
    """

    def __call__(self, data: Data):
        data.num_nodes = data.x.shape[0]
        return data


@registry.register_transform("LineGraphMod")
class LineGraphMod(object):
    """
    Adds line graph attributes to the data object
    """

    def __call__(self, data: Data):
        # CODE FROM PYG LINEGRAPH TRANSFORM (DIRECTED)
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        _, edge_attr = coalesce(edge_index, edge_attr, N, N)

        # compute bond angles
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        triplet_pairs = torch.stack([idx_kj, idx_ji], dim=0)

        data.edge_index_lg = triplet_pairs
        data.x_lg = data.edge_attr
        data.num_nodes_lg = edge_index.size(1)

        # assign bond angles to edge attributes
        data.edge_attr_lg = angles.reshape(-1, 1)

        return data


@registry.register_transform("ToFloat")
class ToFloat(object):
    """
    Convert non-int attributes to float
    """

    def __call__(self, data: Data):
        data.x = data.x.float()
        data.x_lg = data.x_lg.float()

        data.distances = data.distances.float()
        data.pos = data.pos.float()

        data.edge_attr = data.edge_attr.float()
        data.edge_attr_lg = data.edge_attr_lg.float()

        return data
