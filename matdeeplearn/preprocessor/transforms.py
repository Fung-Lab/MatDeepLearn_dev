import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, remove_self_loops
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
        virtual_box_increment: int = 3,
        vv_cutoff: float = 5,
        vv_neighbors: int = 10,
        rv_cutoff: float = 5,
        rv_neighbors: int = 10,
        edge_steps: int = 50,
    ):
        self.virtual_box_increment = virtual_box_increment
        self.vv_cutoff = vv_cutoff
        self.vv_neighbors = vv_neighbors
        self.rv_cutoff = rv_cutoff
        self.rv_neighbors = rv_neighbors
        self.edge_steps = edge_steps
        # processing is done on cpu
        self.device = torch.device("cpu")

    def __call__(self, data: Data) -> Data:
        vpos, virtual_z = generate_virtual_nodes(
            data.cell, self.virtual_box_increment, self.device
        )

        # append virtual nodes positions and atomic numbers
        data.rv_pos = torch.cat((data.pos, vpos), dim=0)
        data.z_rv = torch.cat((data.z, virtual_z), dim=0)

        # use cutoffs to determine edge attributes
        cd_matrix_vv, _ = get_cutoff_distance_matrix(
            vpos, data.cell, self.vv_cutoff, self.vv_neighbors, self.device
        )
        cd_matrix_rv, _ = get_cutoff_distance_matrix(
            data.pos, data.cell, self.rv_cutoff, self.rv_neighbors, self.device
        )

        print("cd_matrix_vv", cd_matrix_vv)
        print("cd_matrix_rv", cd_matrix_rv)

        data.edge_index_vv, edge_weight_vv = dense_to_sparse(cd_matrix_vv)
        data.edge_index_rv, edge_weight_rv = dense_to_sparse(cd_matrix_rv)

        # create edge attributes
        data.x_vv, data.edge_attr_vv = custom_node_edge_feats(
            virtual_z,
            len(virtual_z),
            self.vv_neighbors,
            edge_weight_vv,
            data.edge_index_vv,
            self.edge_steps,
            self.vv_cutoff,
            self.device,
        )
        data.x_rv, data.edge_attr_rv = custom_node_edge_feats(
            data.z_rv,
            len(data.z_rv),
            self.rv_neighbors,
            edge_weight_rv,
            data.edge_index_rv,
            self.edge_steps,
            self.vv_cutoff,
            self.device,
        )

        # remove self loops
        data.edge_index_vv, data.edge_attr_vv = remove_self_loops(
            data.edge_index_vv, data.edge_attr_vv
        )
        data.edge_index_rv, data.edge_attr_rv = remove_self_loops(
            data.edge_index_rv, data.edge_attr_rv
        )

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
