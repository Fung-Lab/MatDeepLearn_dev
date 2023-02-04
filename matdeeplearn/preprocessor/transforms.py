import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, remove_self_loops
from torch_sparse import coalesce

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.graph_data import VirtualNodeGraph
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
        vv_cutoff: float = 3,
        vv_neighbors: int = 50,
        rv_cutoff: float = 3,
        rv_neighbors: int = 50,
        n_neighbors: int = 50,
        cutoff: float = 5,
        edge_steps: int = 50,
    ):
        self.virtual_box_increment = virtual_box_increment
        self.vv_cutoff = vv_cutoff
        self.vv_neighbors = vv_neighbors
        self.rv_cutoff = rv_cutoff
        self.rv_neighbors = rv_neighbors
        self.n_neighbors = n_neighbors
        self.cutoff = cutoff
        self.edge_steps = edge_steps
        # processing is done on cpu
        self.device = torch.device("cpu")

    def __call__(self, data: Data) -> Data:
        # make sure n_neigbhors for node embeddings were specified equally for all nodes
        # TODO find better way to assert, perhaps read from config file
        # assert (
        #     self.vv_neighbors == self.rv_neighbors == self.rr_neighbors
        # ), "n_neighbors must be the same for all node types"

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
            data.rv_pos, data.cell, self.rv_cutoff, self.rv_neighbors, self.device
        )
        cd_matrix, _ = get_cutoff_distance_matrix(
            torch.cat((data.pos, vpos), dim=0),
            data.cell,
            self.cutoff,
            self.n_neighbors,
            self.device,
        )

        data.edge_index_vv, edge_weight_vv = dense_to_sparse(cd_matrix_vv)
        edge_index_rv, edge_weight_rv = dense_to_sparse(cd_matrix_rv)
        data.edge_index_both, edge_weight_both = dense_to_sparse(cd_matrix)

        # create node and edge attributes
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
            edge_index_rv,
            self.edge_steps,
            self.rv_cutoff,
            self.device,
        )

        # original method without specific cutoffs
        data.x_both, data.edge_attr_both = custom_node_edge_feats(
            torch.cat((data.z, virtual_z), dim=0),
            len(data.z_rv),
            self.n_neighbors,
            edge_weight_both,
            data.edge_index_both,
            self.edge_steps,
            self.cutoff,
            self.device,
        )

        rv_edge_mask = torch.argwhere(data.z_rv[edge_index_rv[0]] == 100)
        vr_edge_mask = torch.argwhere(data.z_rv[edge_index_rv[0]] != 100)
        rr_edge_mask = torch.argwhere(
            (data.z_rv[edge_index_rv[0]] != 100) & (data.z_rv[edge_index_rv[1]] != 100)
        )

        # create real->virtual directional edges
        data.edge_index_rv = torch.clone(edge_index_rv)
        data.edge_index_rv[0, rv_edge_mask] = edge_index_rv[1, rv_edge_mask]
        # find real->real directional edges for removal
        data.edge_index_rv[0, rr_edge_mask] = edge_index_rv[1, rr_edge_mask]
        # create virtual->real directional edges
        data.edge_index_vr = torch.clone(edge_index_rv)
        data.edge_index_vr[0, vr_edge_mask] = edge_index_rv[1, vr_edge_mask]

        # remove self loops
        # TODO check if this is necessary for VV interactions
        data.edge_index_vv, data.edge_attr_vv = remove_self_loops(
            data.edge_index_vv, data.edge_attr_vv
        )
        data.edge_index_rv, data.edge_attr_rv = remove_self_loops(
            data.edge_index_rv, data.edge_attr_rv
        )
        data.edge_index_vr, _ = remove_self_loops(data.edge_index_vr)

        # assign descriptive attributes
        data.n_vv_nodes = torch.tensor([len(data.x_vv)])
        data.n_rv_nodes = torch.tensor([len(data.x_rv)])
        data.n_both_nodes = torch.tensor([len(data.x_both)])

        return VirtualNodeGraph(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.y,
            data.pos,
            data.n_atoms,
            data.cell,
            data.z,
            data.u,
            data.edge_weight,
            data.cell_offsets,
            data.distances,
            data.structure_id,
            data.rv_pos,
            data.z_rv,
            data.edge_index_vv,
            data.x_vv,
            data.edge_attr_vv,
            data.x_rv,
            data.edge_attr_rv,
            data.edge_index_rv,
            data.edge_index_vr,
            data.x_both,
            data.edge_index_both,
            data.edge_attr_both,
            data.n_vv_nodes,
            data.n_rv_nodes,
            data.n_both_nodes,
        )


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
