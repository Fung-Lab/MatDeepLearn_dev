import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    compute_bond_angles,
    generate_virtual_nodes,
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

    def __init__(self, virtual_box_increment=3):
        self.virtual_box_increment = virtual_box_increment

    def __call__(self, data: Data) -> Data:
        pos, atomic_numbers = generate_virtual_nodes(
            data.cell, data.pos, data.z, self.virtual_box_increment
        )

        virtual_indices = torch.argwhere(data.z == 100).squeeze(1)
        real_indices = torch.argwhere(data.z != 100).squeeze(1)

        # Create edge indices for edge pairings
        data.edge_index_vv = torch.stack(
            (
                virtual_indices.repeat_interleave(len(virtual_indices), dim=0),
                virtual_indices.repeat(len(virtual_indices)),
            ),
            dim=-1
        )
        data.edge_index_vr = torch.stack(
            (
                virtual_indices.repeat_interleave(len(real_indices), dim=0),
                real_indices.repeat(len(virtual_indices)),
            ),
            dim=-1
        )

        # remove self loops
        data.edge_index_vv = remove_self_loops(data.edge_index_vv)
        data.edge_index_vr = remove_self_loops(data.edge_index_vr)

        data.pos = pos
        data.z = atomic_numbers

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
