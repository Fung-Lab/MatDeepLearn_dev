import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    compute_bond_angles,
    custom_edge_feats,
    custom_node_feats,
    generate_virtual_nodes,
    get_cutoff_distance_matrix,
    get_mask,
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
        edge_steps: int = 50,
        attrs=["rv", "rr"],
        rv_cutoff=5.0,
        rr_cutoff=5.0,
        neighbors=50,
    ):
        self.virtual_box_increment = virtual_box_increment
        self.edge_steps = edge_steps
        self.attrs = attrs
        self.rv_cutoff = rv_cutoff
        self.rr_cutoff = rr_cutoff
        self.neighbors = neighbors
        # processing is done on cpu
        self.device = torch.device("cpu")
        # attrs that remain the same for all cases
        self.keep_attrs = [
            "x",
            "pos",
            "edge_index",
            "edge_attr",
            "z",
            "cell",
            "cell2",
            "structure_id",
            "u",
            "cell_offsets",
            "y",
        ]

    def __call__(self, data: Data) -> Data:
        # NOTE use cell2 instead of cell for correct VN generation
        vpos, virtual_z = generate_virtual_nodes(
            data.cell2, self.virtual_box_increment, self.device
        )

        data.rv_pos = torch.cat((data.o_pos, vpos), dim=0)
        data.z = torch.cat((data.o_z, virtual_z), dim=0)

        # create edges
        for attr in self.attrs["attrs"]:
            cd_matrix, _ = get_cutoff_distance_matrix(
                data.rv_pos,
                data.cell,
                getattr(self, f"{attr}_cutoff"),
                self.neighbors,
                self.device,
            )

            edge_index, edge_weight = dense_to_sparse(cd_matrix)
            edge_attr = custom_edge_feats(
                edge_weight,
                self.edge_steps,
                getattr(self, f"{attr}_cutoff"),
                self.device,
            )

            # apply mask to compute desired edges for interaction
            mask = get_mask(attr, data, edge_index[0], edge_index[1])

            setattr(
                data,
                f"edge_index_{attr}",
                torch.index_select(edge_index, 1, mask),
            )
            setattr(
                data,
                f"edge_attr_{attr}",
                torch.index_select(edge_attr, 0, mask),
            )

        # compute node embeddings
        participating_edges = torch.cat(
            [
                getattr(data, ei)
                for ei in [f"edge_index_{attr}" for attr in self.mp_attrs["attrs"]]
            ],
            dim=1,
        )

        data.x = custom_node_feats(
            data.z,
            participating_edges,
            len(data.z),
            self.mp_attrs["neighbors"],  # any neighbor suffices
            self.device,
        )

        # make original edge_attr and edge_index sentinel values for object compatibility
        data.edge_attr = torch.zeros(1, self.edge_steps)
        data.edge_index = torch.zeros(1, 2)

        # remove unnecessary attributes to reduce memory overhead
        edge_index_attrs = [f"edge_index_{s['name']}" for s in self.mp_attrs["attrs"]]
        edge_attr_attrs = [f"edge_attr_{s['name']}" for s in self.mp_attrs["attrs"]]

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
