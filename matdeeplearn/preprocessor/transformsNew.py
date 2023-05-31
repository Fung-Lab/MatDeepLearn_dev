import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_sparse import coalesce

from matdeeplearn.common.graph_data import CustomData
from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (compute_bond_angles,
                                               custom_edge_feats,
                                               custom_node_feats,
                                               generate_virtual_nodes,
                                               get_cutoff_distance_matrix,
                                               get_mask)

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
        offset_number=1,
    ):
        self.virtual_box_increment = virtual_box_increment
        self.edge_steps = edge_steps
        self.attrs = attrs
        self.rv_cutoff = rv_cutoff
        self.rr_cutoff = rr_cutoff
        self.neighbors = neighbors
        self.offset_number = offset_number
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

    def __call__(self, data: Data) -> CustomData:
        # NOTE use cell2 instead of cell for correct VN generation
        vpos, virtual_z = generate_virtual_nodes(
            data.cell2, self.virtual_box_increment, self.device
        )

        data.rv_pos = torch.cat((data.o_pos, vpos), dim=0)
        data.z = torch.cat((data.o_z, virtual_z), dim=0)

        # create edges
        for attr in self.attrs:
            # TODO: optimize this by using the same cutoff distance matrix for all edges or incorporating all_neighbors efficient computation
            # causes a large slowdown during processing since we recompute the same matrix for each edge type
            cd_matrix, _, _ = get_cutoff_distance_matrix(
                data.rv_pos,
                data.cell,
                getattr(self, f"{attr}_cutoff"),
                self.neighbors,
                self.device,
                experimental=False,
                offset_number=self.offset_number,
                remove_virtual_edges=False
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
            [getattr(data, ei) for ei in [f"edge_index_{attr}" for attr in self.attrs]],
            dim=1,
        )

        data.x = custom_node_feats(
            data.z,
            participating_edges,
            len(data.z),
            self.neighbors,  # any neighbor suffices
            self.device,
        )

        # make original edge_attr and edge_index sentinel values for object compatibility
        data.edge_attr = torch.zeros(1, self.edge_steps)
        data.edge_index = torch.zeros(1, 2)

        # remove unnecessary attributes to reduce memory overhead
        edge_index_attrs = [f"edge_index_{s}" for s in self.attrs]
        edge_attr_attrs = [f"edge_attr_{s}" for s in self.attrs]

        for attr in list(data.__dict__.get("_store").keys()):
            if (
                attr not in edge_index_attrs
                and attr not in edge_attr_attrs
                and attr not in self.keep_attrs
            ):
                data.__dict__.get("_store")[attr] = None

        # compile all generated edges
        edge_kwargs = {
            attr: getattr(data, attr) for attr in edge_index_attrs + edge_attr_attrs
        }

        return CustomData(
            pos=data.pos,
            cell=data.cell,
            cell2=data.cell2,
            y=data.y,
            z=data.z,
            u=data.u,
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            cell_offsets=data.cell_offsets,
            structure_id=data.structure_id,
            n_atoms=len(data.x),
            **edge_kwargs,
        )


@registry.register_transform("NumNodes")
class NumNodes(object):
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
class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If `degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axes (int, optional): The rotation axes. (default: `[0, 1, 2]`)
    """

    def __init__(self, degrees, axes=[0, 1, 2]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axes = axes

    def __call__(self, data):
        if data.pos.size(-1) == 2:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)
            matrix = [[cos, sin], [-sin, cos]]
        else:
            m1, m2, m3 = torch.eye(3), torch.eye(3), torch.eye(3)
            if 0 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m1 = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            if 1 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m2 = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            if 2 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m3 = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

            matrix = torch.mm(torch.mm(m1, m2), m3)

        data_rotated = LinearTransformation(matrix)(data)
        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_rotated, "cell"):
            data_rotated.cell = torch.matmul(data_rotated.cell, matrix)

        return (
            data_rotated,
            matrix,
            torch.inverse(matrix),
        )

    def __repr__(self):
        return "{}({}, axis={})".format(
            self.__class__.__name__, self.degrees, self.axis
        )