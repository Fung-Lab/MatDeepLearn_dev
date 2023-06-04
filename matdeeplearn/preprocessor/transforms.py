import torch
from torch_sparse import coalesce

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import compute_bond_angles

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

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            #print("0", data.y.shape, data.y[:][self.index].shape, data.y[:, self.index].shape)
            #data.y = data.y[:][self.index]
            data.y = data.y[:, self.index]
            #print("1", data.y.shape)
            assert (data.y.dim() <= 2), "data.y dimension is incorrect"            
            if data.y.dim() == 1 and data.y.shape[0] == data.x.shape[0]:
                data.y = data.y.unsqueeze(1)  
            elif data.y.dim() == 1 and data.y.shape[0] == 1:
                data.y = data.y.unsqueeze(0)   
            #print("2", data.y.shape)
            #data.y = data.y[0][self.index]            
        return data


@registry.register_transform("NumNodeTransform")
class NumNodeTransform(object):
    """
    Adds the number of nodes to the data object
    """

    def __call__(self, data):
        data.num_nodes = data.x.shape[0]
        return data


@registry.register_transform("LineGraphMod")
class LineGraphMod(object):
    """
    Adds line graph attributes to the data object
    """

    def __call__(self, data):
        # CODE FROM PYG LINEGRAPH TRANSFORM (DIRECTED)
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        _, edge_attr = coalesce(edge_index, edge_attr, N, N)

        # compute bond angles
        try:
            angles, idx_kj, idx_ji = compute_bond_angles(
                data.pos, data.cell_offsets, data.edge_index, data.num_nodes
            )
        except:
            angles, idx_kj, idx_ji = compute_bond_angles(
                data.pos, None, data.edge_index, data.num_nodes
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

    def __call__(self, data):
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