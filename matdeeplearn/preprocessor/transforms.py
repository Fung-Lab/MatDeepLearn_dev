import torch
import math
import numbers
import torch_geometric
import random
from ase import Atoms
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_sparse import coalesce

from matdeeplearn.common.graph_data import (
    CustomBatchingData,
    VirtualNodeData,
    TokenGTData,
)
from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    calculate_edges_master,
    compute_bond_angles,
    custom_edge_feats,
    custom_node_feats,
    generate_virtual_nodes,
    generate_virtual_nodes_ase,
    get_mask,
    one_hot_degree,
    convert_to_single_emb,
    lap_eig,
)

"""
here resides the transform classes needed for data processing

From PyG:
    Transform: A function/transform that takes in an torch_geometric.data.Data
    object and returns a transformed version.
    The data object will be transformed before every access.
"""


@registry.register_transform("TokenGTGeneration")
class TokenGTGeneration(object):
    def __init__(self, **kwargs: dict) -> None:
        self.max_n = kwargs.get("max_n", 512)
        self.edge_bin_size = kwargs.get("edge_bin_size", 100)

    def __call__(self, data: Data) -> Data:
        """
        See
        https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/data/wrapper.py
        for details
        """

        if data.batch:
            raise ValueError("Batching not supported for TokenGTGeneration")

        # dimensions of node and edge features should be same
        assert (
            data.x.shape[1] == data.edge_attr.shape[1]
        ), "node and edge features should have same dimensions"

        edge_feature, edge_index, node_feature = (
            data.edge_attr,
            data.edge_index,
            data.x,
        )

        # bin the Gaussian basis edge features into discrete indices
        edge_int_feature = torch.round(edge_feature * self.edge_bin_size).to(
            node_feature.dtype
        )

        node_data = convert_to_single_emb(node_feature)
        node_data = node_feature
        if len(edge_feature.size()) == 1:
            edge_feature = edge_feature[:, None]
        edge_data = convert_to_single_emb(edge_int_feature)

        N = node_feature.size(0)
        dense_adj = torch.zeros([N, N], dtype=torch.bool)
        dense_adj[edge_index[0, :], edge_index[1, :]] = True
        in_degree = dense_adj.long().sum(dim=1).view(-1)
        lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)

        # these are empirically better than ORFs
        lap_eigvec = F.pad(
            lap_eigvec, (0, self.max_n - lap_eigvec.size(1)), value=float("0")
        )
        lap_eigval = F.pad(
            lap_eigval, (0, self.max_n - lap_eigval.size(1)), value=float("0")
        )

        data.node_data = node_data
        data.edge_data = edge_data
        data.edge_index = edge_index
        data.in_degree = in_degree
        data.out_degree = in_degree
        data.lap_eigvec = lap_eigvec
        data.lap_eigval = lap_eigval

        data.edge_num = int(edge_data.size(0))
        data.node_num = int(node_data.size(0))

        return TokenGTData.from_dict(data.to_dict())


@registry.register_transform("GetY")
class GetY(object):
    """Get the target from the data object."""

    def __init__(self, **kwargs: dict):
        self.index = kwargs.get("index", 0)

    def __call__(self, data: Data):
        if data.batch:
            raise NotImplementedError("Batching not supported yet for GetY")

        # Specify target.
        if self.index != -1:
            # print("0", data.y.shape, data.y[:][self.index].shape, data.y[:, self.index].shape)
            # data.y = data.y[:][self.index]
            data.y = data.y[:, self.index]
            # print("1", data.y.shape)
            assert data.y.dim() <= 2, "data.y dimension is incorrect"
            if data.y.dim() == 1 and data.y.shape[0] == data.num_nodes:
                data.y = data.y.unsqueeze(1)
            elif data.y.dim() == 1 and data.y.shape[0] == 1:
                data.y = data.y.unsqueeze(0)
            # print("2", data.y.shape)
            # data.y = data.y[0][self.index]
        return data


@registry.register_transform("RTTransform")
class RTTransform(object):
    def __init__(self, **kwargs: dict) -> None:
        self.max_nodes = kwargs.get("max_nodes", 256)

    def __call__(self, data: Data) -> Data:
        if data.batch:
            raise NotImplementedError("Batching not supported yet for RTTransform")

        # pad node features
        padded_x = torch.zeros((self.max_nodes, data.x.size(1)))
        padded_x[: data.x.size(0), :] = data.x

        # pad edge features, [2, E + N]
        src = torch.arange(data.x.size(0), self.max_nodes).repeat_interleave(
            self.max_nodes - data.x.size(0)
        )
        dest = torch.flatten(
            torch.arange(data.x.size(0), self.max_nodes)
            .unsqueeze(0)
            .repeat(self.max_nodes - data.x.size(0), 1)
        )

        new_edges = torch.stack([src, dest], dim=0)
        new_edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        new_edge_attr = torch.zeros((new_edge_index.size(1), data.edge_attr.size(1)))
        new_edge_attr[: data.edge_attr.size(0), :] = data.edge_attr

        src_key_padding_mask = torch.zeros((self.max_nodes, self.max_nodes))
        src_key_padding_mask[: data.x.size(0), : data.x.size(0)] = 1

        null_rows = (
            torch.logical_not(torch.all(src_key_padding_mask == 0, dim=-1))
            .unsqueeze(-1)
            .repeat(1, src_key_padding_mask.size(-1))
        )

        place_inf = torch.logical_and(
            null_rows,
            src_key_padding_mask == 0,
        )

        src_key_padding_mask.masked_fill_(place_inf, float("-inf"))

        data.src_key_padding_mask = src_key_padding_mask.unsqueeze(0)
        data.null_row_mask = torch.logical_not(null_rows).unsqueeze(0)

        data.edge_index = new_edge_index

        data.x = padded_x
        data.edge_attr = new_edge_attr

        data.num_nodes = self.max_nodes

        return data


@registry.register_transform("DegreeNodeAttr")
class DegreeNodeAttr(object):
    """Add degree as node attribute."""

    def __init__(self, **kwargs: dict):
        self.n_neighbors = kwargs.get("n_neighbors")

    def __call__(self, data: Data):
        if data.batch:
            raise NotImplementedError("Batching not supported yet for DegreeNodeAttr")

        one_hot_degree(data, self.n_neighbors)
        return data


@registry.register_transform("VirtualNodeGeneration")
class VirtualNodeGeneration(object):
    def __init__(self, **kwargs) -> None:
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        if data.batch:
            raise NotImplementedError("Batching not supported yet for GetY")

        method = self.kwargs.get("method", "ase")

        if method == "ase":
            structure = Atoms(
                numbers=data.z,
                positions=data.pos,
                cell=data.cell.view(3, 3),
                pbc=[1, 1, 1],
            )
            vpos, virtual_z = generate_virtual_nodes_ase(
                structure, self.kwargs.get("virtual_box_increment"), self.device
            )
        elif method == "pytorch":
            vpos, virtual_z = generate_virtual_nodes(
                data.cell2, self.kwargs.get("virtual_box_increment"), device=self.device
            )
        else:
            raise ValueError("Invalid method for virtual node generation")

        data.pos = torch.cat((data.pos, vpos), dim=0)
        data.z = torch.cat((data.z, virtual_z), dim=0)
        data.n_atoms = torch.tensor([data.z.shape[0]])

        return data


@registry.register_transform("VirtualEdgeGeneration")
class VirtualEdgeGeneration(object):
    """Generate virtual nodes along the unit cell."""

    def __init__(
        self,
        **kwargs,
    ):
        self.device = torch.device(kwargs.get("device", "cpu"))
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
            "neighbors",
            "cell_offsets",
            "cell_offset_distances",
            "y",
        ]
        self.kwargs = kwargs

    def __call__(self, data: Data) -> VirtualNodeData:
        use_batching = isinstance(data, CustomBatchingData) and hasattr(
            data, "_slice_dict"
        )
        if use_batching:
            # compute edge slicing based on node slicing
            slice_partitions = data._slice_dict["z"]
            # compute slicing indices for edges
            slices = [
                slice(slice_partitions[i - 1].data(), slice_partitions[i].data())
                for i in range(1, len(slice_partitions))
            ]
            batch_size = len(data.batch.unique(return_counts=False))
        # compute edges for each mp interaction
        for attr in self.kwargs.get("attrs"):
            cutoff = self.kwargs.get(f"{attr}_cutoff")
            # compute edges (expensive)
            edge_gen_out = calculate_edges_master(
                self.kwargs.get("edge_calc_method"),
                self.kwargs.get("all_neighbors"),
                data,
                cutoff,
                self.kwargs.get("n_neighbors"),
                self.kwargs.get("num_offsets"),
                remove_virtual_edges=False,
                experimental_distance=False,
                device=self.device,
            )

            edge_index = edge_gen_out["edge_index"]
            edge_weight = edge_gen_out["edge_weights"]
            edge_vec = edge_gen_out["edge_vec"]
            cell_offsets = edge_gen_out["cell_offsets"]
            edge_vec = edge_gen_out["edge_vec"]

            # create mask to compute desired edges for interaction
            edge_mask = get_mask(attr, data, edge_index[0], edge_index[1])
            # apply mask
            edge_index = torch.index_select(edge_index, 1, edge_mask)
            edge_weight = torch.index_select(edge_weight, 0, edge_mask)
            edge_attr = custom_edge_feats(
                edge_weight,
                self.kwargs.get("edge_steps"),
                cutoff,
                self.device,
            )
            edge_vec = torch.index_select(edge_vec, 0, edge_mask)
            cell_offsets = torch.index_select(cell_offsets, 0, edge_mask)

            if use_batching:
                # make an index list for edges
                edges_sliced_order = torch.empty(size=(0, 1))
                edge_partitions = [0]
                # compute slicing indices for edges
                for sl in slices:
                    indices = torch.argwhere(
                        (edge_index[0] >= sl.start) & (edge_index[0] < sl.stop)
                    )
                    edges_sliced_order = torch.vstack((edges_sliced_order, indices))
                    # normalize edge indices to reflect individual graph counts
                    edges_target = edge_index[:, indices.squeeze()]
                    edge_index[:, indices.squeeze()] = edges_target - edges_target.min()
                    edge_partitions.append(edge_partitions[-1] + indices.shape[0])
                # compute slicing indices for edges
                edge_partitions = torch.tensor(edge_partitions)
                edges_sliced_order = edges_sliced_order.long().squeeze()
                # update batching track dicts
                for x in [
                    f"edge_index_{attr}",
                    f"edge_attr_{attr}",
                    f"edge_vec_{attr}",
                    f"cell_offsets_{attr}",
                    f"edge_weights_{attr}",
                ]:
                    data._slice_dict[x] = edge_partitions
                    data._inc_dict[x] = torch.zeros(size=(batch_size,))
                # arrange edge attributes to reflect slicing order
                edge_index = edge_index[:, edges_sliced_order]
                edge_attr = edge_attr[edges_sliced_order]
                edge_weight = edge_weight[edges_sliced_order]
                edge_vec = edge_vec[edges_sliced_order]
                cell_offsets = cell_offsets[edges_sliced_order]

            setattr(
                data,
                f"edge_index_{attr}",
                edge_index,
            )
            setattr(
                data,
                f"edge_attr_{attr}",
                edge_attr,
            )
            setattr(
                data,
                f"edge_weights_{attr}",
                edge_weight,
            )
            setattr(
                data,
                f"edge_vec_{attr}",
                edge_vec,
            )
            setattr(
                data,
                f"cell_offsets_{attr}",
                cell_offsets,
            )

        participating_edges = torch.cat(
            [
                getattr(data, ei)
                for ei in [f"edge_index_{attr}" for attr in self.kwargs.get("attrs")]
            ],
            dim=1,
        )

        # compute node embeddings
        data.x = custom_node_feats(
            data.z,
            participating_edges,
            len(data.z),
            self.kwargs.get("n_neighbors"),  # any neighbor suffices
            self.device,
            use_degree=self.kwargs.get("use_degree", False),
        )

        if use_batching:
            data._slice_dict["x"] = data._slice_dict["z"]
            data._inc_dict["x"] = data._inc_dict["z"]

        # make original edge_attr and edge_index sentinel values for object compatibility
        data.edge_attr = torch.zeros(1, self.kwargs.get("edge_steps"))
        data.edge_index = torch.zeros(1, 2)

        # remove unnecessary attributes to reduce memory overhead
        edge_index_attrs = [f"edge_index_{s}" for s in self.kwargs.get("attrs")]
        edge_attr_attrs = [f"edge_attr_{s}" for s in self.kwargs.get("attrs")]
        edge_weight_attrs = [f"edge_weights_{s}" for s in self.kwargs.get("attrs")]
        edge_vec_attrs = [f"edge_vec_{s}" for s in self.kwargs.get("attrs")]
        cell_offset_attrs = [f"cell_offsets_{s}" for s in self.kwargs.get("attrs")]

        edge_related_attrs = (
            edge_index_attrs
            + edge_attr_attrs
            + edge_weight_attrs
            + edge_vec_attrs
            + cell_offset_attrs
        )

        if self.kwargs.get("optimize_memory", False):
            for cutoff in list(data.__dict__.get("_store").keys()):
                if cutoff not in edge_related_attrs and cutoff not in self.keep_attrs:
                    data.__dict__.get("_store")[cutoff] = None

        # compile all generated edges and related attributes
        edge_kwargs = {
            attr: getattr(data, attr)
            for attr in edge_related_attrs
            if hasattr(data, attr)
        }

        for attr, value in edge_kwargs.datas():
            setattr(data.__dict__["_store"], attr, value)

        data.num_edge_features = self.kwargs.get("edge_steps")

        return data


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
        # from PyG linegraph transform
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
