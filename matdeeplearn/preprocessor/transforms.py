import torch
from torch_geometric.data import Data
from torch_sparse import coalesce

from matdeeplearn.common.graph_data import CustomData
from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (calculate_edges_master,
                                               compute_bond_angles,
                                               custom_edge_feats,
                                               custom_node_feats,
                                               generate_virtual_nodes,
                                               get_mask, one_hot_degree)

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

    def __init__(self, **kwargs: dict):
        self.index = kwargs.get("index", 0)

    def __call__(self, data: Data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data


@registry.register_transform("DegreeNodeAttr")
class DegreeNodeAttr(object):
    """Add degree as node attribute."""

    def __init__(self, **kwargs: dict):
        self.n_neighbors = kwargs.get("n_neighbors")

    def __call__(self, data: Data):
        one_hot_degree(data, self.n_neighbors)
        return data


@registry.register_transform("VirtualNodes")
class VirtualNodes(object):
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

    def __call__(self, data: Data) -> CustomData:
        # NOTE use cell2 instead of cell for correct VN generation
        vpos, virtual_z = generate_virtual_nodes(
            data.cell2, self.kwargs.get("virtual_box_increment"), self.device
        )

        try:
            data.rv_pos = torch.cat((data.o_pos, vpos), dim=0)
            data.z = torch.cat((data.o_z, virtual_z), dim=0)
        except AttributeError:
            pass

        # TODO figure out how to resolve neighbors for different MP interactions
        for attr in self.kwargs.get("attrs"):
            cutoff = self.kwargs.get(f"{attr}_cutoff")

            # compute edges
            edge_gen_out = calculate_edges_master(
                self.kwargs.get("edge_calc_method"),
                self.kwargs.get("all_neighbors"),
                cutoff,
                self.kwargs.get("n_neighbors"),
                self.kwargs.get("num_offsets"),
                data.structure_id,
                data.cell,
                data.pos,
                data.z,
                remove_virtual_edges=False,
                experimental_distance=False,
                device=self.device,
            )

            edge_index = edge_gen_out["edge_index"]
            edge_weight = edge_gen_out["edge_weights"]
            edge_vec = edge_gen_out["edge_vec"]
            cell_offsets = edge_gen_out["cell_offsets"]
            cell_offset_distances = edge_gen_out["offsets"]
            neighbors = edge_gen_out["neighbors"]

            edge_attr = custom_edge_feats(
                edge_weight,
                self.kwargs.get("edge_steps"),
                cutoff,
                self.device,
            )

            # apply mask to compute desired edges for interaction
            edge_mask = get_mask(attr, data, edge_index[0], edge_index[1])

            setattr(
                data,
                f"edge_index_{attr}",
                torch.index_select(edge_index, 1, edge_mask),
            )
            setattr(
                data,
                f"edge_attr_{attr}",
                torch.index_select(edge_attr, 0, edge_mask),
            )
            setattr(
                data,
                f"edge_weights_{attr}",
                torch.index_select(edge_weight, 0, edge_mask),
            )
            setattr(
                data,
                f"edge_vec_{attr}",
                torch.index_select(edge_vec, 0, edge_mask),
            )
            setattr(
                data,
                f"cell_offsets_{attr}",
                torch.index_select(cell_offsets, 0, edge_mask),
            )

            # in case edge calculation method does not return cell_offset_distances
            try:
                setattr(
                    data,
                    f"cell_offset_distances_{attr}",
                    torch.index_select(cell_offset_distances, 0, edge_mask),
                )
            except IndexError:
                setattr(
                    data,
                    f"cell_offset_distances_{attr}",
                    None,
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
            self.kwargs.get("neighbors"),  # any neighbor suffices
            self.device,
            use_degree=self.kwargs.get("use_degree", False),
        )

        # make original edge_attr and edge_index sentinel values for object compatibility
        data.edge_attr = torch.zeros(1, self.kwargs.get("edge_steps"))
        data.edge_index = torch.zeros(1, 2)

        # remove unnecessary attributes to reduce memory overhead
        edge_index_attrs = [f"edge_index_{s}" for s in self.kwargs.get("attrs")]
        edge_attr_attrs = [f"edge_attr_{s}" for s in self.kwargs.get("attrs")]
        edge_weight_attrs = [f"edge_weights_{s}" for s in self.kwargs.get("attrs")]
        edge_vec_attrs = [f"edge_vec_{s}" for s in self.kwargs.get("attrs")]
        cell_offset_attrs = [f"cell_offsets_{s}" for s in self.kwargs.get("attrs")]
        cell_offset_distance_attrs = [
            f"cell_offset_distances_{s}" for s in self.kwargs.get("attrs")
        ]

        if self.kwargs.get("optimize_memory", False):
            for attr in list(data.__dict__.get("_store").keys()):
                if (
                    attr not in edge_index_attrs
                    and attr not in edge_attr_attrs
                    and attr not in edge_weight_attrs
                    and attr not in edge_vec_attrs
                    and attr not in self.keep_attrs
                    and attr not in cell_offset_attrs
                    and attr not in cell_offset_distance_attrs
                ):
                    data.__dict__.get("_store")[attr] = None

        # compile all generated edges and related attributes
        edge_kwargs = {
            attr: getattr(data, attr)
            for attr in edge_index_attrs
            + edge_attr_attrs
            + edge_weight_attrs
            + edge_vec_attrs
            + cell_offset_attrs
            + cell_offset_distance_attrs
            if hasattr(data, attr)
        }

        return CustomData(
            pos=data.pos,
            neighbors=neighbors,
            cell=data.cell,
            cell2=data.cell2,
            y=data.y,
            z=data.z,
            u=data.u,
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
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
        # from PyG linegraph transform
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
