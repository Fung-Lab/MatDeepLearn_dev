from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class VirtualNodeGraph(Data):
    """Class for virtual node applications to implement correct
    minibatching and type recognition.
    """

    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        n_atoms: OptTensor = None,
        cell: OptTensor = None,
        z: OptTensor = None,
        u: OptTensor = None,
        edge_weight: OptTensor = None,
        cell_offsets: OptTensor = None,
        distances: OptTensor = None,
        structure_id: OptTensor = None,
        rv_pos: OptTensor = None,
        zv: OptTensor = None,
        edge_index_vv: OptTensor = None,
        x_vv: OptTensor = None,
        edge_attr_vv: OptTensor = None,
        x_rv: OptTensor = None,
        edge_attr_rv: OptTensor = None,
        edge_index_rv: OptTensor = None,
        edge_index_vr: OptTensor = None,
        x_both: OptTensor = None,
        edge_index_both: OptTensor = None,
        edge_attr_both: OptTensor = None,
        total_atoms: OptTensor = None,
        n_vv_nodes: OptTensor = None,
        n_rv_nodes: OptTensor = None,
        n_both_nodes: OptTensor = None,
    ):
        super().__init__()

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.n_atoms = n_atoms
        self.cell = cell
        self.z = z
        self.u = u
        self.edge_weight = edge_weight
        self.cell_offsets = cell_offsets
        self.distances = distances
        self.structure_id = structure_id

        # Properties specific to virtual nodes
        self.rv_pos = rv_pos
        self.zv = zv
        self.edge_index_vv = edge_index_vv
        self.x_vv = x_vv
        self.edge_attr_vv = edge_attr_vv
        self.x_rv = x_rv
        self.edge_attr_rv = edge_attr_rv
        self.edge_index_rv = edge_index_rv
        self.edge_index_vr = edge_index_vr
        self.x_both = x_both
        self.edge_index_both = edge_index_both
        self.edge_attr_both = edge_attr_both

        # assign descriptive attributes
        self.n_vv_nodes = n_vv_nodes
        self.n_rv_nodes = n_rv_nodes
        self.n_both_nodes = n_both_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if "rv" in key:
            return self.n_rv_nodes
        if "vv" in key:
            return self.n_vv_nodes
        if "both" in key or "zv" in key:
            return self.n_both_nodes

        return super().__inc__(key, value, *args, **kwargs)
