from __future__ import annotations

from torch_geometric.data import Data


class VirtualNodeGraph(Data):
    """Data class for virtual node applications to implement correct
    minibatching and type recognition.
    """

    def __init__(
        self,
        data: Data,
    ):
        super().__init__()

        self.x = data.x
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.y = data.y
        self.pos = data.pos
        self.n_atoms = data.n_atoms
        self.cell = data.cell
        self.z = data.z
        self.u = data.u
        self.edge_weight = data.edge_weight
        self.cell_offsets = data.cell_offsets
        self.distances = data.distances
        self.structure_id = data.structure_id

        # Properties specific to virtual nodes
        self.rv_pos = data.rv_pos
        self.z_rv = data.z_rv
        self.edge_index_vv = data.edge_index_vv
        self.x_vv = data.x_vv
        self.edge_attr_vv = data.edge_attr_vv
        self.x_rv = data.x_rv
        self.edge_attr_rv = data.edge_attr_rv
        self.edge_index_rv = data.edge_index_rv
        # self.edge_index_rr = data.edge_index_rr
        # self.edge_attr_rr = data.edge_attr_rr

        # assign descriptive attributes
        self.n_vv_nodes = data.n_vv_nodes
        self.n_rv_nodes = data.n_rv_nodes

    def __inc__(self, key, value):
        if "rv" in key:
            return self.n_rv_nodes
        if "vv" in key:
            return self.n_vv_nodes

        return super().__inc__(key, value)
