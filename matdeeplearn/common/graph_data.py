from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from typing import Optional


class CustomData(Data):
    """
    Custom graph data object which performs correct batching

    Args:
        Data (torch_geometric.data.Data): data object to wrap and perform correct batching
    """

    def __init__(
        self,
        pos: OptTensor = None,
        cell: OptTensor = None,
        cell2: OptTensor = None,
        neighbors: OptTensor = None,
        y: OptTensor = None,
        z: OptTensor = None,
        u: OptTensor = None,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        edge_index_rr: OptTensor = None,
        edge_attr_rr: OptTensor = None,
        edge_weights_rr: OptTensor = None,
        edge_vec_rr: OptTensor = None,
        cell_offsets_rr: OptTensor = None,
        cell_offset_distances_rr: OptTensor = None,
        neighbors_rr: OptTensor = None,
        edge_index_rv: OptTensor = None,
        edge_attr_rv: OptTensor = None,
        edge_weights_rv: OptTensor = None,
        edge_vec_rv: OptTensor = None,
        cell_offsets_rv: OptTensor = None,
        cell_offset_distances_rv: OptTensor = None,
        neighbors_rv: OptTensor = None,
        edge_index_vv: OptTensor = None,
        edge_attr_vv: OptTensor = None,
        edge_weights_vv: OptTensor = None,
        edge_vec_vv: OptTensor = None,
        cell_offsets_vv: OptTensor = None,
        cell_offset_distances_vv: OptTensor = None,
        neighbors_vv: OptTensor = None,
        edge_index_vr: OptTensor = None,
        edge_attr_vr: OptTensor = None,
        edge_weights_vr: OptTensor = None,
        edge_vec_vr: OptTensor = None,
        cell_offsets_vr: OptTensor = None,
        cell_offset_distances_vr: OptTensor = None,
        neighbors_vr: OptTensor = None,
        structure_id: Optional[list] = None,
        n_atoms: OptTensor = None,
        o_pos: OptTensor = None,
        o_z: OptTensor = None,
    ):
        super().__init__()

        self.n_atoms: OptTensor = n_atoms
        self.pos: OptTensor = pos
        self.cell: OptTensor = cell
        self.cell2: OptTensor = cell2
        self.y: OptTensor = y
        self.z: OptTensor = z
        self.u: OptTensor = u
        self.x: OptTensor = x

        # set edge attributes
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.edge_index_rr = edge_index_rr
        self.edge_attr_rr = edge_attr_rr
        self.edge_weights_rr = edge_weights_rr
        self.edge_vec_rr = edge_vec_rr
        self.cell_offsets_rr = cell_offsets_rr
        self.cell_offset_distances_rr = cell_offset_distances_rr
        self.neighbors_rr = neighbors_rr

        self.edge_index_rv = edge_index_rv
        self.edge_attr_rv = edge_attr_rv
        self.edge_weights_rv = edge_weights_rv
        self.edge_vec_rv = edge_vec_rv
        self.cell_offsets_rv = cell_offsets_rv
        self.cell_offset_distances_rv = cell_offset_distances_rv
        self.neighbors_rv = neighbors_rv

        self.edge_index_vv = edge_index_vv
        self.edge_attr_vv = edge_attr_vv
        self.edge_weights_vv = edge_weights_vv
        self.edge_vec_vv = edge_vec_vv
        self.cell_offsets_vv = cell_offsets_vv
        self.cell_offset_distances_vv = cell_offset_distances_vv
        self.neighbors_vv = neighbors_vv

        self.edge_index_vr = edge_index_vr
        self.edge_attr_vr = edge_attr_vr
        self.edge_weights_vr = edge_weights_vr
        self.edge_vec_vr = edge_vec_vr
        self.cell_offsets_vr = cell_offsets_vr
        self.cell_offset_distances_vr = cell_offset_distances_vr
        self.neighbors_vr = neighbors_vr

        self.neighbors: OptTensor = neighbors
        self.structure_id: list = structure_id
        self.o_pos: OptTensor = o_pos
        self.o_z: OptTensor = o_z

    def __inc__(self, key, value, *args, **kwargs) -> int:
        if "index" in key:
            return self.n_atoms
        else:
            return 0
