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
        y: OptTensor = None,
        z: OptTensor = None,
        u: OptTensor = None,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        edge_index_rr: OptTensor = None,
        edge_attr_rr: OptTensor = None,
        edge_index_rv: OptTensor = None,
        edge_attr_rv: OptTensor = None,
        edge_index_vv: OptTensor = None,
        edge_attr_vv: OptTensor = None,
        edge_index_vr: OptTensor = None,
        edge_attr_vr: OptTensor = None,
        edge_vec: OptTensor = None,
        cell_offsets: OptTensor = None,
        distances: OptTensor = None,
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

        # set edges
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_index_rr = edge_index_rr
        self.edge_attr_rr = edge_attr_rr
        self.edge_index_rv = edge_index_rv
        self.edge_attr_rv = edge_attr_rv
        self.edge_index_vv = edge_index_vv
        self.edge_attr_vv = edge_attr_vv
        self.edge_index_vr = edge_index_vr
        self.edge_attr_vr = edge_attr_vr

        self.edge_vec: OptTensor = edge_vec
        self.cell_offsets: OptTensor = cell_offsets
        self.distances: OptTensor = distances
        self.structure_id: list = structure_id
        self.o_pos: OptTensor = o_pos
        self.o_z: OptTensor = o_z

    def __inc__(self, key, value, *args, **kwargs) -> int:
        if 'index' in key:
            return self.n_atoms
        else:
            return 0