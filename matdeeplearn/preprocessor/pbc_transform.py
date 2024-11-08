from dataclasses import dataclass, replace
from itertools import chain
from typing import List, Optional, TypedDict
from typing_extensions import NotRequired

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data as TorchGeoData, Batch as TorchGeoBatch
import torch.autograd.profiler as profiler


# fmt: off
atom_list = list(range(1, 101))
# fmt: on

unk_idx = len(atom_list) + 1
atom_mapper = torch.full((128,), unk_idx)
for idx, atom in enumerate(atom_list):
    atom_mapper[atom] = idx + 1  # reserve 0 for paddin


def get_cell_offsets(num_offsets: int):
    cell_offsets = torch.tensor([
        [x, y, z] for x in range(-num_offsets, num_offsets + 1)
                   for y in range(-num_offsets, num_offsets + 1)
                   for z in range(-num_offsets, num_offsets + 1)
        if not (x == 0 and y == 0 and z == 0)
    ]).float()
    n_cells = cell_offsets.size(0)
    return cell_offsets, n_cells


class ExpandPBCConfig(TypedDict):
    cutoff: NotRequired[float]
    filter_by_tag: NotRequired[bool]
    num_offsets: NotRequired[int]


@dataclass
class Data:
    pos: torch.Tensor  # (N, 3)
    atoms: torch.Tensor  # (N,)
    cell: torch.Tensor # (1, 3, 3)
    real_mask: torch.Tensor  # (N,)
    y: torch.Tensor  # (1,)
    n_atoms: torch.Tensor  # (1,)
    structure_id: List[str] = None
    forces: Optional[torch.Tensor] = None  # (N, 3)
    stress: Optional[torch.Tensor] = None  # (3, 3)

    def to(self, device):
        return Data(
            structure_id=self.structure_id,
            pos=self.pos.to(device),
            cell=self.cell.to(device),
            atoms=self.atoms.to(device),
            real_mask=self.real_mask.to(device),
            n_atoms=self.n_atoms.to(device),
            y=self.y.to(device),
            forces=self.forces.to(device) if self.forces is not None else None,
            stress=self.stress.to(device) if self.stress is not None else None,
        )

    def clone(self):
        return Data(
            structure_id=self.structure_id,
            pos=self.pos.clone(),
            cell=self.cell.clone(),
            atoms=self.atoms.clone(),
            real_mask=self.real_mask.clone(),
            n_atoms=self.n_atoms.clone(),
            y=self.y.clone(),
            forces=self.forces.clone() if self.forces is not None else None,
            stress=self.stress.clone() if self.stress is not None else None,
        )

    @classmethod
    def from_torch_geometric_data(
        cls,
        data: TorchGeoData,
        *,
        pbc: ExpandPBCConfig = {},
    ):
        cutoff = pbc.get("cutoff", 8.)

        pos = data.pos
        cell = data.cell
        atoms = data.z.long()

        global atom_mapper
        atoms = atom_mapper[atoms]
        cell_offsets, n_cells = get_cell_offsets(pbc.get("num_offsets", 2))
        
        # with profiler.record_function("APPLY CELL OFFSETS"):
        offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        src_pos = pos

        # with profiler.record_function("FILTER BY CUTOFF"):
        dist: torch.Tensor = (
            src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)
        ).norm(dim=-1)
        used_mask = (dist < cutoff).any(dim=0)
        used_expand_pos = expand_pos[used_mask]
        
        # with profiler.record_function("CREATE DATA"):
        result = cls(
            structure_id=data.structure_id,
            pos=torch.cat([pos, used_expand_pos], dim=0),
            cell=cell,
            atoms=torch.cat([atoms, atoms.repeat(n_cells)[used_mask]]),
            real_mask=torch.cat(
                [
                    torch.ones(pos.shape[0], dtype=torch.bool),
                    torch.zeros(used_expand_pos.shape[0], dtype=torch.bool),
                ]
            ),
            y=torch.tensor([data.y], dtype=torch.float),
            n_atoms=torch.tensor([data.num_nodes], dtype=torch.long),
            forces=data.forces if hasattr(data, "forces") else None,
            stress=data.stress if hasattr(data, "stress") else None,
        )
        return result
        
def _pad(
    data_list: List[Data],
    attr: str,
    batch_first: bool = True,
    padding_value: float = 0,
):
    return pad_sequence(
        [getattr(d, attr) for d in data_list],
        batch_first=batch_first,
        padding_value=padding_value,
    )


@dataclass
class Batch:
    pos: torch.Tensor
    cell: torch.Tensor
    atoms: torch.Tensor
    real_mask: torch.Tensor
    n_atoms: torch.Tensor
    y: torch.Tensor
    structure_id: List[str] = None
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None

    def to(self, device):
        return Batch(
            structure_id=self.structure_id,
            pos=self.pos.to(device),
            cell=self.cell.to(device),
            atoms=self.atoms.to(device),
            real_mask=self.real_mask.to(device),
            n_atoms=self.n_atoms.to(device),
            y=self.y.to(device),
            forces=self.forces.to(device) if self.forces is not None else None,
            stress=self.stress.to(device) if self.stress is not None else None,
        )

    @classmethod
    def from_batch(cls, batch: TorchGeoBatch, pbc: ExpandPBCConfig = {}):
        data_list = batch.to_data_list()
        data_list = [Data.from_torch_geometric_data(data, pbc=pbc) for data in data_list]
        batch = cls(
            pos=_pad(data_list, "pos"),
            cell=torch.cat([d.cell for d in data_list], dim=0),
            atoms=_pad(data_list, "atoms"),
            real_mask=_pad(data_list, "real_mask"),
            n_atoms=_pad(data_list, "n_atoms"),
            y=torch.cat([d.y for d in data_list], dim=0),
        )
        return batch
            
    @classmethod
    def from_datalist(cls, data_list: List[Data]):
        batch = cls(
            pos=_pad(data_list, "pos"),
            cell=torch.cat([d.cell for d in data_list], dim=0),
            atoms=_pad(data_list, "atoms"),
            real_mask=_pad(data_list, "real_mask"),
            n_atoms=torch.cat([d.n_atoms for d in data_list], dim=0),
            y=torch.cat([d.y for d in data_list], dim=0),
        )
        if hasattr(data_list[0], "forces"):
            # batch.forces = _pad(data_list, "forces")
            batch.forces = torch.cat([d.forces for d in data_list], dim=0)
        if hasattr(data_list[0], "stress"):
            # batch.stress = _pad(data_list, "stress")
            batch.stress = torch.cat([d.stress for d in data_list], dim=0)
        new_batch = TorchGeoBatch()
        for key, value in batch.__dict__.items():
            setattr(new_batch, key, value)
            
        new_batch.structure_id = list(chain(*[d.structure_id for d in data_list]))
        new_batch.y = new_batch.y.transpose(0, 1)
        new_batch.z = data_list[0].z
        return new_batch

    def clone(self):
        return Batch(
            structure_id=self.structure_id,
            pos=self.pos.clone(),
            cell=self.cell.clone(),
            atoms=self.atoms.clone(),
            real_mask=self.real_mask.clone(),
            n_atoms=self.n_atoms.clone(),
            y=self.y.clone(),
            forces=self.forces.clone() if self.forces is not None else None,
            stress=self.stress.clone() if self.stress is not None else None,
        )