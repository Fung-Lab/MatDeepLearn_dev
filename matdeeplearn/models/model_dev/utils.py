from dataclasses import dataclass, replace
from typing import List, Optional, TypedDict
from typing_extensions import NotRequired

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data as TorchGeoData, Batch as TorchGeoBatch

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
    real_mask: torch.Tensor  # (N,)
    y: torch.Tensor  # (1,)
    natoms: torch.Tensor  # (1,)

    def to(self, device):
        return Data(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            real_mask=self.real_mask.to(device),
            natoms=self.natoms.to(device),
            y=self.y.to(device),
        )

    def clone(self):
        return Data(
            pos=self.pos.clone(),
            atoms=self.atoms.clone(),
            real_mask=self.real_mask.clone(),
            natoms=self.natoms.clone(),
            y=self.y.clone(),
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
        offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        src_pos = pos

        dist: torch.Tensor = (
            src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)
        ).norm(dim=-1)
        used_mask = (dist < cutoff).any(dim=0)
        used_expand_pos = expand_pos[used_mask]
        
        return cls(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(n_cells)[used_mask]]),
            real_mask=torch.cat(
                [
                    torch.ones(pos.shape[0], dtype=torch.bool),
                    torch.zeros(used_expand_pos.shape[0], dtype=torch.bool),
                ]
            ),
            y=torch.tensor([data.y], dtype=torch.float),
            natoms=torch.tensor([data.num_nodes], dtype=torch.long),
        )
        
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
    atoms: torch.Tensor
    real_mask: torch.Tensor
    natoms: torch.Tensor
    y: torch.Tensor

    def to(self, device):
        return Batch(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            real_mask=self.real_mask.to(device),
            natoms=self.natoms.to(device),
            y=self.y.to(device),
        )

    @classmethod
    def from_batch(cls, batch: TorchGeoBatch, pbc: ExpandPBCConfig = {}):
        data_list = batch.to_data_list()
        data_list = [Data.from_torch_geometric_data(data.to("cpu"), pbc=pbc) for data in data_list]
        batch = cls(
            pos=_pad(data_list, "pos"),
            atoms=_pad(data_list, "atoms"),
            real_mask=_pad(data_list, "real_mask"),
            natoms=_pad(data_list, "natoms"),
            y=torch.cat([d.y for d in data_list], dim=0),
        )
        return batch

    def clone(self):
        return Batch(
            pos=self.pos.clone(),
            atoms=self.atoms.clone(),
            real_mask=self.real_mask.clone(),
            natoms=self.natoms.clone(),
            y=self.y.clone(),
        )