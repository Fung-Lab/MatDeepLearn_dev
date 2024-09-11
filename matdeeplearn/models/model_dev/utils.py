from dataclasses import dataclass, replace
from typing import List, Optional, TypedDict
from typing_extensions import NotRequired

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data as TorchGeoData

# fmt: off
atom_list = list(range(1, 101))
# fmt: on

unk_idx = len(atom_list) + 1
atom_mapper = torch.full((128,), unk_idx)
for idx, atom in enumerate(atom_list):
    atom_mapper[atom] = idx + 1  # reserve 0 for paddin


cell_offsets = torch.tensor(
    [
        [-1, -1, 0],
        [-1, 0, 0],
        [-1, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [1, -1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ],
).float()
n_cells = cell_offsets.size(0)


class ExpandPBCConfig(TypedDict):
    cutoff: NotRequired[float]
    filter_by_tag: NotRequired[bool]


@dataclass
class Data:
    pos: torch.Tensor  # (N, 3)
    atoms: torch.Tensor  # (N,)
    # tags: torch.Tensor  # (N,)
    # real_mask: torch.Tensor  # (N,)
    # pos_relaxed: torch.Tensor  # (N, 3)
    # y_relaxed: torch.Tensor  # (N, 3)
    # fixed: torch.Tensor  # (N,)
    y: torch.Tensor  # (1,)
    natoms: torch.Tensor  # (1,)

    def to(self, device):
        return Data(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            # tags=self.tags.to(device),
            # real_mask=self.real_mask.to(device),
            # pos_relaxed=self.pos_relaxed.to(device),
            # y_relaxed=self.y_relaxed.to(device),
            # fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
            y=self.y.to(device),
        )

    def clone(self):
        return Data(
            pos=self.pos.clone(),
            atoms=self.atoms.clone(),
            # tags=self.tags.clone(),
            # real_mask=self.real_mask.clone(),
            # pos_relaxed=self.pos_relaxed.clone(),
            # y_relaxed=self.y_relaxed.clone(),
            # fixed=self.fixed.clone(),
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
        # filter_by_tag = pbc.get("filter_by_tag", False)
        cutoff = pbc.get("cutoff", 8.)

        pos = data.pos
        # pos_relaxed = data.pos_relaxed
        cell = data.cell
        atoms = data.z.long()
        # tags = data.tags.long()
        # fixed = data.fixed.long()

        global atom_mapper, cell_offsets, n_cells
        atoms = atom_mapper[atoms]
        offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        # expand_pos_relaxed = (
        #     pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets
        # ).view(-1, 3)
        src_pos = pos

        dist: torch.Tensor = (
            src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)
        ).norm(dim=-1)
        used_mask = (dist < cutoff).any(dim=0) #& tags.ne(2).repeat(
        #     n_cells
        # )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        # used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        # used_expand_tags = tags.repeat(n_cells)[used_mask]
        # used_expand_fixed = fixed.repeat(n_cells)[used_mask]
        return cls(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(n_cells)[used_mask]]),
            # tags=torch.cat([tags, used_expand_tags]),
            # real_mask=torch.cat(
            #     [
            #         torch.ones_like(tags, dtype=torch.bool),
            #         torch.zeros_like(used_expand_tags, dtype=torch.bool),
            #     ]
            # ),
            # pos_relaxed=torch.cat(
            #     [pos_relaxed, used_expand_pos_relaxed], dim=0
            # ),
            y=torch.tensor([data.y], dtype=torch.float),
            natoms=torch.tensor([data.num_nodes], dtype=torch.long),
            # fixed=torch.cat([fixed, used_expand_fixed]),
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
    # tags: torch.Tensor
    # real_mask: torch.Tensor
    # pos_relaxed: torch.Tensor
    # y_relaxed: torch.Tensor
    # fixed: torch.Tensor
    natoms: torch.Tensor
    y: torch.Tensor

    def to(self, device):
        return Batch(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            # tags=self.tags.to(device),
            # real_mask=self.real_mask.to(device),
            # pos_relaxed=self.pos_relaxed.to(device),
            # y_relaxed=self.y_relaxed.to(device),
            # fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
            y=self.y.to(device),
        )

    @classmethod
    def from_data_list(cls, data_list: List[Data]):
        batch = cls(
            pos=_pad(data_list, "pos"),
            atoms=_pad(data_list, "atoms"),
            # tags=_pad(data_list, "tags"),
            # real_mask=_pad(data_list, "real_mask"),
            # pos_relaxed=_pad(data_list, "pos_relaxed"),
            # y_relaxed=torch.cat([d.y_relaxed for d in data_list], dim=0),
            # fixed=_pad(data_list, "fixed"),
            natoms=_pad(data_list, "natoms"),
            y=torch.cat([d.y for d in data_list], dim=0),
        )
        return [batch]

    def clone(self):
        return Batch(
            pos=self.pos.clone(),
            atoms=self.atoms.clone(),
            # tags=self.tags.clone(),
            # real_mask=self.real_mask.clone(),
            # pos_relaxed=self.pos_relaxed.clone(),
            # y_relaxed=self.y_relaxed.clone(),
            # fixed=self.fixed.clone(),
            natoms=self.natoms.clone(),
            y=self.y.clone(),
        )