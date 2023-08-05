import numpy as np
import torch
from torch_geometric.data import Data
import random
import copy
from ase import Atoms
from ase.build import make_supercell
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


def node_masking(subdata: Data, mask_node_ratio):
    subdata = copy.deepcopy(subdata)
    x = subdata.x
    num_nodes = x.size(0)
    mask = torch.randperm(num_nodes) < max(1, int(num_nodes * mask_node_ratio))
    subdata.x[mask] = 0
    return subdata


def edge_masking(subdata: Data, mask_edge_ratio):
    subdata = copy.deepcopy(subdata)
    edge_index, edge_attr = subdata.edge_index, subdata.edge_attr
    num_edges = edge_index.size(1)
    mask = torch.randperm(num_edges) < max(int(num_edges * mask_edge_ratio), 1)
    subdata.edge_index = subdata.edge_index[:, ~mask]
    subdata.edge_attr = subdata.edge_attr[~mask]
    return subdata
    
def perturb_positions(pos, distance, min_distance):
    random_vectors = torch.randn(pos.size(0), 3)
    random_vectors /= torch.norm(random_vectors, dim=1, keepdim=True)
    if min_distance is not None:
        perturbation_distances = torch.empty(pos.size(0)).uniform_(min_distance, distance)
    else:
        perturbation_distances = torch.full((pos.size(0),), distance)
    pos += perturbation_distances.view(-1, 1) * random_vectors
    return pos
    
        
def column_replacement(atomic_numbers, column_replace_num):
    elements = [[1, 3, 11, 19, 37, 55, 87], [4, 12, 20, 38, 56, 88], [21, 39, 71],
             [22, 40, 72], [23, 41, 73], [24, 42, 74], [25, 43, 75], [26, 44, 76],
             [27, 45, 77], [28, 46, 78], [29, 47, 79], [30, 48, 80], [5, 13, 31, 49, 81],
             [6, 14, 32, 50, 82], [7, 15, 33, 51, 83], [8, 16, 34, 52, 84],
             [9, 17, 35, 53, 85], [2, 10, 18, 36, 54, 86],
             [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
             [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
    
    def get_atom_of_same_group(atom_num):
        for row in elements:
            if atom_num in row:
                r = random.choice(row)
                while r == atom_num:
                    r = random.choice(row)
                return r
    
    atomic_numbers = atomic_numbers.clone()
    num_replacements = min(column_replace_num, atomic_numbers.numel())
    for _ in range(num_replacements):
        random_index = torch.randint(0, atomic_numbers.numel(), (1,)).item()
        atomic_numbers[random_index] = get_atom_of_same_group(atomic_numbers[random_index].item())
    return atomic_numbers


def strain_cell(atomic_numbers, pos, cell, strain_strength):
    ase_crystal_object = Atoms(numbers = atomic_numbers, positions=pos, cell=cell.squeeze(), pbc=True)
    cell_l_and_a = ase_crystal_object.get_cell_lengths_and_angles()
    
    while True:
      count = 0
      try:
          rand_factor = 1 + torch.rand(1) * strain_strength
          new_cell_l_and_a = cell_l_and_a.copy()
          if count > 100:
              new_cell_l_and_a = cell_l_and_a[:3] * rand_factor.item()
          else:
              new_cell_l_and_a = cell_l_and_a * rand_factor.item()
          ase_crystal_object.set_cell(new_cell_l_and_a, scale_atoms=True)
          break
      except Exception as e:
          pass
      count += 1
      if count % 20 == 0:
          print(f"Have retried straining for {count} times.")
          
    pos = ase_crystal_object.get_positions()
    cell = ase_crystal_object.get_cell().cellpar()[:3]
    pos = torch.tensor(pos, dtype=torch.float32)
    cell = torch.diag(torch.tensor(cell, dtype=torch.float32))
    return pos, cell.unsqueeze(dim=0)
    
    
def generate_supercell(atomic_numbers, scale_factor, pos, cell):
    super_cell = make_supercell(Atoms(numbers=atomic_numbers, positions=pos, cell=cell.squeeze(), pbc=True), np.identity(3) * scale_factor)
    atomic_numbers = torch.from_numpy(super_cell.get_atomic_numbers())
    pos = torch.tensor(super_cell.get_positions(), dtype=torch.float32)
    cell = torch.diag(torch.tensor(super_cell.get_cell().cellpar()[:3], dtype=torch.float32)).unsqueeze(dim=0)
    del super_cell
    return atomic_numbers, pos, cell