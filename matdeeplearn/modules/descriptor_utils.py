import torch, itertools
import numpy as np

def get_distances(positions, pbc_offsets, device):
    '''
    Get atomic distances

        Parameters:
            positions (numpy.ndarray/torch.Tensor): positions attribute of ase.Atoms
            pbc_offsets (numpy.ndarray/torch.Tensor): periodic boundary condition offsets

        Returns:
            TODO
    '''

    if isinstance(positions, np.ndarray):
        positions = torch.tensor(positions, device=device, dtype=torch.float)

    n_atoms = len(positions)
    n_cells = len(pbc_offsets)
    
    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)
    pbc_offsets = pbc_offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + pbc_offsets

    # calculate the distance between target atom and the periodic images of the other atom
    atom_distance_sqr = torch.linalg.norm(pos1 - pos2, dim=-1)
    # get the minimum distance
    atom_distance_sqr_min, min_indices = torch.min(atom_distance_sqr, dim=-1)

    atom_rij = pos1 - pos2
    min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()

    return atom_distance_sqr_min, atom_rij

def get_all_distances(positions, pbc_offsets, device):
    '''
    Get atomic distances

        Parameters:
            positions (numpy.ndarray/torch.Tensor): positions attribute of ase.Atoms
            pbc_offsets (numpy.ndarray/torch.Tensor): periodic boundary condition offsets

        Returns:
            TODO
    '''

    if isinstance(positions, np.ndarray):
        positions = torch.tensor(positions, device=device, dtype=torch.float)

    n_atoms = len(positions)
    n_cells = len(pbc_offsets)
    
    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)
    pbc_offsets = pbc_offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + pbc_offsets

    # calculate the distance between target atom and the periodic images of the other atom
    atom_distance_sqr = torch.linalg.norm(pos1 - pos2, dim=-1)
    # get the minimum distance
    atom_distance_sqr_min, min_indices = torch.min(atom_distance_sqr, dim=-1)

    atom_rij = pos1 - pos2
    # min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    # atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()

    return atom_distance_sqr, atom_rij

def get_pbc_offsets(cell, offset_num, device):
    '''
    Get periodic boundary condition (PBC) offsets

        Parameters:
            cell (np.ndarray/torch.Tensor): unit cell vectors of ase.cell.Cell
            offset_num:
        
        Returns:
            TODO
    '''
    if isinstance(cell, np.ndarray):
        cell = torch.tensor(np.array(cell), device=device, dtype=torch.float)

    unit_cell = []
    offset_range = np.arange(-offset_num, offset_num + 1)

    for prod in itertools.product(offset_range, offset_range, offset_range):
        unit_cell.append(list(prod))
    
    unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)

    return torch.mm(unit_cell, cell.to(device))

# Obtain unit cell offsets for distance calculation
class PBC_offsets():
    def __init__(self, cell, device, supercell_max=4):
        # set up pbc offsets for minimum distance in pbc
        self.pbc_offsets = []

        for offset_num in range(0, supercell_max):
            unit_cell = []
            offset_range = np.arange(-offset_num, offset_num+1)

            for prod in itertools.product(offset_range, offset_range, offset_range):
                unit_cell.append(list(prod))

            unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)
            self.pbc_offsets.append(torch.mm(unit_cell, cell.to(device)))

    def get_offset(self, offset_num):
        return self.pbc_offsets[offset_num]