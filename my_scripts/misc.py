import torch
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import norm
from pymatgen.optimization.neighbors import find_points_in_spheres

import ase
from ase import Atoms
from ase.md import Langevin as LangevinASE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from torch_geometric.data import Data
from matdeeplearn.common.batch_langevin import Langevin
from matdeeplearn.common.ase_utils import MDLCalculator


def get_rdf(structure: Atoms, σ = 0.05, dr = 0.01, max_r = 12.0):
    rmax = max_r + 3 * σ + dr
    rs = np.arange(0.5, rmax + dr, dr)
    nr = len(rs) - 1
    natoms = len(structure)

    normalization = 4 / structure.get_cell().volume * np.pi
    normalization *= (natoms * rs[0:-1]) ** 2

    rdf = np.zeros(nr, dtype = int)
    lattice_matrix = np.array(structure.get_cell(), dtype=float)
    cart_coords = np.array(structure.get_positions(), dtype=float)

    for i in range(natoms):
        rdf += np.histogram(find_points_in_spheres(
            all_coords = cart_coords,
            center_coords = np.array([cart_coords[i]], dtype=float),
            r = rmax,
            pbc = np.array([1, 1, 1], dtype=int),
            lattice = lattice_matrix,
            tol = 1e-8,
        )[3], rs)[0]
        
    return np.convolve(rdf / normalization,
                        norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ),
                        mode="same")[0:(nr - int((3 * σ) / dr) - 1)]
    
    
def get_rdf_test(structure: Atoms, σ = 0.05, dr = 0.01, max_r = 12.0):
    rmax = max_r + 3 * σ + dr
    rs = np.arange(0.5, rmax + dr, dr)
    nr = len(rs) - 1
    natoms = len(structure)

    normalization = 4 / structure.get_cell().volume * np.pi
    normalization *= (natoms * rs[0:-1]) ** 2

    lattice_matrix = np.array(structure.get_cell(), dtype=float)
    cart_coords = np.array(structure.get_positions(), dtype=float)

    rdf = sum(np.histogram(find_points_in_spheres(
            all_coords = cart_coords,
            center_coords = np.array([cart_coords[i]], dtype=float),
            r = rmax,
            pbc = np.array([1, 1, 1], dtype=int),
            lattice = lattice_matrix,
            tol = 1e-8,
        )[3], rs)[0] for i in range(natoms))

    return np.convolve(rdf / normalization,
                        norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ),
                        mode="same")[0:(nr - int((3 * σ) / dr) - 1)]


if __name__ == '__main__':    
    data = torch.load('data/Silica_data/data.pt')[0]
    atoms_list = MDLCalculator.data_to_atoms_list(data)
    data_list = []
    
    atoms = atoms_list[0]
    print(np.allclose(get_rdf(atoms), get_rdf_test(atoms)))