import torch
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.build import make_supercell
from ase import Atoms
import numpy as np
import psutil
from tqdm import tqdm


if __name__ == '__main__':
    pid = psutil.Process()
    dta = torch.load('data/force_data/data.pt')
    atoms_list = MDLCalculator.data_to_atoms_list(dta[0])
    calc_str = './configs/calculator/config_cgcnn.yml'
    calculator = MDLCalculator(config=calc_str)
    
    N = 20
    
    atoms = atoms_list[0]
    super_cell = make_supercell(atoms, [[N, 0, 0],[0, 1, 0],[0, 0, 1]])
    print(len(super_cell.get_atomic_numbers()))
    calculator.calculate(super_cell)
    
    memory_info = pid.memory_info()
    print(f"Memory used: {memory_info.rss / (1024 ** 2):.2f} MB")  # in megabytes
    
