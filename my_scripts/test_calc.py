import torch
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.build import make_supercell
from ase import Atoms
import numpy as np
import psutil
from tqdm import tqdm


if __name__ == '__main__':
    dta = torch.load('data/force_data/data.pt')
    atoms_list = MDLCalculator.data_to_atoms_list(dta[0])
    calc_str = './configs/calculator/config_torchmd.yml'
    calculator = MDLCalculator(config=calc_str)
    
    for a in atoms_list:
        if a.structure_id == 'mp-772822-0-2':
            calculator.calculate(a)
            print(calculator.results)

    
    
