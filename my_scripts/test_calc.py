import torch
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.build import make_supercell
from ase import Atoms
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    dta = torch.load('data/Silica_data/data.pt')
    atoms_list = MDLCalculator.data_to_atoms_list(dta[0])
    calc_str = './configs/silica/config_hybrid_edge_torchmd.yml'
    # calc_str = './configs/silica/config_cgcnn_hybrid_new.yml'
    calculator = MDLCalculator(config=calc_str)
    
    for a in atoms_list:
        if a.get_positions().shape[0] < 20:
            print('structure:', a.structure_id)
            calculator.calculate(a)
            print(calculator.results['energy'])
            # print(calculator.results['forces'])
            break

    
    
