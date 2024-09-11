import torch
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.build import make_supercell
from ase import Atoms
import numpy as np
from tqdm import tqdm
from time import time


if __name__ == '__main__':
    dta = torch.load('/net/csefiles/coc-fung-cluster/Qianyu/data/Silica/original/data.pt')
    atoms_list = MDLCalculator.data_to_atoms_list(dta[0])
    calc_str = 'configs/calculator/Silica/config_cgcnn_quant.yml'
    # calc_str = './configs/silica/config_cgcnn_hybrid_new.yml'
    calculator = MDLCalculator(config=calc_str)
    
    s_time = time()
    for a in atoms_list:
        calculator.calculate(a)
    print('Time:', time()-s_time)
        

    
    
