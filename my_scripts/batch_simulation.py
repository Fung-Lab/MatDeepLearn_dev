from matdeeplearn.common.ase_utils import MDLCalculator, MDSimulator
from time import time
from helpers import build_atoms_list, evaluate, get_test_structures
import logging
import copy
import random
import pandas as pd
import torch

from torch_geometric.data import Data
from matdeeplearn.common.batch_langevin import Langevin

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    # unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft,\
    #     unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json', 1)
    device = 'cuda:0'
    dyn = Langevin(
        timestep=5.,
        config_str='configs/calculator/config_cgcnn.yml',
        temperature_K=2500,
        friction=0.1,
        rank=device
    )
    
    data = torch.load('data/Silica_data/data.pt')[0]
    atoms_list = MDLCalculator.data_to_atoms_list(data)
    data_list = []
    
    for atoms in atoms_list:
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device)
        pos = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())
        data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
            z=atomic_numbers)
        
        data_list.append(data)

    # random.seed(42)
    test_pred_csv_path = 'results/2024-04-06-17-34-58-184-cgcnn_sam_sio2/train_results/test_predictions.csv'
    data_pt_path = './data/Silica_data/data.pt'
    idx = get_test_structures(test_pred_csv_path, data_pt_path)
    data_list = [data_list[i] for i in idx]

    num_steps = 1000
    start = time()
    max_energies, min_energies = dyn.run_simulation(data_list[:8], num_steps, batch_size=1)
    end = time()
    print(max_energies, min_energies)
    print(f"Time elapsed: {end - start:.3f}")
    