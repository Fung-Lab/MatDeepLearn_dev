from helpers import get_rdf, get_test_structures
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from ase_utils_dev import MDLCalculator, MDSimulator

# Load the silica dataset
data = torch.load('data/Silica_data/data.pt')[0]
test_pred_csv_path = 'results/2024-04-06-10-48-11-973-cgcnn_sio2/train_results/test_predictions.csv'
silica_dataset = 'data/Silica_data/data.pt'

# Get the test structures from the silica dataset
idx = get_test_structures(test_pred_csv_path, silica_dataset)
atoms_list = MDLCalculator.data_to_atoms_list(data)
atoms_list = [a for i, a in enumerate(atoms_list) if i in idx]

config_path = 'configs/sim_configs/cgcnn_sam.yml'
simulator = MDSimulator(config_path)

# timesteps = list(range(num_steps))
start = time()
energy, rdf, min_dists, forces_gt_threshold = simulator.run_simulation(atoms_list[107])
print(f"Time taken for simulation: {time() - start:.3f}")

# Previous id: 46