from helpers import get_rdf, get_test_structures
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from ase_utils_dev import MDLCalculator, MDSimulator

# Load the silica dataset
# Replace 'path_to_silica_dataset' with the actual path to your silica dataset
data = torch.load('data/Silica_data/data.pt')[0]
test_pred_csv_path = 'results/2024-04-06-10-48-11-973-cgcnn_sio2/train_results/test_predictions.csv'
silica_dataset = 'data/Silica_data/data.pt'

# Get the test structures from the silica dataset
idx = get_test_structures(test_pred_csv_path, silica_dataset)
atoms_list = MDLCalculator.data_to_atoms_list(data)
atoms_list = [a for i, a in enumerate(atoms_list) if i in idx]

calc_str = './configs/calculator/config_cgcnn.yml'
simulation_type = 'NVT'
num_steps = 3000
temperature = 2000
calculator = MDLCalculator(config=calc_str)
simulator = MDSimulator(simulation_type, 5., temperature=temperature, calculator=calculator)

# atoms = atoms_list[3]
# print(atoms.structure_id)

# si_mask = atoms.get_atomic_numbers() == 14
# o_mask = atoms.get_atomic_numbers() == 8

# energy, rdf, min_dists = simulator.run_simulation(atoms, num_steps)
# Create a list of timesteps
timesteps = list(range(num_steps))
plt.figure(figsize=(20, 15))

atoms_idx = [2]
for idx in range(len(atoms_idx)):
    energy, rdf, min_dists = simulator.run_simulation(atoms_list[atoms_idx[idx]], num_steps)
    timesteps = list(range(num_steps))[750:1250]
    energy = energy[750:1250]
    rdf = rdf[750:1250]
    min_dists = min_dists[750:1250]
    
    plt.subplot(2, 3, idx * 3 + 1)
    plt.plot(timesteps, energy, label='Energy')
    plt.xlabel('Timesteps')
    plt.ylabel('Energy')
    plt.title('Energy over Time')
    
    plt.subplot(2, 3, idx * 3 + 2)
    plt.plot(timesteps, rdf, label='RDF MAE', color='orange')
    plt.xlabel('Timesteps')
    plt.ylabel('RDF MAE')
    plt.title('RDF over Time')
    
    plt.subplot(2, 3, idx * 3 + 3)
    plt.plot(timesteps, min_dists, label='Min Dists', color='green')
    plt.xlabel('Timesteps')
    plt.ylabel('Min Dists')
    plt.title('Min Dists over Time')

plt.tight_layout()
plt.savefig('energy_rdf_min_dists_zoomed.png')

# energy, rdf = np.array(energy), np.array(rdf)

# rdf_drastic_change, normal_rdf = rdf[energy > 200], rdf[energy <= 200]

# print(rdf_drastic_change.shape, normal_rdf.shape)

# # Plotting the two different distributions of rdf on the same plot
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.hist(rdf_drastic_change, bins=50, color='r', alpha=0.7, label='RDF Drastic Change')
# plt.title(f'RDF Drastic Change Distribution: {len(rdf_drastic_change)} steps')
# plt.ylabel('Frequency')
# plt.xlim(right=0.07)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.hist(normal_rdf, bins=50, color='b', alpha=0.7, label='Normal RDF')
# plt.title(f'Normal RDF Distribution: {len(normal_rdf)} steps')
# plt.xlabel('RDF')
# plt.ylabel('Frequency')
# plt.xlim(right=0.07)
# plt.legend()
# plt.savefig('rdf_normal_4.png')
