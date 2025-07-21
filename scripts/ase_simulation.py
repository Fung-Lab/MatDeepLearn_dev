import logging
import argparse
import warnings
import random
import json
from time import time


import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import Cell
from ase.calculators.singlepoint import SinglePointCalculator

import torch
from matdeeplearn.common.simulators import MetricMDSimulator

logging.basicConfig(level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def load_mp_subset_test_data(
    json_path=None, test_csv_path=None, n_structures=100
) -> list[Atoms]:
    if json_path is None:
        json_path = '/net/csefiles/coc-fung-cluster/Qianyu/data/Li-O-P-Mn-Subset/test_data.json'

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if test_csv_path is not None:
        df = pd.read_csv(test_csv_path)
        structure_ids = df['structure_id'].tolist()
        structure_ids = set(map(lambda x: str(x), structure_ids))
        data = [d for d in data if d['structure_id'] in structure_ids]
        
    atoms_list = []
    for d in data:
        cell = Cell(d['cell'])
        atoms = Atoms(
            symbols=d['atomic_numbers'],
            positions=d['positions'],
            cell=cell,
            pbc=[True, True, True]
        )
        atoms.structure_id = d['structure_id']
        atoms_list.append(atoms)
    return atoms_list[:n_structures]

def load_test_silica_data(json_path=None) -> list[Atoms]:
    if json_path is None:
        json_path = '/net/csefiles/coc-fung-cluster/Qianyu/data/Silica/original/test_data.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    atoms_list = []
    for d in data:
        atoms = Atoms(
            symbols=d['atomic_numbers'],
            positions=d['positions'],
            cell=Cell(d['cell']),
            pbc=[True, True, True]
        )
        atoms.structure_id = d['structure_id']
        atoms.calc = SinglePointCalculator(
            atoms, energy=d['y'], forces=d['forces'], stress=d['stress'][0]
        )
        atoms_list.append(atoms)
        
    return atoms_list[-100:]

def load_test_phosphorous_data(json_path=None) -> list[Atoms]:
    if json_path is None:
        json_path = '/net/csefiles/coc-fung-cluster/Qianyu/data/p/test_data.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    atoms_list = []
    for d in data:
        atoms = Atoms(
            symbols=d['atomic_numbers'],
            positions=d['positions'],
            cell=Cell(d['cell'][0]),
            pbc=[True, True, True]
        )
        atoms.structure_id = d['structure_id']
        atoms.calc = SinglePointCalculator(
            atoms, energy=d['y'], forces=d['forces']
        )
        atoms_list.append(atoms)
        
    return atoms_list
                         
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--dataset', type=str, choices=['silica', 'silica_test', 'mp', 'mp_subset', 'mp_subset_test', 'p', 'p_test'], default='silica')
    parser.add_argument('--save', type=str, default='simulation.csv')
    args = parser.parse_args()

    if args.dataset == "mp_subset_test":
        atoms_list = load_mp_subset_test_data()
    elif args.dataset == "silica_test":
        atoms_list = load_test_silica_data()
    elif args.dataset == "p_test":
        atoms_list = load_test_phosphorous_data()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    simulator = MetricMDSimulator(args.config_path)
    save_to = args.save
 
    # if save:
    logging.info(f"Running simulation with {len(atoms_list)} structures")
    logging.info(f"Saving simulation results to: {save_to}")
    logging.info(f"Simulation type: {simulator.simulation}, num_steps: {simulator.total_steps}, temperature: {simulator.temperature} K")
    logging.info(f"Running on device: {simulator.device}")

    start = time()
    
    accumulated_metrics = {}

    seed_everything(42)
    for i, atoms_sim in enumerate(atoms_list):
        metrics = simulator.run_simulation(atoms_sim)
        for key, value in metrics.items():
            if key not in accumulated_metrics:
                accumulated_metrics[key] = []
            accumulated_metrics[key].append(value)
        
        if (i + 1) % 5 == 0:
            print(f"Saving first {i + 1} results...")
            df = pd.DataFrame(accumulated_metrics)
            header = i + 1 == 5
            df.to_csv(save_to, mode='a', header=header, index=False)
            accumulated_metrics = {key: [] for key in accumulated_metrics}
    
    end = time()
    
    print(f"Time elapsed: {end - start:.3f}")
    df = pd.DataFrame(accumulated_metrics)
    df.to_csv(save_to, mode='a', header=False, index=False)