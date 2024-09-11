from matdeeplearn.common.ase_utils import MDLCalculator
from simulators import MetricMDSimulator
from time import time
from helpers import get_test_structures_from_pt, build_atoms_list
import logging
import pandas as pd
import torch
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    # ------ Silica ------
    # data = torch.load('data/Silica_data/data.pt', map_location='cpu')[0]
    # test_pred_csv_path = 'results/cgcnn/2024-06-14-09-41-25-357-cgcnn_sio2/train_results/test_predictions.csv'
    # dataset = 'data/Silica/original/data.pt'
    # idx = get_test_structures_from_pt(test_pred_csv_path, dataset)
    # atoms_list = MDLCalculator.data_to_atoms_list(data)
    # ------ Silica ------
    
    # ------ Materials Project ------
    # unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft_unrelaxed,\
    #     unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json')
    
    # np.random.seed(123)
    # idx = np.random.choice(len(relaxed), 400, replace=False)
    # atoms_list = relaxed
    # atoms_list = [a for i, a in enumerate(atoms_list) if i in idx]
    # ------ Materials Project ------

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--dataset', type=str, choices=['silica', 'mp'])
    parser.add_argument('--save', type=str, default='simulation.csv')
    args = parser.parse_args()
    
    if args.dataset == "silica":
        pt_path = "/net/csefiles/coc-fung-cluster/Qianyu/data/Silica/original/data.pt"
        data = torch.load(pt_path, map_location='cpu')[0]
        test_pred_csv_path = 'results/finetuned/2024-08-21-12-08-21-040-finetune_filtered_silica_0_0.5_2_aug/train_results/test_predictions.csv'
        idx = get_test_structures_from_pt(test_pred_csv_path, pt_path)
        atoms_list = MDLCalculator.data_to_atoms_list(data)
        atoms_list = [a for i, a in enumerate(atoms_list) if i in idx]
    else:
        unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft_unrelaxed,\
            unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('/net/csefiles/coc-fung-cluster/Qianyu/data//optimization_data/data.json')
    
        np.random.seed(42)
        idx = np.random.choice(len(relaxed), 200, replace=False)
        atoms_list = relaxed
        atoms_list = [a for i, a in enumerate(atoms_list) if i in idx]
        print([a.structure_id for a in atoms_list[:5]])
    
    simulator = MetricMDSimulator(args.config_path)
    save_to = args.save
 
    # if save:
    logging.info(f"Running simulation with {len(atoms_list)} structures")
    logging.info(f"Saving simulation results to: {save_to}")
    logging.info(f"Simulation type: {simulator.simulation}, num_steps: {simulator.total_steps}, temperature: {simulator.temperature} K")
    logging.info(f"Running on device: {simulator.device}")

    start = time()
    
    accumulated_metrics = {}
        
    for i, atoms_sim in enumerate(atoms_list):
        
        metrics = simulator.run_simulation(atoms_sim)
        for key, value in metrics.items():
            if key not in accumulated_metrics:
                accumulated_metrics[key] = []
            accumulated_metrics[key].append(value)
        
        if (i + 1) % 5 == 0:
            logging.info(f"Saving first {i + 1} results...")
            df = pd.DataFrame(accumulated_metrics)
            header = i + 1 == 5
            df.to_csv(save_to, mode='a', header=header, index=False)
            accumulated_metrics = {key: [] for key in accumulated_metrics}
    
    end = time()
    
    print(f"Time elapsed: {end - start:.3f}")
    
    df = pd.DataFrame(accumulated_metrics)
    df.to_csv(save_to, mode='a', header=False, index=False)

    # for key in result.keys():
    #     print(f"{key}: {result[key]:.4f}")