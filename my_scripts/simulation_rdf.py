from matdeeplearn.common.ase_utils import MDLCalculator, MDSimulator
from time import time
from helpers import build_atoms_list, evaluate, get_test_structures, get_rdf
import logging
import copy
import random
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    # unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft,\
    #     unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json', 1)
    data = torch.load('./data/Silica_data/data.pt')
    
    relaxed = MDLCalculator.data_to_atoms_list(data[0])
    relaxed_ids = data[0].structure_id

    # random.seed(42)
    test_pred_csv_path = 'results/2024-04-10-15-17-44-421-torchmd_silica_batch_2/train_results/test_predictions.csv'
    data_pt_path = './data/Silica_data/data.pt'
    idx = get_test_structures(test_pred_csv_path, data_pt_path)[:5]
    
    # for i, atoms in enumerate(relaxed):
    #     if atoms.structure_id == 'mp-18929':
    #         idx = i

    # Configurations below
    calc_str = './configs/calculator/config_cgcnn_pretrained.yml'
    simulation_type = 'NVT'
    num_steps = 8000
    temperature = 2000
    
    save_to = 'train_outs/cgcnn_pretrained.csv'
    save = False
    # Configurations above
    
    calculator = MDLCalculator(config=calc_str)
    
    if save:
        logging.info(f"Saving simulation results to: {save_to}")
        logging.info(f"Simulation type: {simulation_type}, num_steps: {num_steps}, temperature: {temperature} K")

    original_atoms, optimized_atoms = [], []
    
    simulator = MDSimulator(simulation_type, 5., temperature=temperature, calculator=calculator)
    start = time()
    times = []
    startings, lows, highs = [], [], []
    ids = []
    
    cols = {
        'structure_id': ids,
        "Starting energy": startings,
        "Highest energy": highs, 
        "Lowest energy": lows
    }
        
    for i, atoms_idx in enumerate(idx):
        cols['structure_id'].append(relaxed[atoms_idx].structure_id)
        relaxed[atoms_idx].set_calculator(calculator)
        cols['Starting rdf'].append(relaxed[atoms_idx])
        to_optim = copy.deepcopy(relaxed[atoms_idx])
        final_atoms, time_per_step, h, l = simulator.run_simulation(to_optim, num_steps=num_steps)
        cols['Highest energy'].append(h)
        cols['Lowest energy'].append(l)
        print(f"start={relaxed[atoms_idx].get_potential_energy():.4f}, high={h:.4f}, low={l:.4f}")
        if (i + 1) % 10 == 0:
            logging.info(f"Completed simulating {i + 1} structures.")
            
        if (i + 1) % 10 == 0:
            logging.info(f"Saving first {i + 1} results...")
            df = pd.DataFrame(cols)
            try:
                df.to_csv(save_to, mode='a', header=False, index=False)
            except FileNotFoundError:
                df.to_csv(save_to, header=True, index=False)
                
            cols = {key: [] for key in cols}
    
    end = time()
    
    print(f"Time elapsed: {end - start:.3f}")
    
    df = pd.DataFrame(cols)
    df.to_csv(save_to, mode='a', header=False, index=False)

    # for key in result.keys():
    #     print(f"{key}: {result[key]:.4f}")