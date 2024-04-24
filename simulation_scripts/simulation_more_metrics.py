from matdeeplearn.common.ase_utils import MDLCalculator
from simulators import MetricMDSimulator
from time import time
from helpers import get_test_structures
import logging
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    # Load the silica dataset
    data = torch.load('data/Silica_data/data.pt')[0]
    test_pred_csv_path = 'results/2024-04-06-10-48-11-973-cgcnn_sio2/train_results/test_predictions.csv'
    silica_dataset = 'data/Silica_data/data.pt'

    # Get the test structures from the silica dataset
    idx = get_test_structures(test_pred_csv_path, silica_dataset)
    atoms_list = MDLCalculator.data_to_atoms_list(data)
    atoms_list = [a for i, a in enumerate(atoms_list) if i in idx][:10]

    config_path = 'configs/sim_configs/cgcnn.yml'
    simulator = MetricMDSimulator(config_path)

    # Configurations below
    save_to = 'cgcnn_all_metrics_si.csv'
    save = True
    # Configurations above
    
    # if save:
    logging.info(f"Saving simulation results to: {save_to}")
    logging.info(f"Simulation type: {simulator.simulation}, num_steps: {simulator.total_steps}, temperature: {simulator.temperature} K")

    start = time()
    
    accumulated_metrics = {}
        
    for i, atoms_sim in enumerate(atoms_list):
        
        metrics = simulator.run_simulation(atoms_sim)
        for key, value in metrics.items():
            if key not in accumulated_metrics:
                accumulated_metrics[key] = []
            accumulated_metrics[key].append(value)
        
        # print("Finished simulating structure")
        # print(f"start={relaxed[atoms_idx].get_potential_energy():.4f}, high={h:.4f}, low={l:.4f}")
        if (i + 1) % 5 == 0:
            logging.info(f"Completed simulating {i + 1} structures.")
            
        if save and (i + 1) % 5 == 0:
            logging.info(f"Saving first {i + 1} results...")
            df = pd.DataFrame(accumulated_metrics)
            header = i + 1 == 5
            df.to_csv(save_to, mode='a', header=header, index=False)
            accumulated_metrics = {key: [] for key in accumulated_metrics}
    
    end = time()
    
    print(f"Time elapsed: {end - start:.3f}")
    
    df = pd.DataFrame(accumulated_metrics)
    print(df.head(5))
    if save:
        df.to_csv(save_to, mode='a', header=False, index=False)

    # for key in result.keys():
    #     print(f"{key}: {result[key]:.4f}")