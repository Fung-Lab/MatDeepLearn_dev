from matdeeplearn.common.ase_utils import MDLCalculator, MDSimulator
from time import time
from helpers import build_atoms_list, evaluate
import logging
import copy
import random
import pandas as pd

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft,\
        unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json')
    
    random.seed(42)
    idx = [random.randint(0, len(relaxed_ids) - 1) for _ in range(500)]
    
    # for i, atoms in enumerate(relaxed):
    #     if atoms.structure_id == 'mp-18929':
    #         idx = i

    # Configurations below
    calc_str = './configs/calculator/config_cgcnn_lj.yml'
    simulation_type = 'NVT'
    num_steps = 5000
    temperature = 1000
    
    save_to = 'train_outs/lj_sim_no_coef.csv'
    save = True
    # Configurations above
    
    calculator = MDLCalculator(config=calc_str)
    
    if save:
        logging.info(f"Saving simulation results to: {save_to}")

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
        cols['Starting energy'].append(relaxed[atoms_idx].get_potential_energy()[0][0])
        to_optim = copy.deepcopy(relaxed[atoms_idx])
        final_atoms, time_per_step, h, l = simulator.run_simulation(to_optim, num_steps=num_steps)
        cols['Highest energy'].append(h)
        cols['Lowest energy'].append(l)
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