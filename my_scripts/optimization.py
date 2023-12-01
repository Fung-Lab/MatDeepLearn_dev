from matdeeplearn.common.ase_utils import MDLCalculator, StructureOptimizer
from time import time
from helpers import build_atoms_list, evaluate
import logging
import copy

logging.basicConfig(level=logging.INFO)

                         
if __name__ == '__main__':
    unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft,\
        unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json')

    # Configurations below
    calc_str = './configs/calculator/config_cgcnn_morse.yml'
    
    save = True
    folder = './train_outs/relaxed_data_cgcnn_morse'
    # folder = './train_outs/test'
    optim_type = 'relax'
    # Configurations above
    
    calculator = MDLCalculator(config=calc_str)
    
    if save:
        logging.info(f"Saving optimizaion results for {optim_type}ed structures to: {folder}")

    original_atoms, optimized_atoms = [], []
    
    optim = StructureOptimizer(calculator, relax_cell=True)
    start = time()
    times = []
    
    to_optim = relaxed if optim_type == 'relax' else unrelaxed
        
    for idx, atoms in enumerate(to_optim):
        atoms.set_calculator(calculator)
        original_atoms.append(atoms)
        to_optim = copy.deepcopy(atoms)
        optimized, time_per_step = optim.optimize(to_optim)
        times.append(time_per_step)
        optimized_atoms.append(optimized)
        if (idx + 1) % 20 == 0:
            logging.info(f"Completed optimizing {idx + 1} structures.")
    end = time()
    
    structure_ids = relaxed_ids if optim_type == 'relax' else unrelaxed_ids

    print(f"Time elapsed: {end - start:.3f}")
    result = evaluate(structure_ids, optim_type, original_atoms, optimized_atoms, times, dft_relaxed=dft,
                      save=save, optim=optim, folder=folder, filename=optim_type)

    for key in result.keys():
        print(f"{key}: {result[key]:.4f}")