from ase import Atoms
from matdeeplearn.common.ase_utils import MDLCalculator, StructureOptimizer
from time import time
import numpy as np
from tqdm import tqdm
import json
from ase.geometry import Cell
import copy
import os

    
def build_atoms_list_from_json(json_path, num=-1):
    with open(json_path, 'r') as f:
        data = json.load(f)
    total = num if num != -1 else len(data)
    atoms_list, id_list = [], []
    for structure in data[:total]:
        atoms = Atoms(symbols=structure['atomic_numbers'],
                                positions=structure['positions'],
                                cell=Cell(structure['cell']))
        atoms.structure_id = structure['structure_id']
        atoms_list.append(atoms)
        id_list.append(structure['structure_id'])
    return atoms_list, id_list

def save(optim, folder, filename, structure_ids, original_atoms, optimized_atoms):
    original_energy = [atoms.get_potential_energy() for atoms in original_atoms]
    optimized_energy = [atoms.get_potential_energy() for atoms in optimized_atoms]
    original_pos = [atoms.get_positions() for atoms in original_atoms]
    optimized_pos = [atoms.get_positions() for atoms in optimized_atoms]
    original_cell = [atoms.get_cell() for atoms in original_atoms]
    optimized_cell = [atoms.get_cell() for atoms in optimized_atoms]
    if optim is not None and save:
        os.makedirs(folder, exist_ok=True)
        optim.save_results(original_energy, optimized_energy, structure_ids, os.path.join(folder, f"{filename}_energy.csv"), header="strucure_id, original_energy, optimized_energy")
        optim.save_results(original_pos, optimized_pos, structure_ids, os.path.join(folder, f"{filename}_positions.csv"), header="strucure_id, original_pos, original_pos, original_pos, optimized_pos, optimized_pos, optimized_pos")
        optim.save_results(original_cell, optimized_cell, structure_ids,os.path.join(folder, f"{filename}_cell.csv"), header="strucure_id, original_cell, original_cell, original_cell, optimized_cell, optimized_cell, optimized_cell")

                         
if __name__ == '__main__':
    atoms_list, id_list = build_atoms_list_from_json('./data/force_data/data.json', 5)
    calc_str = './configs/calculator/config_calculator.yml'
    print('Using calculator config from', calc_str)
    calculator = MDLCalculator(config=calc_str)

    original_atoms, optimized_atoms = [], []
    
    optim = StructureOptimizer(calculator, relax_cell=True)
    start = time()
    total_iterations = len(atoms_list)
    times = []
        
    for idx, atoms in enumerate(tqdm(atoms_list, total=total_iterations)):
        atoms.set_calculator(calculator)
        original_atoms.append(atoms)
        to_optim = copy.deepcopy(atoms)
        optimized, time_per_step = optim.optimize(to_optim)
        times.append(time_per_step)
        optimized_atoms.append(optimized)
    end = time()
    
    save(optim, 'train_outs/test_folder', 'test', id_list, original_atoms, optimized_atoms)
    print(f"Time elapsed: {end - start:.3f}")
    