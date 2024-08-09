import copy
import os
import logging
from typing import Tuple, List
from time import time
import json

import numpy as np
import pandas as pd
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.io.trajectory import Trajectory
from ase.io import read, write

from helpers import get_test_structures_from_json
from matdeeplearn.common.ase_utils import MDLCalculator

logging.basicConfig(level=logging.INFO)

class StructureOptimizer:
    """
    This class provides functionality to optimize the structure of an Atoms object using a specified calculator.
    """
    def __init__(self,
                 calculator,
                 relax_cell: bool = False,
                 ):
        """
        Initialize the StructureOptimizer.

        Parameters:
        - calculator (Calculator): A calculator object for performing energy and force calculations.
        - relax_cell (bool): If True, the cell will be relaxed in addition to positions during the optimization process.
        """
        self.calculator = calculator
        self.relax_cell = relax_cell
        
    def optimize(self, atoms: Atoms, logfile=None, write_traj_name=None) -> Tuple[Atoms, float]:
        """
        This method optimizes the structure of the given Atoms object using the specified calculator.
        If `relax_cell` is True, the cell will be relaxed. Trajectory information can be written to a file.

        Parameters:
        - atoms: An Atoms object representing the structure to be optimized.
        - logfile: File to write optimization log information.
        - write_traj_name: File to write trajectory of the optimization.

        Returns:
        - atoms: The optimized Atoms object.
        - time_per_step: The average time taken per optimization step.
        """
        atoms.calc = self.calculator       
        if self.relax_cell:
            atoms = ExpCellFilter(atoms)

        optimizer = FIRE(atoms, logfile=logfile)
        
        if write_traj_name is not None:
            traj = Trajectory(write_traj_name + '.traj', 'w', atoms)
            optimizer.attach(traj.write, interval=1)

        start_time = time()
        optimizer.run(fmax=0.001, steps=500)
        end_time = time()
        num_steps = optimizer.get_number_of_steps()
        
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
        return atoms, time_per_step
    
def traj_to_xdatcar(traj_file) -> None:
    """
    This method reads a trajectory file and saves it in the XDATCAR format.

    Parameters:
    - traj_file (str): Path to the trajectory file.

    Returns:
    - None
    """
    traj = read(traj_file, index=':')
    file_name, _ = os.path.splitext(traj_file)
    write(file_name + '.XDATCAR', traj)
    
def get_min_interatomic_distance(structure: Atoms):
    all_dist = structure.get_all_distances()
    np.fill_diagonal(all_dist, np.inf)
    return all_dist.min().min()
    
if __name__ == '__main__':
    
    device = 'cuda:0'
    
    # config for calculator
    calculator_config = './configs/calculator/config_cgcnn_pretrained.yml'
    calculator = MDLCalculator(calculator_config, rank=device)
    
    # You need to turn structures into a list of Atoms objects
    # MDLCalculator has a data_to_atoms_list that can convert a Data object to a list of Atoms objects
    # unrelaxed_ids, relaxed_ids, unrelaxed, relaxed, dft,\
    #     unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list('./data/optimization_data/data.json', 5000)
    
    dataset = 'data/silica_optimization/data.json'
    test_pred_csv_path = 'results/2024-04-20-16-37-53-453-cgcnn_pretrain_sio2/train_results/test_predictions.csv'
    test_indices = set(get_test_structures_from_json(test_pred_csv_path, dataset))
    
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    original_atoms: List[Atoms] = []
    
    for i, dt in enumerate(data):
        if i in test_indices:
            atoms = Atoms(symbols=dt['atomic_numbers'],
                          positions=dt['positions'],
                          cell=dt['cell'])
            atoms.structure_id = dt['structure_id']
            original_atoms.append(atoms)
    
    optimized_atoms: List[Atoms] = []
    
    optim = StructureOptimizer(calculator, relax_cell=True)
    start = time()
    times = []
    
    results = []
        
    for idx, atoms in enumerate(original_atoms):
        atoms.set_calculator(calculator)
        entry = {
            "structure_id": atoms.structure_id,
            "original_energy": atoms.get_potential_energy(),
            "starting_min_dist": get_min_interatomic_distance(atoms)
        }
        to_optim = copy.deepcopy(atoms)
        optimized, time_per_step = optim.optimize(to_optim, write_traj_name='optimization')
        times.append(time_per_step)
        optimized_atoms.append(optimized)
        entry["optimized_energy"] = optimized.get_potential_energy()
        entry["ending_min_dist"] = get_min_interatomic_distance(optimized)
        results.append(entry)
        if (idx + 1) % 20 == 0:
            logging.info(f"Completed optimizing {idx + 1} structures.")
    end = time()
    
    df = pd.DataFrame(results)
    df.to_csv('optimization_silica_pretrain.csv', index=False)
    