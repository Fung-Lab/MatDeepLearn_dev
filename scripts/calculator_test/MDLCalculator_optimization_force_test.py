from ase import Atoms
from ase.optimize import FIRE
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
import numpy as np
import torch
from time import time
import numpy as np


def energy_drop(original_energy, optimized_energy, per_atom=False, n_atoms=None):
    original_energy = np.array(original_energy)
    optimized_energy = np.array(optimized_energy)
    n_atoms = np.array(n_atoms)
    if per_atom and n_atoms is not None:
        return np.mean((original_energy - optimized_energy) / n_atoms)
    return np.mean(original_energy - optimized_energy)
    
def cartesian_distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2, axis=1))

def perform_optimization(atoms, calculator, logfile=None, write_traj_name=None):
    optimizer = FIRE(atoms, logfile=logfile)
    
    if write_traj_name is not None:
        traj = Trajectory(write_traj_name + '.traj', 'w', atoms)
        optimizer.attach(traj.write, interval=1)
        
    optimizer.run(fmax=0.001, steps=500)
    
    if write_traj_name is not None:
        traj = read(write_traj_name + '.traj', index=':')
        write(write_traj_name + '.XDATCAR', traj)
    return atoms

dta = torch.load('./data/force_data/data.pt')
atoms_list = MDLCalculator.data_to_atoms_list(data=dta[0]) 
calculator = MDLCalculator(config='./configs/config_calculator.yml')

#print(f'Original positions: {atoms.get_positions()}')
#print(f'Original energy: {atoms.get_potential_energy()}')

original_energy, optimized_energy, n_atoms = [], [], []
original_pos, optimized_pos = [], []

start = time()

for atoms in atoms_list[0:20]:
    atoms.calc = calculator
    
    original_energy.append(atoms.get_potential_energy())
    original_pos.append(atoms.get_positions())
    n_atoms.append(len(atoms.get_atomic_numbers()))
    
    optimized_atoms = perform_optimization(atoms, calculator)
    
    optimized_energy.append(optimized_atoms.get_potential_energy())
    optimized_pos.append(optimized_atoms.get_positions())
end = time()

#print(f'Optimized positions: {atoms.get_positions()}')
#print(f'Optimized energy: {atoms.get_potential_energy()}')

print(f'Average energy drop: {energy_drop(original_energy, optimized_energy):.5f}')
print(f'Average energy drop per atom: {energy_drop(original_energy, optimized_energy, True, n_atoms):.5f}')
#print(f'Cartesian dist: {cartesian_distance(original_pos[0], optimized_pos[0])}')
print(f'Time elapsed: {end - start:.2f}s.')

