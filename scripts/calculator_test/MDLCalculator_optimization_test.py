from ase import Atoms
from ase.optimize import FIRE
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
import numpy as np
import torch


dta = torch.load('./data/test_data/data.pt')
atoms_list = MDLCalculator.data_to_atoms_list(data=dta[0]) 
atoms = atoms_list[0]
atoms.calc = MDLCalculator(config='./configs/config_calculator.yml')

print(f'Original positions: {atoms.get_positions()}')
print(f'Original energy: {atoms.get_potential_energy()}')

optimizer = FIRE(atoms)
traj = Trajectory('./trajs/optim_CGCNN_graph_scalar.traj', 'w', atoms)
optimizer.attach(traj.write, interval=1)
optimizer.run(fmax=0.001, steps=1000)

traj = read('./trajs/optim_CGCNN_graph_scalar.traj', index=':')
write('./trajs/optim_CGCNN_graph_scalar.XDATCAR', traj)

print(f'Optimized positions: {atoms.get_positions()}')
print(f'Optimized energy: {atoms.get_potential_energy()}')
