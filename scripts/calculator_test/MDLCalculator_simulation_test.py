from matdeeplearn.common.ase_utils import MDLCalculator, Simulator
from ase.md.langevin import Langevin
from ase import units
import torch


dta = torch.load('./data/force_data/data.pt')
atoms_list = MDLCalculator.data_to_atoms_list(data=dta[0]) 
atoms = atoms_list[0]
initial_t = 100
calculator = MDLCalculator(config='./configs/config_calculator.yml')
dyn = Langevin(atoms, 0.8 * units.fs, initial_t * units.kB, 0.002)
save_freq = 50

sim = Simulator(atoms, calculator, dyn, initial_t, save_frequency=save_freq, save_dir='.')

# Now run the dynamics
sim.run(5000)