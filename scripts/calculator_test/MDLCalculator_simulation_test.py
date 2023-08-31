from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
from matdeeplearn.common.MDLCalculator import MDLCalculator
from asap3 import EMT  # Way too slow with ase.EMT !
import torch


dta = torch.load('./data/data.pt')
atoms_list = MDLCalculator.data_to_atoms_list(data=dta[0]) 
atoms = atoms_list[0]

size = 10
T = 500  # Kelvin

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = MDLCalculator(config='./configs/config_calculator.yml')

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 5 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):  
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temperature = ekin / (1.5 * units.kB)
    etot = epot + ekin
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  Etot = %.3feV' % (epot.item(), ekin, temperature, etot.item()))

dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('moldyn3.traj', 'w', atoms)
dyn.attach(traj.write, interval=50)

# Now run the dynamics
printenergy()
dyn.run(5000)