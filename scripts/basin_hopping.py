import numpy as np
import ase
from ase import Atoms
from matdeeplearn.common.ase_utils import MDLCalculator
from ase.optimize import FIRE
from time import time


if __name__ == '__main__':

    atomic_numbers = [22, 22, 8, 8, 8, 8]
    scaled_positions = np.random.uniform(0, 1, size=(len(atomic_numbers), 3))
    cell = [5, 5, 5, 90, 90, 90]
    atoms = Atoms(numbers=atomic_numbers, scaled_positions=scaled_positions, cell=cell, pbc=(True, True, True))

    calc_str = './configs/config_calculator.yml'
    print('Using calculator config from', calc_str)
    calculator = MDLCalculator(config=calc_str)

    hops = 20
    steps = 500
    dr = 0.5
    atoms.calc = calculator

    optimizer = FIRE(atoms, logfile='basin_hopping.log')
    minAtoms = atoms.copy()
    currAtoms = atoms.copy()
    currAtoms.set_calculator(calculator)
    minEnergy = currAtoms.get_potential_energy()

    for i in range(hops):
        oldEnergy = currAtoms.get_potential_energy()
        optimizer = FIRE(currAtoms, logfile='hop' + str(i) + '.log')
        start_time = time()
        optimizer.run(fmax=0.001, steps=steps)
        end_time = time()
        num_steps = optimizer.get_number_of_steps()
        time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
        print('HOP', i, 'took', end_time - start_time, 'seconds')
        print('HOP', i, 'took', time_per_step, 'seconds per step')
        optimizedEnergy = currAtoms.get_potential_energy()
        print('HOP', i, 'old energy', oldEnergy)
        print('HOP', i, 'optimized energy', optimizedEnergy)
        if optimizedEnergy < minEnergy:
            minAtoms = currAtoms.copy()
            minEnergy = optimizedEnergy
        disp = np.random.uniform(-1., 1., (len(atoms), 3)) * dr
        currAtoms.set_positions(currAtoms.get_positions() + disp)
    
    print('Minimum energy', minEnergy)
    ase.io.write('min.cif', minAtoms, format='cif')



