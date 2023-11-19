from ase import Atoms
from matdeeplearn.common.ase_utils import MDLCalculator, StructureOptimizer
import numpy as np
from time import time
import numpy as np
from tqdm import tqdm
import json
from ase.geometry import Cell
import copy
import os
from scipy.stats import norm
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_rdf(structure: Atoms, σ = 0.05, dr = 0.01, max_r = 12.0):
    rmax = max_r + 3 * σ + dr
    rs = np.arange(0.5, rmax + dr, dr)
    nr = len(rs) - 1
    natoms = len(structure)

    normalization = 4 / structure.get_cell().volume * np.pi
    normalization *= (natoms * rs[0:-1]) ** 2

    rdf = np.zeros(nr, dtype = int)
    lattice_matrix = np.array(structure.get_cell(), dtype=float)
    cart_coords = np.array(structure.get_positions(), dtype=float)

    for i in range(natoms):
        rdf += np.histogram(find_points_in_spheres(
            all_coords = cart_coords,
            center_coords = np.array([cart_coords[i]], dtype=float),
            r = rmax,
            pbc = np.array([1, 1, 1], dtype=int),
            lattice = lattice_matrix,
            tol = 1e-8,
        )[3], rs)[0]

    return np.convolve(rdf / normalization,
                        norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ),
                        mode="same")[0:(nr - int((3 * σ) / dr) - 1)]

def rdf_mse(rdf1, rdf2):
    if isinstance(rdf1, list):
        rdf1 = np.array(rdf1)
    if isinstance(rdf2, list):
        rdf2 = np.array(rdf2)
    return np.mean((rdf1 - rdf2) ** 2)

def wright_factor(rdf, rdf_ref):
    return np.sum((rdf - rdf_ref) ** 2) / np.sum(rdf_ref ** 2)
    
def cartesian_distance(original, optimized):
    original, optimized = np.array(original), np.array(optimized)
    dist = np.sqrt(np.sum((original - optimized) ** 2, axis=1))
    return np.mean(dist)
    
def build_atoms_list(num=-1):
    with open('./data/relax_test_data/data.json', 'r') as f:
        data = json.load(f)
    total = num if num != -1 else len(data)
    unrelaxed = []
    dft_unrelaxed = []
    dft_unrelaxed_energy = []
    relaxed = []
    unrelaxed_energy = []
    relaxed_energy = []
    ids = []
    for structure in data[:total]:
        if 'type' not in structure.keys():
            continue

        # unrelaxed structures: need its unrelaxed and ground truth relaxed information
        if structure['type'] == 'unrelaxed_in_test':
            unrelaxed_atoms = Atoms(symbols=structure['atomic_numbers'],
                                    positions=structure['unrelaxed_positions'],
                                    cell=Cell(structure['unrelaxed_cell']))
            dft_atoms = Atoms(symbols=structure['atomic_numbers'],
                                    positions=structure['relaxed_positions'],
                                    cell=Cell(structure['relaxed_cell']))
            
            unrelaxed_atoms.structure_id = structure['structure_id']
            dft_atoms.structure_id = structure['structure_id']
            unrelaxed.append(unrelaxed_atoms)
            dft_unrelaxed.append(dft_atoms)
            unrelaxed_energy.append(structure['unrelaxed_energy'])
            dft_unrelaxed_energy.append(structure['relaxed_energy'])
        
        # relaxed structures: only need its ground truth relaxed information
        if structure['type'] == 'relaxed_in_test':
            relaxed_atoms = Atoms(symbols=structure['atomic_numbers'],
                                  positions=structure['relaxed_positions'],
                                  cell=Cell(structure['relaxed_cell']))
            relaxed_atoms.structure_id = structure['structure_id']
            relaxed.append(relaxed_atoms)
            relaxed_energy.append(structure['relaxed_energy'])
        
        ids.append(structure['structure_id'])
    return ids, unrelaxed, relaxed, dft_unrelaxed,\
        unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy


def evaluate(task, original_atoms, optimized_atoms, times, dft_relaxed=None,
             save=False, optim=None, folder=None, filename=None):
    original_energy = [atoms.get_potential_energy() for atoms in original_atoms]
    optimized_energy = [atoms.get_potential_energy() for atoms in optimized_atoms]
    n_atoms = [len(atoms.get_atomic_numbers()) for atoms in optimized_atoms]
    original_pos = [atoms.get_positions() for atoms in original_atoms]
    optimized_pos = [atoms.get_positions() for atoms in optimized_atoms]
    original_cell = [atoms.get_cell() for atoms in original_atoms]
    optimized_cell = [atoms.get_cell() for atoms in optimized_atoms]

    result = None

    e_drop = np.mean((np.array(original_energy) - np.array(optimized_energy)) / n_atoms)

    if task == 'relaxed':
        e_drop = np.mean((np.array(original_energy) - np.array(optimized_energy)) / n_atoms)
        pos_delta = np.mean([cartesian_distance(pos1, pos2) for pos1, pos2 in zip(original_pos, optimized_pos)])
        cell_delta = np.mean([cartesian_distance(c1, c2) for c1, c2 in zip(original_cell, optimized_cell)])

        rdf_optimized = [get_rdf(atoms) for atoms in optimized_atoms]
        rdf_original = [get_rdf(atoms) for atoms in original_atoms]
        rdf_compared_to_dft = np.mean([wright_factor(x, y) for x, y in zip(rdf_optimized, rdf_original)])

        result = {
            'Average energy drop per atom': e_drop,
            'Average change in atoms cartesian positions': pos_delta,
            'Average change in cell parameters': cell_delta,
            'MSE for RDF': rdf_mse(rdf_original, rdf_optimized),
            'Wright Factor': rdf_compared_to_dft,
            'Averge time per step': np.mean(times)
        }
    elif task == 'unrelaxed':
        assert dft_relaxed is not None
        dft_pos = [atoms.get_positions() for atoms in dft_relaxed]
        pos_compared_to_dft = np.mean([cartesian_distance(pos1, pos2) for pos1, pos2 in zip(optimized_pos, dft_pos)])
        
        dft_cell = [atoms.get_cell()[:] for atoms in dft_relaxed]
        cell_compared_to_dft = np.mean([cartesian_distance(c1, c2) for c1, c2 in zip(optimized_cell, dft_cell)])

        rdf_optimized = [get_rdf(atoms) for atoms in optimized_atoms]
        rdf_dft = [get_rdf(atoms) for atoms in dft_relaxed]
        rdf_compared_to_dft = np.mean([wright_factor(x, y) for x, y in zip(rdf_dft, rdf_optimized)])

        result = {
            'Average energy drop per atom': e_drop,
            'Average cartesian distance for atom positions compared to dft': pos_compared_to_dft,
            'Average cartesian distance for cell positions compared to dft': cell_compared_to_dft,
            'MSE for RDF': rdf_mse(rdf_dft, rdf_optimized),
            'Wright Factor': rdf_compared_to_dft,
            'Averge time per step': np.mean(times)
        }
    else:
        raise NotImplementedError("Not implemented evaluate type.")
    
    if optim is not None and save:
        os.makedirs(folder, exist_ok=True)
        optim.save_results(original_energy, optimized_energy, structure_ids, f"{folder}/{filename}_energy.csv", header="strucure_id, original_energy, optimized_energy")
        optim.save_results(original_pos, optimized_pos, structure_ids, f"{folder}/{filename}_positions.csv", header="strucure_id, original_pos, original_pos, original_pos, optimized_pos, optimized_pos, optimized_pos")
        optim.save_results(original_cell, optimized_cell, structure_ids, f"{folder}/{filename}_cell.csv", header="strucure_id, original_cell, original_cell, original_cell, optimized_cell, optimized_cell, optimized_cell")
    return result

                         
if __name__ == '__main__':
    '''
    What you need to adjust:
    - A data.json containing relaxed/unrelaxed version of structures (line 58)
    - A config file containing information for calculator (line 174)
    - Choose whether optimizing relaxed or unrelaxed structures (line 183 and 186)
    - Folder to save results (line 197)
    '''
    structure_ids, unrelaxed, relaxed, dft,\
        unrelaxed_energy, relaxed_energy, dft_unrelaxed_energy = build_atoms_list(5)
    
    calc_str = './configs/config_calculator.yml'
    print('Using calculator config from', calc_str)
    calculator = MDLCalculator(config=calc_str)

    original_atoms, optimized_atoms = [], []
    
    optim = StructureOptimizer(calculator, relax_cell=True)
    start = time()
    total_iterations = len(relaxed)
    times = []
        
    for idx, atoms in enumerate(tqdm(relaxed, total=total_iterations)):
        atoms.set_calculator(calculator)
        original_atoms.append(atoms)
        to_optim = copy.deepcopy(atoms)
        optimized, time_per_step = optim.optimize(to_optim)
        times.append(time_per_step)
        optimized_atoms.append(optimized)
    end = time()

    print(f"Time elapsed: {end - start:.3f}")
    result = evaluate('relaxed', original_atoms, optimized_atoms, times, dft_relaxed=dft,
                      save=True, optim=optim, folder='./train_outs/test_folder', filename='relax')

    for key in result.keys():
        print(f"{key}: {result[key]:.4f}")