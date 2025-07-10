#!/usr/bin/env python3
"""
Diatomic Augmentation Script

This script performs data augmentation on molecular structures by applying 
pulling and perturbation operations, then calculating properties using 
a machine learning potential.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import Any
from tqdm import tqdm

from ase.cell import Cell
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from matdeeplearn.common.ase_utils import MDLCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Molecular structure data augmentation')
    parser.add_argument('--rank', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save augmented data (default: data/silica_10%_5aug/)')
    
    parser.add_argument('--pull_max_strength', type=float, default=0.5,
                        help='Strength of pulling for each round (default: 0.5)')
    parser.add_argument('--perturb_max_strength', type=float, default=0.5,
                        help='Strength of perturbation for each round (default: 0.5)')
    
    parser.add_argument('--n_pairs_to_choose', type=int, default=4,
                        help='Number of closest pairs to choose for pulling (default: 4)')
    parser.add_argument('--n_perturb_sites', type=int, default=5,
                        help='Number of sites to perturb (default: 5)')
    parser.add_argument('--n_round_augmentation', type=int, default=5,
                        help='Number of augmentation rounds (default: 5)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--dataset', type=str,
                        choices=['Silica', 'P', 'mp_subset'],
                        help='Dataset to use')
    parser.add_argument('--train_results_dir', type=str, required=False,
                        help='Directory containing training results CSV files')
    
    return parser.parse_args()


def setup_dataset_paths(dataset: str) -> tuple[str, str, tuple[str, ...]]:
    if dataset == "Silica":
        original_data_path = "data/Silica/original/data.json"
        calculator_path = 'configs/calculator/Silica/potential_only/config_eam_interaction.yml'
        calculated_properties = ('energy', 'forces', 'stress')
    elif dataset == "P":
        original_data_path = "data/p/train_val_data.json"
        calculator_path = 'configs/calculator/phosphorous/potential_only/config_eam_interaction.yml'
        calculated_properties = ('energy', 'forces')
    elif dataset == 'mp_subset':
        original_data_path = "data/Li-O-P-Mn-Subset/data.json"
        calculator_path = 'configs/calculator/mp_subset/potential_only/config_eam_interaction.yml'
        calculated_properties = ('energy', 'forces', 'stress')
    
    return original_data_path, calculator_path, calculated_properties


def load_and_filter_structures(original_data_path: str, train_results_dir: str) -> list[dict[str, Any]]:
    """Load structures and filter based on training results."""
    # Load original data
    with open(original_data_path, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} structures from original data")
    
    # Load training results to get structure IDs
    if train_results_dir is not None:
        df_train = pd.read_csv(os.path.join(train_results_dir, 'train_predictions.csv'))
        df_val = pd.read_csv(os.path.join(train_results_dir, 'val_predictions.csv'))
        names = set(df_train['structure_id'].apply(lambda x: str(x)).values).union(
            set(df_val['structure_id'].apply(lambda x: str(x)).values))
        
        print(f"Found {len(names)} structure IDs from training results")
            
        # Filter data based on structure IDs
        data = [d for d in data if d['structure_id'] in names]
        
    structures = [{key: np.array(val) if isinstance(val, list) else val 
                for key, val in dt.items()} for dt in data]
    print(f"Filtered to {len(structures)} structures")

    return structures


def get_atoms(structure: dict[str, Any]) -> Atoms:
    """Convert structure dictionary to ASE Atoms object."""
    if structure['cell'].ndim == 3:
        cell = Cell(structure['cell'][0])
    else:
        cell = Cell(structure['cell'])
    
    atoms = Atoms(
        symbols=structure['atomic_numbers'],
        positions=structure['positions'],
        cell=cell,
        pbc=[True] * 3
    )
    atoms.structure_id = structure['structure_id']
    return atoms


def get_closest_pairs(atoms: Atoms, n_pairs: int) -> list[tuple[int, int, float]]:
    """Get closest pairs of atoms with no shared atoms."""
    all_distances = atoms.get_all_distances(mic=True)

    # Create list of all unique pairs and their distances
    n_atoms = len(atoms)
    pairs_and_distances = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            pairs_and_distances.append((i, j, all_distances[i,j]))
            
    pairs_and_distances.sort(key=lambda x: x[2])

    # Select n pairs with no shared atoms
    closest_pairs = []
    used_atoms = set()
    
    for i, j, dist in pairs_and_distances:
        if len(closest_pairs) >= n_pairs:
            break
        
        # Skip if either atom is already used
        if i in used_atoms or j in used_atoms:
            continue
            
        closest_pairs.append((i, j, dist))
        used_atoms.add(i)
        used_atoms.add(j)
    
    return closest_pairs


def pull_atoms(atoms: Atoms, eps: float, n_pairs_to_choose: int=4) -> Atoms:
    """Apply pulling augmentation to atoms."""
    a = atoms.copy() 
    closest_pairs = get_closest_pairs(a, n_pairs_to_choose)
    
    positions = a.get_positions()

    for i, j, _ in closest_pairs:
        # Sample random pull distance
        pull_dist = np.random.uniform(0, eps)
        pos_i = positions[i]
        pos_j = positions[j]
        
        if any(a.get_pbc()):
            diff = a.get_distance(i, j, mic=True, vector=True)
        else:
            diff = pos_j - pos_i
            
        direction = diff / np.linalg.norm(diff)        
        positions[i] += direction * (pull_dist/2)
        positions[j] -= direction * (pull_dist/2)
        
    a.set_positions(positions)
    return a


def random_perturb(atoms: Atoms, n_perturb_sites: int, eps: float) -> Atoms:
    """Apply random perturbation to atoms."""
    a = atoms.copy()
    positions = a.get_positions()
    n_atoms = len(a)
    
    perturbed_sites = np.random.choice(n_atoms, n_perturb_sites, replace=False)
    rand_unit_vectors = np.random.randn(n_perturb_sites, 3)
    rand_unit_vectors /= np.linalg.norm(rand_unit_vectors, axis=1)[:, np.newaxis]

    perturb_strengths = np.random.uniform(0, eps, n_perturb_sites)
    positions[perturbed_sites] += rand_unit_vectors * perturb_strengths[:, np.newaxis]
        
    a.set_positions(positions)
    return a


def atoms2dict(atoms: Atoms, structure_id: str) -> dict[str, Any]:
    """Convert ASE Atoms object to dictionary format."""
    return {
        'structure_id': structure_id,
        'atomic_numbers': atoms.get_atomic_numbers().tolist(),
        'positions': atoms.get_positions().tolist(),
        'cell': atoms.get_cell().array.tolist(),
        'pbc': atoms.get_pbc().tolist(),
        'y': atoms.get_potential_energy().item(),
        'forces': atoms.get_forces().tolist(),
        'stress': [atoms.get_stress(voigt=False).tolist()]
    }


def print_force_statistics(structures: list[dict[str, Any]], title_prefix: str="Original"):
    """Print force magnitude statistics."""
    orig_forces = np.concatenate([structure['forces'] for structure in structures])
    force_magnitude = np.linalg.norm(orig_forces, axis=1)
    
    print(f"\n{title_prefix} Force Statistics:")
    print(f"  Mean: {np.mean(force_magnitude):.4f}")
    print(f"  Std:  {np.std(force_magnitude):.4f}")
    print(f"  Min:  {np.min(force_magnitude):.4f}")
    print(f"  Max:  {np.max(force_magnitude):.4f}")
    print(f"  Median: {np.median(force_magnitude):.4f}")
    print(f"  Total force vectors: {len(force_magnitude)}")
    
    return force_magnitude


def print_energy_statistics(energies: np.ndarray, title_prefix: str="Energy"):
    """Print energy statistics."""
    print(f"\n{title_prefix} Statistics:")
    print(f"  Mean: {np.mean(energies):.4f}")
    print(f"  Std:  {np.std(energies):.4f}")
    print(f"  Min:  {np.min(energies):.4f}")
    print(f"  Max:  {np.max(energies):.4f}")
    print(f"  Median: {np.median(energies):.4f}")
    print(f"  Total structures: {len(energies)}")


def print_structure_statistics(structures: list[dict[str, Any]], title: str="Structure"):
    """Print general structure statistics."""
    n_atoms_list = [len(s['atomic_numbers']) for s in structures]
    unique_elements = set()
    for s in structures:
        unique_elements.update(s['atomic_numbers'])
    
    print(f"\n{title} Statistics:")
    print(f"  Total structures: {len(structures)}")
    print(f"  Atoms per structure - Mean: {np.mean(n_atoms_list):.2f}, "
          f"Min: {np.min(n_atoms_list)}, Max: {np.max(n_atoms_list)}")
    print(f"  Unique elements: {sorted(unique_elements)}")
    print(f"  Number of unique elements: {len(unique_elements)}")


def main():
    """Main function to run the augmentation pipeline."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup dataset paths
    original_data_path, calculator_path, calculated_properties = setup_dataset_paths(args.dataset)
    
    # Load and filter structures
    structures = load_and_filter_structures(original_data_path, args.train_results_dir)
    
    # Print original structure and force statistics
    print("="*60)
    print("ORIGINAL DATA ANALYSIS")
    print("="*60)
    print_structure_statistics(structures, "Original Structure")
    print_force_statistics([{'forces': s['forces']} for s in structures], "Original")
    
    # Perform augmentation
    print("\n" + "="*60)
    print("STARTING AUGMENTATION")
    print("="*60)
    print(f"Augmentation parameters:")
    print(f"  Pull strengths: {args.pull_max_strength}")
    print(f"  Perturb strengths: {args.perturb_max_strength}")
    print(f"  Number of rounds: {args.n_round_augmentation}")
    print(f"  Pairs to choose: {args.n_pairs_to_choose}")
    print(f"  Perturb sites: {args.n_perturb_sites}")
    print(f"  Random seed: {args.seed}")
    
    modified_atoms_list = []
    
    if args.n_round_augmentation <= 1:
        # Fractional augmentation
        indices = np.random.choice(
            len(structures),
            int(args.n_round_augmentation * len(structures)),
            replace=False
        )
        original_atoms_list = [get_atoms(structures[i]) for i in indices]
        modified_atoms_list = [
            random_perturb(
                pull_atoms(
                    atoms, args.pull_max_strength, 'uniform', 
                    n_pairs_to_choose=args.n_pairs_to_choose
                ),
                min(args.n_perturb_sites, len(atoms)), args.perturb_max_strength, 
                'uniform', std=args.std
            ) for atoms in original_atoms_list
        ]
    else:
        # Multiple rounds of augmentation
        perturb_strengths = np.linspace(0, args.perturb_max_strength, args.n_round_augmentation)
        pull_strengths = np.linspace(0, args.pull_max_strength, args.n_round_augmentation)
        for i in range(args.n_round_augmentation):
            print(f"Augmentation round {i+1}/{args.n_round_augmentation}")
            original_atoms_list = [get_atoms(structure) for structure in structures]
            
            round_modified = [
                random_perturb(
                    pull_atoms(
                        atoms, pull_strengths[i],
                        n_pairs_to_choose=args.n_pairs_to_choose
                    ),
                    min(args.n_perturb_sites, len(atoms)), perturb_strengths[i], 
                ) for atoms in tqdm(original_atoms_list)
            ]
            modified_atoms_list.extend(round_modified)
    
    # Filter out structures with <= 1 atom
    modified_atoms_list = [atoms for atoms in modified_atoms_list if len(atoms) > 1]
    print(f"\nGenerated {len(modified_atoms_list)} valid augmented structures")
    
    # Setup calculator and compute properties
    print("\n" + "="*60)
    print("COMPUTING PROPERTIES")
    print("="*60)
    print("Loading calculator...")
    calc = MDLCalculator(calculator_path)
    
    print("Computing properties for augmented structures...")
    successful_calculations = 0
    for atoms in tqdm(modified_atoms_list):
        if len(atoms) <= 1:
            continue
        try:
            calc.calculate(atoms, properties=calculated_properties)
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=calc.results['energy'],
                forces=calc.results['forces'],
                stress=calc.results.get('stress', None)
            )
            successful_calculations += 1
        except Exception as e:
            print(f"Warning: Failed to calculate properties for structure: {e}")
    
    print(f"Successfully calculated properties for {successful_calculations}/{len(modified_atoms_list)} structures")
    
    # Convert to dictionary format
    print("\nConverting structures to dictionary format...")
    structures_to_write = []
    for idx, atoms in enumerate(modified_atoms_list):
        if atoms.calc is not None:  # Only include structures with successful calculations
            structures_to_write.append(atoms2dict(atoms, str(idx)))
    
    # Print augmented statistics
    print("\n" + "="*60)
    print("AUGMENTED DATA ANALYSIS")
    print("="*60)
    
    if structures_to_write:
        energies = np.array([s['y'] for s in structures_to_write])
        force_magnitudes = np.array([np.linalg.norm(s['forces']) for s in structures_to_write])
        
        print_structure_statistics(structures_to_write, "Augmented Structure")
        print_energy_statistics(energies, "Augmented Energy")
        print_force_statistics(structures_to_write, "Augmented")
        
        # Additional augmentation-specific statistics
        print(f"\nAugmentation Summary:")
        print(f"  Original structures: {len(structures)}")
        print(f"  Generated structures: {len(modified_atoms_list)}")
        print(f"  Successfully computed: {len(structures_to_write)}")
        print(f"  Success rate: {len(structures_to_write)/len(modified_atoms_list)*100:.1f}%")
        print(f"  Augmentation ratio: {len(structures_to_write)/len(structures):.2f}x")
    else:
        print("No structures with successful property calculations!")
    
    # Save augmented data
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    print(f"Saving {len(structures_to_write)} structures to {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(os.path.join(args.save_dir, 'data.json'), 'w') as f:
        json.dump(structures_to_write, f)
    
    # Save summary statistics to text file
    summary_file = os.path.join(args.save_dir, 'augmentation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Augmentation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Dataset: {args.dataset}\n")
        f.write(f"  Pull strengths: {args.pull_max_strength}\n")
        f.write(f"  Perturb strengths: {args.perturb_max_strength}\n")
        f.write(f"  Number of rounds: {args.n_round_augmentation}\n")
        f.write(f"  Pairs to choose: {args.n_pairs_to_choose}\n")
        f.write(f"  Perturb sites: {args.n_perturb_sites}\n")
        f.write(f"  Random seed: {args.seed}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Original structures: {len(structures)}\n")
        f.write(f"  Generated structures: {len(modified_atoms_list)}\n")
        f.write(f"  Successfully computed: {len(structures_to_write)}\n")
        f.write(f"  Success rate: {len(structures_to_write)/len(modified_atoms_list)*100:.1f}%\n")
        f.write(f"  Augmentation ratio: {len(structures_to_write)/len(structures):.2f}x\n")
    
    print(f"Summary saved to: {summary_file}")
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"Final dataset size: {len(structures_to_write)} structures")
    print(f"Data saved to: {os.path.join(args.save_dir, 'data.json')}")
    print("="*60)


if __name__ == "__main__":
    main()