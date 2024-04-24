from abc import ABC, abstractmethod
import logging
import os
from time import time
import yaml

import numpy as np
from scipy.stats import norm

from ase import Atoms, units
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet  import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.optimization.neighbors import find_points_in_spheres

from matdeeplearn.common.ase_utils import MDLCalculator


logging.basicConfig(level=logging.INFO)

class AbstractSimulator(ABC):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.simulation = config['simulation_type']
        self.timestep = config['timestep']
        self.temperature = config['temperature']
        self.device = config.get('device', 'cpu')
        self.calculator = MDLCalculator(config['calculator_config_path'], self.device)
        self.total_steps = config['total_steps']
        
        self.log_console = config.get('log_console', False)
        self.save_metrics_per = config.get('save_metrics_per', self.total_steps // 5)
        self.continue_simulation = config.get('continue_simulation', False)
        self.state_path = config.get('state_path', None)
        self.metrics = config.get('metrics', ['energy', 'mae_rdf', 'min_dist', 'large_forces'])
        
        self.additional_args = config.get('additional_args', {})
        force_n_std = config.get('force_n_std', 5)
        force_mean = config.get('force_mean', 2.7553)
        force_std = config.get('force_std', 2.1227)
        self.force_threshold = force_mean + force_n_std * force_std
        self.simulation_class = self.setup_dynamics()
        
        self.save = config.get('save', False)
        self.save_atomic_state_folder = config.get('save_atomic_state_folder', None)
        self.save_metrics_folder = config.get('save_metrics_folder', None)
        
        if self.save:
            os.makedirs(self.save_atomic_state_folder, exist_ok=True)
            os.makedirs(self.save_metrics_folder, exist_ok=True)
            
            if self.save_metrics_per != 0:
                assert self.save_metrics_folder is not None, "Path to save metrics must be provided."
                assert self.save_atomic_state_folder is not None, "Path to save atomic state must be provided."
            
            if self.continue_simulation and self.state_path is None:
                raise ValueError("Path must be provided to continue simulation.")
        
    def get_rdf(self, structure: Atoms, σ = 0.05, dr = 0.01, max_r = 12.0):
        rmax = max_r + 3 * σ + dr
        rs = np.arange(0.5, rmax + dr, dr)
        nr = len(rs) - 1
        natoms = len(structure)

        normalization = 4 / structure.get_cell().volume * np.pi
        normalization *= (natoms * rs[0:-1]) ** 2

        lattice_matrix = np.array(structure.get_cell(), dtype=float)
        cart_coords = np.array(structure.get_positions(), dtype=float)

        rdf = sum(np.histogram(find_points_in_spheres(
                all_coords = cart_coords,
                center_coords = np.array([cart_coords[i]], dtype=float),
                r = rmax,
                pbc = np.array([1, 1, 1], dtype=int),
                lattice = lattice_matrix,
                tol = 1e-8,
            )[3], rs)[0] for i in range(natoms))

        return np.convolve(rdf / normalization,
                            norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ),
                            mode="same")[0:(nr - int((3 * σ) / dr) - 1)]
    
    def get_min_interatomic_distance(self, structure: Atoms):
        all_dist = structure.get_all_distances()
        np.fill_diagonal(all_dist, np.inf)
        return all_dist.min().min()
    
    def calc_energy(self, structure: Atoms):
        epot = structure.get_potential_energy()
        ekin = structure.get_kinetic_energy()
        etotal = epot + ekin
        return etotal
    
    @abstractmethod
    def run_simulation(self, atoms: Atoms):
        pass
    
    def save_atoms_state(self, atoms: Atoms, curr_step: int):
        np.savez(os.path.join(self.save_metrics_folder, f'atoms_state_at_step_{curr_step}.npz'), curr_step=curr_step,
                 positions=atoms.get_positions(), momenta=atoms.get_momenta(),
                 velocities=atoms.get_velocities())
        
    def load_atoms_state(self, atoms: Atoms, file_name: str):
        """
        This method loads the state of the atoms from a .npz file and returns the current step.
        """
        state = np.load(file_name)
        atoms.set_positions(state['positions'])
        atoms.set_momenta(state['momenta'])
        atoms.set_velocities(state['velocities'])
        return state['curr_step']
    
    def setup_dynamics(self):
        common_valid_params = ['trajectory', 'logfile', 'loginterval', 'append_trajectory', 'temperature_K']
        valid_params = common_valid_params

        if self.simulation == 'NVE':
            valid_params.append('dt')
            simulation_class = VelocityVerlet
        elif self.simulation == 'NVT':
            valid_params += ['friction', 'fixcm', 'communicator', 'rng']
            self.additional_args.setdefault('friction', 0.001 / units.fs)
            self.additional_args.setdefault('temperature_K', self.temperature)
            simulation_class = Langevin
        elif self.simulation == 'NPT':
            valid_params += ['externalstress', 'ttime', 'pfactor', 'mask']
            self.additional_args.setdefault('externalstress', 0.01623)
            self.additional_args.setdefault('pfactor', 0.6)
            self.additional_args.setdefault('temperature_K', self.temperature)
            simulation_class = NPT
        else:
            raise NotImplementedError("Currently unimplemented simulation type")

        invalid_params = [key for key in self.additional_args.keys() if key not in valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_params)}")

        return simulation_class
        
class LongMDSimulator(AbstractSimulator):
    """
    This class provides a simple interface for running molecular dynamics simulations.
    It currently supports microcanonical ('NVE'), canonical ('NVT'), or isothermal-isobaric ('NPT') simulations.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)


    def run_simulation(self, atoms: Atoms):
        atoms.set_calculator(self.calculator)

        dyn = self.simulation_class(atoms, timestep=self.timestep * units.fs, **self.additional_args)

        energy, rdf_changes, min_dists, forces_gt_threshold = [], [], [], []
        rolling_rdf = LongMDSimulator.get_rdf(atoms)
        
        if not self.continue_simulation:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
            starting_step = 1
        else:
            logging.info("Loading atoms state from " + self.state_path)
            curr_step = self.load_atoms_state(atoms, self.state_path)
            assert self.total_steps > curr_step, "Total steps must be greater than the current step."
            starting_step = curr_step + 1
            logging.info(f"Have already run {curr_step} steps. Continuing for {self.total_steps - curr_step} more steps.")
        
        start = time()
        for i in range(starting_step, self.total_steps + 1, 1):
            e_total = self.calc_energy(atoms)
            current_rdf = self.get_rdf(atoms)
            if rolling_rdf.shape[0] != 10:
                rolling_rdf = np.vstack((rolling_rdf, current_rdf))
                rdf_changes.append(0)
            else:
                mean_prev_10_rdf = np.mean(rolling_rdf, axis=0)
                rdf_change = np.mean(np.abs(current_rdf - mean_prev_10_rdf))
                rdf_changes.append(rdf_change)
                rolling_rdf[0] = current_rdf
                rolling_rdf = np.roll(rolling_rdf, -1, axis=0)
                
            min_dist = self.get_min_interatomic_distance()
            energy.append(e_total)
            min_dists.append(min_dist)
            dyn.step()
            
            forces = atoms.get_calculator().results['forces']
            force_magnitute = np.linalg.norm(forces, axis=1)
            if len(forces_gt_threshold) == 0:
                forces_gt_threshold.append(any(force_magnitute > self.force_threshold))
            else:
                forces_gt_threshold.append(
                    forces_gt_threshold[-1] + any(force_magnitute > self.force_threshold)
                )
            
            if self.save_metrics_per != 0 and i != starting_step and i % self.save_metrics_per == 0:
                logging.info(f"Step: {i} out of {self.total_steps}. Time taken: {time() - start:.3f}")
                start = time()
                metrics = {
                    'energy': energy,
                    'rolling_rdf': rolling_rdf,
                    'num_gt_force_threshold': forces_gt_threshold[-1],
                }
                self.save_info_to_npz(energy, rdf_changes, min_dists, forces_gt_threshold, i)
                self.save_atoms_state(atoms, i)
                for i in [energy, rdf_changes, min_dists]:
                    i.clear()
                last_thres = forces_gt_threshold[-1]
                forces_gt_threshold.clear()
                forces_gt_threshold.append(last_thres)
        end = time()
        
        if self.log_console:
            logging.info(f"Time: {end-start:.4f}")
            
        return energy, rdf_changes, min_dists, forces_gt_threshold
        
    def save_info_to_npz(self, energy, rdf_changes, min_dists, forces_gt_threshold, curr_step):
        """
        This method saves the energy, RDF changes, and minimum interatomic distances to a .npz file.

        Parameters:
        - energy (list): List of energy values.
        - rdf_changes (list): List of RDF changes.
        - min_dists (list): List of minimum interatomic distances.
        - file_name (str): Name of the .npz file to save the information.
        """
        if len(forces_gt_threshold) == self.save_metrics_per + 1:
            forces_gt_threshold = forces_gt_threshold[1:]
        np.savez(os.path.join(self.save_atomic_state_folder,
                              f'metrics_at_step_{curr_step}.npz'),
                 energy=energy, rdf_changes=rdf_changes,
                 min_dists=min_dists, num_gt_force_threshold=forces_gt_threshold)
     
    
class MetricMDSimulator(AbstractSimulator):
    """
    This class provides a simple interface for running molecular dynamics simulations.
    It currently supports microcanonical ('NVE'), canonical ('NVT'), or isothermal-isobaric ('NPT') simulations.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def run_simulation(self,
                       atoms: Atoms) -> None:
        dyn = self.simulation_class(atoms, timestep=self.timestep * units.fs, **self.additional_args)
        final_metrics = {'structure_id': atoms.structure_id}
        atoms.set_calculator(self.calculator)
        
        if 'energy' in self.metrics:
            starting = atoms.get_potential_energy()
            final_metrics['staring_energy'] = starting
            final_metrics['highest_e'] = starting
            final_metrics['lowest_e'] = starting
        if 'mae_rdf' in self.metrics:
            rolling_rdf = self.get_rdf(atoms)
            final_metrics['highest_rdf_mae'] = 0
        if 'min_dist' in self.metrics:
            starting = self.get_min_interatomic_distance(atoms)
            final_metrics['starting_min_dist'] = starting
            final_metrics['min_min_dist'] = starting
            final_metrics['max_min_dist'] = starting
        if 'large_forces' in self.metrics:
            final_metrics['large_forces'] = 0
            
        def update_metics(a=atoms):
            nonlocal final_metrics, rolling_rdf
            if 'energy' in self.metrics:
                etotal = self.calc_energy(a)
                final_metrics['highest_e'] = max(final_metrics['highest_e'], etotal) 
                final_metrics['lowest_e'] = min(final_metrics['lowest_e'], etotal)
            if 'mae_rdf' in self.metrics:
                current_rdf = self.get_rdf(atoms)
                if rolling_rdf.shape[0] != 10:
                    rolling_rdf = np.vstack((rolling_rdf, current_rdf))
                else:
                    mean_prev_10_rdf = np.mean(rolling_rdf, axis=0)
                    rdf_change = np.mean(np.abs(current_rdf - mean_prev_10_rdf))
                    final_metrics['highest_rdf_mae'] = max(final_metrics['highest_rdf_mae'], rdf_change)
                    rolling_rdf[0] = current_rdf
                    rolling_rdf = np.roll(rolling_rdf, -1, axis=0)
            if 'min_dist' in self.metrics:
                curr_min_dist = self.get_min_interatomic_distance(a)
                final_metrics['min_min_dist'] = min(final_metrics['min_min_dist'], curr_min_dist)
                final_metrics['max_min_dist'] = max(final_metrics['max_min_dist'], curr_min_dist)
            if 'large_forces' in self.metrics:
                forces = a.get_calculator().results['forces']
                force_magnitute = np.linalg.norm(forces, axis=1)
                final_metrics['large_forces'] += any(force_magnitute > self.force_threshold)
            
        dyn.attach(update_metics, interval=1)

        start = time()
        dyn.run(steps=self.total_steps)
        end = time()
        final_metrics['duration'] = end - start
        return final_metrics


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