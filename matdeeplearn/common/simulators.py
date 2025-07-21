import logging
import os
import yaml
from time import time
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms, units
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet  import VelocityVerlet
from ase.md.npt import NPT

from matdeeplearn.common.ase_utils import MDLCalculator


logging.basicConfig(level=logging.INFO)

class AbstractSimulator(ABC):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.simulation = config['simulation_type']
        self.timestep = config['timestep']
        self.temperature = config['temperature']
        self.save_traj_dir = config.get('save_traj_dir', None)
        if self.save_traj_dir is not None:
            os.makedirs(self.save_traj_dir, exist_ok=True)
        self.device = config.get('device', 'cpu')
        self.calculator = MDLCalculator(config['calculator_config_path'], rank=self.device)
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
    
    
    def setup_dynamics(self):
        common_valid_params = ['trajectory', 'logfile', 'loginterval', 'append_trajectory', 'temperature_K']
        valid_params = common_valid_params

        if self.simulation == 'NVE':
            valid_params.append('dt')
            simulation_class = VelocityVerlet
        elif self.simulation == 'NVT':
            valid_params += ['friction', 'fixcm', 'communicator', 'rng']
            self.additional_args.setdefault('friction', 0.01 / units.fs)
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

    
class MetricMDSimulator(AbstractSimulator):
    """
    This class provides a simple interface for running molecular dynamics simulations.
    It currently supports microcanonical ('NVE'), canonical ('NVT'), or isothermal-isobaric ('NPT') simulations.
    """
    def run_simulation(self,
                       atoms: Atoms) -> None:
        dyn = self.simulation_class(atoms, timestep=self.timestep * units.fs, **self.additional_args)
        final_metrics = {'structure_id': atoms.structure_id}
        atoms.set_calculator(self.calculator)
        
        if 'energy' in self.metrics:
            starting = atoms.get_potential_energy() / len(atoms.get_atomic_numbers())
            final_metrics['starting_energy'] = final_metrics['highest_e'] = final_metrics['lowest_e'] = starting
        if 'min_dist' in self.metrics:
            starting = self.get_min_interatomic_distance(atoms)
            final_metrics['starting_min_dist'] = final_metrics['min_min_dist'] = final_metrics['max_min_dist'] = starting
        if 'large_forces' in self.metrics:
            final_metrics['large_forces'] = 0
        
        metrics = {
            key: [val] for key, val in final_metrics.items()
        }
        for key in ['energy', 'rdf_change', 'min_dist']:
            metrics[key] = [0]
        trajectory = [atoms.copy()]
        save_traj = False
        
        def update_metics(a=atoms):
            nonlocal final_metrics, metrics, save_traj
            if 'energy' in self.metrics:
                etotal = self.calc_energy(a) / len(atoms.get_atomic_numbers())
                final_metrics['highest_e'] = max(final_metrics['highest_e'], etotal) 
                final_metrics['lowest_e'] = min(final_metrics['lowest_e'], etotal)
                metrics['energy'].append(etotal)
            if 'min_dist' in self.metrics:
                curr_min_dist = self.get_min_interatomic_distance(a)
                metrics['min_dist'].append(curr_min_dist)
                final_metrics['min_min_dist'] = min(final_metrics['min_min_dist'], curr_min_dist)
                final_metrics['max_min_dist'] = max(final_metrics['max_min_dist'], curr_min_dist)
            if 'large_forces' in self.metrics:
                forces = a.get_calculator().results['forces']
                force_magnitude = np.linalg.norm(forces, axis=1)
                final_metrics['large_forces'] += any(force_magnitude > self.force_threshold)

            for key in final_metrics.keys():
                metrics[key].append(final_metrics[key])
            trajectory.append(a.copy())
            
        dyn.attach(update_metics, interval=1)

        start = time()
        try:
            dyn.run(steps=self.total_steps)
            final_metrics['exception_at'] = -1
        except Exception as e:
            final_metrics['exception_at'] = dyn.nsteps
            print(e)
        end = time()
        final_metrics['duration'] = end - start
        if self.save_traj_dir is not None:
            self.save_trajectory(atoms.structure_id, trajectory)
        return final_metrics
    
    def save_trajectory(self, id: str, traj: list[Atoms]):
        dirname = os.path.join(self.save_traj_dir, f"trajectory_{id}")
        filename = os.path.join(dirname, f"trajectory_{id}.traj")
        os.makedirs(dirname, exist_ok=True)
        ase_traj = Trajectory(filename, 'w')

        for i in range(len(traj)):
            ase_traj.write(traj[i])

        ase_traj.close()
        print(f"Trajectory with metrics saved as {dirname}")