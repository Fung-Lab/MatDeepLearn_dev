import torch
import numpy as np
import yaml
from ase import Atoms, units
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
from matdeeplearn.preprocessor.helpers import generate_node_features
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
import logging
from typing import List
from matdeeplearn.common.registry import registry
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet  import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.minimahopping import MinimaHopping
import os
from time import time


logging.basicConfig(level=logging.INFO)


class MDLCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(self, config):
        """
        Initialize the MDLCalculator instance.

        Args:
            config (str or dict): Configuration settings for the MDLCalculator.

        Raises:
            AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        """
        Calculator.__init__(self)
        if isinstance(config, str):
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
                
        gradient = config["model"].get("gradient", False)
        otf_edge_index = config["model"].get("otf_edge_index", False)
        otf_edge_attr = config["model"].get("otf_edge_attr", False)
        self.otf_node_attr = config["model"].get("otf_node_attr", False)
        assert otf_edge_index and otf_edge_attr and gradient, "To use this calculator to calculate forces and stress, you should set otf_edge_index, oft_edge_attr and gradient to True."
        
        trainer_name = config.get("trainer", "matdeeplearn.trainers.PropertyTrainer")
        assert trainer_name.count(".") >= 1, "Trainer name should be in format {module}.{trainer_name}, like matdeeplearn.trainers.PropertyTrainer"
        
        trainer_cls = registry.get_trainer_class(trainer_name)
        load_state = config['task'].get('checkpoint_path', None)
        assert trainer_cls is not None, "Trainer not found"
        self.trainer = trainer_cls.from_config(config)
        
        try:
            self.trainer.load_checkpoint()
        except ValueError:
            logging.warning("No checkpoint.pt file is found, and an untrained model is used for prediction.")
        
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)
        self.device = 'cpu'

    def calculate(self, atoms: Atoms, properties=implemented_properties, system_changes=None):
        """
        Calculate energy, forces, and stress for a given ase.Atoms object.

        Args:
            atoms (ase.Atoms): The atomic structure for which calculations are to be performed.
            properties (list): List of properties to calculate. Defaults to ['energy', 'forces', 'stress'].
            system_changes: Not supported in the current implementation.

        Returns:
            None: The results are stored in the instance variable 'self.results'.

        Note:
            This method performs energy, forces, and stress calculations using a neural network-based calculator.
            The results are stored in the instance variable 'self.results' as 'energy', 'forces', and 'stress'.
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())

        data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
            z=atomic_numbers, structure_id=atoms.info.get('structure_id', None))
        
        # Generate node features
        if not self.otf_node_attr:
            generate_node_features(data, self.n_neighbors, device=self.device)
            data.x = data.x.to(torch.float32)
        
        data_list = [data]
        loader = DataLoader(data_list, batch_size=1)

        out = self.trainer.predict_by_calculator(loader)
        self.results['energy'] = out['energy']
        self.results['forces'] = out['forces']
        self.results['stress'] = out['stress']
        self.results['free_energy'] = out['energy']
        
    @staticmethod
    def data_to_atoms_list(data: Data) -> List[Atoms]:
        """
        This helper method takes a 'torch_geometric.data.Data' object containing information about atomic structures
        and converts it into a list of 'ase.Atoms' objects. Each 'Atoms' object represents an atomic structure
        with its associated properties such as positions and cell.
        
        Args:
            data (Data): A data object containing information about atomic structures.
            
        Returns:
            List[Atoms]: A list of 'ase.Atoms' objects, each representing an atomic structure
                         with positions and associated properties.
        """
        cells = data.cell.numpy()
        
        split_indices = np.cumsum(data.n_atoms)[:-1]
        positions_per_structure = np.split(data.pos.numpy(), split_indices)
        symbols_per_structure = np.split(data.z.numpy(), split_indices)
        
        atoms_list = [Atoms(
                        symbols=symbols_per_structure[i],
                        positions=positions_per_structure[i],
                        cell=Cell(cells[i])) for i in range(len(data.structure_id))]
        for i in range(len(data.structure_id)):
            atoms_list[i].structure_id = data.structure_id[i][0]
        return atoms_list
    
            
class StructureOptimizer:
    def __init__(self,
                 calculator,
                 relax_cell: bool = False,
                 ):
        self.calculator = calculator
        self.relax_cell = relax_cell
        
    def optimize(self, atoms: Atoms, logfile=None, write_traj_name=None, minima_traj='minima.traj'):
        atoms.calc = self.calculator       
        #if self.relax_cell:
        #    atoms = ExpCellFilter(atoms)

        #optimizer = FIRE(atoms, logfile=logfile)
        optimizer = MinimaHopping(atoms, logfile=logfile, minima_traj=minima_traj)
        
        if write_traj_name is not None:
            traj = Trajectory(write_traj_name + '.traj', 'w', atoms)
            optimizer.attach(traj.write, interval=1)

        start_time = time()
        #optimizer.run(fmax=0.001, steps=500)
        optimizer(totalsteps=10)
        end_time = time()
        #num_steps = optimizer.get_number_of_steps()
        num_steps = 10
        
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
        return atoms, time_per_step
        
    @staticmethod
    def save_results(original, optimized, structure_ids, file_path, header=None):
        
        if header == None:
            header = "strucure_id, original_pos, original_pos, original_pos, optimized_pos, optimized_pos, optimized_pos"
        for i in range(len(original)):
            with open(file_path, 'a') as f:
                if i == 0:
                    f.write(header + '\n')
                n = len(original[i]) if isinstance(original, list) else 1
                ids = np.array([structure_ids[i]] * n).reshape(-1, 1)
                data = np.hstack((ids, original[i], optimized[i]))
                np.savetxt(f, data, fmt='%s', delimiter=', ')
        
        
class MDSimulator:
    def __init__(self,
                 simulation_constant_type: str,
                 timestep: float,
                 temperature: float,
                 calculator: Calculator
                 ):
        self.simulation = simulation_constant_type
        self.timestep = timestep
        self.temperature = temperature
        self.calculator = calculator
        self.results = []

    def run_simulation(self,
                       atoms: Atoms,
                       num_steps: int,
                       log_console: int = 0,
                       save_energy: bool = False,
                       **kwargs):
        atoms.set_calculator(self.calculator)
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        
        common_valid_params = ['trajectory', 'logfile', 'loginterval', 'append_trajectory', 'temperature_K']
        valid_params = common_valid_params

        if self.simulation == 'NVE':
            valid_params.append('dt')
            simulation_class = VelocityVerlet
        elif self.simulation == 'NVT':
            valid_params += ['friction', 'fixcm', 'communicator', 'rng']
            kwargs.setdefault('friction', 1e-3)
            kwargs.setdefault('temperature_K', self.temperature)
            simulation_class = Langevin
        elif self.simulation == 'NPT':
            valid_params += ['externalstress', 'ttime', 'pfactor', 'mask']
            kwargs.setdefault('externalstress', 0.01623)
            kwargs.setdefault('pfactor', 0.6)
            kwargs.setdefault('temperature_K', self.temperature)
            simulation_class = NPT
        else:
            raise NotImplementedError("Currently unimplemented simulation type")

        invalid_params = [key for key in kwargs.keys() if key not in valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_params)}")

        dyn = simulation_class(atoms, timestep=self.timestep * units.fs, **kwargs)
        time_ps = 0

        if log_console:
            def printenergy(a=atoms):
                epot = a.get_potential_energy()
                ekin = a.get_kinetic_energy()
                temperature = 2 * ekin / (3 * len(a) * units.kB)
                if save_energy:
                    self.results.append({'time': time_ps, 'Etot': epot + ekin,
                                         'Epot': epot, 'Ekin': ekin, 'T': temperature})
                print('Energy of system: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                    'Etot = %.3feV' % (epot, ekin, temperature, epot + ekin))
                time_ps += 0.005
            dyn.attach(printenergy, interval=log_console)

        start = time()
        dyn.run(steps=num_steps)
        end = time()
        print(f"Time: {end-start:.4f}")

def traj_to_xdatcar(traj_file):
    traj = read(traj_file, index=':')
    file_name, _ = os.path.splitext(traj_file)
    write(file_name + '.XDATCAR', traj)