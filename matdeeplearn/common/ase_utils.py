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
from ase.md import MDLogger
from pathlib import Path
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory


logging.basicConfig(level=logging.INFO)


class MDLCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

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
    
        

class SimulationLogger(MDLogger):
    """
    A custom logger for recording simulation data during molecular dynamics simulations using the ASE library.

    Parameters:
    - args: Positional arguments passed to the parent class `MDLogger`.
    - start_time (float): The starting time of the simulation (default is 0).
    - verbose (bool): Whether to display log messages (default is True).
    - fmt (str): The format string for logging simulation data (default is '%-10.4f %12.4f %12.4f %12.4f  %6.1f').
    - kwargs: Keyword arguments passed to the parent class `MDLogger`.

    Keyword Arguments (kwargs):
    - header (bool): Whether to include a header in the log file.
    - peratom (bool): Whether to show energy per atom.
    - stress (bool): Whether to log stress values.
    - interval (int): The interval for logging data.

    Usage:
    1. Create an instance of SimulationLogger and attach it to your ASE dynamics object.
    2. As the simulation progresses, it will log data to console and to the logfile according to the specified format.
    """
    def __init__(self, *args, start_time=0, verbose=True, fmt='%-10.4f %12.4f %12.4f %12.4f  %6.1f\n', **kwargs):
        header = start_time == 0
        super().__init__(header=header, *args, **kwargs)
        self.start_time = start_time
        self.verbose = verbose
        self.fmt = fmt
        self.n_atoms = len(self.atoms)
        
        if verbose:
            logging.info(self.hdr)
    
    def __call__(self):
        t = self.dyn.get_time() / (1000 * units.fs) + self.start_time
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.n_atoms)
        
        if self.peratom:
            epot /= self.n_atoms
            ekin /= self.n_atoms

        dat = (t, epot + ekin, epot, ekin, temp)
        
        if self.stress:
            stress_values = tuple(self.atoms.get_stress() / units.GPa)
            dat += stress_values
        
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            logging.info(self.fmt % dat)


class Simulator:
    def __init__(self, 
                 atoms,
                 calculator,
                 dyn,
                 initial_temperature,
                 start_time=0,
                 save_dir='./log',
                 restart=False,
                 save_frequency=100,
                 min_temp=0.1,
                 max_temp=100000):
        self.atoms = atoms
        atoms.calc = calculator
        self.dyn = dyn
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_atoms = self.atoms.get_global_number_of_atoms()

        # intialize system momentum 
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, initial_temperature * units.kB)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.save_dir / 'atoms.traj', 'a', self.atoms)
        self.dyn.attach(self.traj.write, interval=save_frequency)
        
        # attach log file
        self.dyn.attach(SimulationLogger(self.dyn, self.atoms, 
                                        self.save_dir / 'thermo.log', 
                                        start_time=start_time), interval=save_frequency)
        
    def run(self, steps):
        early_stop = False
        step = 0
        for step in range(steps):
            self.dyn.run(1)
            ekin = self.atoms.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB * self.n_atoms)
            if temp < self.min_temp or temp > self.max_temp:
                logging.warning(f'Temprature {temp:.2f} is out of range: [{self.min_temp:.2f}, {self.max_temp:.2f}]. Early stopping the simulation.')
                early_stop = True
                break
            
        self.traj.close()
        return early_stop, (step + 1)
        