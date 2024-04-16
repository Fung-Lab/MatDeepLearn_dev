from typing import List, Tuple
import logging
import os
from time import time
import yaml

import numpy as np
from scipy.stats import norm

from ase import Atoms, units
from ase import units
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet  import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.optimization.neighbors import find_points_in_spheres

import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import generate_node_features
from matdeeplearn.models.base_model import BaseModel



logging.basicConfig(level=logging.INFO)


class MDLCalculator(Calculator):
    """
    A neural networked based Calculator that calculates the energy, forces and stress of a crystal structure.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, config, rank='cuda:0'):
        """
        Initialize the MDLCalculator instance.

        Args:
        - config (str or dict): Configuration settings for the MDLCalculator.
        - rank (str): Rank of device the calculator calculates properties. Defaults to 'cuda:0'
        - config (str or dict): Configuration settings for the MDLCalculator.
        - rank (str): Rank of device the calculator calculates properties. Defaults to 'cuda:0'

        Raises:
        - AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        - AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        """
        Calculator.__init__(self)
        
        
        if isinstance(config, str):
            logging.info(f'MDLCalculator instantiated from config: {config}')
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
        elif isinstance(config, dict):
            logging.info('MDLCalculator instantiated from a dictionary.')
        else:
            raise NotImplementedError('Unsupported config type.')
                
        gradient = config["model"].get("gradient", False)
        otf_edge_index = config["model"].get("otf_edge_index", False)
        otf_edge_attr = config["model"].get("otf_edge_attr", False)
        self.otf_node_attr = config["model"].get("otf_node_attr", False)
        assert otf_edge_index and otf_edge_attr and gradient, "To use this calculator to calculate forces and stress, you should set otf_edge_index, oft_edge_attr and gradient to True."
        
        self.device = rank if torch.cuda.is_available() else 'cpu'
        self.models = MDLCalculator._load_model(config, self.device)
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)

    def calculate(self, atoms: Atoms, properties=implemented_properties, system_changes=None) -> None:
        """
        Calculate energy, forces, and stress for a given ase.Atoms object.

        Args:
        - atoms (ase.Atoms): The atomic structure for which calculations are to be performed.
        - properties (list): List of properties to calculate. Defaults to ['energy', 'forces', 'stress'].
        - system_changes: Not supported in the current implementation.
        - atoms (ase.Atoms): The atomic structure for which calculations are to be performed.
        - properties (list): List of properties to calculate. Defaults to ['energy', 'forces', 'stress'].
        - system_changes: Not supported in the current implementation.

        Returns:
        - None: The results are stored in the instance variable 'self.results'.
        - None: The results are stored in the instance variable 'self.results'.

        Note:
        - This method performs energy, forces, and stress calculations using a neural network-based calculator.
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
        loader_iter = iter(loader)
        batch = next(loader_iter).to(self.device)
        
        out_list = []
        for model in self.models:   
            out_list.append(model(batch))

        energy = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)
        forces = torch.stack([entry["pos_grad"] for entry in out_list]).mean(dim=0)
        stresses = torch.stack([entry["cell_grad"] for entry in out_list]).mean(dim=0)
        
        self.results['energy'] = energy.detach().cpu().numpy().squeeze()
        self.results['forces'] = forces.detach().cpu().numpy().squeeze()
        self.results['stress'] = stresses.squeeze().detach().cpu().numpy().squeeze()
        
    @staticmethod
    def data_to_atoms_list(data: Data) -> List[Atoms]:
        """
        This helper method takes a 'torch_geometric.data.Data' object containing information about atomic structures
        and converts it into a list of 'ase.Atoms' objects. Each 'Atoms' object represents an atomic structure
        with its associated properties such as positions and cell.
        
        Args:
        - data (Data): A data object containing information about atomic structures.
            
        Returns:
        - List[Atoms]: A list of 'ase.Atoms' objects, each representing an atomic structure
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
    
    @staticmethod
    def _load_model(config: dict, rank: str) -> List[BaseModel]:
        """
        This static method loads a model based on the provided configuration.

        Parameters:
        - config (dict): Configuration dictionary containing model and dataset parameters.
        - rank: Rank information for distributed training.

        Returns:
        - model_list: A list of loaded models.
        """
        
        graph_config = config['dataset']['preprocess_params']
        model_config = config['model']
        
        model_list = []
        model_name = model_config["name"]
        logging.info(f'MDLCalculator: setting up {model_name} for calculation')
        # Obtain node, edge, and output dimensions for model initialization   
        for _ in range(model_config["model_ensemble"]): 
            node_dim = graph_config["node_dim"]
            edge_dim = graph_config["edge_dim"]   

            model_cls = registry.get_model_class(model_name)
            model = model_cls(
                    node_dim=node_dim, 
                    edge_dim=edge_dim, 
                    output_dim=1, 
                    cutoff_radius=graph_config["cutoff_radius"], 
                    n_neighbors=graph_config["n_neighbors"], 
                    graph_method=graph_config["edge_calc_method"], 
                    num_offsets=graph_config["num_offsets"], 
                    **model_config
                    )
            model = model.to(rank)
            model_list.append(model)
        
        checkpoints = config['task']["checkpoint_path"].split(',')
        if len(checkpoints) == 0:
            logging.warning("MDLCalculator: No checkpoint.pt file is found, and untrained models are used for prediction.")
        else:
            for i in range(len(checkpoints)):
                try:
                    checkpoint = torch.load(checkpoints[i])
                    model_list[i].load_state_dict(checkpoint["state_dict"])
                    logging.info(f'MDLCalculator: weights for model No.{i+1} loaded from {checkpoints[i]}')
                except ValueError:
                    logging.warning(f"MDLCalculator: No checkpoint.pt file is found for model No.{i+1}, and an untrained model is used for prediction.")

        return model_list
    
            
class StructureOptimizer:
    """
    This class provides functionality to optimize the structure of an Atoms object using a specified calculator.
    """
    def __init__(self,
                 calculator,
                 relax_cell: bool = False,
                 ):
        """
        Initialize the StructureOptimizer.

        Parameters:
        - calculator (Calculator): A calculator object for performing energy and force calculations.
        - relax_cell (bool): If True, the cell will be relaxed in addition to positions during the optimization process.
        """
        self.calculator = calculator
        self.relax_cell = relax_cell
        
    def optimize(self, atoms: Atoms, logfile=None, write_traj_name=None) -> Tuple[Atoms, float]:
        """
        This method optimizes the structure of the given Atoms object using the specified calculator.
        If `relax_cell` is True, the cell will be relaxed. Trajectory information can be written to a file.

        Parameters:
        - atoms: An Atoms object representing the structure to be optimized.
        - logfile: File to write optimization log information.
        - write_traj_name: File to write trajectory of the optimization.

        Returns:
        - atoms: The optimized Atoms object.
        - time_per_step: The average time taken per optimization step.
        """
        atoms.calc = self.calculator       
        if self.relax_cell:
            atoms = ExpCellFilter(atoms)

        optimizer = FIRE(atoms, logfile=logfile)
        
        if write_traj_name is not None:
            traj = Trajectory(write_traj_name + '.traj', 'w', atoms)
            optimizer.attach(traj.write, interval=1)

        start_time = time()
        optimizer.run(fmax=0.001, steps=500)
        end_time = time()
        num_steps = optimizer.get_number_of_steps()
        
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
        return atoms, time_per_step
        
    @staticmethod
    def save_results(original, optimized, structure_ids, file_path, header=None) -> None:
        """
        This static method saves the original and optimized atomic positions along with structure IDs to a file.

        Parameters:
        - original: List of original atomic positions or a single set of positions.
        - optimized: List of optimized atomic positions or a single set of positions.
        - structure_ids: List of structure IDs corresponding to each set of positions.
        - file_path: Path to the file where results will be saved.
        - header: Optional header for the saved file.
        """
        if header == None:
            header = "structure_id, original_pos, original_pos, original_pos, optimized_pos, optimized_pos, optimized_pos"
        for i in range(len(original)):
            with open(file_path, 'a') as f:
                if i == 0:
                    f.write(header + '\n')
                n = len(original[i]) if isinstance(original, list) else 1
                ids = np.array([structure_ids[i]] * n).reshape(-1, 1)
                data = np.hstack((ids, original[i], optimized[i]))
                np.savetxt(f, data, fmt='%s', delimiter=', ')
        
        
class MDSimulator:
    """
    This class provides a simple interface for running molecular dynamics simulations.
    It currently supports microcanonical ('NVE'), canonical ('NVT'), or isothermal-isobaric ('NPT') simulations.
    """
    def __init__(self,
                 simulation_constant_type: str,
                 timestep: float,
                 temperature: float,
                 calculator: Calculator
                 ):
        """
        Initialize a Molecular Dynamics Simulator.

        Parameters:
        - simulation_constant_type (str): Type of molecular dynamics simulation ('NVE', 'NVT', 'NPT').
        - timestep (float): Time step for the simulation in femtoseconds.
        - temperature (float): Initial temperature for the simulation in Kelvin.
        - calculator (Calculator): A calculator object for energy and force calculations.
        """
        self.simulation = simulation_constant_type
        self.timestep = timestep
        self.temperature = temperature
        self.calculator = calculator
        
    @staticmethod
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

    def run_simulation(self,
                       atoms: Atoms,
                       num_steps: int,
                       log_console: int = 0,
                       save_energy: bool = False,
                       **kwargs) -> None:
        """
        This method runs a molecular dynamics simulation for the specified number of steps using the chosen ensemble.
        It currently supports the 'NVE', 'NVT', and 'NPT' ensembles. Energy information can be logged to the console and saved.

        Parameters:
        - atoms (Atoms): An Atoms object representing the molecular system.
        - num_steps (int): Number of simulation steps to run.
        - log_console (int): Interval for logging energy information to the console (0 for no logging).
        - save_energy (bool): If True, save energy information during the simulation.
        - **kwargs: Additional parameters specific to the chosen simulation ensemble.
        """
        atoms.set_calculator(self.calculator)
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        
        common_valid_params = ['trajectory', 'logfile', 'loginterval', 'append_trajectory', 'temperature_K']
        valid_params = common_valid_params

        if self.simulation == 'NVE':
            valid_params.append('dt')
            simulation_class = VelocityVerlet
        elif self.simulation == 'NVT':
            valid_params += ['friction', 'fixcm', 'communicator', 'rng']
            kwargs.setdefault('friction', 0.001 / units.fs)
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

        starting = atoms.get_potential_energy()
        starting_rdf = MDSimulator.get_rdf(atoms)
        highest_e = starting
        lowest_e = starting
        
        def calc_energy(a=atoms):
            # nonlocal highest_e, lowest_e
            epot = a.get_potential_energy()
            ekin = a.get_kinetic_energy()
            etotal = epot + ekin
            return etotal
            # highest_e = max(highest_e, etotal) 
            # lowest_e = min(lowest_e, etotal)
        # dyn.attach(calc_energy, interval=1)
        
        def get_min_interatomic_distance(a=atoms):
            all_dist = a.get_all_distances()
            np.fill_diagonal(all_dist, np.inf)
            return all_dist.min().min()
        
        # if log_console:
        #     def printenergy(a=atoms):
        #         epot = a.get_potential_energy()
        #         ekin = a.get_kinetic_energy()
        #         temperature = 2 * ekin / (3 * len(a) * units.kB)
        #         etotal = epot + ekin
            
                # if save_energy:
                #     self.results.append({'time': time_ps, 'Etot': epot + ekin,
                #                          'Epot': epot, 'Ekin': ekin, 'T': temperature})
            #     logging.info('Energy of system: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            #         'Etot = %.3feV' % (epot, ekin, temperature, etotal))
            # dyn.attach(printenergy, interval=log_console)

        start = time()
        prev_energy = starting
        
        energy_changes, rdf_changes, min_dists = [], [], []
        
        for i in range(num_steps):
            
            e_total = calc_energy()
            current_rdf = MDSimulator.get_rdf(atoms)
            min_dist = get_min_interatomic_distance()
            rdf_change = np.mean(np.abs(current_rdf - starting_rdf))
            # if np.abs(e_total - prev_energy) > 200:
            #     logging.info(f"Step {i}: energy fluctuation is {np.abs(e_total - prev_energy):.4f}")
            #     logging.info(f"RDF fluctuation at step {i} is {rdf_change:.4f}")   
            #     logging.info(f"Min interatomic distance: {get_min_interatomic_distance():.4f}")
            energy_changes.append(e_total)
            rdf_changes.append(rdf_change)
            min_dists.append(min_dist)
            prev_energy = e_total
            starting_rdf = current_rdf
            dyn.step()
        end = time()
        
        if log_console:
            logging.info(f"Time: {end-start:.4f}")
            
        return energy_changes, rdf_changes, min_dists
        # return atoms, (end - start) / num_steps, highest_e, lowest_e


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