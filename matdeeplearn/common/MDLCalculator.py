import ase
import torch
import yaml
from ase import Atoms
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
from matdeeplearn.preprocessor.helpers import (
    clean_up,
    calculate_edges_master,
    generate_node_features,
    generate_edge_features
)
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
import logging
import yaml
import os
from typing import List
from matdeeplearn.common.registry import registry


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
                
        otf_edge = config["model"].get("otf_edge", False)
        gradient = config["model"].get("gradient", False)
        assert otf_edge and gradient, "To use this calculator to calculate forces and stress, you must set otf_edge and gradient to True."
        
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
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)

        data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
            z=atomic_numbers, structure_id=atoms.info.get('structure_id', None))

        # Generate node features
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
        structure_ids = data.structure_id
        positions = data.pos.numpy()
        cells = data.cell.numpy()
    
        pos_idx = 0
        atoms_list = []
    
        for i in range(len(structure_ids)):
            n_atoms = data.n_atoms[i]
            positions_i = positions[pos_idx : pos_idx + n_atoms]
            pos_idx += n_atoms
            cell = Cell(cells[i])
            atoms = Atoms(positions=positions_i, cell=cell)
            atoms.structure_id = structure_ids[i]
            atoms_list.append(atoms)
        
        return atoms_list
