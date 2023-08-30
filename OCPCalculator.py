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


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, config):
        """
        Initialize the OCPCalculator instance.

        Args:
            config (str or dict): Configuration settings for the OCPCalculator.

        Raises:
            AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        """
        Calculator.__init__(self)
        if isinstance(config, str):
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
        
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
        
        self.cutoff_radius = config['dataset']['preprocess_params'].get('cutoff_radius', 8.)
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)
        self.offset_number = config['dataset']['preprocess_params'].get('num_offsets', 2)
        self.edge_steps = config['dataset']['preprocess_params'].get('edge_steps', 50)
        self.otf_edge = config['model'].get('otf_edge', False) 
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

        if self.otf_edge:
            data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
                z=atomic_numbers, structure_id=atoms.info.get('structure_id', None))
        else:
            # Generate edges
            edge_gen_out = calculate_edges_master(
                method='ocp',
                all_neighbors=True,
                r=self.cutoff_radius,
                n_neighbors=self.n_neighbors,
                offset_number=self.offset_number,
                structure_id=atoms.structure_id,
                cell=cell,
                pos=pos,
                z=atomic_numbers,
            )
    
            edge_indices = edge_gen_out["edge_index"]
            edge_weights = edge_gen_out["edge_weights"]
            cell_offsets = edge_gen_out["cell_offsets"]
            edge_vec = edge_gen_out["edge_vec"]
            neighbors = edge_gen_out["neighbors"]
            if edge_vec.dim() > 2:
                edge_vec = edge_vec[edge_indices[0], edge_indices[1]]
    
            data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
                        z=atomic_numbers, structure_id=atoms.structure_id,
                        edge_index=edge_indices, edge_weight=edge_weights,
                        edge_vec=edge_vec, cell_offsets=cell_offsets,
                        neighbors=neighbors, edge_descriptor={"distance": edge_weights},
                        distances=edge_weights)
            generate_edge_features(data, self.edge_steps, self.cutoff_radius, device=self.device)

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


if __name__ == '__main__':
    config_path = './configs/config.yml'
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    c = OCPCalculator(config_path)

    dta = torch.load('./data/data.pt')
    
    atoms_list = OCPCalculator.data_to_atoms_list(data=dta[0])
    
    for i in range(len(atoms_list)):
        print('Label:', dta[0].y[i])
        c.calculate(atoms_list[i])
        print(c.results)

