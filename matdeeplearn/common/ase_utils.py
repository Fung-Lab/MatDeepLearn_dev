import torch
import numpy as np
import yaml
from ase import Atoms
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
from matdeeplearn.preprocessor.helpers import generate_node_features
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
import logging
from typing import List
from matdeeplearn.common.registry import registry


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
        #model_name = 'matdeeplearn.models.' + model_config["name"]
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
