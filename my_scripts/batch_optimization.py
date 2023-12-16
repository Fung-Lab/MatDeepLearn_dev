import logging
import json
import yaml
from tqdm import tqdm
from time import time

import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from matdeeplearn.models import BaseModel
from matdeeplearn.common.registry import registry


logging.basicConfig(level=logging.INFO)


def load_structures(file_path, num_structures=-1, device='cpu'):
    logging.info("Reading JSON file for structures.")

    f = open(file_path)
    original_structures = json.load(f)
    f.close()

    n_structures = len(original_structures) if num_structures == -1 else num_structures
    data_list = [Data() for _ in range(n_structures)]
    
    original_energy, target_energy = [], []

    for i, s in enumerate(tqdm(original_structures[:n_structures])):
        data = data_list[i]

        pos = torch.tensor(s["unrelaxed_positions"], device=device, dtype=torch.float)
        cell = torch.tensor(s["unrelaxed_cell"], device=device, dtype=torch.float).unsqueeze(0)
        atomic_numbers = torch.LongTensor(s["atomic_numbers"])
        structure_id = s["structure_id"]
                
        data.n_atoms = len(atomic_numbers)
        data.pos = pos
        data.cell = cell   
        data.structure_id = [structure_id]  
        data.z = atomic_numbers
        data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
        
        original_energy.append(s['unrelaxed_energy'])
        target_energy.append(s['relaxed_energy'])

    return data_list, original_energy, target_energy

@staticmethod
def load_model(config: dict, rank: str) -> BaseModel:
    """
    This static method loads a machine learning model based on the provided configuration.

    Parameters:
    - config (dict): Configuration dictionary containing model and dataset parameters.
    - rank: Rank information for distributed training.

    Returns:
    - model: Loaded model for further calculations.

    Raises:
    - ValueError: If the 'checkpoint.pt' file is not found, the method issues a warning and uses an untrained model for prediction.
    """
    if isinstance(config, str):
        with open(config, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
    
    graph_config = config['dataset']['preprocess_params']
    model_config = config['model']
    node_dim = graph_config["node_dim"]
    edge_dim = graph_config["edge_dim"]   

    model_name = 'matdeeplearn.models.' + model_config["name"]
    logging.info(f'Setting up {model_name} for calculation')
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
    
    try:
        checkpoint = torch.load(config['task']["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"])
        logging.info(f'Model weights loaded from {config["task"]["checkpoint_path"]}')
    except ValueError:
        logging.warning("No checkpoint.pt file is found, and an untrained model is used for prediction.")

    return model


def optimize(file_path, config, steps, log_per, learning_rate, num_structures=-1, batch_size=4, device='cpu'):
    data_list, original_energy, target_energy = \
        load_structures(file_path, num_structures=num_structures, device=device)
    model = load_model(config, rank=device)
    
    original_energy_per_batch = [sum(original_energy[i:i+batch_size]) for i in range(0, len(data_list), batch_size)]
    target_energy_per_batch = [sum(target_energy[i:i+batch_size]) for i in range(0, len(data_list), batch_size)]
    
    # Created a data list
    loader = DataLoader(data_list, batch_size=batch_size)
    loader_iter = iter(loader)
    for i in range(len(loader_iter)):
        batch = next(loader_iter).to(device)
        pos, cell = batch.pos, batch.cell
        
        opt = torch.optim.LBFGS([pos, cell], lr=learning_rate)

        pos.requires_grad_(True)
        cell.requires_grad_(True)

        def closure(step):
            opt.zero_grad()
            energy = model(batch.to(device))['output'].sum()
            energy.backward(retain_graph=True)
            if log_per > 0 and step[0] % log_per == 0:
                print("{0:4d}   {1: 3.6f}".format(step[0], energy.item()))
            step[0] += 1
            batch.pos, batch.cell = pos, cell
            return energy
        
        print(f"Batch {i}:\nOriginal Energy: {original_energy_per_batch[i]:.4f}\
              \nTarget Energy: {target_energy_per_batch[i]:.4f}")
        
        step = [0]
        for _ in range(steps):
            opt.step(lambda: closure(step))
            
        
if __name__ == '__main__':
    logging.info("Batch original/target energies are ground truth energies.")
    
    # Configurations
    data_path = './data/optimization_data/data.json'
    model_config = './configs/calculator/config_torchmd.yml'
    steps = 25
    log_per = 50
    num_structures = 50
    batch_size = 8
    learning_rate = 0.05
    device = 'cuda:0'
    
    start = time()
    optimize(data_path,
             model_config,
             steps=steps,
             log_per=log_per,
             learning_rate=learning_rate,
             num_structures=num_structures,
             batch_size=batch_size,
             device=device)
    logging.info(f"Optimize {num_structures} structures took {time() - start:.4} seconds.")