import logging
import time
import os

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from matdeeplearn.common.data import get_dataloader
from matdeeplearn.common.registry import registry



@registry.register_trainer("preprocess")
class TeacherPreprocessTrainer():
    def __init__(self, teacher_model):
        self.teacher_model = teacher_model
        self.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_config(cls, config):
        config['teacher_model']['use_distill'] = True
        config['teacher_model']['id_mapping'] = True

        if config["task"]["parallel"] == True:
            # os.environ["MASTER_ADDR"] = "localhost"
            # os.environ["MASTER_PORT"] = "12355"
            local_world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
            local_world_size = int(local_world_size)
            dist.init_process_group(
                "nccl", world_size=local_world_size, init_method="env://"
            )
            rank = int(dist.get_rank())
        else:
            rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_world_size = 1

        # set up teacher model and mapping funtion
        teacher_model = cls._load_teacher_model(config["teacher_model"], config["dataset"]["preprocess_params"], local_world_size, rank)
        teacher_model.setup_distillation()
        teacher_model.to(rank)

        return cls(teacher_model=teacher_model)

    @staticmethod
    def _load_teacher_model(model_config, graph_config, world_size, rank):
        """Loads teacher model from a config file."""
        
        node_dim = graph_config["node_dim"]
        edge_dim = graph_config["edge_dim"] 
        output_dim = 1
        model_config["prediction_level"] = "graph"

        model_cls = registry.get_model_class(model_config["name"])
        model = model_cls(
                  node_dim=node_dim, 
                  edge_dim=edge_dim, 
                  output_dim=output_dim, 
                  cutoff_radius=graph_config["cutoff_radius"], 
                  n_neighbors=graph_config["n_neighbors"], 
                  graph_method=graph_config["edge_calc_method"], 
                  num_offsets=graph_config["num_offsets"], 
                  is_teacher=True,
                  **model_config
                  )
        model = model.to(rank)
        if world_size > 1:
            model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
            )
        
        if not model_config['teacher_checkpoint_path']:
            raise ValueError("No checkpoint directory specified in teacher config.")

        # Load teacher model parameters
        checkpoint = torch.load(model_config['teacher_checkpoint_path'])
        model.load_state_dict(checkpoint["state_dict"])

        # Freeze teacher parameters
        for param in model.parameters():
            param.requires_grad = False
    
        return model

    def get_embedding(self, loader):
        self.teacher_model.eval()
        loader_iter = iter(loader)
        embeddings = []
        with torch.no_grad(): 
            for i in range(0, len(loader_iter)):
                batch = next(loader_iter).to(self.rank)
                out = self._forward(batch)
                out_cpu = {key: [tensor.cpu() for tensor in value] for key, value in out.items()}
                embeddings.append(out_cpu) 
                del batch
            torch.cuda.empty_cache()

        return embeddings
        
    def _forward(self, batch_data):
        output = self.teacher_model.extract_feature(batch_data)
        return output
