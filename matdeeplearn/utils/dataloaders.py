from concurrent.futures import process
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform

##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

def base_loader(
    dataset,
    rank,
    world_size,
    train_test_split: float = 0.8,
    val_test_split: float = 0.25,
    batch_size: int = 100,
    seed: int = 0
):
    # train_dataset, val_dataset, test_dataset = 

    # distributed data parallel
    if rank not in ['cpu', 'cuda']:
        pass