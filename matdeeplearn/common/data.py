import gc
import json
import warnings
import numpy as np
from typing import List
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.datasets import LargeStructureDataset, StructureDataset


# train test split
def dataset_split(
    dataset,
    train_size: float = 0.8,
    valid_size: float = 0.05,
    test_size: float = 0.15,
):
    """
    Splits an input dataset into 3 subsets: train, validation, test.
    Requires train_size + valid_size + test_size = 1

    Parameters
    ----------
        dataset: matdeeplearn.preprocessor.datasets.StructureDataset
            a dataset object that contains the target data

        train_size: float
            a float between 0.0 and 1.0 that represents the proportion
            of the dataset to use as the training set

        valid_size: float
            a float between 0.0 and 1.0 that represents the proportion
            of the dataset to use as the validation set

        test_size: float
            a float between 0.0 and 1.0 that represents the proportion
            of the dataset to use as the test set
    """
    
    if train_size + valid_size + test_size > 1:
        warnings.warn("Invalid sizes detected (ratios add up to larger than one). Using default split of 0.8/0.05/0.15.")
        train_size, valid_size, test_size = 0.8, 0.05, 0.15

    dataset_size = len(dataset)

    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = int(test_size * dataset_size)
    unused_len = dataset_size - train_len - valid_len - test_len

    (train_dataset, val_dataset, test_dataset, unused_dataset) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
    )

    return train_dataset, val_dataset, test_dataset

def get_otf_transforms(transform_list: List[dict]):
    """
    get on the fly specific transforms

    Parameters
    ----------

    transform_list: transformation function/classes to be applied
    """

    transforms = []
    # set transform method
    for transform in transform_list:
        if transform.get("otf", False):
            transforms.append(
                registry.get_transform_class(
                    transform["name"],
                    **transform.get("args", {})
                )
            )
            
    return transforms

def get_dataset(
    data_path,
    processed_file_name,
    transform_list: List[dict] = [],
    large_dataset=False,
):
    """
    get dataset according to data_path
    this assumes that the data has already been processed and
    data.pt file exists in data_path/processed/ folder

    Parameters
    ----------

    data_path: str
        path to the folder containing data.pt file

    transform_list: transformation function/classes to be applied
    """

    # get on the fly transforms for use on dataset access
    otf_transforms = get_otf_transforms(transform_list)

    # check if large dataset is needed
    if large_dataset:
        Dataset = LargeStructureDataset
    else:
        Dataset = StructureDataset

    composition = Compose(otf_transforms) if len(otf_transforms) >= 1 else None
        
    dataset = Dataset(data_path, processed_data_path="", processed_file_name=processed_file_name, transform=composition)

    return dataset


def get_dataloader(
    dataset, batch_size: int, num_workers: int = 0, sampler=None, shuffle=True
):
    """
    Returns a single dataloader for a given dataset

    Parameters
    ----------
        dataset: matdeeplearn.preprocessor.datasets.StructureDataset
            a dataset object that contains the target data

        batch_size: int
            size of each batch

        num_workers: int
            how many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
    """

    # load data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is not None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return loader

# ported from MLC project
class PrepareData(Dataset):
    def __init__(
        self, 
        species, 
        positions, 
        list_y,
        list_natoms,
        list_cell,
        list_structure_id,
        std,
        loss_on
    ):
        self.species = species
        self.positions = positions
        self.list_y = list_y
        self.list_natoms = list_natoms
        self.list_cell = list_cell
        self.list_structure_id = list_structure_id
        self.std = std
        self.loss_on = loss_on
    
    def __getitem__(self, index):
        ori_pos = self.positions[index]
        atoms = self.species[index]
        targets = self.list_y[index]
        natoms = self.list_natoms[index]
        cells = self.list_cell[index]
        structure_id = self.list_structure_id[index]

        # add noise
        if self.loss_on == "noise":
            noise = np.random.normal(0, self.std, ori_pos.shape)
            pos = ori_pos + noise
        elif self.loss_on == "y":
            noise = np.random.normal(0, 0, ori_pos.shape)
            pos = ori_pos

        x = torch.tensor(atoms, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        noise = torch.tensor(noise, dtype=torch.float)
        y = torch.tensor(targets, dtype=torch.float)
        cells = torch.tensor(cells, dtype=torch.float)
        cells = cells.view(1, 3, 3)
        
        data = Data(x=x, pos=pos, y=y, noise=noise, cell=cells, n_atoms=natoms, structure_id=structure_id)
        return data

    def __len__(self):
        return len(self.positions)
    
class DataWrapper(object):
    def __init__(self, batch_size, num_workers, std, loss_on, paths=None, seed=1234, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.std = std
        self.loss_on = loss_on
        self.paths = paths
        self.seed = seed
        self.__dict__.update(kwargs)

    def get_dataset(self, path):
        f = open(path, 'r')
        data = json.load(f)
        f.close()

        list_atomic_numbers = []
        list_positions = []
        list_y = []
        list_natoms = []
        list_cell = []
        list_structure_id = []

        for i, d in enumerate(tqdm(data)):
            list_atomic_numbers.append(d['atomic_numbers'])
            list_positions.append(np.array(d['positions']))

            if isinstance(d['y'], list):
                list_y.append(d['y'])
            else:
                list_y.append([d['y']])
            list_natoms.append(len(d['atomic_numbers']))
            list_cell.append(d['cell'])
            list_structure_id.append(d['structure_id'])

        dataset = PrepareData(
            list_atomic_numbers, list_positions, list_y, list_natoms, list_cell, list_structure_id, self.std, self.loss_on
        )

        return dataset

    def get_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=True, pin_memory=True,
        )

        gc.collect()
        return dataloader
        
    def get_all_loaders(self):
        if isinstance(self.paths, str):
            dataset = self.get_dataset(self.paths)
            if not hasattr(self, "split_ratio"):
                train_size = 0.8
                valid_size = 0.1
                test_size = 0.1
            else:
                train_size = self.split_ratio['train']
                valid_size = self.split_ratio['val']
                test_size = self.split_ratio['test']
            
            train_size = int(train_size * len(dataset))
            valid_size = int(valid_size * len(dataset))
            test_size = len(dataset) - train_size - valid_size

            if self.seed is not None:
                torch.manual_seed(self.seed)

            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size, test_size]
            )

        elif isinstance(self.paths, dict):
            train_dataset = self.get_dataset(self.paths['train'])
            valid_dataset = self.get_dataset(self.paths['val'])
            test_dataset = self.get_dataset(self.paths['test'])

        train_dataloader = self.get_dataloader(train_dataset)
        valid_dataloader = self.get_dataloader(valid_dataset)
        test_dataloader = self.get_dataloader(test_dataset)

        return (
            train_dataset, valid_dataset, test_dataset,
            train_dataloader, valid_dataloader, test_dataloader
        )