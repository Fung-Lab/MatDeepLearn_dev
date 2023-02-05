import warnings

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from matdeeplearn.preprocessor.datasets import LargeStructureDataset, StructureDataset
from matdeeplearn.preprocessor.transforms import TRANSFORM_REGISTRY, GetY


# train test split
def dataset_split(
    dataset,
    train_size: float = 0.8,
    valid_size: float = 0.05,
    test_size: float = 0.15,
    seed: int = 1234,
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
    if train_size + valid_size + test_size != 1:
        warnings.warn("Invalid sizes detected. Using default split of 80/5/15.")
        train_size, valid_size, test_size = 0.8, 0.05, 0.15

    dataset_size = len(dataset)

    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = int(test_size * dataset_size)
    unused_len = dataset_size - train_len - valid_len - test_len

    (train_dataset, val_dataset, test_dataset, unused_dataset) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataset(
    data_path, target_index: int = 0, transform_list=[], otf=False, large_dataset=False
):
    """
    get dataset according to data_path
    this assumes that the data has already been processed and
    data.pt file exists in data_path/processed/ folder

    Parameters
    ----------

    data_path: str
        path to the folder containing data.pt file

    target_index: int
        the index to select the target values
        this is needed because in our target.csv, there might be
        multiple columns of target values available for that
        particular dataset, thus we need to index one column for
        the current run/experiment

    transform_list: transformation function/classes to be applied
    """

    transforms = [GetY(index=target_index)]

    # set transform method
    if otf:
        for transform in transform_list:
            if transform in TRANSFORM_REGISTRY:
                transforms.append(TRANSFORM_REGISTRY[transform]())
            else:
                raise ValueError("No such transform found for {transform}")

    # check if large dataset is needed
    if large_dataset:
        Dataset = LargeStructureDataset
    else:
        Dataset = StructureDataset

    transform = Compose(transforms)
    
    return Dataset(data_path, processed_data_path="", transform=transform)


def get_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    sampler=None,
    shuffle=True
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
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
    )

    return loader
