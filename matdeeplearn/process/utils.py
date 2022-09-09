import os

from .datasets import StructureDataset, LargeStructureDataset
from .transforms import *

def get_dataset(
    data_path, 
    target_index: int = 0, 
    transform_type='GetY',
    large_dataset=False
):
    '''
    get dataset according to data_path
    this assumes that the data has already been processed and
    data.pt file exists in data_path/processed/ folder

    Parameters
    ----------
    
    data_path: str
        path to the root folder of the dataset
        contains subfolders processed/ and raw/

    target_index: int
        the index to select the target values
        this is needed because in our target.csv, there might be
        multiple columns of target values available for that
        particular dataset, thus we need to index one column for
        the current run/experiment

    transform_type: transformation function/class to be applied
    '''
    
    # set transform method
    if transform_type == 'GetY':
        T = GetY
    else:
        raise ValueError('No such transform found for {transform}')

    # check if large dataset is needed
    if large_dataset:
        Dataset = LargeStructureDataset
    else:
        Dataset = StructureDataset

    transform = T(index=target_index)

    return Dataset(data_path, 'processed', transform)