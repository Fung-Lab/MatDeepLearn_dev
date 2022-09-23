import os

from .datasets import *
from .transforms import *
from .data_processor import DataProcessor

def get_dataset(
    data_path, 
    target_index,
    transform_type='GetY', 
    large_dataset=False, 
    reprocess=False
):
    '''
    Get dataset according from data_path

    Parameters
    ----------
        data_path: str

    Returns:
        dataset
    '''

    # REDO
    processed_data_path = 'processed'
    processed_data_fullpath = os.path.join(data_path, processed_data_path, 'data.pt')

    # set transform method
    # need a better lookup system
    if transform_type == 'GetY':
        T = GetY
    else:
        raise ValueError('No transform function/class found for {transform}')
    
    # check if large dataset is needed
    Dataset = StructureDataset
    if large_dataset:
        Dataset = LargeStructureDataset

    # check if data_path exists
    if not os.path.exists(data_path):
        raise ValueError('Invalid data path: {data_path}')

    # check if reprocessing is needed
    if reprocess:
        #TODO
        pass

    transform = T(index=target_index)

    if not os.path.exists(processed_data_fullpath):
        # process
        #TODO
        pass

    return Dataset(data_path, processed_data_path, transform)