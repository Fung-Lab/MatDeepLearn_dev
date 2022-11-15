import torch, os

from torch_geometric.data import InMemoryDataset

class StructureDataset(InMemoryDataset):
    def __init__(
        self,
        root, 
        processed_data_path, 
        transform=None, 
        pre_transform=None, 
        pre_filter=None,
        device=None
    ):
        self.root = root
        self.processed_data_path = processed_data_path
        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if device is None:
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except:
                self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device('cpu'))
        else:
            if device == 'cpu':
                self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device(device))
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        '''
        The name of the files in the self.raw_dir folder 
        that must be present in order to skip downloading.
        '''
        return []

    def download(self):
        '''
        Download required data files; to be implemented
        '''
        pass

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_data_path)

    @property
    def processed_file_names(self):
        '''
        The name of the files in the self.processed_dir 
        folder that must be present in order to skip processing.
        '''
        return ["data.pt"]

class LargeStructureDataset(InMemoryDataset):
    pass