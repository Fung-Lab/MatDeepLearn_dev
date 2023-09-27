import os

import torch
from torch_geometric.data import InMemoryDataset


class StructureDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        processed_data_path,
        processed_file_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        device=None,
    ):
        self.root = root
        self.processed_data_path = processed_data_path
        self.processed_file_name = processed_file_name
        super(StructureDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        if not torch.cuda.is_available() or device == "cpu":
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location=torch.device("cpu")
            )
        else:
            self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)

    @property
    def raw_file_names(self):
        """
        The name of the files in the self.raw_dir folder
        that must be present in order to skip downloading.
        """
        return []

    def download(self):
        """
        Download required data files; to be implemented
        """
        pass

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_data_path)

    @property
    def processed_file_names(self):
        """
        The name of the files in the self.processed_dir
        folder that must be present in order to skip processing.
        """
        return [self.processed_file_name]


class LargeStructureDataset(InMemoryDataset):
    pass
