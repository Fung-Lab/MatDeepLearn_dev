import copy
import itertools
import os
import time

import torch
from torch_geometric.data import InMemoryDataset

from matdeeplearn.preprocessor.helpers import get_pbc_cells


class GeoSSLPretrainDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        processed_data_path,
        processed_file_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        device=None,
        num_offsets=2,
        mu=0,
        sigma=0.5
    ):
        self.root = root
        self.processed_data_path = processed_data_path
        self.processed_file_name = processed_file_name
        self.num_offsets = num_offsets
        self.mu, self.sigma = mu, sigma
        super(GeoSSLPretrainDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if not torch.cuda.is_available() or device == "cpu":
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location=torch.device("cpu")
            )
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

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

    def __getitem__(self, idx):
        data1 = super().__getitem__(idx)
        super_edge_index = list(itertools.combinations(range(len(data1.x)), 2))
        offsets, cell_coors = get_pbc_cells(data1.cell, self.num_offsets)
        offsets = offsets.expand(len(data1.pos), offsets.shape[1], offsets.shape[2])

        data1.super_edge_index = torch.tensor(super_edge_index).t().contiguous()
        data1.offsets = offsets

        data2 = super().__getitem__(idx)
        data2.super_edge_index = torch.tensor(super_edge_index).t().contiguous()
        data2.offsets = offsets
        data2.pos = data2.pos + torch.normal(self.mu, self.sigma, size=data2.pos.size())
        return data1, data2


if __name__ == '__main__':
    dataset = GeoSSLPretrainDataset(root="data/ct_pretrain/processed/",
                                    processed_data_path="",
                                    processed_file_name="data.pt",)
    # print(dataset.num_features   , dataset.num_edge_features)

    # print(dataset[10][0].cell, dataset[10][0].cell_offsets)
    print(dataset[10][1])

    # print(len(dataset))
    # loader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=2,
    #     sampler=None,
    # )
    # print(len(loader))
    # train_loader_iter = iter(loader)
    # batch1, batch2 = next(loader)
    # print(batch1)
    # print(batch2, "\n")