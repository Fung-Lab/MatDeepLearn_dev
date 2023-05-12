import os
import copy
import random
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader

from matdeeplearn.preprocessor import StructureDataset


class CTPretrainDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            processed_data_path,
            processed_file_name,
            mask_node_ratios=None,
            mask_edge_ratios=None,
            distance=0.05,
            min_distance: float = None,
            augmentation_list=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            device=None,
    ):
        if mask_edge_ratios is None:
            mask_edge_ratios = [0.1, 0.1]
        if mask_node_ratios is None:
            mask_node_ratios = [0.1, 0.25]

        self.root = root
        self.processed_data_path = processed_data_path
        self.processed_file_name = processed_file_name
        self.augmentation_list = augmentation_list if augmentation_list else []

        self.mask_node_ratio1 = mask_node_ratios[0]
        self.mask_node_ratio2 = mask_node_ratios[1]
        self.mask_edge_ratio1 = mask_edge_ratios[0]
        self.mask_edge_ratio2 = mask_edge_ratios[1]
        self.distance = distance
        self.min_distance = min_distance
        super(CTPretrainDataset, self).__init__(
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
        slices = self.slices
        data = self.data

        subdata = {}
        for key in data.keys:
            # get the slice corresponding to the m-th structure for the current key
            if key == 'edge_index':
                # print("1:", data[key][:][slices[key][idx]: slices[key][idx + 1]])
                subdata[key] = data[key][:, slices[key][idx]: slices[key][idx + 1]]
                # print(subdata[key])
            else:
                subdata[key] = data[key][slices[key][idx]: slices[key][idx + 1]]

        # create a new Data object using the subdata
        # print(subdata['structure_id'])
        subdata = Data.from_dict(subdata)

        subdata1 = copy.deepcopy(subdata)
        subdata2 = copy.deepcopy(subdata)

        def mask_node(mask_node_ratio1, mask_node_ratio2):
            x = subdata.x
            num_nodes = x.size(0)
            mask1 = torch.randperm(num_nodes) < int(num_nodes * mask_node_ratio1)
            mask2 = torch.randperm(num_nodes) < int(num_nodes * mask_node_ratio2)

            subdata1.x[mask1] = 0
            subdata2.x[mask2] = 0

        def mask_edge(mask_edge_ratio1, mask_edge_ratio2):
            edge_index, edge_attr = subdata.edge_index, subdata.edge_attr
            num_edges = edge_index.size(1)
            mask1 = torch.randperm(num_edges) < int(num_edges * mask_edge_ratio1)
            mask2 = torch.randperm(num_edges) < int(num_edges * mask_edge_ratio2)
            subdata1.edge_index = subdata1.edge_index[:, ~mask1]
            subdata1.edge_attr = subdata1.edge_attr[~mask1]
            subdata2.edge_index = subdata2.edge_index[:, ~mask2]
            subdata2.edge_attr = subdata2.edge_attr[~mask2]

        def perturb_data(distance: float = 0.05, min_distance: float = None):
            for i in range(subdata.pos.size(0)):
                # Perturb subdata1
                # Generate a random unit vector
                random_vector = torch.randn(3)
                random_vector /= torch.norm(random_vector)

                # Calculate the perturbation distance
                if min_distance is not None:
                    perturbation_distance = random.uniform(min_distance, distance)
                else:
                    perturbation_distance = distance

                # Perturb the node position
                subdata1.pos[i] += perturbation_distance * random_vector

                # Perturb subdata2
                # Generate a random unit vector
                random_vector = torch.randn(3)
                random_vector /= torch.norm(random_vector)

                # Calculate the perturbation distance
                if min_distance is not None:
                    perturbation_distance = random.uniform(min_distance, distance)
                else:
                    perturbation_distance = distance

                # Perturb the node position
                subdata2.pos[i] += perturbation_distance * random_vector

        if "node_masking" in self.augmentation_list:
            mask_node(self.mask_node_ratio1, self.mask_node_ratio2)
        if "edge_masking" in self.augmentation_list:
            mask_edge(self.mask_edge_ratio1, self.mask_edge_ratio2)
        if "perturbing" in self.augmentation_list:
            perturb_data(self.distance, self.min_distance)

        # Apply transforms
        if self.transform is not None:
            subdata1 = self.transform(subdata1)
            subdata2 = self.transform(subdata2)

        return subdata1, subdata2

    # def get(self, idx):
    #
    #     slices = self.slices
    #     data = self.data
    #
    #     subdata = {}
    #     for key in data.keys:
    #         # get the slice corresponding to the m-th structure for the current key
    #         if key == 'edge_index':
    #             # print("1:", data[key][:][slices[key][idx]: slices[key][idx + 1]])
    #             subdata[key] = data[key][:, slices[key][idx]: slices[key][idx + 1]]
    #             # print(subdata[key])
    #         else:
    #             subdata[key] = data[key][slices[key][idx]: slices[key][idx + 1]]
    #
    #     # create a new Data object using the subdata
    #     # print(subdata['structure_id'])
    #     subdata = Data.from_dict(subdata)
    #
    #     subdata1 = copy.deepcopy(subdata)
    #     subdata2 = copy.deepcopy(subdata)
    #
    #     # edge_index, edge_attr = subdata.edge_index, subdata.edge_attr
    #     x, y = subdata.x, subdata.y
    #
    #     # Perform node masking augmentation twice
    #     num_nodes = x.size(0)
    #     mask1 = torch.randperm(num_nodes) < int(num_nodes * 0.2)
    #     mask2 = torch.randperm(num_nodes) < int(num_nodes * 0.2)
    #
    #     subdata1.x[mask1] = 0
    #     subdata2.x[mask2] = 0
    #
    #     # Apply transforms
    #     if self.transform is not None:
    #         subdata1 = self.transform(subdata1)
    #         subdata2 = self.transform(subdata2)
    #
    #     return subdata1, subdata2


if __name__ == '__main__':
    # dataset = StructureDataset(root="../../data/test_data/processed/", processed_data_path="",
    #                             processed_file_name="data.pt")
    # print(dataset[0])
    dataset = CTPretrainDataset(root="../../data/test_data/processed/", processed_data_path="",
                                processed_file_name="data.pt")
    # print(dataset.num_edge_features)

    print(dataset.slices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=None,
    )
    print(len(loader))
    for batch1 in loader:
        print(batch1)
        # print(batch2,"\n")
        break
