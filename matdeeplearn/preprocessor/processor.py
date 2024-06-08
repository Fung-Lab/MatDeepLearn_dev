import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import ase
from ase import io, Atoms, neighborlist
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
    calculate_edges_master,
)


def from_config(dataset_config):
    root_path_dict = dataset_config["src"]
    target_path_dict = dataset_config["target_path"]
    pt_path = dataset_config.get("pt_path", None)
    prediction_level = dataset_config.get("prediction_level", "graph")
    preprocess_edges = dataset_config["preprocess_params"]["preprocess_edges"]
    preprocess_edge_features = dataset_config["preprocess_params"]["preprocess_edge_features"]
    preprocess_node_features = dataset_config["preprocess_params"]["preprocess_node_features"]
    cutoff_radius = dataset_config["preprocess_params"]["cutoff_radius"]
    n_neighbors = dataset_config["preprocess_params"]["n_neighbors"]
    num_offsets = dataset_config["preprocess_params"]["num_offsets"]
    edge_dim = dataset_config["preprocess_params"]["edge_dim"]
    data_format = dataset_config.get("data_format", "json")
    image_selfloop = dataset_config.get("image_selfloop", True)
    self_loop = dataset_config.get("self_loop", True)
    node_representation = dataset_config["preprocess_params"].get("node_representation", "onehot")
    additional_attributes = dataset_config.get("additional_attributes", [])
    verbose: bool = dataset_config.get("verbose", True)
    edge_calc_method = dataset_config["preprocess_params"].get("edge_calc_method", "mdl")
    device: str = dataset_config.get("device", "cpu")
    virtual_params = dataset_config["preprocess_params"].get("virtual_params", None)

    processor = DataProcessor(
        root_path=root_path_dict,
        target_file_path=target_path_dict,
        pt_path=pt_path,
        prediction_level=prediction_level,
        preprocess_edges=preprocess_edges,
        preprocess_edge_features=preprocess_edge_features,
        preprocess_node_features=preprocess_node_features,
        r=cutoff_radius,
        n_neighbors=n_neighbors,
        num_offsets=num_offsets,
        edge_dim=edge_dim,
        transforms=dataset_config.get("transforms", []),
        data_format=data_format,
        image_selfloop=image_selfloop,
        self_loop=self_loop,
        node_representation=node_representation,
        additional_attributes=additional_attributes,
        verbose=verbose,
        edge_calc_method=edge_calc_method,
        device=device,
        virtual_params=virtual_params
    )

    return processor


def process_data(dataset_config, seed):
    processor = from_config(dataset_config)
    np.random.seed(seed)
    dataset = processor.process()

    return dataset


def get_sampling_indices(vn_labels, num_samples):
    prob_distribution = vn_labels / np.sum(vn_labels)
    sampled_indices = np.random.choice(len(vn_labels), size=num_samples, p=prob_distribution.squeeze())

    return sampled_indices


class DataProcessor:
    def __init__(
        self,
        root_path: str,
        target_file_path: str,
        pt_path: str,
        prediction_level: str,
        preprocess_edges,
        preprocess_edge_features,
        preprocess_node_features,     
        r: float,
        n_neighbors: int,
        num_offsets: int,
        edge_dim: int,
        transforms: list = [],
        data_format: str = "json",
        image_selfloop: bool = True,
        self_loop: bool = True,
        node_representation: str = "onehot",
        additional_attributes: list = [],
        verbose: bool = True,
        edge_calc_method: str = "mdl",
        device: str = "cpu",
        virtual_params: dict = None
    ) -> None:
        """
        create a DataProcessor that processes the raw data and save into data.pt file.

        Parameters
        ----------
            root_path: str
                a path to the root folder for all data

            target_file_path: str
                a path to a CSV file containing target y values

            pt_path: str
                a path to the directory to which data.pt should be saved

            r: float
                cutoff radius

            n_neighbors: int
                max number of neighbors to be considered
                => closest n neighbors will be kept

            edge_dim: int
                step size for creating Gaussian basis for edges
                used in torch.linspace

            transforms: list
                default []. List of transforms to apply to the data.

            data_format: str
                format of the raw data file

            image_selfloop: bool
                if True, add self loop to node and set the distance to
                the distance between node and its closest image

            self_loop: bool
                if True, every node in a graph will have a self loop

            node_representation: str
                a path to a JSON file containing vectorized representations
                of elements of atomic numbers from 1 to 100 inclusive.

                default: one-hot representation

            additional_attributes: list of str
                additional user-specified attributes to be included in
                a Data() object

            verbose: bool
                if True, certain messages will be printed
        """

        self.root_path_dict = root_path
        self.target_file_path_dict = target_file_path
        self.pt_path = pt_path
        self.r = r
        self.prediction_level = prediction_level
        self.preprocess_edges = preprocess_edges
        self.preprocess_edge_features = preprocess_edge_features
        self.preprocess_node_features = preprocess_node_features
        self.n_neighbors = n_neighbors
        self.num_offsets = num_offsets
        self.edge_dim = edge_dim
        self.data_format = data_format
        self.image_selfloop = image_selfloop
        self.self_loop = self_loop
        self.node_representation = node_representation
        self.additional_attributes = additional_attributes
        self.verbose = verbose
        self.edge_calc_method = edge_calc_method
        self.device = device
        self.transforms = transforms
        self.disable_tqdm = logging.root.level > logging.INFO
        if virtual_params:
            self.num_samples = virtual_params["num_samples"]
            self.sub_batch = virtual_params["num_sub_batch"]

    def src_check(self):
        if self.prediction_level == "virtual":
            return self.chg_wrap()
        elif self.target_file_path:
            return self.ase_wrap()
        else:
            return self.json_wrap()

    def chg_wrap(self):
        dict_structures = []
        if ".json" in self.root_path:
            logging.info("Reading one JSON file for multiple structures.")

            f = open(self.root_path)

            logging.info(
                "Loading json file as dict (this might take a while for large json file size)."
            )
            original_structures = json.load(f)
            f.close()

            y = []
            y_dim = (
                len(original_structures[0]["y"])
                if isinstance(original_structures[0]["y"], list)
                else 1
            )

            logging.info("Converting data to standardized form for downstream processing.")
            for i, s in enumerate(tqdm(original_structures, disable=self.disable_tqdm)):

                charge_density = torch.tensor(s["charge_density"], device=self.device, dtype=torch.float)

                num_virtual_nodes = len(charge_density)
                # num_half = num_virtual_nodes // 2
                # random_indices = torch.randperm(num_virtual_nodes)
                random_indices = torch.arange(0, num_virtual_nodes)
                indices = [random_indices[i: min(i + 200, num_virtual_nodes)] for i in
                           range(0, num_virtual_nodes, 200)]

                for sub_indices in indices:
                    d = {}
                    charge_density_part = charge_density[sub_indices]

                    pos_vn = charge_density_part[:, :3]
                    vn_labels = charge_density_part[:, -1].view(-1, 1)
                    atomic_numbers_vn = torch.LongTensor([100] * pos_vn.shape[0], device=self.device)

                    pos = torch.tensor(s["positions"], device=self.device, dtype=torch.float)
                    if "cell" in s:
                        cell = torch.tensor(s["cell"], device=self.device, dtype=torch.float)
                        if cell.shape[0] != 1:
                            cell = cell.view(1, 3, 3)
                    else:
                        cell = torch.zeros((3, 3)).unsqueeze(0)
                    atomic_numbers = torch.LongTensor(s["atomic_numbers"])

                    d["positions"] = torch.cat((pos, pos_vn), dim=0)
                    d["cell"] = cell
                    d["atomic_numbers"] = torch.cat((atomic_numbers, atomic_numbers_vn), dim=0)
                    d["structure_id"] = s["structure_id"]
                    d["y"] = vn_labels
                    # print(pos_vn.shape, d["y"].shape, pos_vn[0:3], d["y"][0:3])

                    dict_structures.append(d)
        elif "singlet" in self.root_path or "triplet" in self.root_path:
            try:
                # densities = np.genfromtxt(self.root_path+dir_name+"/densities.csv", skip_header=1, delimiter=',')
                df = pd.read_csv(self.root_path + "/densities.csv", header=0).to_numpy()
                vn_coords = df[:, 0:3]
                vn_labels = np.expand_dims((df[:, 5] + df[:, 6]), 1)

                num_virtual_nodes = len(vn_labels)
                random_indices = torch.arange(0, num_virtual_nodes)
                indices = [random_indices[i: min(i + 200, num_virtual_nodes)] for i in
                           range(0, num_virtual_nodes, 200)]

                for sub_indices in indices:
                    d = {}
                    pos_vn = torch.tensor(vn_coords[sub_indices, :], device=self.device, dtype=torch.float)
                    atomic_numbers_vn = torch.LongTensor([100] * pos_vn.shape[0], device=self.device)
                    # d["positions_vn"] = vn_coords[indices, :]
                    # d["atomic_numbers_vn"] = torch.LongTensor([100] * df.shape[0])
                    d["y"] = vn_labels[sub_indices, :]

                    ase_structure = io.read(self.root_path + "/structure.xsf")
                    pos = torch.tensor(ase_structure.get_positions(), device=self.device, dtype=torch.float)
                    cell = torch.tensor(
                        np.array(ase_structure.get_cell()), device=self.device, dtype=torch.float
                    ).view(1, 3, 3)
                    if (np.array(cell) == np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])).all():
                        cell = torch.zeros((3, 3)).unsqueeze(0)
                    atomic_numbers = torch.LongTensor(ase_structure.get_atomic_numbers())

                    d["positions"] = torch.cat((pos, pos_vn), dim=0)
                    d["cell"] = cell
                    d["atomic_numbers"] = torch.cat((atomic_numbers, atomic_numbers_vn), dim=0)
                    str_root_path_list = str(self.root_path).split("/")
                    d["structure_id"] = str_root_path_list[-1] if str_root_path_list[-1] else str_root_path_list[-2]
                    dict_structures.append(d)
                # print(dir_name, df.shape, pos_vn.shape, d["y"].shape, pos_vn[0:3], d["y"][0:3], ase_structure)
            except Exception as e:
                pass
        else:
            # if isinstance(self.root_path_dict, dict):
            #     self.root_path_dict = self.root_path_dict["train"]
            dirs = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
            for i, dir_name in enumerate(tqdm(dirs, disable=self.disable_tqdm)):
                if "singlet" in dir_name:
                    try:
                        #densities = np.genfromtxt(self.root_path+dir_name+"/densities.csv", skip_header=1, delimiter=',')
                        df = pd.read_csv(self.root_path+dir_name+"/densities.csv", header=0).to_numpy()
                        vn_coords = df[:,0:3]
                        vn_labels = np.expand_dims((df[:,5] + df[:,6]), 1)

                        # indices = random.sample(range(0, df.shape[0]), 200)
                        indices = get_sampling_indices(vn_labels, self.num_samples) \
                            if self.num_samples != -1 else np.arange(len(vn_labels))
                        np.random.shuffle(indices)
                        indices = [indices[i: min(i + self.sub_batch, len(indices))] for i in range(0, len(indices), self.sub_batch)]

                        for sub_indices in indices:
                            d = {}
                            pos_vn = torch.tensor(vn_coords[sub_indices, :], device=self.device, dtype=torch.float)
                            atomic_numbers_vn = torch.LongTensor([100] * pos_vn.shape[0], device=self.device)
                            #d["positions_vn"] = vn_coords[indices, :]
                            #d["atomic_numbers_vn"] = torch.LongTensor([100] * df.shape[0])
                            d["y"] = vn_labels[sub_indices, :]

                            ase_structure = io.read(self.root_path+dir_name+"/structure.xsf")
                            pos = torch.tensor(ase_structure.get_positions(), device=self.device, dtype=torch.float)
                            cell = torch.tensor(
                                np.array(ase_structure.get_cell()), device=self.device, dtype=torch.float
                            ).view(1, 3, 3)
                            if (np.array(cell) == np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])).all():
                                cell = torch.zeros((3,3)).unsqueeze(0)
                            atomic_numbers = torch.LongTensor(ase_structure.get_atomic_numbers())

                            d["positions"] = torch.cat((pos, pos_vn), dim=0)
                            d["cell"] = cell
                            d["atomic_numbers"] = torch.cat((atomic_numbers, atomic_numbers_vn), dim=0)
                            d["structure_id"] = str(dir_name)
                            dict_structures.append(d)
                        # print(dir_name, df.shape, pos_vn.shape, d["y"].shape, pos_vn[0:3], d["y"][0:3], ase_structure)
                    except Exception as e:
                        pass

        return dict_structures

    def ase_wrap(self):
        """
        raw files are ase readable and self.target_file_path is not None
        """
        logging.info("Reading individual structures using ASE.")

        df = pd.read_csv(self.target_file_path, header=None)
        file_names = df[0].to_list()
        y = df.iloc[:, 1:].to_numpy()

        dict_structures = []
        ase_structures = []

        logging.info("Converting data to standardized form for downstream processing.")
        for i, structure_id in enumerate(file_names):
            p = os.path.join(self.root_path, str(structure_id) + "." + self.data_format)
            ase_structures.append(io.read(p))

        for i, s in enumerate(tqdm(ase_structures, disable=self.disable_tqdm)):
            d = {}
            pos = torch.tensor(s.get_positions(), device=self.device, dtype=torch.float)
            cell = torch.tensor(
                np.array(s.get_cell()), device=self.device, dtype=torch.float
            ).view(1, 3, 3)
            if (np.array(cell) == np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])).all():
                cell = torch.zeros((3,3)).unsqueeze(0)
            atomic_numbers = torch.LongTensor(s.get_atomic_numbers())

            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = str(file_names[i])

            # add additional attributes
            if self.additional_attributes:
                attributes = self.get_csv_additional_attributes(d["structure_id"])
                for k, v in attributes.items():
                    d[k] = v

            d["y"] = y[i]

            dict_structures.append(d)

        return dict_structures

    def get_csv_additional_attributes(self, structure_id):
        """
        load additional attributes specified by the user
        """

        attributes = {}

        for attr in self.additional_attributes:
            p = os.path.join(self.root_path, structure_id + "_" + attr + ".csv")
            values = np.genfromtxt(p, delimiter=",", dtype=float, encoding=None)
            values = torch.tensor(values, device=self.device, dtype=torch.float)
            attributes[attr] = values

        return attributes

    def json_wrap(self):
        """
        all structures are saved in a single json file
        """
        logging.info("Reading one JSON file for multiple structures.")

        f = open(self.root_path)

        logging.info(
            "Loading json file as dict (this might take a while for large json file size)."
        )
        original_structures = json.load(f)
        f.close()

        dict_structures = []
        y = []
        y_dim = (
            len(original_structures[0]["y"])
            if isinstance(original_structures[0]["y"], list)
            else 1
        )
        logging.info("Converting data to standardized form for downstream processing.")
        for i, s in enumerate(tqdm(original_structures, disable=self.disable_tqdm)):
            d = {}
            if (len(s["atomic_numbers"]) == 1):
                continue
            pos = torch.tensor(s["positions"], device=self.device, dtype=torch.float)
            if "cell" in s:
                cell = torch.tensor(s["cell"], device=self.device, dtype=torch.float)
                if cell.shape[0] != 1:
                    cell = cell.view(1,3,3)
            else:
                cell = torch.zeros((3,3)).unsqueeze(0)
            atomic_numbers = torch.LongTensor(s["atomic_numbers"])

            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = s["structure_id"]

            # add additional attributes
            if self.additional_attributes:
                for attr in self.additional_attributes:
                    d[attr] = torch.tensor(
                        s[attr], device=self.device, dtype=torch.float
                    )

            dict_structures.append(d)

            # check y types
            _y = s["y"]
            if isinstance(_y, list) == False:
                _y = np.array([_y], dtype=np.float32)
            else:
                _y = np.array(_y, dtype=np.float32)
            #if isinstance(_y, str):
            #    _y = float(_y)
            #elif isinstance(_y, list):
            #    _y = [float(each) for each in _y]

            y.append(_y)

            d["y"] = np.array(_y)

        y = np.array(y)
        return dict_structures

    def process(self, save=True):

        data_list={}
        if isinstance(self.root_path_dict, dict):

            if self.root_path_dict.get("train"):
                self.root_path = self.root_path_dict["train"]
                if self.target_file_path_dict:
                    self.target_file_path = self.target_file_path_dict["train"]
                else:
                    self.target_file_path = self.target_file_path_dict
                logging.info("Train dataset found at {}".format(self.root_path))
                logging.info("Processing device: {}".format(self.device))

                dict_structures = self.src_check()
                data_list["train"] = self.get_data_list(dict_structures)
                data, slices = InMemoryDataset.collate(data_list["train"])

                if save:
                    if self.pt_path:
                        save_path = os.path.join(self.pt_path, "data_train.pt")
                    torch.save((data, slices), save_path)
                    logging.info("Processed train data saved successfully.")

            if self.root_path_dict.get("val"):
                self.root_path = self.root_path_dict["val"]
                if self.target_file_path_dict:
                    self.target_file_path = self.target_file_path_dict["val"]
                else:
                    self.target_file_path = self.target_file_path_dict
                logging.info("Val dataset found at {}".format(self.root_path))
                logging.info("Processing device: {}".format(self.device))

                dict_structures = self.src_check()
                data_list["val"] = self.get_data_list(dict_structures)
                data, slices = InMemoryDataset.collate(data_list["val"])

                if save:
                    if self.pt_path:
                        save_path = os.path.join(self.pt_path, "data_val.pt")
                    torch.save((data, slices), save_path)
                    logging.info("Processed val data saved successfully.")

            if self.root_path_dict.get("test"):
                self.root_path = self.root_path_dict["test"]
                if self.target_file_path_dict:
                    self.target_file_path = self.target_file_path_dict["test"]
                else:
                    self.target_file_path = self.target_file_path_dict
                logging.info("Test dataset found at {}".format(self.root_path))
                logging.info("Processing device: {}".format(self.device))

                dict_structures = self.src_check()
                data_list["test"] = self.get_data_list(dict_structures)
                data, slices = InMemoryDataset.collate(data_list["test"])

                if save:
                    if self.pt_path:
                        save_path = os.path.join(self.pt_path, "data_test.pt")
                    torch.save((data, slices), save_path)
                    logging.info("Processed test data saved successfully.")

            if self.root_path_dict.get("predict"):
                self.root_path = self.root_path_dict["predict"]
                if self.target_file_path_dict:
                    self.target_file_path = self.target_file_path_dict["predict"]
                else:
                    self.target_file_path = self.target_file_path_dict
                logging.info("Predict dataset found at {}".format(self.root_path))
                logging.info("Processing device: {}".format(self.device))

                dict_structures = self.src_check()
                data_list["predict"] = self.get_data_list(dict_structures)
                data, slices = InMemoryDataset.collate(data_list["predict"])

                if save:
                    if self.pt_path:
                        save_path = os.path.join(self.pt_path, "data_predict.pt")
                    torch.save((data, slices), save_path)
                    logging.info("Processed predict data saved successfully.")

        else:
            self.root_path = self.root_path_dict
            self.target_file_path = self.target_file_path_dict
            logging.info("Single dataset found at {}".format(self.root_path))
            logging.info("Processing device: {}".format(self.device))

            dict_structures = self.src_check()
            data_list["full"] = self.get_data_list(dict_structures)
            data, slices = InMemoryDataset.collate(data_list["full"])

            if save:
                if self.pt_path:
                    save_path = os.path.join(self.pt_path, "data.pt")
                torch.save((data, slices), save_path)
                logging.info("Processed data saved successfully.")

        return data_list

    def get_data_list(self, dict_structures):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        for i, sdict in enumerate(tqdm(dict_structures, disable=self.disable_tqdm)):
            #target_val = y[i]
            data = data_list[i]

            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            #data.structure_id = [[structure_id] * len(data.y)]
            structure_id = sdict["structure_id"]

            data.n_atoms = len(atomic_numbers)
            data.pos = pos
            data.cell = cell
            data.structure_id = [structure_id]
            data.z = atomic_numbers
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])

            target_val = sdict["y"]
            # Data.y.dim()should equal 2, with dimensions of either (1, n) for graph-level labels or (n_atoms, n) for node level labels, where n is length of label vector (usually n=1)
            data.y = torch.Tensor(np.array(target_val))

            if self.prediction_level == "graph":
                if data.y.dim() > 1 and data.y.shape[0] != 0:
                    raise ValueError('Target labels do not have the correct dimensions for graph-level prediction.')
                elif data.y.dim() == 1:             
                    data.y = data.y.unsqueeze(0)
            elif self.prediction_level == "node":
                if data.y.shape[0] != data.n_atoms:
                    raise ValueError('Target labels do not have the correct dimensions for node-level prediction.')
                elif data.y.dim() == 1:
                    data.y = data.y.unsqueeze(1)            

            if self.preprocess_edges == True:
                edge_gen_out = calculate_edges_master(
                    self.edge_calc_method,
                    self.r,
                    self.n_neighbors,
                    self.num_offsets,
                    structure_id,
                    cell,
                    pos,
                    atomic_numbers,
                )
                edge_indices = edge_gen_out["edge_index"]
                edge_weights = edge_gen_out["edge_weights"]
                cell_offsets = edge_gen_out["cell_offsets"]
                edge_vec = edge_gen_out["edge_vec"]
                neighbors = edge_gen_out["neighbors"]
                if(edge_vec.dim() > 2):
                    edge_vec = edge_vec[edge_indices[0], edge_indices[1]]

                if self.prediction_level == "virtual":
                    edge_mask = torch.zeros_like(edge_indices[0])
                    edge_mask[(atomic_numbers[edge_indices[0]] == 100) & (atomic_numbers[edge_indices[1]] == 100)] = 0  # virtual node to virtual node
                    edge_mask[(atomic_numbers[edge_indices[0]] != 100) & (atomic_numbers[edge_indices[1]] == 100)] = 1  # regular node to virtual node
                    edge_mask[(atomic_numbers[edge_indices[0]] == 100) & (atomic_numbers[edge_indices[1]] != 100)] = 2  # virtual node to regular node
                    edge_mask[(atomic_numbers[edge_indices[0]] != 100) & (atomic_numbers[edge_indices[1]] != 100)] = 3  # regular node to regular node

                    # data.edge_mask = edge_mask
                    indices_rn_to_rn = (edge_mask == 3) & (edge_weights <= 8)
                    indices_rn_to_vn = (edge_mask == 1) & (edge_weights <= 8)
                    # indices_vn_to_vn = (edge_mask == 0) & (edge_weights <= 4)
                    indices_to_keep = indices_rn_to_rn | indices_rn_to_vn  # | indices_vn_to_vn
                    indices_rn_to_rn = indices_rn_to_rn[indices_to_keep]
                    indices_rn_to_vn = indices_rn_to_vn[indices_to_keep]
                    # indices_vn_to_vn = indices_vn_to_vn[indices_to_keep]

                    edge_indices = edge_indices[:, indices_to_keep]
                    edge_weights = edge_weights[indices_to_keep]
                    edge_vec = edge_vec[indices_to_keep, :]
                    data.indices_rn_to_rn = indices_rn_to_rn
                    data.indices_rn_to_vn = indices_rn_to_vn

                data.edge_index, data.edge_weight = edge_indices, edge_weights
                data.edge_vec = edge_vec
                data.cell_offsets = cell_offsets
                data.neighbors = neighbors

                data.edge_descriptor = {}
                # data.edge_descriptor["mask"] = cd_matrix_masked
                data.edge_descriptor["distance"] = edge_weights
                # data.distances = edge_weights




            # add additional attributes
            if self.additional_attributes:
                for attr in self.additional_attributes:
                    data.__setattr__(attr, sdict[attr])


        if self.preprocess_node_features == True:
            logging.info("Generating node features...")
            generate_node_features(data_list, self.n_neighbors, device=self.device)

        if self.preprocess_edge_features == True:
            logging.info("Generating edge features...")
            generate_edge_features(data_list, self.edge_dim, self.r, device=self.device)

        # compile non-otf transforms
        logging.debug("Applying transforms.")
        # Ensure GetY exists to prevent downstream model errors
        assert "GetY" in [
            tf["name"] for tf in self.transforms
        ], "The target transform GetY is required in config."

        transforms_list = []
        for transform in self.transforms:
            if not transform.get("otf_transform", False):
                transforms_list.append(
                    registry.get_transform_class(
                        transform["name"],
                        **({} if transform["args"] is None else transform["args"])
                    )
                )
        composition = Compose(transforms_list)

        # apply transforms
        for data in data_list:
            composition(data)
        clean_up(data_list, ["edge_descriptor"])
        return data_list
