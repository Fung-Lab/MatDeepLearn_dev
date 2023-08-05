import os
import copy
import random
import json
import numpy as np
import pandas as pd
from ase import io
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.utils import dense_to_sparse
from ase import Atoms

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
    calculate_edges_master
)

from matdeeplearn.datasets.augmentations import (
    node_masking, edge_masking, perturb_positions, column_replacement, strain_cell, generate_supercell
)


from matdeeplearn.preprocessor import StructureDataset
from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
import logging


def init_processor(dataset_config):
    root_path_dict = dataset_config["src"]
    target_path_dict = dataset_config["target_path"]
    pt_path = dataset_config.get("pt_path", None)
    prediction_level = dataset_config.get("prediction_level", "graph")
    preprocess_edges = dataset_config["preprocess_params"]["preprocess_edges"]
    preprocess_edge_features = dataset_config["preprocess_params"]["preprocess_edge_features"]
    preprocess_nodes = dataset_config["preprocess_params"]["preprocess_nodes"]
    cutoff_radius = dataset_config["preprocess_params"]["cutoff_radius"]
    n_neighbors = dataset_config["preprocess_params"]["n_neighbors"]
    num_offsets = dataset_config["preprocess_params"]["num_offsets"]
    edge_steps = dataset_config["preprocess_params"]["edge_steps"]
    data_format = dataset_config.get("data_format", "json")
    image_selfloop = dataset_config.get("image_selfloop", True)
    self_loop = dataset_config.get("self_loop", True)
    node_representation = dataset_config["preprocess_params"].get("node_representation", "onehot")
    additional_attributes = dataset_config.get("additional_attributes", [])
    verbose: bool = dataset_config.get("verbose", True)
    all_neighbors = dataset_config["preprocess_params"]["all_neighbors"]
    edge_calc_method = dataset_config["preprocess_params"].get("edge_calc_method", "mdl")
    device: str = dataset_config.get("device", "cpu")

    processor = LargeDataProcessor(
        root_path=root_path_dict,
        target_file_path=target_path_dict,
        pt_path=pt_path,
        prediction_level=prediction_level,
        preprocess_edges=preprocess_edges,
        preprocess_edge_features=preprocess_edge_features,
        preprocess_nodes=preprocess_nodes,
        r=cutoff_radius,
        n_neighbors=n_neighbors,
        num_offsets=num_offsets,
        edge_steps=edge_steps,
        transforms=dataset_config.get("transforms", []),
        data_format=data_format,
        image_selfloop=image_selfloop,
        self_loop=self_loop,
        node_representation=node_representation,
        additional_attributes=additional_attributes,
        verbose=verbose,
        all_neighbors=all_neighbors,
        edge_calc_method=edge_calc_method,
        device=device,
    )

    return processor


class LargeDataProcessor:
    def __init__(
            self,
            root_path: str,
            target_file_path: str,
            pt_path: str,
            prediction_level: str,
            preprocess_edges,
            preprocess_edge_features,
            preprocess_nodes,
            r: float,
            n_neighbors: int,
            num_offsets: int,
            edge_steps: int,
            transforms: list = [],
            data_format: str = "json",
            image_selfloop: bool = True,
            self_loop: bool = True,
            node_representation: str = "onehot",
            additional_attributes: list = [],
            verbose: bool = True,
            all_neighbors: bool = False,
            edge_calc_method: str = "mdl",
            device: str = "cpu",
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

            edge_steps: int
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
        self.preprocess_nodes = preprocess_nodes
        self.n_neighbors = n_neighbors
        self.num_offsets = num_offsets
        self.edge_steps = edge_steps
        self.data_format = data_format
        self.image_selfloop = image_selfloop
        self.self_loop = self_loop
        self.node_representation = node_representation
        self.additional_attributes = additional_attributes
        self.verbose = verbose
        self.all_neighbors = all_neighbors
        self.edge_calc_method = edge_calc_method
        self.device = device
        self.transforms = transforms
        self.disable_tqdm = logging.root.level > logging.INFO

    def src_check(self):
        if isinstance(self.root_path_dict, dict):

            self.root_path = self.root_path_dict["train"]
            if self.target_file_path_dict:
                self.target_file_path = self.target_file_path_dict["train"]
            else:
                self.target_file_path = self.target_file_path_dict
            logging.info("Train dataset found at {}".format(self.root_path))
            logging.info("Processing device: {}".format(self.device))

            self.root_path = self.root_path_dict["val"]
            if self.target_file_path_dict:
                self.target_file_path = self.target_file_path_dict["val"]
            else:
                self.target_file_path = self.target_file_path_dict
            logging.info("Train dataset found at {}".format(self.root_path))
            logging.info("Processing device: {}".format(self.device))

            self.root_path = self.root_path_dict["test"]
            if self.target_file_path_dict:
                self.target_file_path = self.target_file_path_dict["test"]
            else:
                self.target_file_path = self.target_file_path_dict
            logging.info("Train dataset found at {}".format(self.root_path))
            logging.info("Processing device: {}".format(self.device))

        else:
            self.root_path = self.root_path_dict
            self.target_file_path = self.target_file_path_dict
            logging.info("Single dataset found at {}".format(self.root_path))
            logging.info("Processing device: {}".format(self.device))

        if self.target_file_path:
            return self.ase_wrap()
        else:
            return self.json_wrap()

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
            if (np.array(cell) == np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])).all():
                cell = torch.zeros((3, 3)).unsqueeze(0)
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
                    cell = cell.view(1, 3, 3)
            else:
                cell = torch.zeros((3, 3)).unsqueeze(0)
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
            # if isinstance(_y, str):
            #    _y = float(_y)
            # elif isinstance(_y, list):
            #    _y = [float(each) for each in _y]

            y.append(_y)

            d["y"] = np.array(_y)

        y = np.array(y)
        return dict_structures

    def process(self, dict_structures,
                      column_replace=False, column_replace_num=0,
                      perturb=False, perturbing_distance=0.05, min_perturb_distance=0.,
                      strain=False, strain_strength=0.05,
                      supercell=False, supercell_num=2,
                      mask_node=False, mask_node_ratio=0.1,
                      mask_edge=False, mask_edge_ratio=0.1):
        data_list = self.get_data_list(dict_structures,
                                      column_replace, column_replace_num,
                                      perturb, perturbing_distance, min_perturb_distance,
                                      strain, strain_strength,
                                      supercell,supercell_num,
                                      mask_node, mask_node_ratio,
                                      mask_edge, mask_edge_ratio)
        data, slices = InMemoryDataset.collate(data_list)
        return data

    def get_data_list(self, dict_structures,
                            column_replace, column_replace_num,
                            perturb, perturbing_distance, min_perturb_distance,
                            strain, strain_strength,
                            supercell, supercell_num,
                            mask_node, mask_node_ratio,
                            mask_edge, mask_edge_ratio):
    
        #logging.debug(perturb, column_replacement, strain)
        
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        # logging.info("Getting torch_geometric.data.Data() objects.")

        for i, sdict in enumerate(dict_structures):
            # target_val = y[i]
            data = data_list[i]

            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            
            if supercell:
                atomic_numbers, pos, cell = generate_supercell(atomic_numbers, supercell_num, pos, cell)

            if perturb:
                pos = perturb_positions(pos, perturbing_distance, min_perturb_distance)
                        
            if column_replace:
                atomic_numbers = column_replacement(atomic_numbers, column_replace_num)
                
            if strain:
                pos, cell = strain_cell(atomic_numbers, pos, cell, strain_strength)
              
            # data.structure_id = [[structure_id] * len(data.y)]
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
                    self.all_neighbors,
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
                if (edge_vec.dim() > 2):
                    edge_vec = edge_vec[edge_indices[0], edge_indices[1]]

                data.edge_index, data.edge_weight = edge_indices, edge_weights
                data.edge_vec = edge_vec
                data.cell_offsets = cell_offsets
                data.neighbors = neighbors
                

                data.edge_descriptor = {}
                # data.edge_descriptor["mask"] = cd_matrix_masked
                data.edge_descriptor["distance"] = edge_weights
                data.distances = edge_weights

            # add additional attributes
            if self.additional_attributes:
                for attr in self.additional_attributes:
                    data.__setattr__(attr, sdict[attr])
            

        if self.preprocess_nodes == True:
            # logging.info("Generating node features...")
            generate_node_features(data_list, self.n_neighbors, device=self.device)

        if self.preprocess_edge_features == True:
            # logging.info("Generating edge features...")
            generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)
            
        if mask_node or mask_edge:
            for i in range(len(data_list)):
                data = data_list[i]
                if mask_node:
                    data = node_masking(data, mask_node_ratio)
                if mask_edge:
                    data = edge_masking(data, mask_edge_ratio)
                data_list[i] = data
        
            
        # compile non-otf transforms
        # logging.debug("Applying transforms.")

        # Ensure GetY exists to prevent downstream model errors
        assert "GetY" in [
            tf["name"] for tf in self.transforms
        ], "The target transform GetY is required in config."

        transforms_list = []
        for transform in self.transforms:
            if not transform.get("otf", False):
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

class LargeCTPretrainDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            processed_data_path,
            processed_file_name,
            mask_node_ratios=None,
            mask_edge_ratios=None,
            augmentation_list=None,
            random_augmentation=False,
            column_replace_num=1,
            perturbing_distance=0.05,
            min_perturb_distance=0.,
            strain_strength=0.05,
            supercell_num=2,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            device=None,
            dataset_config=None
    ):
        if mask_edge_ratios is None:
            mask_edge_ratios = [0.1, 0.1]
        if mask_node_ratios is None:
            mask_node_ratios = [0.1, 0.1]

        self.root = root
        self.processed_data_path = processed_data_path
        self.processed_file_name = processed_file_name
        self.augmentation_list = augmentation_list if augmentation_list else []
        self.random_augmentation = random_augmentation
        self.column_replace_num = column_replace_num
        self.perturbing_distance = perturbing_distance
        self.min_perturb_distance = min_perturb_distance
        self.strain_strength = strain_strength
        self.supercell_num = supercell_num
        self.mask_node_ratio1 = mask_node_ratios[0]
        self.mask_node_ratio2 = mask_node_ratios[1]
        self.mask_edge_ratio1 = mask_edge_ratios[0]
        self.mask_edge_ratio2 = mask_edge_ratios[1]
        self.device = device
        
        logging.info(f"augmentation: {augmentation_list}")
        logging.info(f"Random augmentation for each step: {random_augmentation}")
        
        log_str = ""
        if "column_replacement" in self.augmentation_list:
            log_str += f"Column replacement atom number: {self.column_replace_num}\n"
        if "perturbing" in self.augmentation_list:
            log_str += f"Perturbing distance: {self.perturbing_distance}, Minumum distance: {self.min_perturb_distance}\n"
        if "strain" in self.augmentation_list:
            log_str += f"Strain strength: {self.strain_strength}\n"
        if "supercell" in self.augmentation_list:
            log_str += f"Construct supercell: {self.supercell_num} times {self.supercell_num}\n"
        if "node_masking" in self.augmentation_list:
            log_str += f"Node masking ratios: [{self.mask_node_ratio1}, {self.mask_node_ratio2}]\n"
        if "edge_masking" in self.augmentation_list:
            log_str += f"Edge masking ratios: [{self.mask_edge_ratio1}, {self.mask_edge_ratio2}]\n"
        if log_str != "":
            logging.info(log_str)
            
        super(LargeCTPretrainDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        self.processer = init_processor(dataset_config)
        self.dict_structures = self.processer.src_check()

    def __len__(self):
        return len(self.dict_structures)

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

        actual_augmentation = self.augmentation_list
        if self.random_augmentation:
            actual_augmentation = np.random.choice(self.augmentation_list, 1)
            
        #actual_augmentation = [element for element in self.augmentation_list if random.random() < 0.5]
        #if not actual_augmentation:
        #    random_element = random.choice(self.augmentation_list)
        #    actual_augmentation.append(random_element)

            
        #print(f"augmentations {idx}: {actual_augmentation}")

        perturbing = "perturbing" in actual_augmentation
        column_replace = "column_replacement" in actual_augmentation
        strain = "strain" in actual_augmentation
        supercell = "supercell" in actual_augmentation
        mask_node = "node_masking" in actual_augmentation
        mask_edge = "edge_masking" in actual_augmentation
        
        #print(perturbing, column_replace, strain, supercell)
        
        if perturbing or column_replace or strain or supercell or mask_node or mask_edge:
            subdata1 = self.processer.process([self.dict_structures[idx]],
                                              column_replace=column_replace, column_replace_num=self.column_replace_num,
                                              perturb=perturbing,
                                              perturbing_distance=self.perturbing_distance, min_perturb_distance=self.min_perturb_distance,
                                              strain=strain, strain_strength=self.strain_strength,
                                              mask_node=mask_node, mask_node_ratio=self.mask_node_ratio1,
                                              mask_edge=mask_edge, mask_edge_ratio=self.mask_edge_ratio1,
                                              supercell=supercell, supercell_num=self.supercell_num)
            subdata2 = self.processer.process([self.dict_structures[idx]],
                                              column_replace=column_replace, column_replace_num=self.column_replace_num,
                                              perturb=perturbing,
                                              perturbing_distance=self.perturbing_distance, min_perturb_distance=self.min_perturb_distance,
                                              strain=strain, strain_strength=self.strain_strength,
                                              mask_node=mask_node, mask_node_ratio=self.mask_node_ratio2,
                                              mask_edge=mask_edge, mask_edge_ratio=self.mask_edge_ratio2,
                                              supercell=supercell, supercell_num=self.supercell_num)
            return subdata1, subdata2

        subdata = self.processer.process([self.dict_structures[idx]])

        subdata1 = copy.deepcopy(subdata)
        subdata2 = copy.deepcopy(subdata)

        # Apply transforms
        if self.transform is not None:
            subdata1 = self.transform(subdata1)
            subdata2 = self.transform(subdata2)
            
        return subdata1, subdata2