import json
import logging
import os

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
)


def process_data(dataset_config):
    root_path = dataset_config["src"]
    target_path = dataset_config["target_path"]
    pt_path = dataset_config.get("pt_path", None)
    cutoff_radius = dataset_config["cutoff_radius"]
    n_neighbors = dataset_config["n_neighbors"]
    edge_steps = dataset_config["edge_steps"]
    data_format = dataset_config.get("data_format", "json")
    image_selfloop = dataset_config.get("image_selfloop", True)
    self_loop = dataset_config.get("self_loop", True)
    node_representation = dataset_config.get("node_representation", "onehot")
    additional_attributes = dataset_config.get("additional_attributes", [])
    verbose: bool = dataset_config.get("verbose", True)
    all_neighbors = dataset_config["all_neighbors"]
    device: str = dataset_config.get("device", "cpu")

    processor = DataProcessor(
        root_path=root_path,
        target_file_path=target_path,
        pt_path=pt_path,
        r=cutoff_radius,
        n_neighbors=n_neighbors,
        edge_steps=edge_steps,
        transforms=dataset_config.get("transforms", []),
        data_format=data_format,
        image_selfloop=image_selfloop,
        self_loop=self_loop,
        node_representation=node_representation,
        additional_attributes=additional_attributes,
        verbose=verbose,
        all_neighbors=all_neighbors,
        device=device,
    )
    processor.process()


class DataProcessor:
    def __init__(
        self,
        root_path: str,
        target_file_path: str,
        pt_path: str,
        r: float,
        n_neighbors: int,
        edge_steps: int,
        transforms: list = [],
        data_format: str = "json",
        image_selfloop: bool = True,
        self_loop: bool = True,
        node_representation: str = "onehot",
        additional_attributes: list = [],
        verbose: bool = True,
        all_neighbors: bool = False,
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

        self.root_path = root_path
        self.target_file_path = target_file_path
        self.pt_path = pt_path
        self.r = r
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.data_format = data_format
        self.image_selfloop = image_selfloop
        self.self_loop = self_loop
        self.node_representation = node_representation
        self.additional_attributes = additional_attributes
        self.verbose = verbose
        self.all_neighbors = all_neighbors
        self.device = device
        self.transforms = transforms
        self.disable_tqdm = logging.root.level > logging.INFO

    def src_check(self):
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

            dict_structures.append(d)

        return dict_structures, y

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
            pos = torch.tensor(s["positions"], device=self.device, dtype=torch.float)
            cell = torch.tensor(s["cell"], device=self.device, dtype=torch.float)
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
            if isinstance(_y, str):
                _y = float(_y)
            elif isinstance(_y, list):
                _y = [float(each) for each in _y]
            y.append(_y)

        y = np.array(y).reshape(-1, y_dim)
        return dict_structures, y

    def process(self, save=True):
        logging.info("Data found at {}".format(self.root_path))
        logging.info("Processing device: {}".format(self.device))

        dict_structures, y = self.src_check()
        data_list = self.get_data_list(dict_structures, y)
        data, slices = InMemoryDataset.collate(data_list)

        if save:
            if self.pt_path:
                save_path = os.path.join(self.pt_path, "data.pt")

            torch.save((data, slices), save_path)
            logging.info("Processed data saved successfully.")

        return data_list

    def get_data_list(self, dict_structures, y):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        for i, sdict in enumerate(tqdm(dict_structures, disable=self.disable_tqdm)):
            target_val = y[i]
            data = data_list[i]

            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            structure_id = sdict["structure_id"]
            if self.all_neighbors == False:
                cd_matrix, cell_offsets, atom_rij = get_cutoff_distance_matrix(
                    pos,
                    cell,
                    self.r,
                    self.n_neighbors,
                    image_selfloop=self.image_selfloop,
                    device=self.device,
                )
                edge_indices, edge_weights = dense_to_sparse(cd_matrix)
                if(atom_rij.dim() > 1):
                  edge_vec = atom_rij[edge_indices[0], edge_indices[1]]
            elif self.all_neighbors == True:
                #cd_matrix, cell_offsets, _ = get_cutoff_distance_matrix(
                #    pos,
                #    cell,
                #    self.r,
                #    self.n_neighbors,
                #    image_selfloop=self.image_selfloop,
                #    device=self.device,
                #)
                #edge_indices, edge_weights = dense_to_sparse(cd_matrix)
                
                first_idex, second_idex, rij, rij_vec, shifts = ase.neighborlist.primitive_neighbor_list("ijdDS", (True,True,True), ase.geometry.complete_cell(cell.squeeze()), pos.numpy(), cutoff=self.r, self_interaction=False, use_scaled_positions=False)   
                # Eliminate true self-edges that don't cross periodic boundaries (https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py)
                bad_edge = first_idex == second_idex
                bad_edge &= np.all(shifts == 0, axis=1)
                keep_edge = ~bad_edge
                first_idex = first_idex[keep_edge]
                second_idex = second_idex[keep_edge]
                rij = rij[keep_edge] 
                rij_vec = rij_vec[keep_edge]
                shifts = shifts[keep_edge]
                # get closest n neighbors
                if len(rij) > self.n_neighbors:
                    _, topk_indices = torch.topk(torch.tensor(rij), self.n_neighbors, largest=False, sorted=False)
                    first_idex = first_idex[topk_indices]
                    second_idex = second_idex[topk_indices]
                    rij = rij[topk_indices] 
                    rij_vec = rij_vec[topk_indices]
                    shifts = shifts[topk_indices]    
                                  
                edge_indices = torch.stack([torch.LongTensor(torch.tensor(first_idex)), torch.LongTensor(torch.tensor(second_idex))], dim=0)
                edge_vec = torch.tensor(rij_vec).float()
                edge_weights = torch.tensor(rij).float()   
                cell_offsets = torch.tensor(shifts).int()                
                
            data.n_atoms = len(atomic_numbers)
            data.pos = pos
            data.cell = cell
            data.y = torch.Tensor(np.array([target_val]))
            data.z = atomic_numbers
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights
            data.edge_vec = edge_vec
            data.cell_offsets = cell_offsets

            data.edge_descriptor = {}
            # data.edge_descriptor["mask"] = cd_matrix_masked
            data.edge_descriptor["distance"] = edge_weights
            data.distances = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]

            # add additional attributes
            if self.additional_attributes:
                for attr in self.additional_attributes:
                    data.__setattr__(attr, sdict[attr])

        logging.info("Generating node features...")
        generate_node_features(data_list, self.n_neighbors, device=self.device)

        logging.info("Generating edge features...")
        generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)

        # compile non-otf transforms
        logging.debug("Applying transforms.")

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
