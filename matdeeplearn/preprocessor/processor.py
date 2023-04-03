import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import pathlib
import wandb
from ase import io
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    generate_virtual_nodes,
    calculate_edges_master,
    PerfTimer,
)


def process_data(dataset_config, wandb_config):
    root_path = dataset_config["src"]
    target_path = dataset_config["target_path"]
    pt_path = dataset_config.get("pt_path", None)
    cutoff_radius = dataset_config["preprocess_params"]["cutoff_radius"]
    n_neighbors = dataset_config["preprocess_params"]["n_neighbors"]
    num_offsets = dataset_config["preprocess_params"]["num_offsets"]
    edge_steps = dataset_config["preprocess_params"]["edge_steps"]
    data_format = dataset_config.get("data_format", "json")
    image_selfloop = dataset_config.get("image_selfloop", True)
    self_loop = dataset_config.get("self_loop", True)
    node_representation = dataset_config.get("node_representation", "onehot")
    additional_attributes = dataset_config.get("additional_attributes", [])
    verbose: bool = dataset_config.get("verbose", True)
    all_neighbors = dataset_config["preprocess_params"]["all_neighbors"]
    use_degree = dataset_config["preprocess_params"]["use_degree"]
    edge_calc_method = dataset_config["preprocess_params"].get(
        "edge_calc_method", "mdl"
    )
    apply_pre_transform_processing = dataset_config.get(
        "apply_pre_transform_processing", True
    )
    device: str = dataset_config.get("device", "cpu")

    # wandb config
    use_sweep_params = wandb_config["sweep"].get("do_sweep", False)
    use_wandb = wandb_config.get("use_wandb", False)

    preprocess_kwargs = dataset_config.get("preprocess_params", {})

    processor = DataProcessor(
        root_path=root_path,
        target_file_path=target_path,
        pt_path=pt_path,
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
        use_degree=use_degree,
        edge_calc_method=edge_calc_method,
        apply_pre_transform_processing=apply_pre_transform_processing,
        use_sweep_params=use_sweep_params,
        use_wandb=use_wandb,
        preprocess_kwargs=preprocess_kwargs,
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
        use_degree: bool = False,
        edge_calc_method: str = "mdl",
        apply_pre_transform_processing: bool = True,
        use_sweep_params: bool = False,
        use_wandb: bool = False,
        preprocess_kwargs: dict = {},
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

            otf: bool
                default False. Whether or not to transform the data on the fly.

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
        self.num_offsets = num_offsets
        self.edge_steps = edge_steps
        self.data_format = data_format
        self.image_selfloop = image_selfloop
        self.self_loop = self_loop
        self.node_representation = node_representation
        self.additional_attributes = additional_attributes
        self.verbose = verbose
        self.all_neighbors = all_neighbors
        self.use_degree = use_degree
        self.edge_calc_method = edge_calc_method
        self.apply_pre_transform_processing = apply_pre_transform_processing
        self.use_sweep_params = use_sweep_params
        self.use_wandb = use_wandb
        self.preprocess_kwargs = preprocess_kwargs
        self.device = device
        self.transforms = transforms
        self.disable_tqdm = logging.root.level > logging.INFO

        # construct metadata signature
        self.metadata = self.preprocess_kwargs
        # find non-OTF transforms
        transforms = [t.get("args") for t in self.transforms if not t.get("otf")]
        for t_args in transforms:
            self.metadata.update(t_args)

    def src_check(self):
        if self.target_file_path:
            logging.debug("ASE wrap")
            return self.ase_wrap()
        else:
            logging.debug("JSON wrap")
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

        logging.info(
            "(ASE) Converting data to standardized form for downstream processing."
        )
        for i, structure_id in enumerate(file_names):
            p = os.path.join(self.root_path, str(structure_id) + "." + self.data_format)
            ase_structures.append(io.read(p))

        for i, s in enumerate(tqdm(ase_structures, disable=self.disable_tqdm)):
            d = {}
            pos = torch.tensor(s.get_positions(), device=self.device, dtype=torch.float)
            cell = torch.tensor(
                np.array(s.get_cell()), device=self.device, dtype=torch.float
            )
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

        logging.info(
            "(JSON) Converting data to standardized form for downstream processing."
        )
        for i, s in enumerate(tqdm(original_structures, disable=self.disable_tqdm)):
            d = {}
            pos = torch.tensor(s["positions"], device=self.device, dtype=torch.float)
            cell = torch.tensor(s["cell"], device=self.device, dtype=torch.float)
            cell2 = s["cell2"]
            atomic_numbers = torch.LongTensor(s["atomic_numbers"])

            d["positions"] = pos
            d["cell"] = cell
            d["cell2"] = cell2
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
        logging.info("Checking for existing processed data with matching metadata...")

        found_existing = False
        data_dir = pathlib.Path(self.pt_path).parent
        for proc_dir in data_dir.glob("**/"):
            if proc_dir.is_dir():
                try:
                    with open(proc_dir / "metadata.json", "r") as f:
                        metadata = json.load(f)
                    # check for matching metadata of processed datasets
                    if metadata == self.metadata:
                        logging.info(
                            "Found existing processed data with matching metadata. Loading..."
                        )
                        self.save_path = proc_dir / "data.pt"
                        found_existing = True
                        break
                except FileNotFoundError:
                    continue

        if not found_existing:
            logging.info("No existing processed data found. Processing...")
            dict_structures, y = self.src_check()
            data_list = self.get_data_list(dict_structures, y)
            data, slices = InMemoryDataset.collate(data_list)

            if save:
                if os.path.exists(self.pt_path):
                    logging.warn("Found existing processed data dir with same name, creating new dir.")
                    original_path = self.pt_path
                    idx = 1
                    while os.path.exists(original_path + "_" + str(idx)):
                        idx += 1
                        self.pt_path = original_path + "_" + str(idx)
                    else:
                        self.pt_path = original_path + "_" + str(idx)
                    os.mkdir(self.pt_path)
                if self.pt_path:
                    if not os.path.exists(self.pt_path):
                        os.mkdir(self.pt_path)
                    save_path = os.path.join(self.pt_path, "data.pt")
                torch.save((data, slices), save_path)
                # save metadata
                with open(os.path.join(self.pt_path, "metadata.json"), "w") as f:
                    json.dump(self.metadata, f)
                logging.info("Processed data saved successfully.")

    def get_data_list(self, dict_structures, y):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        # find the virtual nodes transform (workaround for now)
        transforms = [
            (i, t)
            for (i, t) in enumerate(
                wandb.config.get("transforms")
                if self.use_sweep_params
                else self.transforms
            )
        ]
        virtual_nodes_transform = None
        for i, t in transforms:
            if t.get("name") == "VirtualNodes":
                virtual_nodes_transform = t
                break

        for i, sdict in enumerate(tqdm(dict_structures, disable=self.disable_tqdm)):
            with PerfTimer() as t:
                target_val = y[i]
                data = data_list[i]

                pos = sdict["positions"]
                cell = sdict["cell"]
                cell2 = sdict["cell2"]
                atomic_numbers = sdict["atomic_numbers"]
                structure_id = sdict["structure_id"]

                data.o_pos = pos.clone()
                data.o_z = atomic_numbers.clone()

                # only apply basic preprocessing if needed
                # not used when using virtual nodes,
                # avoid unnecessary processing overhead
                if self.apply_pre_transform_processing:
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
                        remove_virtual_edges=False,
                        experimental_distance=False,
                        device=self.device,
                    )

                    edge_indices = edge_gen_out["edge_index"]
                    edge_weights = edge_gen_out["edge_weights"]
                    cell_offsets = edge_gen_out["cell_offsets"]
                    cell_offset_distances = edge_gen_out["offsets"]
                    edge_vec = edge_gen_out["edge_vec"]

                    data.neighbors = edge_gen_out["neighbors"]
                    data.cell_offsets = cell_offsets
                    data.cell_offset_distances = cell_offset_distances
                    data.edge_index, data.edge_weight = edge_indices, edge_weights
                    data.distances = edge_weights
                    data.edge_vec = edge_vec
                    data.edge_descriptor = {}
                    data.edge_descriptor["distance"] = edge_weights

                # virtual node generation (workaround for now)
                if virtual_nodes_transform:
                    vpos, virtual_z = generate_virtual_nodes(
                        cell2,
                        virtual_nodes_transform["args"].get(
                            "virtual_box_increment", 3.0
                        ),
                        self.device,
                    )
                    pos = torch.cat([pos, vpos], dim=0)
                    atomic_numbers = torch.cat([atomic_numbers, virtual_z], dim=0)

                data.n_atoms = len(atomic_numbers)
                data.pos = pos
                data.cell = cell
                data.cell2 = cell2
                data.y = torch.Tensor(np.array([target_val]))
                data.z = atomic_numbers
                data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
                data.structure_id = [[structure_id] * len(data.y)]

                # add additional attributes
                if self.additional_attributes:
                    for attr in self.additional_attributes:
                        data.__setattr__(attr, sdict[attr])

            if self.use_wandb:
                wandb.log({"process_times": t.elapsed})

        if self.apply_pre_transform_processing:
            logging.info("Generating node features...")
            generate_node_features(
                data_list,
                self.n_neighbors,
                device=self.device,
                use_degree=self.use_degree,
            )
            logging.info("Generating edge features...")
            generate_edge_features(
                data_list, self.edge_steps, self.r, device=self.device
            )

        # compile non-otf transforms
        logging.info("Applying transforms.")
        transforms_list = []
        for transform in self.transforms:
            # go through dict and overwrite with wandb config for sweep purposes
            if self.use_sweep_params:
                for key in transform["args"]:
                    if key in wandb.config:
                        transform["args"][key] = wandb.config[key]

            if not transform["otf"]:
                transforms_list.append(
                    registry.get_transform_class(
                        transform["name"],
                        # merge config parameters with global processing parameters
                        **{**transform.get("args", {}), **self.preprocess_kwargs},
                    )
                )
        composition = Compose(transforms_list)

        # apply transforms
        for i, data in enumerate(tqdm(data_list, disable=self.disable_tqdm)):
            with PerfTimer() as t:
                data_list[i] = composition(data)

            if self.use_wandb:
                wandb.log({"transforms_times": t.elapsed})

        clean_up(data_list, ["edge_descriptor"])

        return data_list
