import json
import logging
import os
import pathlib
from typing import Union

import numpy as np
import pandas as pd
import torch
import wandb
from ase import io
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from matdeeplearn.common.graph_data import CustomBatchingData
from matdeeplearn.common.registry import registry
from matdeeplearn.common.utils import DictTools
from matdeeplearn.preprocessor.helpers import (
    PerfTimer,
    calculate_edges_master,
    clean_up,
    generate_edge_features,
    generate_node_features,
)


def from_config(dataset_config, wandb_config):
    use_wandb = wandb_config.get("use_wandb", False)

    # modify config to reflect sweep parameters if being run
    if use_wandb and wandb_config["sweep"].get("do_sweep", False):
        sweep_params = wandb_config["sweep"].get("params", {})
        for key in sweep_params:
            DictTools._mod_recurse(dataset_config, key, wandb.config.get(key, None))

    preprocess_kwargs = dataset_config["preprocess_params"]

    pt_path = dataset_config.get("pt_path", None)
    cutoff_radius = preprocess_kwargs["cutoff_radius"]
    n_neighbors = preprocess_kwargs["n_neighbors"]
    num_offsets = preprocess_kwargs["num_offsets"]
    edge_steps = preprocess_kwargs["edge_steps"]
    root_path_dict = dataset_config["src"]
    target_file_path_dict = dataset_config["target_path"]
    pt_path = dataset_config.get("pt_path", None)
    prediction_level = dataset_config.get("prediction_level", "graph")
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
    all_neighbors = preprocess_kwargs["all_neighbors"]
    use_degree = preprocess_kwargs["use_degree"]
    edge_calc_method = preprocess_kwargs.get("edge_calc_method", "mdl")
    apply_pre_transform_processing = dataset_config.get(
        "apply_pre_transform_processing", True
    )
    batch_process = dataset_config.get("batch_process", False)
    device: str = dataset_config.get("device", "cpu")

    processor = DataProcessor(
        root_path_dict=root_path_dict,
        target_file_path_dict=target_file_path_dict,
        pt_path=pt_path,
        prediction_level=prediction_level,
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
        batch_process=batch_process,
        batch_size=preprocess_kwargs.get("process_batch_size", 1),
        use_wandb=use_wandb,
        preprocess_kwargs=preprocess_kwargs,
        force_preprocess=dataset_config.get("force_preprocess", False),
        num_examples=dataset_config.get("num_examples", 0),
        device=device,
    )

    return processor


def process_data(dataset_config, wandb_config):
    processor = from_config(dataset_config, wandb_config)
    dataset = processor.process()

    return dataset


class DataProcessor:
    def __init__(
        self,
        root_path_dict: Union[str, dict],
        target_file_path_dict: Union[str, dict],
        pt_path: str,
        prediction_level: str,
        r: float,
        n_neighbors: int,
        num_offsets: int,
        edge_steps: int,
        transforms: dict = {},
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
        batch_size: int = 1,
        batch_process: bool = False,
        use_wandb: bool = False,
        preprocess_kwargs: dict = {},
        force_preprocess: bool = False,
        num_examples: int = None,
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

            transforms: dict
                default {}. index-dict of transforms to apply to the data.

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

        self.root_path_dict = root_path_dict
        self.target_file_path_dict = target_file_path_dict
        self.pt_path = pt_path
        self.r = r
        self.prediction_level = prediction_level
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
        self.batch_process = batch_process
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        self.preprocess_kwargs = preprocess_kwargs
        self.device = device
        self.transforms = transforms
        self.disable_tqdm = logging.root.level > logging.INFO

        self.force_preprocess = force_preprocess
        self.num_examples = num_examples

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
        self.num_examples = (
            len(file_names) if self.num_examples == 0 else self.num_examples
        )
        for i, structure_id in enumerate(file_names[: self.num_examples]):
            p = os.path.join(self.root_path, str(structure_id) + "." + self.data_format)
            ase_structures.append(io.read(p))

        for i, s in enumerate(tqdm(ase_structures, disable=self.disable_tqdm)):
            d = {}
            pos = torch.tensor(s.get_positions(), device=self.device, dtype=torch.float)
            cell = torch.tensor(
                np.array(s.get_cell()), device=self.device, dtype=torch.float
            ).view(1, 3, 3)
            if (
                np.array(cell)
                == np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            ).all():
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

        logging.info(
            "(JSON) Converting data to standardized form for downstream processing."
        )
        self.num_examples = (
            len(original_structures) if self.num_examples == 0 else self.num_examples
        )
        for i, s in enumerate(
            tqdm(original_structures[: self.num_examples], disable=self.disable_tqdm)
        ):
            d = {}
            if len(s["atomic_numbers"]) == 1:
                continue
            pos = torch.tensor(s["positions"], device=self.device, dtype=torch.float)
            cell2 = None
            if "cell2" in s:
                cell2 = s["cell2"]
            if "cell" in s:
                cell = torch.tensor(s["cell"], device=self.device, dtype=torch.float)
                if cell.shape[0] != 1:
                    cell = cell.view(1, 3, 3)
            else:
                cell = torch.zeros((3, 3)).unsqueeze(0)
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

        return dict_structures

    def process_split(self, data_list: dict, split: str, save: bool):
        self.root_path = (
            self.root_path_dict[split]
            if isinstance(self.root_path_dict, dict)
            else self.root_path_dict
        )

        logging.info("Working on split {}".format(split))
        logging.info(f"{split} dataset found at {self.root_path}")
        logging.info("Processing device: {}".format(self.device))
        logging.info("Checking for existing processed data with matching metadata...")

        if self.target_file_path_dict:
            self.target_file_path = (
                self.target_file_path_dict[split]
                if isinstance(self.target_file_path_dict, dict)
                else self.target_file_path_dict
            )
        else:
            self.target_file_path = self.target_file_path_dict

        found_existing = False
        data_dir = pathlib.Path(self.pt_path).parent
        # If forcing preprocess, we ignore the metadata search procedure
        if not self.force_preprocess:
            for proc_dir in data_dir.glob("**/"):
                if proc_dir.is_dir():
                    try:
                        with open(proc_dir / "metadata.json", "r") as f:
                            metadata = json.load(f)
                        # check for matching metadata of processed datasets
                        if metadata == self.metadata:
                            logging.info(
                                f"Found existing processed data with matching metadata ({proc_dir}), skipping processing. Loading..."
                            )
                            self.save_path = proc_dir / f"data_{split}.pt"
                            found_existing = True
                            break
                    except FileNotFoundError:
                        continue
        else:
            logging.info("Forcing preprocessing of data.")

        if not found_existing:
            logging.info("No existing processed data found. Processing...")
            dict_structures = self.src_check()
            data_list[split] = self.get_data_list(dict_structures)
            data, slices = InMemoryDataset.collate(data_list[split])

            if save:
                if os.path.exists(self.pt_path):
                    logging.warn(
                        "Found existing processed data dir with same name, creating new dir."
                    )
                    original_path = self.pt_path
                    idx = 1
                    while os.path.exists(original_path + "_" + str(idx)):
                        idx += 1
                        self.pt_path = original_path + "_" + str(idx)
                    else:
                        self.pt_path = original_path + "_" + str(idx)
                    logging.debug(f"New processed data dir: {self.pt_path}")
                    os.makedirs(self.pt_path)

                # save processed data
                if self.pt_path:
                    if not os.path.exists(self.pt_path):
                        os.makedirs(self.pt_path)
                    save_path = os.path.join(self.pt_path, f"data_{split}.pt")
                torch.save((data, slices), save_path)

                # save metadata
                with open(os.path.join(self.pt_path, "metadata.json"), "w") as f:
                    json.dump(self.metadata, f)
                logging.info(f"Processed {split} data saved successfully.")

    def process(self, save=True):
        data_list = {}
        if isinstance(self.root_path_dict, dict):
            if self.root_path_dict.get("train"):
                self.process_split(data_list, "train", save)

            if self.root_path_dict.get("val"):
                self.process_split(data_list, "val", save)

            if self.root_path_dict.get("test"):
                self.process_split(data_list, "test", save)

            if self.root_path_dict.get("predict"):
                self.process_split(data_list, "predict", save)

        else:
            self.process_split(data_list, "full", save)

        return data_list

    def get_data_list(self, dict_structures):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        for i, sdict in enumerate(tqdm(dict_structures, disable=self.disable_tqdm)):
            with PerfTimer() as perf_timer:
                # target_val = y[i]
                data = data_list[i]

                pos = sdict["positions"]
                cell = sdict["cell"]
                cell2 = sdict.get("cell2", None)
                atomic_numbers = sdict["atomic_numbers"]
                structure_id = sdict["structure_id"]
                target_val = sdict["y"]

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

                    if edge_vec.dim() > 2:
                        edge_vec = edge_vec[edge_indices[0], edge_indices[1]]

                    data.edge_index, data.edge_weight = edge_indices, edge_weights
                    data.edge_vec = edge_vec
                    data.cell_offsets = cell_offsets
                    data.neighbors = neighbors

                    data.edge_descriptor = {}
                    # data.edge_descriptor["mask"] = cd_matrix_masked
                    data.edge_descriptor["distance"] = edge_weights
                    data.distances = edge_weights

                data.n_atoms = len(atomic_numbers)
                data.pos = pos
                data.cell = cell
                data.cell2 = cell2

                # Data.y.dim()should equal 2, with dimensions of either (1, n) for graph-level labels or (n_atoms, n) for node level labels, where n is length of label vector (usually n=1)
                data.y = torch.Tensor(np.array(target_val))
                if self.prediction_level == "graph":
                    if data.y.dim() > 1 and data.y.shape[0] != 0:
                        raise ValueError(
                            "Target labels do not have the correct dimensions for graph-level prediction."
                        )
                    elif data.y.dim() == 1:
                        data.y = data.y.unsqueeze(0)
                elif self.prediction_level == "node":
                    if data.y.shape[0] != data.n_atoms:
                        raise ValueError(
                            "Target labels do not have the correct dimensions for node-level prediction."
                        )
                    elif data.y.dim() == 1:
                        data.y = data.y.unsqueeze(1)
                # print(data.y.shape)

                data.z = atomic_numbers
                data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
                # data.structure_id = [[structure_id] * len(data.y)]
                data.structure_id = [structure_id]

                # add additional attributes
                if self.additional_attributes:
                    for attr in self.additional_attributes:
                        data.__setattr__(attr, sdict[attr])

            if self.use_wandb:
                wandb.log({"process_times": perf_timer.elapsed})

        # compile non-otf transforms
        logging.debug("Applying transforms.")

        # Ensure GetY exists to prevent downstream model errors
        assert "GetY" in [
            tf["name"] for tf in self.transforms
        ], "The target transform GetY is required in config."

        # compile transforms
        transforms_list_unbatched = []
        transforms_list_batched = []

        for transform in self.transforms:
            entry = registry.get_transform_class(
                transform["name"],
                # merge config parameters with global processing parameters
                **{**transform.get("args", {}), **self.preprocess_kwargs},
            )
            if not transform["otf"] and not transform.get("batch", False):
                transforms_list_unbatched.append(entry)
            elif transform.get("batch", False):
                transforms_list_batched.append(entry)

        composition_unbatched = Compose(transforms_list_unbatched)
        composition_batched = Compose(transforms_list_batched)

        # perform unbatched transforms
        logging.info("Applying non-batch transforms...")
        for i, data in enumerate(tqdm(data_list, disable=self.disable_tqdm)):
            with PerfTimer() as perf_timer:
                data_list[i] = composition_unbatched(data)
            if self.use_wandb:
                wandb.log({"transforms_times": perf_timer.elapsed})

        # convert to custom data object
        for i, data in enumerate(data_list):
            data_list[i] = CustomBatchingData.from_dict(data.to_dict())

        if len(transforms_list_batched) > 0:
            # perform batch transforms
            logging.info(
                f"Applying batch transforms with batch size {self.batch_size}..."
            )
            for i in tqdm(
                range(0, len(data_list), self.batch_size),
                disable=self.disable_tqdm,
            ):
                batch = Batch.from_data_list(data_list[i : i + self.batch_size])
                # apply transforms
                batch = composition_batched(batch)
                # convert back to list of Data() objects
                data_list[i : i + self.batch_size] = batch.to_data_list()

        clean_up(data_list, ["edge_descriptor"])

        return data_list
