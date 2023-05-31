import copy
import csv
import json
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
import random
import numpy as np

import psutil
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch_geometric
from torch_geometric.data import Dataset

import wandb
from matdeeplearn.common.data import (
    DataLoader,
    dataset_split,
    get_dataloader,
    get_dataset,
)
from matdeeplearn.common.registry import registry
from matdeeplearn.common.utils import min_alloc_gpu
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.modules.scheduler import LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        optimizer: Optimizer,
        sampler: DistributedSampler,
        scheduler: LRScheduler,
        data_loader: DataLoader,
        loss: nn.Module,
        max_epochs: int,
        max_checkpoint_epochs: int = None,
        identifier: str = None,
        verbosity: int = None,
        device: str = None,
        save_dir: str = None,
        checkpoint_path: str = None,
        wandb_config: dict = None,
        model_config: dict = None,
        opt_config: dict = None,
        dataset_config: dict = None,
    ):
        self.device = min_alloc_gpu(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_sampler = sampler
        self.data_loader = data_loader
        self.scheduler = scheduler
        self.loss_fn = loss
        self.max_epochs = max_epochs
        self.max_checkpoint_epochs = max_checkpoint_epochs
        self.train_verbosity = verbosity

        self.save_dir = save_dir
        self.wandb_config = wandb_config

        self.model_config = model_config
        self.opt_config = opt_config
        self.dataset_config = dataset_config

        # non passable params
        self.epoch = 0
        self.step = 0
        self.metrics = {}
        self.epoch_time = None
        self.best_metric = 1e10
        self.best_model_state = None

        self.save_dir = save_dir if save_dir else os.getcwd()
        self.checkpoint_path = checkpoint_path

        self.evaluator = Evaluator()

        if self.train_sampler == None:
            self.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.rank = self.train_sampler.rank

        timestamp = torch.tensor(datetime.now().timestamp()).to(self.device)
        self.timestamp_id = datetime.fromtimestamp(timestamp.int()).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        if identifier:
            self.timestamp_id = f"{identifier}-{self.timestamp_id}"

        self.identifier = identifier

        if self.train_verbosity:
            # MPS and CUDA support
            if self.device.type == "cuda":
                logging.info(
                    f"GPU is available: {torch.cuda.is_available()}, Quantity: {torch.cuda.device_count()}"
                )
                logging.info(
                    f"GPU: {self.device} ({torch.cuda.get_device_name(device)}), "
                    f"available memory: {1e-9 * torch.cuda.mem_get_info(device)[0]:3f} GB"
                )
            elif self.device.type == "cpu":
                logging.warning("Training on CPU, this will be slow")
                logging.info(f"available CPUs: {os.cpu_count()}")
                stats = psutil.virtual_memory()  # returns a named tuple
                available = getattr(stats, "available")
                logging.info(f"available memory: {1e-9 * available:3f} GB")
            elif self.device.type == "mps":
                logging.info("Training with MPS backend")

            logging.info(f"Dataset used: {self.dataset}")
            if self.dataset.get("train"):
                logging.debug(self.dataset["train"][0])
                logging.debug(self.dataset["train"][0].x[0])
                logging.debug(self.dataset["train"][0].y[0])
            else:
                logging.debug(self.dataset[list(self.dataset.keys())[0]][0])
                logging.debug(self.dataset[list(self.dataset.keys())[0]][0].x[0])
                logging.debug(self.dataset[list(self.dataset.keys())[0]][0].y[0])

            if str(self.rank) not in ("cpu", "cuda"):
                logging.debug(self.model.module)
            else:
                logging.debug(self.model)

    @classmethod
    def from_config(cls, config):
        """Class method used to initialize BaseTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
            scheduler
            dataset
        """
        dataset_config = config["dataset"]

        # find a matching dataset metadata signature
        metadata = dataset_config.get("preprocess_params", {})
        # find non-OTF transforms
        transforms = [
            t.get("args")
            for t in dataset_config.get("transforms", [])
            if not t.get("otf")
        ]
        for t_args in transforms:
            metadata.update(t_args)

        cls.set_seed(config["task"].get("seed"))

        if config["task"]["parallel"] == True:
            # os.environ["MASTER_ADDR"] = "localhost"
            # os.environ["MASTER_PORT"] = "12355"
            local_world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
            local_world_size = int(local_world_size)
            dist.init_process_group(
                "nccl", world_size=local_world_size, init_method="env://"
            )
            rank = int(dist.get_rank())
        else:
            rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_world_size = 1

        dataset = cls._load_dataset(config["dataset"], config["task"]["run_mode"])
        sweep_config = (
            wandb.config.get("hyperparams", None)
            if wandb.run and config["task"]["wandb"]["sweep"]["do_sweep"]
            else None
        )
        model = cls._load_model(
            config["model"],
            dataset["train"],
            sweep_config,
            local_world_size,
            rank,
        )
        optimizer = cls._load_optimizer(config["optim"], model, local_world_size)
        sampler = cls._load_sampler(dataset, local_world_size, rank)
        data_loader = cls._load_dataloader(
            config["optim"],
            dataset,
            sampler,
            config["task"]["run_mode"],
        )

        scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)
        loss = cls._load_loss(config["optim"]["loss"])
        max_epochs = config["optim"]["max_epochs"]
        verbosity = config["optim"].get("verbosity", None)
        max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
        identifier = config["task"].get("identifier", None)

        # pass in custom results home dir and load in prev checkpoint dir
        save_dir = config["task"].get("save_dir", None)

        if local_world_size > 1:
            dist.barrier()

        device = config["task"].get("gpu", None)

        # pass in custom results home dir and load in prev checkpoint dir
        save_dir = config["task"].get("save_dir", None)
        checkpoint_dir = config["task"].get("run_name", None)

        return cls(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            sampler=sampler,
            scheduler=scheduler,
            data_loader=data_loader,
            loss=loss,
            max_epochs=max_epochs,
            max_checkpoint_epochs=max_checkpoint_epochs,
            identifier=identifier,
            verbosity=verbosity,
            device=device,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
            wandb_config=config["task"].get("wandb"),
            model_config=config["model"],
            opt_config=config["optim"],
            dataset_config=config["dataset"],
        )

    @staticmethod
    def _load_dataset(dataset_config: dict, metadata: dict, task: str):
        """Loads the dataset if from a config file."""

        dataset_path = dataset_config["pt_path"]

        # search for a metadata match, else use provided path
        data_dir = pathlib.Path(dataset_path).parent
        found = False
        for proc_dir in data_dir.glob("**/"):
            if proc_dir.is_dir():
                try:
                    with open(proc_dir / "metadata.json", "r") as f:
                        found_metadata = json.load(f)
                    # check for matching metadata of processed datasets
                    if found_metadata == metadata:
                        logging.debug(
                            "Found dataset with matching metadata when attempting to load dataset. Loading..."
                        )
                        dataset_path = proc_dir
                        found = True
                        break
                except FileNotFoundError:
                    continue

        if not found:
            logging.info(
                "No existing processed dataset with matching metadata found. Defaulting to config..."
            )

        if not os.path.exists(os.path.join(dataset_path, "data.pt")):
            raise FileNotFoundError(
                f"Dataset path {dataset_path} does not exist. Specify processed=False in config to process data."
            )

        dataset = {}
        if isinstance(dataset_config["src"], dict):
            if dataset_config["src"].get("train"):
                dataset["train"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_train.pt",
                    transform_list=dataset_config.get("transforms", []),
                )
            if dataset_config["src"].get("val"):
                dataset["val"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_val.pt",
                    transform_list=dataset_config.get("transforms", []),
                )
            if dataset_config["src"].get("test"):
                dataset["test"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_test.pt",
                    transform_list=dataset_config.get("transforms", []),
                )
            if dataset_config["src"].get("predict"):
                dataset["predict"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_predict.pt",
                    transform_list=dataset_config.get("transforms", []),
                )

        else:
            if task != "predict":
                dataset_full = get_dataset(
                    dataset_path,
                    processed_file_name="data.pt",
                    transform_list=dataset_config.get("transforms", []),
                )
                train_ratio = dataset_config["train_ratio"]
                val_ratio = dataset_config["val_ratio"]
                test_ratio = dataset_config["test_ratio"]
                dataset["train"], dataset["val"], dataset["test"] = dataset_split(
                    dataset_full,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
            else:
                # if running in predict mode, then no data splitting is performed
                dataset["predict"] = get_dataset(
                    dataset_path,
                    processed_file_name="data.pt",
                    transform_list=dataset_config.get("transforms", []),
                )

        return dataset

    @staticmethod
    def _load_model(model_config, dataset, sweep_config, world_size, rank):
        """Loads the model if from a config file."""

        if dataset.get("train"):
            dataset = dataset["train"]
        else:
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

        # Obtain node, edge, and output dimensions for model initialization
        node_dim = dataset.num_features
        edge_dim = dataset.num_edge_features
        if dataset[0]["y"].ndim == 0:
            output_dim = 1
        else:
            output_dim = dataset[0]["y"].shape[1]

        # Determine if this is a node or graph level model
        if dataset[0]["y"].shape[0] == dataset[0]["x"].shape[0]:
            model_config["prediction_level"] = "node"
        elif dataset[0]["y"].shape[0] == 1:
            model_config["prediction_level"] = "graph"
        else:
            raise ValueError(
                "Target labels do not have the correct dimensions for node or graph-level prediction."
            )

        model_cls = registry.get_model_class(model_config["name"])
        model_params = model_config["hyperparams"] if not sweep_config else sweep_config
        model = model_cls(
            node_dim=node_dim, edge_dim=edge_dim, output_dim=output_dim, **model_params
        )
        model = model.to(rank)
        # model = torch_geometric.compile(model)
        # if model_config["load_model"] == True:
        #    checkpoint = torch.load(model_config["model_path"])
        #    model.load_state_dict(checkpoint["state_dict"])
        if world_size > 1:
            model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
            )
        return model

    @staticmethod
    def _load_optimizer(optim_config, model, world_size):
        # Some issues with DDP learning rate
        # Unclear regarding the best practice
        # Currently, effective batch size per epoch is batch_size * world_size
        # Some discussions here:
        # https://github.com/Lightning-AI/lightning/discussions/3706
        # https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15
        if world_size > 1:
            optim_config["lr"] = optim_config["lr"] * world_size

        optimizer = getattr(optim, optim_config["optimizer"]["optimizer_type"])(
            model.parameters(),
            lr=optim_config["lr"],
            **optim_config["optimizer"].get("optimizer_args", {}),
        )
        return optimizer

    @staticmethod
    def _load_sampler(dataset, world_size, rank):
        # TODO: write sampler, look into BalancedBatchSampler in
        #  OCP for their implementation of train_sampler batches
        #  (part of self.train_loader)
        # TODO: update sampler with more attributes like rank and num_replicas (world_size)
        if dataset.get("train"):
            dataset = dataset["train"]
        else:
            dataset = dataset[list(dataset.keys())[0]]

        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = None

        return sampler

    @staticmethod
    def _load_dataloader(optim_config, dataset, sampler, run_mode):
        data_loader = {}
        batch_size = optim_config.get("batch_size")
        if dataset.get("train"):
            data_loader["train_loader"] = get_dataloader(
                dataset["train"], batch_size=batch_size, sampler=sampler
            )
        if dataset.get("val"):
            data_loader["val_loader"] = get_dataloader(
                dataset["val"], batch_size=batch_size, sampler=None
            )
        if dataset.get("test"):
            data_loader["test_loader"] = get_dataloader(
                dataset["test"], batch_size=batch_size, sampler=None
            )
        if run_mode == "predict" and dataset.get("predict"):
            data_loader["predict_loader"] = get_dataloader(
                dataset["predict"], batch_size=batch_size, sampler=None
            )

        return data_loader

    @staticmethod
    def _load_scheduler(scheduler_config, optimizer):
        scheduler_type = scheduler_config["scheduler_type"]
        scheduler_args = scheduler_config["scheduler_args"]
        scheduler = LRScheduler(optimizer, scheduler_type, scheduler_args)
        return scheduler

    @staticmethod
    def _load_loss(loss_config):
        """Loads the loss from either the TorchLossWrapper or custom loss functions in matdeeplearn"""
        loss_cls = registry.get_loss_class(loss_config["loss_type"])
        # if there are other params for loss type, include in call
        if loss_config.get("loss_args"):
            return loss_cls(**loss_config["loss_args"])
        else:
            return loss_cls()

    @abstractmethod
    def _load_task(self):
        """Initializes task-specific info. Implemented by derived classes."""

    @abstractmethod
    def train(self):
        """Implemented by derived classes."""

    @abstractmethod
    def validate(self):
        """Implemented by derived classes."""

    @abstractmethod
    def predict(self):
        """Implemented by derived classes."""

    def update_best_model(self, metric):
        """Updates the best val metric and model, saves the best model, and saves the best model predictions"""
        self.best_metric = metric[type(self.loss_fn).__name__]["metric"]
        if str(self.rank) not in ("cpu", "cuda"):
            self.best_model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.save_model("best_checkpoint.pt", metric, True)

        logging.debug(
            f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/train_results/"
        )

        self.predict(self.data_loader["train_loader"], "train")
        if self.data_loader.get("val_loader"):
            self.predict(self.data_loader["val_loader"], "val")
        if self.data_loader.get("test_loader"):
            self.predict(self.data_loader["test_loader"], "test")

    def save_model(self, checkpoint_file, metric=None, training_state=True):
        """Saves the model state dict"""
        if str(self.rank) not in ("cpu", "cuda"):
            if training_state:
                state = {
                    "epoch": self.epoch,
                    "step": self.step,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.scheduler.state_dict(),
                    "best_metric": self.best_metric,
                    "identifier": self.timestamp_id,
                    "seed": torch.random.initial_seed(),
                }
            else:
                state = {"state_dict": self.model.module.state_dict(), "metric": metric}
        else:
            if training_state:
                state = {
                    "epoch": self.epoch,
                    "step": self.step,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.scheduler.state_dict(),
                    "best_metric": self.best_metric,
                    "identifier": self.timestamp_id,
                    "seed": torch.random.initial_seed(),
                }
            else:
                state = {"state_dict": self.model.state_dict(), "metric": metric}

        curr_checkpt_dir = os.path.join(
            self.save_dir, "results", self.timestamp_id, "checkpoint"
        )
        os.makedirs(curr_checkpt_dir, exist_ok=True)
        filename = os.path.join(curr_checkpt_dir, checkpoint_file)

        torch.save(state, filename)

        if wandb.run is not None:
            # No need to save the model to W&B at every step
            wandb.save(filename, policy="end")

        return filename

    def save_results(self, output, results_dir, filename, node_level_predictions=False):
        results_path = os.path.join(
            self.save_dir, "results", self.timestamp_id, results_dir
        )
        os.makedirs(results_path, exist_ok=True)
        filename = os.path.join(results_path, filename)
        shape = output.shape

        id_headers = ["structure_id"]
        if node_level_predictions:
            id_headers += ["node_id"]
        num_cols = (shape[1] - len(id_headers)) // 2
        headers = id_headers + ["target"] * num_cols + ["prediction"] * num_cols

        with open(filename, "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(output)):
                if i == 0:
                    csvwriter.writerow(headers)
                elif i > 0:
                    csvwriter.writerow(output[i - 1, :])
        return filename

    def load_checkpoint(self, load_training_state=True):
        """Loads the model from a checkpoint.pt file"""
        # Load params from checkpoint
        checkpoint_file = None
        if wandb.run and wandb.run.resumed:
            checkpoint_obj = wandb.restore("checkpoint.pt")
            if checkpoint_obj:
                checkpoint_file = checkpoint_obj.name
            else:
                logging.info(
                    "No checkpoint file found in W&B run history. Defaulting to local checkpoint file."
                )
        if not checkpoint_file:
            if not self.checkpoint_dir:
                raise ValueError("No checkpoint directory specified in config.")

            checkpoint_dir = os.path.join("results", self.checkpoint_dir)
            checkpoint_file = os.path.join(
                checkpoint_dir, "checkpoint", "checkpoint.pt"
            )

        # Load params from checkpoint
        checkpoint = torch.load(self.checkpoint_dir)

        if str(self.rank) not in ("cpu", "cuda"):
            self.model.module.load_state_dict(checkpoint["state_dict"])
            self.best_model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
            self.best_model_state = copy.deepcopy(self.model.state_dict())

        if load_training_state == True:
            if checkpoint.get("optimizer"):
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scheduler"):
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
                self.scheduler.update_lr()
            if checkpoint.get("epoch"):
                self.epoch = checkpoint["epoch"]
            if checkpoint.get("step"):
                self.step = checkpoint["step"]
            if checkpoint.get("best_metric"):
                self.best_metric = checkpoint["best_metric"]
            if checkpoint.get("seed"):
                seed = checkpoint["seed"]
                self.set_seed(seed)

            self._load_dataset

    @staticmethod
    def set_seed(seed):
        # https://pytorch.org/docs/stable/notes/randomness.html
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
