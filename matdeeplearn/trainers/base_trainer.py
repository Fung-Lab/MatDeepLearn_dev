import copy
import csv
import glob
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch_geometric
from torch_geometric.data import Dataset

from matdeeplearn.common.data import (
    DataLoader,
    dataset_split,
    get_dataloader,
    get_dataset,
)
from matdeeplearn.common.registry import registry
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
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        loss: nn.Module,
        max_epochs: int,
        max_checkpoint_epochs: int = None,
        identifier: str = None,
        verbosity: int = None,
        save_dir: str = None,
        checkpoint_dir: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_sampler = sampler
        self.train_loader, self.val_loader, self.test_loader = (
            train_loader,
            val_loader,
            test_loader,
        )
        self.scheduler = scheduler
        self.loss_fn = loss
        self.max_epochs = max_epochs
        self.max_checkpoint_epochs = max_checkpoint_epochs
        self.train_verbosity = verbosity

        self.epoch = 0
        self.step = 0
        self.metrics = {}
        self.epoch_time = None
        self.best_val_metric = 1e10
        self.best_model_state = None

        self.save_dir = save_dir if save_dir else os.getcwd()
        self.checkpoint_dir = checkpoint_dir

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
            self.timestamp_id = f"{self.timestamp_id}-{identifier}"

        if self.train_verbosity:
            logging.info(
                f"GPU is available: {torch.cuda.is_available()}, Quantity: {os.environ.get('LOCAL_WORLD_SIZE', None)}"
            )
            logging.info(f"Dataset used: {self.dataset}")
            logging.debug(self.dataset["train"][0])
            logging.debug(self.dataset["train"][0].x[0])
            logging.debug(self.dataset["train"][0].x[-1])
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

        
        if config["task"]["parallel"] == True:
            #os.environ["MASTER_ADDR"] = "localhost"
            #os.environ["MASTER_PORT"] = "12355"
            local_world_size = os.environ.get('LOCAL_WORLD_SIZE', None)
            local_world_size= int(local_world_size)
            dist.init_process_group("nccl", world_size=local_world_size, init_method='env://')    
            rank = int(dist.get_rank())            
        else:        
            rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_world_size = 1
            
        dataset = cls._load_dataset(config["dataset"], config["task"]["seed"])
        model = cls._load_model(config["model"], dataset["train"], local_world_size, rank)
        optimizer = cls._load_optimizer(config["optim"], model, local_world_size)
        sampler = cls._load_sampler(config["optim"], dataset["train"], local_world_size, rank)
        train_loader, val_loader, test_loader = cls._load_dataloader(
            config["optim"], config["dataset"], dataset, sampler,
        )               

        scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)
        loss = cls._load_loss(config["optim"]["loss"])
        max_epochs = config["optim"]["max_epochs"]
        max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
        identifier = config["task"].get("identifier", None)
        verbosity = config["task"].get("verbosity", None)
        # pass in custom results home dir and load in prev checkpoint dir
        save_dir = config["task"].get("save_dir", None)
        checkpoint_dir = config["task"].get("checkpoint_dir", None)
        
        if local_world_size > 1:
            dist.barrier()

        return cls(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            sampler=sampler,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss=loss,
            max_epochs=max_epochs,
            max_checkpoint_epochs=max_checkpoint_epochs,
            identifier=identifier,
            verbosity=verbosity,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
        )

    @staticmethod
    def _load_dataset(dataset_config, seed):
        """Loads the dataset if from a config file."""

        dataset_path = dataset_config["pt_path"]
        dataset={}
        if isinstance(dataset_config["src"], dict):
            dataset["train"] = get_dataset(
                dataset_path,
                processed_file_name="data_train.pt",
                transform_list=dataset_config.get("transforms", []),
            )
            dataset["val"] = get_dataset(
                dataset_path,
                processed_file_name="data_val.pt",
                transform_list=dataset_config.get("transforms", []),
            )
            dataset["test"] = get_dataset(
                dataset_path,
                processed_file_name="data_test.pt",
                transform_list=dataset_config.get("transforms", []),
            )

        else:
            dataset["train"] = get_dataset(

                dataset_path,
                processed_file_name="data.pt",
                transform_list=dataset_config.get("transforms", []),
            )
            train_ratio = dataset_config["train_ratio"]
            val_ratio = dataset_config["val_ratio"]
            test_ratio = dataset_config["test_ratio"]
            dataset["train"], dataset["val"], dataset["test"] = dataset_split(
                dataset_full, train_ratio, val_ratio, test_ratio, seed,
            )       
            
        return dataset

    @staticmethod
    def _load_model(model_config, dataset, world_size, rank):
        """Loads the model if from a config file."""
        
        model_cls = registry.get_model_class(model_config["name"])
        model = model_cls(data=dataset, **model_config)
        model = model.to(rank)
        #model = torch_geometric.compile(model)
        #if model_config["load_model"] == True:
        #    checkpoint = torch.load(model_config["model_path"])
        #    model.load_state_dict(checkpoint["state_dict"])
        if world_size > 1:            
            model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
            )
                        

        return model

    @staticmethod
    def _load_optimizer(optim_config, model, world_size):
        #Some issues with DDP learning rate
        #Unclear regarding the best practice
        #Currently, effective batch size per epoch is batch_size * world_size
        #Some discussions here:
        #https://github.com/Lightning-AI/lightning/discussions/3706
        #https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15
        if world_size > 1:
            optim_config["lr"] = optim_config["lr"] * world_size
        optimizer = getattr(optim, optim_config["optimizer"]["optimizer_type"])(
            model.parameters(),
            lr=optim_config["lr"],
            **optim_config["optimizer"].get("optimizer_args", {}),
        )
        return optimizer

    @staticmethod
    def _load_sampler(optim_config, dataset, world_size, rank):
        # TODO: write sampler, look into BalancedBatchSampler in
        #  OCP for their implementation of train_sampler batches
        #  (part of self.train_loader)
        # TODO: update sampler with more attributes like rank and num_replicas (world_size)
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank
            )
        else:
            sampler = None        
        return sampler

    @staticmethod
    def _load_dataloader(optim_config, dataset_config, dataset, sampler):

        batch_size = optim_config.get("batch_size")
        if isinstance(dataset_config["src"], dict):
            train_loader = get_dataloader(
                dataset["train"], batch_size=batch_size, sampler=sampler
            )
            val_loader = get_dataloader(dataset["val"], batch_size=batch_size, sampler=sampler)
            test_loader = get_dataloader(
                dataset["test"], batch_size=batch_size, sampler=sampler
            )

        else:
            train_ratio = dataset_config["train_ratio"]
            val_ratio = dataset_config["val_ratio"]
            test_ratio = dataset_config["test_ratio"]
            train_dataset, val_dataset, test_dataset = dataset_split(
                dataset["train"], train_ratio, val_ratio, test_ratio
            )

            train_loader = get_dataloader(
                train_dataset, batch_size=batch_size, sampler=sampler
            )
            val_loader = get_dataloader(val_dataset, batch_size=batch_size, sampler=sampler)
            test_loader = get_dataloader(
                test_dataset, batch_size=batch_size, sampler=sampler
            )

        return train_loader, val_loader, test_loader

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

    def update_best_model(self, val_metrics):
        """Updates the best val metric and model, saves the best model, and saves the best model predictions"""
        self.best_val_metric = val_metrics[type(self.loss_fn).__name__]["metric"]
        if str(self.rank) not in ("cpu", "cuda"): 
            self.best_model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.save_model("best_checkpoint.pt", val_metrics, False)

        logging.debug(
            f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/train_results/"
        )

        self.predict(self.train_loader, "train")
        self.predict(self.val_loader, "val")
        self.predict(self.test_loader, "test")

    def save_model(self, checkpoint_file, val_metrics=None, training_state=True):
        """Saves the model state dict"""

        
        if str(self.rank) not in ("cpu", "cuda"): 
            if training_state:
                state = {
                    "epoch": self.epoch,
                    "step": self.step,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.scheduler.state_dict(),
                    "best_val_metric": self.best_val_metric,
                    "identifier": self.timestamp_id,
                }
            else:
                state = {"state_dict": self.model.module.state_dict(), "val_metrics": val_metrics}
      
        curr_checkpt_dir = os.path.join(
            self.save_dir, "results", self.timestamp_id, "checkpoint"
        )
        os.makedirs(curr_checkpt_dir, exist_ok=True)
        filename = os.path.join(curr_checkpt_dir, checkpoint_file)

        torch.save(state, filename)
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

    # TODO: streamline this from PR #12
    def load_checkpoint(self):
        """Loads the model from a checkpoint.pt file"""

        if not self.checkpoint_dir:
            raise ValueError("No checkpoint directory specified in config.")

        #checkpoint_dir = glob.glob(os.path.join(self.checkpoint_dir, "results", "*"))
        #checkpoint_file = os.path.join(checkpoint_dir, "checkpoint", "checkpoint.pt")

        # Load params from checkpoint
        checkpoint = torch.load(self.checkpoint_dir)
        
        if str(self.rank) not in ("cpu", "cuda"): 
            self.model.module.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_metric = checkpoint["best_val_metric"]
