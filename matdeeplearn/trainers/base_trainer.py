import copy
import csv
import logging
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset

from matdeeplearn.common.data import (DataLoader, dataset_split,
                                      get_dataloader, get_dataset)
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
        data_loader: DataLoader,        
        loss: nn.Module,
        max_epochs: int,
        clip_grad_norm: float = None,
        max_checkpoint_epochs: int = None,
        identifier: str = None,
        verbosity: int = None,
        batch_tqdm: bool = False,        
        write_output: list = ["train", "val", "test"],
        output_frequency: int = 1,
        model_save_frequency: int = 1,
        save_dir: str = None,
        checkpoint_path: str = None,
        use_amp: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_sampler = sampler
        self.data_loader = data_loader
        self.scheduler = scheduler
        self.loss_fn = loss
        self.max_epochs = max_epochs
        self.clip_grad_norm = clip_grad_norm
        self.max_checkpoint_epochs = max_checkpoint_epochs
        self.train_verbosity = verbosity
        self.batch_tqdm = batch_tqdm
        self.write_output = write_output
        self.output_frequency = output_frequency
        self.model_save_frequency = model_save_frequency

        self.epoch = 0
        self.step = 0
        self.epoch_time = None

        self.metrics = [{} for _ in range(len(self.model))]  
        self.best_metric = [1e10 for _ in range(len(self.model))]
        self.best_model_state = [None for _ in range(len(self.model))]

        self.save_dir = save_dir if save_dir else os.getcwd()
        self.checkpoint_path = checkpoint_path
        self.use_amp = use_amp

        if self.use_amp:
            logging.info("Using PyTorch automatic mixed-precision")

        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        self.evaluator = Evaluator()

        if self.train_sampler == None:
            self.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.rank = self.train_sampler.rank

        timestamp = datetime.now().timestamp()
        self.timestamp_id = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )[:-3]
        if identifier:
            self.timestamp_id = f"{self.timestamp_id}-{identifier}"

        if self.train_verbosity:
            logging.info(
                f"GPU is available: {torch.cuda.is_available()}, Quantity: {os.environ.get('LOCAL_WORLD_SIZE', None)}"
            )
            if not (self.dataset is None):
                logging.info("Dataset(s) used:")
                for key in self.dataset:
                    logging.info(f"Dataset length: {key, len(self.dataset[key])}")
                if self.dataset.get("train"):
                    logging.debug(self.dataset["train"][0])
                    logging.debug(self.dataset["train"][0].z[0])
                    logging.debug(self.dataset["train"][0].y[0])
                else:
                    logging.debug(self.dataset[list(self.dataset.keys())[0]][0])
                    logging.debug(self.dataset[list(self.dataset.keys())[0]][0].x[0])
                    logging.debug(self.dataset[list(self.dataset.keys())[0]][0].y[0])

            if str(self.rank) not in ("cpu", "cuda"):
                logging.debug(self.model[0].module)
            else:
                logging.debug(self.model[0])

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
        dataset = cls._load_dataset(config["dataset"], config["task"]["run_mode"]) if "src" in config["dataset"] else None
        model = cls._load_model(config["model"], config["dataset"]["preprocess_params"], dataset, local_world_size, rank)
        optimizer = cls._load_optimizer(config["optim"], model, local_world_size)
        sampler = cls._load_sampler(config["optim"], dataset, local_world_size, rank) if "src" in config["dataset"] else None
        data_loader = cls._load_dataloader(
            config["optim"],
            config["dataset"],
            dataset,
            sampler,
            config["task"]["run_mode"],
            config["model"]
        ) if "src" in config["dataset"] else None

        scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)
        loss = cls._load_loss(config["optim"]["loss"])
        max_epochs = config["optim"]["max_epochs"]
        clip_grad_norm = config["optim"].get("clip_grad_norm", None)
        verbosity = config["optim"].get("verbosity", None)
        batch_tqdm = config["optim"].get("batch_tqdm", False)
        write_output = config["task"].get("write_output", [])
        output_frequency = config["task"].get("output_frequency", 0)
        model_save_frequency = config["task"].get("model_save_frequency", 0)
        max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
        identifier = config["task"].get("identifier", None)

        # pass in custom results home dir and load in prev checkpoint dir
        save_dir = config["task"].get("save_dir", None)
        checkpoint_path = config["task"].get("checkpoint_path", None)

        if local_world_size > 1:
            dist.barrier()
            
        return cls(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            sampler=sampler,
            scheduler=scheduler,
            data_loader=data_loader,
            loss=loss,
            max_epochs=max_epochs,
            clip_grad_norm=clip_grad_norm,
            max_checkpoint_epochs=max_checkpoint_epochs,
            identifier=identifier,
            verbosity=verbosity,
            batch_tqdm=batch_tqdm,
            write_output=write_output,
            output_frequency=output_frequency,
            model_save_frequency=model_save_frequency,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path,
            use_amp=config["task"].get("use_amp", False),
        )

    @staticmethod
    def _load_dataset(dataset_config, task):
        """Loads the dataset if from a config file."""
        if dataset_config.get("dataset_device", "cpu"):
            logging.info("Loading dataset to "+dataset_config.get("dataset_device", "cpu"))
        else:
            logging.info("Loading dataset to default device")
        
        dataset_path = dataset_config["pt_path"]
        dataset = {}
        if isinstance(dataset_config["src"], dict):
            if dataset_config["src"].get("train"):
                dataset["train"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_train.pt",
                    transform_list=dataset_config.get("transforms", []),
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
                )
            if dataset_config["src"].get("val"):
                dataset["val"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_val.pt",
                    transform_list=dataset_config.get("transforms", []),
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
                )
            if dataset_config["src"].get("test"):
                dataset["test"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_test.pt",
                    transform_list=dataset_config.get("transforms", []),
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
                )                
            if dataset_config["src"].get("predict"):
                dataset["predict"] = get_dataset(
                    dataset_path,
                    processed_file_name="data_predict.pt",
                    transform_list=dataset_config.get("transforms", []),
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
                )

        else:
            if task != "predict":
                dataset_full = get_dataset(
                    dataset_path,
                    processed_file_name="data.pt",
                    transform_list=dataset_config.get("transforms", []),
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
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
                    dataset_device=dataset_config.get("dataset_device", "cpu"),
                )

        return dataset

    @staticmethod
    def _load_model(model_config, graph_config, dataset, world_size, rank):
        """Loads the model if from a config file."""

        if not (dataset is None):
            if dataset.get("train"):
                dataset = dataset["train"]
            else:
                dataset = dataset[list(dataset.keys())[0]]
                    
        if isinstance(dataset, torch.utils.data.Subset): 
            dataset = dataset.dataset 
        
        model_list = []
        # Obtain node, edge, and output dimensions for model initialization   
        for mod in range(model_config["model_ensemble"]): 
            rand_seed = random.randint(1,10000)
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            #torch.autograd.set_detect_anomaly(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if graph_config["node_dim"]:
                node_dim = graph_config["node_dim"]
            else:
                node_dim = dataset.num_features
            edge_dim = graph_config["edge_dim"]
            if not (dataset is None):
                if dataset[0]["y"].ndim == 0:
                    output_dim = 1
                else:
                    output_dim = dataset[0]["y"].shape[1]
            else:
                output_dim = graph_config["output_dim"]

            # Determine if this is a node or graph level model
            if not (dataset is None):
                if dataset[0]["y"].shape[0] == dataset[0]["z"].shape[0]:
                    model_config["prediction_level"] = "node"
                elif dataset[0]["y"].shape[0] == 1:
                    model_config["prediction_level"] = "graph"
                else:
                    raise ValueError(
                        "Target labels do not have the correct dimensions for node or graph-level prediction."
                    )
            else:
                model_config["prediction_level"] = graph_config["prediction_level"]

            model_cls = registry.get_model_class(model_config["name"])
            model = model_cls(
                    node_dim=node_dim, 
                    edge_dim=edge_dim, 
                    output_dim=output_dim, 
                    cutoff_radius=graph_config["cutoff_radius"], 
                    n_neighbors=graph_config["n_neighbors"], 
                    graph_method=graph_config["edge_calc_method"], 
                    num_offsets=graph_config["num_offsets"], 
                    **model_config
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

            model_list.append(model)

        return model_list

    @staticmethod
    def _load_optimizer(optim_config, model, world_size):
        # Some issues with DDP learning rate
        # Unclear regarding the best practice
        # Currently, effective batch size per epoch is batch_size * world_size
        # Some discussions here:
        # https://github.com/Lightning-AI/lightning/discussions/3706
        # https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15
        optim_list = []
        for i in range(len(model)):
            if world_size > 1:
                optim_config["lr"] = optim_config["lr"] * world_size

            optimizer = getattr(optim, optim_config["optimizer"]["optimizer_type"])(
                model[i].parameters(),
                lr=optim_config["lr"],
                **optim_config["optimizer"].get("optimizer_args", {}),
            )
            optim_list.append(optimizer)

        return optim_list

    @staticmethod
    def _load_sampler(optim_config, dataset, world_size, rank):
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
    def _load_dataloader(optim_config, dataset_config, dataset, sampler, run_mode, model_config):
        data_loader = [{} for _ in range(model_config["model_ensemble"])]
    
        batch_size = optim_config.get("batch_size")
        
        for i in range(model_config["model_ensemble"]):
            if dataset.get("train"):
                data_loader[i]["train_loader"] = get_dataloader(
                    dataset["train"], batch_size=batch_size, num_workers=dataset_config.get("num_workers", 0), sampler=sampler
                )
            if dataset.get("val"):
                data_loader[i]["val_loader"] = get_dataloader(
                    dataset["val"], batch_size=batch_size, num_workers=dataset_config.get("num_workers", 0), sampler=None
                )
            if dataset.get("test"):
                data_loader[i]["test_loader"] = get_dataloader(
                    dataset["test"], batch_size=batch_size, num_workers=dataset_config.get("num_workers", 0), sampler=None
            )
            if run_mode == "predict" and dataset.get("predict"):
                data_loader[i]["predict_loader"] = get_dataloader(
                    dataset["predict"], batch_size=batch_size, num_workers=dataset_config.get("num_workers", 0), sampler=None, shuffle=True
            )

        return data_loader

    @staticmethod
    def _load_scheduler(scheduler_config, optimizer):
        scheduler_type = scheduler_config["scheduler_type"]
        scheduler_args = scheduler_config["scheduler_args"]
        scheduler = []
        for i in range(len(optimizer)):
            scheduler.append(LRScheduler(optimizer[i], scheduler_type, scheduler_args))
        
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

    def update_best_model(self, metric, index=None, write_model=False, write_csv=False):
        """Updates the best val metric and model, saves the best model, and saves the best model predictions"""
        self.best_metric[index] = metric[type(self.loss_fn).__name__]["metric"]
        
        if str(self.rank) not in ("cpu", "cuda"):
            self.best_model_state[index] = copy.deepcopy(self.model[index].module.state_dict())
        else:
            self.best_model_state[index] = copy.deepcopy(self.model[index].state_dict())
        
        if write_model == True:
            self.save_model("best_checkpoint.pt", index, metric, True)
            
        if write_csv == True:
            logging.debug(
                f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/train_results/"
            )
            
            if "train" in self.write_output:
                self.predict(self.data_loader[index]["train_loader"], "train")
            if "val" in self.write_output and self.data_loader[index].get("val_loader"):
                self.predict(self.data_loader[index]["val_loader"], "val")
            if "test" in self.write_output and self.data_loader[index].get("test_loader"):
                self.predict(self.data_loader[index]["test_loader"], "test")

    def save_model(self, checkpoint_file, index=None, metric=None, training_state=True):
        """Saves the model state dict"""
        if index != None:
            if str(self.rank) not in ("cpu", "cuda"):
                if training_state:
                    state = {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model[index].module.state_dict(),
                        "optimizer": self.optimizer[index].state_dict(),
                        "scheduler": self.scheduler[index].scheduler.state_dict(),
                        "scaler":self.scaler.state_dict(),
                        "best_metric": self.best_metric[index],
                        "identifier": self.timestamp_id,
                        "seed": torch.random.initial_seed(),
                    }
                else:
                    state = {"state_dict": self.model[index].module.state_dict(), "metric": metric}
            else:
                if training_state:
                    state = {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model[index].state_dict(),
                        "optimizer": self.optimizer[index].state_dict(),
                        "scheduler": self.scheduler[index].scheduler.state_dict(),
                        "scaler":self.scaler.state_dict(),
                        "best_metric": self.best_metric[index],
                        "identifier": self.timestamp_id,
                        "seed": torch.random.initial_seed(),
                    }
                else:
                    state = {"state_dict": self.model[index].state_dict(), "metric": metric}
            
            num = str(index)
            curr_checkpt_dir = os.path.join(
                self.save_dir, "results", self.timestamp_id, f"checkpoint_{num}"
            )
            os.makedirs(curr_checkpt_dir, exist_ok=True)
            filename = os.path.join(curr_checkpt_dir, checkpoint_file)

            torch.save(state, filename)
            del state
        else:
            state = []
            for i in range(len(self.model)):
                if str(self.rank) not in ("cpu", "cuda"):
                    if training_state:
                        state.append({
                            "epoch": self.epoch,
                            "step": self.step,
                            "state_dict": self.model.module[i].state_dict(),
                            "optimizer": self.optimizer[i].state_dict(),
                            "scheduler": self.scheduler[i].scheduler.state_dict(),
                            "scaler":self.scaler.state_dict(),
                            "best_metric": self.best_metric[i],
                            "identifier": self.timestamp_id,
                            "seed": torch.random.initial_seed(),
                        })
                    else:
                        state.append({"state_dict": self.model[i].module.state_dict(), "metric": metric[i]})
                else:   
                    if training_state:
                        state.append({
                            "epoch": self.epoch,
                            "step": self.step,
                            "state_dict": self.model[i].state_dict(),
                            "optimizer": self.optimizer[i].state_dict(),
                            "scheduler": self.scheduler[i].scheduler.state_dict(),
                            "scaler": self.scaler.state_dict(),
                            "best_metric": self.best_metric[i],
                            "identifier": self.timestamp_id,
                            "seed": torch.random.initial_seed(),
                        })
                    else:
                        state.append({"state_dict": self.model[i].state_dict(), "metric": metric})

            for x in range(len(self.model)):
                num = str(x)
                curr_checkpt_dir = os.path.join(
                    self.save_dir, "results", self.timestamp_id, f"checkpoint_{num}"
                )
                os.makedirs(curr_checkpt_dir, exist_ok=True)
                filename = os.path.join(curr_checkpt_dir, checkpoint_file)

                torch.save(state[x], filename)
            del state
        
        return filename

    def save_results(self, output, results_dir, filename, node_level_predictions=False, labels=True, std=False):
        results_path = os.path.join(
            self.save_dir, "results", self.timestamp_id, results_dir
        )
        os.makedirs(results_path, exist_ok=True)
        filename = os.path.join(results_path, filename)
        shape = output.shape

        id_headers = ["structure_id"]
        if node_level_predictions:
            id_headers += ["node_id"]
            
        if labels==True:
            if std == True:
                num_cols = (shape[1] - len(id_headers)) // 3
                headers = id_headers + ["target"] * num_cols + ["prediction"] * num_cols + ["std"] * num_cols
            else:
                num_cols = (shape[1] - len(id_headers)) // 2
                headers = id_headers + ["target"] * num_cols + ["prediction"] * num_cols            
        else:
            if std == True:
                num_cols = (shape[1] - len(id_headers)) // 2
                headers = id_headers + ["prediction"] * num_cols + ["std"] * num_cols        
            else:
                num_cols = (shape[1] - len(id_headers))
                headers = id_headers + ["prediction"] * num_cols     
                        
        with open(filename, "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(output)+1):
                if i == 0:
                    csvwriter.writerow(headers)
                elif i > 0:
                    csvwriter.writerow(output[i - 1, :])
        return filename

    # TODO: streamline this from PR #12
    def load_checkpoint(self, load_training_state=True):
        """Loads the model from a checkpoint.pt file"""

        if not self.checkpoint_path:
            raise ValueError("No checkpoint directory specified in config.")

        # checkpoint_path = glob.glob(os.path.join(self.checkpoint_path, "results", "*"))
        # checkpoint_file = os.path.join(checkpoint_path, "checkpoint", "checkpoint.pt")

        # Load params from checkpoint
        self.checkpoint_path = self.checkpoint_path.split(",")
        checkpoint = [torch.load(i) for i in self.checkpoint_path]

        for n, check in enumerate(checkpoint):
            if str(self.rank) not in ("cpu", "cuda"):
                self.model[n].module.load_state_dict(checkpoint[n]["state_dict"])
                self.best_model_state[n] = copy.deepcopy(self.model[n].module.state_dict())
            else:
                self.model[n].load_state_dict(checkpoint[n]["state_dict"])
                self.best_model_state[n] = copy.deepcopy(self.model[n].state_dict())
            if load_training_state == True:
                if checkpoint[n].get("optimizer"):
                    self.optimizer[n].load_state_dict(checkpoint[n]["optimizer"])
                if checkpoint[n].get("scheduler"):
                    self.scheduler[n].scheduler.load_state_dict(checkpoint[n]["scheduler"])
                    self.scheduler[n].update_lr()
                if checkpoint[n].get("epoch"):
                    self.epoch = checkpoint[n]["epoch"]
                if checkpoint[n].get("step"):
                    self.step = checkpoint[n]["step"]
                if checkpoint[n].get("best_metric"):
                    self.best_metric[n] = checkpoint[n]["best_metric"]
                if checkpoint[n].get("seed"):
                    seed = checkpoint[n]["seed"]
                    self.set_seed(seed)
                if checkpoint[n].get("scaler"):
                    self.scaler.load_state_dict(checkpoint[n]["scaler"])
                #todo: load dataset to recreate the same split as the prior run
                #self._load_dataset(dataset_config, task)
        
    # Loads portion of model dict into a new model for fine tuning
    def load_pre_trained_weights(self, load_training_state=False):
        """Loads the pre-trained model from a checkpoint.pt file"""

        if not self.checkpoint_path:
            raise ValueError("No checkpoint directory specified in config.")
            checkpoints_folder = os.path.join(self.fine_tune_from, 'checkpoint')
         
        self.checkpoint_path = self.checkpoint_path.split(",")
        load_model = [torch.load(i, map_location=self.device) for i in self.checkpoint_path]
        load_state = [i["state_dict"] for i in load_model]
        model_state = [i.state_dict() for i in self.model]

        for x in range(len(self.model)):
            for name, param in load_state[x].items():
                #if name not in model_state or name.split('.')[0] in "post_lin_list":
                if name not in model_state[x]:
                    logging.debug('NOT loaded: %s', name)
                    continue
                else:
                    logging.debug('loaded: %s', name)
                if isinstance(param, torch.nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[x][name].copy_(param)
            logging.info("Loaded pre-trained model with success.")
            if load_training_state == True:
                if checkpoint.get("optimizer"): 
                    self.optimizer[x].load_state_dict(checkpoint["optimizer"])
                if checkpoint.get("scheduler"):     
                    self.scheduler[x].scheduler.load_state_dict(checkpoint["scheduler"])
                    self.scheduler[x].update_lr()
                    #if checkpoint.get("epoch"): 
                    #   self.epoch = checkpoint["epoch"]
                    #if checkpoint.get("step"): 
                    #    self.step = checkpoint["step"]
                    #if checkpoint.get("best_metric"): 
                    #    self.best_metric = checkpoint["best_metric"]
                if checkpoint.get("seed"): 
                    seed = checkpoint["seed"]
                    self.set_seed(seed)
                if checkpoint.get("scaler"):
                    self.scaler.load_state_dict(checkpoint["scaler"])

    @staticmethod
    def set_seed(seed):
        # https://pytorch.org/docs/stable/notes/randomness.html
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
