from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.modules.scheduler import LRScheduler
from matdeeplearn.preprocessor.utils import get_dataset

from abc import ABC, abstractmethod

import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler


@registry.register_trainer("base")
class BaseTrainer(ABC):

    def __init__(self, model: BaseModel, dataset: Dataset, optimizer: Optimizer, sampler: DistributedSampler,
                 scheduler: LRScheduler, train_loader: DataLoader, loss: str, max_epochs: int):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_sampler = sampler
        self.train_loader = train_loader
        self.scheduler = scheduler

        self.load_loss(loss)
        self.max_epochs = max_epochs
        self.rank = "cpu"

        self.epoch = 0
        self.step = 0
        self.metrics = {}

        self.evaluator = Evaluator()

    @classmethod
    def from_config(cls, task_config, model_config, dataset_config, sampler_config, loader_config, optim_config):
        """ Class method used to initialize BaseTrainer from a config object """

        # TODO: figure out what configs are passed in and how they're structured
        #  (one overall config, or individual components)
        dataset = cls._load_dataset(dataset_config)
        model = cls._load_model(model_config)
        optimizer = cls._load_optimizer(optim_config)
        sampler = cls._load_sampler(sampler_config)
        loader = cls._load_loader(loader_config)
        scheduler = cls._load_scheduler(optim_config["optim"]["scheduler"])
        loss = optim_config["optim"]["loss_fn"]
        max_epochs = optim_config["optim"]["max_epochs"]

        return cls(model, dataset, sampler, scheduler, loader, loss, max_epochs)

    def _load_dataset(self, dataset_config):
        """ Loads the dataset if from a config file. """
        dataset_path = dataset_config['src']
        target_index = dataset_config.get('target_index', 0)
        return get_dataset(dataset_path, target_index)

    def _load_model(self, model_config):
        """ Loads the model if from a config file. """
        model_cls = registry.get_model_class(model_config["name"])
        model = model_cls(**model_config)
        return model

    def _load_optimizer(self, optimizer_config):
        optimizer = getattr(optim, optimizer_config["optimizer"])(
            self.model.parameters(),
            lr=optimizer_config["lr"],
            **optimizer_config.get("optimizer_args", {})
        )
        return optimizer

    def _load_sampler(self, sampler_config):
        # TODO: write sampler, look into BalancedBatchSampler in
        #  OCP for their implementation of train_sampler batches
        #  (part of self.train_loader)
        pass

    def _load_loader(self, loader_config):
        #TODO: write loader
        pass

    def load_loss(self, loss_name):
        try:
            self.loss_fn = getattr(F, loss)
        except Exception as e:
            raise NotImplementedError(f"Unknown loss function name: {loss_name}")

    def _load_scheduler(self, scheduler_type, scheduler_args):
        scheduler = LRScheduler(self.optimizer, scheduler_type, scheduler_args)
        return scheduler

    @abstractmethod
    def _load_task(self):
        """ Initializes task-specific info. Implemented by derived classes. """

    @abstractmethod
    def train(self):
        """ Implemented by derived classes. """

    @abstractmethod
    def validate(self):
        """ Implemented by derived classes. """

    @abstractmethod
    def predict(self):
        """ Implemented by derived classes. """
