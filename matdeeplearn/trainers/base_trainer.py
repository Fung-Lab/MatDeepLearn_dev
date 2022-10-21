from abc import ABC, abstractmethod
import logging
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.modules.scheduler import LRScheduler
from matdeeplearn.common.data import *


@registry.register_trainer("base")
class BaseTrainer(ABC):

    def __init__(self, model: BaseModel, dataset: Dataset, optimizer: Optimizer, sampler: DistributedSampler,
                 scheduler: LRScheduler, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 loss: str, max_epochs: int, verbosity: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_sampler = sampler
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.scheduler = scheduler

        self.loss_fn = self.load_loss(loss)
        self.max_epochs = max_epochs
        self.train_verbosity = verbosity

        self.epoch = 0
        self.step = 0
        self.metrics = {}
        self.epoch_time = None

        self.evaluator = Evaluator()

        if self.train_verbosity:
            logging.info(f"GPU is available: {torch.cuda.is_available()}, Quantity: {torch.cuda.device_count()}")
            logging.info(f"Dataset used: {self.dataset}")
            logging.debug(self.dataset[0])
            logging.debug(self.dataset[0].x[0])
            logging.debug(self.dataset[0].x[-1])
            logging.debug(self.model)

    @classmethod
    def from_config(cls, config):
        """ Class method used to initialize BaseTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
                scheduler
            dataset
        """
        # TODO: figure out what configs are passed in and how they're structured
        #  (one overall config, or individual components)

        dataset = cls._load_dataset(config["dataset"])
        model = cls._load_model(config["model"], dataset)
        optimizer = cls._load_optimizer(config["optim"], model)
        sampler = cls._load_sampler(config["optim"], dataset)
        train_loader, val_loader, test_loader = cls._load_dataloader(config["optim"], dataset, sampler)
        scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)

        loss = config["optim"]["loss_fn"]
        max_epochs = config["optim"]["max_epochs"]
        verbosity = config["task"].get("verbosity", None)

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
            verbosity=verbosity
        )

    @staticmethod
    def _load_dataset(dataset_config):
        """ Loads the dataset if from a config file. """
        dataset_path = dataset_config['pt_path']
        target_index = dataset_config.get('target_index', 0)

        dataset = get_dataset(dataset_path, target_index)

        return dataset

    @staticmethod
    def _load_model(model_config, dataset):
        """ Loads the model if from a config file. """

        model_cls = registry.get_model_class(model_config["name"])
        model = model_cls(data=dataset, **model_config)
        return model

    @staticmethod
    def _load_optimizer(optim_config, model):
        optimizer = getattr(optim, optim_config["optimizer"]["optimizer_type"])(
            model.parameters(),
            lr=optim_config["lr"],
            **optim_config["optimizer"].get("optimizer_args", {})
        )
        return optimizer

    @staticmethod
    def _load_sampler(optim_config, dataset):
        # TODO: write sampler, look into BalancedBatchSampler in
        #  OCP for their implementation of train_sampler batches
        #  (part of self.train_loader)
        # TODO: update sampler with more attributes like rank and num_replicas (world_size)

        # sampler = DistributedSampler(dataset, rank=0)

        # TODO: for testing purposes, return None
        return None

    @staticmethod
    def _load_dataloader(optim_config, dataset, sampler):
        train_dataset, val_dataset, test_dataset = dataset_split(dataset)

        batch_size = optim_config.get("batch_size")
        
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = get_dataloader(val_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size, sampler=sampler)

        return train_loader, val_loader, test_loader

    @staticmethod
    def _load_scheduler(scheduler_config, optimizer):
        scheduler_type = scheduler_config["scheduler_type"]
        scheduler_args = scheduler_config["scheduler_args"]
        scheduler = LRScheduler(optimizer, scheduler_type, scheduler_args)
        return scheduler

    def load_loss(self, loss_name):
        try:
            return getattr(F, loss_name)
        except Exception as e:
            raise NotImplementedError(f"Unknown loss function name: {loss_name}")

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
