import logging
import time
import os
import copy
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset

from matdeeplearn.common.data import get_otf_transforms, dataset_split
from matdeeplearn.datasets.LargeCTPretain_dataset import LargeCTPretrainDataset
from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.datasets.CTPretain_dataset import CTPretrainDataset

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.modules.scheduler import LRScheduler


def get_dataset(
        data_path,
        processed_file_name,
        transform_list: List[dict] = [],
        augmentation_list=None,
        large_dataset=False,
        dataset_config=None
):
    """
    get dataset according to data_path
    this assumes that the data has already been processed and
    data.pt file exists in data_path/processed/ folder

    Parameters
    ----------

    data_path: str
        path to the folder containing data.pt file

    transform_list: transformation function/classes to be applied
    """

    # get on the fly transforms for use on dataset access
    otf_transforms = get_otf_transforms(transform_list)
    composition = Compose(otf_transforms) if len(otf_transforms) >= 1 else None

    # check if large dataset is needed
    if large_dataset:
        Dataset = LargeCTPretrainDataset
        dataset = Dataset(data_path, processed_data_path="", processed_file_name=processed_file_name,
                          augmentation_list=augmentation_list, transform=composition,
                          mask_node_ratios=[0.1, 0.1],
                          mask_edge_ratios=[0.1, 0.1],
                          distance=0.05,
                          min_distance=0,
                          dataset_config=dataset_config)
    else:
        Dataset = CTPretrainDataset
        dataset = Dataset(data_path, processed_data_path="", processed_file_name=processed_file_name,
                          augmentation_list=augmentation_list, transform=composition,
                          mask_node_ratios=[0.1, 0.1],
                          mask_edge_ratios=[0.1, 0.1],
                          distance=0.05,
                          min_distance=0)

    return dataset

def get_dataloader(
    dataset, batch_size: int, num_workers: int = 0, sampler=None, shuffle=True
):
    """
    Returns a single dataloader for a given dataset

    Parameters
    ----------
        dataset: matdeeplearn.preprocessor.datasets.StructureDataset
            a dataset object that contains the target data

        batch_size: int
            size of each batch

        num_workers: int
            how many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
    """

    # load data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True
    )

    return loader


@registry.register_trainer("ct_pretrain")
class CTPretrainer(BaseTrainer):
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
            clip_grad_norm: int = None,
            max_checkpoint_epochs: int = None,
            identifier: str = None,
            verbosity: int = None,
            batch_tqdm: bool = False,
            write_output: list = ["train", "val", "test"],
            save_dir: str = None,
            checkpoint_path: str = None,
            use_amp: bool = False,
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
        self.clip_grad_norm = clip_grad_norm
        self.max_checkpoint_epochs = max_checkpoint_epochs
        self.train_verbosity = verbosity
        self.batch_tqdm = batch_tqdm
        self.write_output = write_output

        self.epoch = 0
        self.step = 0
        self.metrics = {}
        self.epoch_time = None
        self.best_metric = 1e10
        self.best_model_state = None

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

        timestamp = torch.tensor(datetime.now().timestamp()).to(self.device)
        self.timestamp_id = datetime.fromtimestamp(timestamp.int()).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        if identifier:
            self.timestamp_id = f"{self.timestamp_id}-{identifier}"

        if self.train_verbosity:
            logging.info(
                f"GPU is available: {torch.cuda.is_available()}, Quantity: {torch.cuda.device_count()}"
            )
            logging.info(f"Dataset used: {self.dataset}")
            logging.debug(self.dataset["train"][0][0])
            logging.debug(self.dataset["train"][0][0].x[0])
            logging.debug(self.dataset["train"][0][0].x[-1])
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

        cls.set_seed(config["task"].get("seed"))

        if config["task"]["parallel"] == True and os.environ.get("LOCAL_WORLD_SIZE", None):
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
        dataset = cls._load_dataset(config["dataset"])
        model = cls._load_model(config["model"], config["dataset"]["preprocess_params"], dataset, local_world_size,
                                rank)
        optimizer = cls._load_optimizer(config["optim"], model, local_world_size)
        sampler = cls._load_sampler(config["optim"], dataset, local_world_size, rank)
        train_loader, val_loader, test_loader = cls._load_dataloader(
            config["optim"],
            config["dataset"],
            dataset,
            sampler,
        )

        scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)
        loss = cls._load_loss(config["optim"]["loss"])
        max_epochs = config["optim"]["max_epochs"]
        clip_grad_norm = config["optim"].get("clip_grad_norm", None)
        verbosity = config["optim"].get("verbosity", None)
        batch_tqdm = config["optim"].get("batch_tqdm", False)
        write_output = config["task"].get("write_output", [])
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
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss=loss,
            max_epochs=max_epochs,
            clip_grad_norm=clip_grad_norm,
            max_checkpoint_epochs=max_checkpoint_epochs,
            identifier=identifier,
            verbosity=verbosity,
            batch_tqdm=batch_tqdm,
            write_output=write_output,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path,
            use_amp=config["task"].get("use_amp", False),
        )

    @staticmethod
    def _load_model(model_config, graph_config, dataset, world_size, rank):
        """Loads the model if from a config file."""

        if dataset.get("train"):
            dataset = dataset["train"]
        else:
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

            # Obtain node, edge, and output dimensions for model initialization
        node_dim = dataset.num_features
        edge_dim = graph_config["edge_steps"]
        if dataset[0][0]["y"].ndim == 0:
            output_dim = 1
        else:
            output_dim = dataset[0][0]["y"].shape[1]

            # Determine if this is a node or graph level model
        if dataset[0][0]["y"].shape[0] == dataset[0][0]["x"].shape[0]:
            model_config["prediction_level"] = "node"
        elif dataset[0][0]["y"].shape[0] == 1:
            model_config["prediction_level"] = "graph"
        else:
            raise ValueError(
                "Target labels do not have the correct dimensions for node or graph-level prediction."
            )

        model_cls = registry.get_model_class(model_config["name"])
        model = model_cls(
            node_dim=node_dim,
            edge_dim=edge_dim,
            output_dim=output_dim,
            data=dataset,
            cutoff_radius=graph_config["cutoff_radius"],
            n_neighbors=graph_config["n_neighbors"],
            edge_steps=graph_config["edge_steps"],
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
        return model


    @staticmethod
    def _load_dataset(dataset_config):
        """Loads the dataset if from a config file."""

        dataset_path = dataset_config["pt_path"]  # data/test_data/processed/
        dataset = {}
        if isinstance(dataset_config["src"], dict):
            dataset["train"] = get_dataset(
                dataset_path,
                processed_file_name="data_train.pt",
                transform_list=dataset_config.get("transforms", []),
                augmentation_list=dataset_config.get("augmentation", None),
                large_dataset=dataset_config.get("large_dataset", False),
                dataset_config=dataset_config
            )
            dataset["val"] = get_dataset(
                dataset_path,
                processed_file_name="data_val.pt",
                transform_list=dataset_config.get("transforms", []),
                augmentation_list=dataset_config.get("augmentation", None),
                large_dataset=dataset_config.get("large_dataset", False),
                dataset_config=dataset_config
            )
            dataset["test"] = get_dataset(
                dataset_path,
                processed_file_name="data_test.pt",
                transform_list=dataset_config.get("transforms", []),
                augmentation_list=dataset_config.get("augmentation", None),
                large_dataset=dataset_config.get("large_dataset", False),
                dataset_config=dataset_config
            )

        else:
            print("augmentation: ", dataset_config.get("augmentation"))
            dataset["train"] = get_dataset(
                dataset_path,
                processed_file_name="data.pt",
                transform_list=dataset_config.get("transforms", []),
                augmentation_list=dataset_config.get("augmentation", None),
                large_dataset=dataset_config.get("large_dataset", False),
                dataset_config=dataset_config
            )

        return dataset

    @staticmethod
    def _load_dataloader(optim_config, dataset_config, dataset, sampler):

        batch_size = optim_config.get("batch_size")
        if isinstance(dataset_config["src"], dict):
            train_loader = get_dataloader(
                dataset["train"], batch_size=batch_size, sampler=sampler
            )
            val_loader = get_dataloader(
                dataset["val"], batch_size=batch_size, sampler=sampler
            )
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
            ) if test_ratio != 0 else None

        return train_loader, val_loader, test_loader

    def train(self):
        # Start training over epochs loop
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        end_epoch = (
            self.max_checkpoint_epochs + start_epoch
            if self.max_checkpoint_epochs
            else self.max_epochs
        )
        # print(self.dataset["train"].data)
        # print(self.dataset["train"].slices)

        if self.train_verbosity:
            logging.info("Starting regular training")
            logging.info(
                f"running for {end_epoch - start_epoch} epochs on {type(self.model).__name__} model"
            )
        self.model = self.model.to(self.device)
        print(len(self.dataset))

        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            # metrics for every epoch
            _metrics = {}

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch + (i + 1) / len(self.train_loader)
                self.step = epoch * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch of train data
                batch1, batch2 = next(train_loader_iter)
                if len(batch1) == 0:
                    continue
                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)

                # Compute forward, loss, backward
                out1 = self._forward(batch1)
                out2 = self._forward(batch2)
                loss = self._compute_loss(out1, out2)
                # print("out1 shape: ", out1.size(), " out2 shape: ", out2.size(), " loss: ", loss.item())
                logging.info("Epoch: {:04d}, Step: {:04d}, Loss: {:.5f}".format(int(self.epoch - 1), i, loss.item()))
                self._backward(loss)

                # Compute metrics
                # TODO: revert _metrics to be empty per batch, so metrics are logged per batch, not per epoch
                #  keep option to log metrics per epoch
                _metrics = self._compute_metrics(out1, out2, _metrics)
                self.metrics = self.evaluator.update("loss", loss.item(), _metrics)

            # TODO: could add param to eval and save on increments instead of every time
            # Save current model
            self.save_model(checkpoint_file="checkpoint.pt", training_state=True)

            # Evaluate on validation set if it exists
            if self.val_loader:
                val_metrics = self.validate()

                # Train loop timings
                self.epoch_time = time.time() - epoch_start_time
                # Log metrics
                if epoch % self.train_verbosity == 0:
                    self._log_metrics(val_metrics)

                # Update best val metric and model, and save best model and predicted outputs
                if (
                        val_metrics[type(self.loss_fn).__name__]["metric"]
                        < self.best_metric
                ):
                    # self.update_best_model(val_metrics)
                    self.best_metric = val_metrics[type(self.loss_fn).__name__]["metric"]
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

                    self.save_model("best_checkpoint.pt", val_metrics, False)

                    logging.debug(
                        f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/"
                    )

                # step scheduler, using validation error
                self._scheduler_step()

        return self.best_model_state

    def validate(self, split="val"):
        self.model.eval()
        evaluator, metrics = Evaluator(), {}

        loader_iter = (
            iter(self.val_loader) if split == "val" else iter(self.test_loader)
        )

        for i in range(0, len(loader_iter)):
            with torch.no_grad():
                batch1, batch2 = next(loader_iter)
                if len(batch1) == 1:
                    continue
                out1 = self._forward(batch1.to(self.device))
                out2 = self._forward(batch2.to(self.device))
                loss = self._compute_loss(out1, out2)
                # Compute metrics
                metrics = self._compute_metrics(out1, out2, metrics)
                metrics = evaluator.update("loss", loss.item(), metrics)

        return metrics

    # '''
    @torch.no_grad()
    def predict(self, loader, split):
        # TODO: make predict method work as standalone task
        if not loader: return
        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)

        self.model.eval()
        predict, target = None, None
        ids = []
        node_level_predictions = False
        _metrics_predict = {}
        for i, batch1, batch2 in enumerate(loader):
            out1 = self._forward(batch1.to(self.device))
            out2 = self._forward(batch2.to(self.device))
            loss = self._compute_loss(out1, out2)
            _metrics_predict = self._compute_metrics(out1, out2, _metrics_predict)
            self._metrics_predict = self.evaluator.update(
                "loss", loss.item(), _metrics_predict
            )

            # if out1 is a tuple, then it's scaled data
            if type(out1) == tuple:
                out1 = out1[0] * out1[1].view(-1, 1).expand_as(out1[0])

            batch_p = out1.data.cpu().numpy()
            batch_t = batch1[self.model.target_attr].cpu().numpy()

            batch_ids = np.array(
                [item for sublist in batch1.structure_id for item in sublist]
            )

            # if shape is 2D, then it has node-level predictions
            if batch_p.ndim == 2:
                node_level_predictions = True
                node_ids = batch1.z.cpu().numpy()
                structure_ids = np.repeat(
                    batch_ids, batch1.n_atoms.cpu().numpy(), axis=0
                )
                batch_ids = np.column_stack((structure_ids, node_ids))

            ids = batch_ids if i == 0 else np.row_stack((ids, batch_ids))
            predict = batch_p if i == 0 else np.concatenate((predict, batch_p), axis=0)
            target = batch_t if i == 0 else np.concatenate((target, batch_t), axis=0)

        predictions = np.column_stack((ids, target, predict))

        self.save_results(
            predictions, f"{split}_predictions.csv", node_level_predictions
        )
        predict_loss = self._metrics_predict[type(self.loss_fn).__name__]["metric"]
        logging.debug("Saved {:s} error: {:.5f}".format(split, predict_loss))
        return predictions

    # '''

    def _forward(self, batch_data):
        output = self.model(batch_data)
        return output

    def _compute_loss(self, out1, out2):
        out1 = torch.nn.functional.normalize(out1, dim=1)
        out2 = torch.nn.functional.normalize(out2, dim=1)
        loss = self.loss_fn(out1, out2)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_metrics(self, out1, out2, metrics):
        # TODO: finish this method
        # property_target = batch_data.to(self.device)

        metrics = self.evaluator.eval(
            out1, out2, self.loss_fn, prev_metrics=metrics
        )

        return metrics

    def _log_metrics(self, val_metrics=None):
        if not val_metrics:
            logging.info(f"epoch: {self.epoch}, learning rate: {self.scheduler.lr}")
            logging.info(self.metrics[type(self.loss_fn).__name__]["metric"])
        else:
            train_loss = self.metrics[type(self.loss_fn).__name__]["metric"]
            val_loss = val_metrics[type(self.loss_fn).__name__]["metric"]
            logging.info(
                "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                    int(self.epoch - 1),
                    self.scheduler.lr,
                    train_loss,
                    val_loss,
                    self.epoch_time,
                )
            )

    def _load_task(self):
        """Initializes task-specific info. Implemented by derived classes."""
        pass

    def _scheduler_step(self):
        if self.scheduler.scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(
                metrics=self.metrics[type(self.loss_fn).__name__]["metric"]
            )
        else:
            self.scheduler.step()
