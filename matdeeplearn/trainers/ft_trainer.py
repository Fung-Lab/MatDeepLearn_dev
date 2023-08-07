import csv
import logging
import time
import os
import copy
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from matdeeplearn.common.data import get_otf_transforms, dataset_split
from matdeeplearn.common.registry import registry
from matdeeplearn.datasets.LargeCTPretain_dataset import LargeCTPretrainDataset
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.preprocessor import LargeStructureDataset, StructureDataset
from matdeeplearn.trainers.property_trainer import PropertyTrainer


def get_dataset(
        data_path,
        processed_file_name,
        dataset_config,
        transform_list: List[dict] = [],
        large_dataset=False,
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
                          transform=composition, dataset_config=dataset_config, finetune=True)
    else:
        Dataset = StructureDataset
        dataset = Dataset(data_path, processed_data_path="", processed_file_name=processed_file_name,
                          transform=composition)

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
        drop_last=False
    )

    return loader


@registry.register_trainer("finetune")
class FinetuneTrainer(PropertyTrainer):
    def __init__(
            self,
            model,
            dataset,
            optimizer,
            sampler,
            scheduler,
            data_loader,
            loss,
            max_epochs,
            clip_grad_norm,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            batch_tqdm,
            write_output,
            save_dir,
            checkpoint_path,
            use_amp,
            seed,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.fine_tune_from = checkpoint_path if checkpoint_path else ""
        self._load_pre_trained_weights(self.model)
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
        self.seed = seed

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
            self.identifier = identifier
            self.timestamp_id = f"{self.timestamp_id}-{identifier}"

        if self.train_verbosity:
            logging.info(
                f"GPU is available: {torch.cuda.is_available()}, Quantity: {torch.cuda.device_count()}"
            )
            logging.info(f"Dataset used: {self.dataset}")
            logging.debug(self.dataset["train"][0])
            logging.debug(self.dataset["train"][0].x[0])
            logging.debug(self.dataset["train"][0].y[0])
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
        seed = config["task"].get("seed")
        print("seed:",seed)

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
        data_loader = cls._load_dataloader(
            config["optim"],
            config["dataset"],
            dataset,
            sampler,
            # config["task"]["run_mode"],
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
            data_loader=data_loader,
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
            seed=seed
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
        if dataset[0]["y"].ndim == 0:
            output_dim = 1
        else:
            output_dim = dataset[0]["y"].shape[0]

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
                dataset_config=dataset_config,
                large_dataset=dataset_config.get("large_dataset", False)
            )
            dataset["val"] = get_dataset(
                dataset_path,
                processed_file_name="data_val.pt",
                transform_list=dataset_config.get("transforms", []),
                dataset_config=dataset_config,
                large_dataset=dataset_config.get("large_dataset", False)
            )
            dataset["test"] = get_dataset(
                dataset_path,
                processed_file_name="data_test.pt",
                transform_list=dataset_config.get("transforms", []),
                dataset_config=dataset_config,
                large_dataset=dataset_config.get("large_dataset", False)
            )

        else:
            print("augmentation: ", dataset_config.get("augmentation", None))
            dataset["train"] = get_dataset(
                dataset_path,
                processed_file_name="data.pt",
                transform_list=dataset_config.get("transforms", []),
                dataset_config=dataset_config,
                large_dataset=dataset_config.get("large_dataset", False)
            )
            print(len(dataset["train"]))

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

        return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}

    # @classmethod
    # def from_config(cls, config):
    #     """Class method used to initialize BaseTrainer from a config object
    #     config has the following sections:
    #         trainer
    #         task
    #         model
    #         optim
    #             scheduler
    #         dataset
    #     """
    #
    #     dataset = cls._load_dataset(config["dataset"])
    #     model = cls._load_model(config["model"], dataset["train"])
    #     optimizer = cls._load_optimizer(config["optim"], model)
    #     sampler = cls._load_sampler(config["optim"], dataset["train"])
    #     train_loader, val_loader, test_loader = cls._load_dataloader(
    #         config["optim"], config["dataset"], dataset, sampler
    #     )
    #     scheduler = cls._load_scheduler(config["optim"]["scheduler"], optimizer)
    #     loss = cls._load_loss(config["optim"]["loss"])
    #     max_epochs = config["optim"]["max_epochs"]
    #     max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
    #     identifier = config["task"].get("identifier", None)
    #     verbosity = config["task"].get("verbosity", None)
    #     # pass in custom results home dir and load in prev checkpoint dir
    #     save_dir = config["task"].get("save_dir", None)
    #     checkpoint_dir = config["task"].get("checkpoint_dir", None)
    #     fine_tune_from = config["model"].get("fine_tune_from", "")
    #
    #     return cls(
    #         model=model,
    #         dataset=dataset,
    #         optimizer=optimizer,
    #         sampler=sampler,
    #         scheduler=scheduler,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         test_loader=test_loader,
    #         loss=loss,
    #         max_epochs=max_epochs,
    #         max_checkpoint_epochs=max_checkpoint_epochs,
    #         identifier=identifier,
    #         verbosity=verbosity,
    #         save_dir=save_dir,
    #         checkpoint_dir=checkpoint_dir,
    #         fine_tune_from=fine_tune_from
    #     )

    def _load_pre_trained_weights(self, model):
        # try:
        #     checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
        #     state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        #     model.load_state_dict(state_dict)
        #     print("Loaded pre-trained model with success.")
        # except FileNotFoundError:
        #     print("Pre-trained weights not found. Training from scratch.")

        try:
            checkpoints_folder = os.path.join(self.fine_tune_from, 'checkpoint')
            load_state = torch.load(os.path.join(checkpoints_folder, 'checkpoint.pt'), map_location=self.device)
            load_state = load_state["state_dict"]

            # checkpoint = torch.load('model_best.pth.tar', map_location=args.gpu)
            # load_state = checkpoint['state_dict']
            model_state = model.state_dict()

            # pytorch_total_params = sum(p.numel() for p in model_state.parameters if p.requires_grad)
            # print(pytorch_total_params)
            for name, param in load_state.items():
                if name not in model_state:
                    logging.info('NOT loaded: %s', name)
                    continue
                else:
                    logging.info('loaded: %s', name)
                if isinstance(param, torch.nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            logging.info("Loaded pre-trained model with success.")
        except FileNotFoundError:
            logging.info("Pre-trained weights not found. Training from scratch.")

        return model

    def save_model(self, checkpoint_file, val_metrics=None, training_state=True):
        """Saves the model state dict"""

        # if training_state:
        #     state = {
        #         "epoch": self.epoch,
        #         "step": self.step,
        #         "state_dict": self.model.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #         "scheduler": self.scheduler.scheduler.state_dict(),
        #         "best_val_metric": self.best_val_metric,
        #     }
        # else:
        #     state = {"state_dict": self.model.state_dict(), "val_metrics": val_metrics}
        #
        # curr_checkpt_dir = os.path.join(
        #     self.save_dir, self.fine_tune_from, self.identifier, self.timestamp_id, "checkpoint"
        # )
        # os.makedirs(curr_checkpt_dir, exist_ok=True)
        # filename = os.path.join(curr_checkpt_dir, checkpoint_file)
        #
        # torch.save(state, filename)
        return

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
            # with open(os.path.join(self.save_dir, self.fine_tune_from, self.identifier, self.timestamp_id, "train.log"), "a+") as f:
            #     f.write(
            #         "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}\n".format(
            #             int(self.epoch - 1),
            #             self.scheduler.lr,
            #             train_loss,
            #             val_loss,
            #             self.epoch_time,
            #         ))

    def update_best_model(self, metric):
        """Updates the best val metric and model, saves the best model, and saves the best model predictions"""
        self.best_metric = metric[type(self.loss_fn).__name__]["metric"]
        if str(self.rank) not in ("cpu", "cuda"):
            self.best_model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        # self.save_model("best_checkpoint.pt", metric, True)

        # logging.debug(
        #     f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/train_results/"
        # )
        # self.predict(self.train_loader, "train")
        # self.predict(self.val_loader, "val")
        # self.predict(self.test_loader, "test")

    def train(self):
        # Start training over epochs loop
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        # start_epoch = self.step // len(self.train_loader)
        start_epoch = int(self.epoch)

        if str(self.rank) not in ("cpu", "cuda"):
            dist.barrier()

        end_epoch = (
            self.max_checkpoint_epochs + start_epoch
            if self.max_checkpoint_epochs
            else self.max_epochs
        )

        if self.train_verbosity:
            logging.info("Starting regular training")
            if str(self.rank) not in ("cpu", "cuda"):
                logging.info(
                    f"running for {end_epoch - start_epoch} epochs on {type(self.model.module).__name__} model"
                )
            else:
                logging.info(
                    f"running for {end_epoch - start_epoch} epochs on {type(self.model).__name__} model"
                )

        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            # skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.data_loader["train_loader"])
            # metrics for every epoch
            _metrics = {}

            # for i in range(skip_steps, len(self.train_loader)):
            pbar = tqdm(range(0, len(self.data_loader["train_loader"])), disable=not self.batch_tqdm)
            for i in pbar:
                # self.epoch = epoch + (i + 1) / len(self.train_loader)
                # self.step = epoch * len(self.train_loader) + i + 1
                # print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))
                self.model.train()
                # Get a batch of train data
                batch = next(train_loader_iter).to(self.rank)
                # print(epoch, i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024), torch.sum(batch.n_atoms))
                # Compute forward, loss, backward
                with autocast(enabled=self.use_amp):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                    # print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))
                grad_norm = self._backward(loss)
                pbar.set_description("Batch Loss {:.4f}, grad norm {:.4f}".format(loss.item(), grad_norm.item()))
                # Compute metrics
                # TODO: revert _metrics to be empty per batch, so metrics are logged per batch, not per epoch
                #  keep option to log metrics per epoch
                _metrics = self._compute_metrics(out, batch, _metrics)
                self.metrics = self.evaluator.update("loss", loss.item(), _metrics)

            self.epoch = epoch + 1

            if str(self.rank) not in ("cpu", "cuda"):
                dist.barrier()

            # TODO: could add param to eval and save on increments instead of every time

            # Save current model
            torch.cuda.empty_cache()
            if str(self.rank) in ("0", "cpu", "cuda"):
                self.save_model(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on validation set if it exists
                if self.data_loader.get("val_loader"):
                    metric = self.validate("val")
                else:
                    metric = self.metrics

                # Train loop timings
                self.epoch_time = time.time() - epoch_start_time
                # Log metrics
                if epoch % self.train_verbosity == 0:
                    if self.data_loader.get("val_loader"):
                        self._log_metrics(metric)
                    else:
                        self._log_metrics()

                # Update best val metric and model, and save best model and predicted outputs
                if metric[type(self.loss_fn).__name__]["metric"] < self.best_metric:
                    self.update_best_model(metric)
                    self.best_epoch = epoch

                # step scheduler, using validation error
                self._scheduler_step()

            torch.cuda.empty_cache()

        if self.best_model_state:
            if str(self.rank) in "0":
                self.model.module.load_state_dict(self.best_model_state)
            elif str(self.rank) in ("cpu", "cuda"):
                self.model.load_state_dict(self.best_model_state)

            if self.data_loader.get("test_loader"):
                metric = self.validate("test")
                test_loss = metric[type(self.loss_fn).__name__]["metric"]
            else:
                test_loss = "N/A"
            logging.info("Test loss: " + str(test_loss))

        best_log_dir_name = os.path.join(self.save_dir, self.fine_tune_from, self.identifier) \
            if self.fine_tune_from else os.path.join(self.save_dir, "results/scratch/", self.identifier)
        if not os.path.exists(best_log_dir_name):
            os.makedirs(best_log_dir_name)

        with open(os.path.join(best_log_dir_name, "best_val_metric.csv"), "a+", encoding="utf-8", newline='') as f:
            new_metric = [self.timestamp_id, test_loss, self.best_epoch, self.seed]
            csv_writer = csv.writer(f)
            if not os.path.getsize(os.path.join(best_log_dir_name, "best_val_metric.csv")):
                csv_head = ["timestamp_id", "best_val_metric", "best_epoch", "seed"]
                csv_writer.writerow(csv_head)
            csv_writer.writerow(new_metric)

        return self.best_model_state
