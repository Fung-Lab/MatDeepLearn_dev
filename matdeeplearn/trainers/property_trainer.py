import logging
import time

import numpy as np
import torch
import wandb
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from matdeeplearn.common.data import get_dataloader
from matdeeplearn.common.registry import registry
from matdeeplearn.modules.evaluator import Evaluator
from matdeeplearn.trainers.base_trainer import BaseTrainer


@registry.register_trainer("property")
class PropertyTrainer(BaseTrainer):
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
        max_checkpoint_epochs,
        identifier,
        verbosity,
        device,
        save_dir,
        checkpoint_path,
        wandb_config,
        model_config,
        opt_config,
        dataset_config,
    ):
        super().__init__(
            model,
            dataset,
            optimizer,
            sampler,
            scheduler,
            data_loader,
            loss,
            max_epochs,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            device,
            save_dir,
            checkpoint_path,
            wandb_config,
            model_config,
            opt_config,
            dataset_config,
        )

        self.use_wandb = self.wandb_config.get("use_wandb", False)

    def train(self):
        # Start training over epochs loop
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        # start_epoch = self.step // len(self.train_loader)
        start_epoch = int(self.epoch)

        if isinstance(self.model, DistributedDataParallel):
            dist.barrier()

        end_epoch = (
            self.max_checkpoint_epochs + start_epoch
            if self.max_checkpoint_epochs
            else self.max_epochs
        )

        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            train_loader_iter = iter(self.data_loader["train_loader"])

        if self.max_checkpoint_epochs:
            logging.info("Starting training from checkpoint")

        if self.use_wandb and wandb.run.resumed:
            logging.info("Resuming W&B run")

        if self.train_verbosity:
            logging.info("Starting regular training")
            if isinstance(self.model, DistributedDataParallel):
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
            for i in range(0, len(self.data_loader["train_loader"])):
                # self.epoch = epoch + (i + 1) / len(self.train_loader)
                # self.step = epoch * len(self.train_loader) + i + 1
                self.model.train()
                # Get a batch of train data
                batch = next(train_loader_iter).to(self.rank)
                # Compute forward, loss, backward
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Compute metrics
                # TODO: revert _metrics to be empty per batch, so metrics are logged per batch, not per epoch
                #  keep option to log metrics per epoch
                _metrics = self._compute_metrics(out, batch, _metrics)
                self.metrics = self.evaluator.update("loss", loss.item(), _metrics)

                # step scheduler, using validation error
                if self.opt_config["scheduler"].get("step", "epoch") == "step":
                    self._scheduler_step()

            self.epoch = epoch + 1

            if isinstance(self.model, DistributedDataParallel):
                dist.barrier()

            # TODO: could add param to eval and save on increments instead of every time

            # Save current model
            if not isinstance(self.model, DistributedDataParallel) or str(
                self.rank
            ) in ("0", "cpu", "cuda"):
                self.save_model(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on validation amd test sets if they exists
                val_metric = (
                    self.validate(split="val")
                    if self.data_loader.get("val_loader")
                    else None
                )
                test_metric = (
                    self.validate(split="test")
                    if self.data_loader.get("test_loader")
                    else None
                )

                # Train loop timings
                self.epoch_time = time.time() - epoch_start_time
                # Log metrics
                if epoch % self.train_verbosity == 0:
                    self._log_metrics(val_metrics=val_metric, test_metrics=test_metric)

                # Update best val metric and model, and save best model and predicted outputs
                if val_metric[type(self.loss_fn).__name__]["metric"] < self.best_metric:
                    self.update_best_model(val_metric)

                # step scheduler, using validation error
                if self.opt_config["scheduler"].get("step", "epoch") == "epoch":
                    self._scheduler_step()

        if self.best_model_state:
            if str(self.rank) in "0":
                self.model.module.load_state_dict(self.best_model_state)
            elif str(self.rank) in ("cpu", "cuda"):
                self.model.load_state_dict(self.best_model_state)

            if self.data_loader.get("test_loader"):
                val_metric = self.validate("test")
                test_loss = val_metric[type(self.loss_fn).__name__]["metric"]
            else:
                test_loss = "N/A"
            logging.info("Test loss: " + str(test_loss))

        return self.best_model_state

    def validate(self, split="val"):
        self.model.eval()
        evaluator, metrics = Evaluator(), {}

        if split == "val":
            loader_iter = iter(self.data_loader["val_loader"])
        elif split == "test":
            loader_iter = iter(self.data_loader["test_loader"])
        elif split == "train":
            loader_iter = iter(self.data_loader["train_loader"])

        for i in range(0, len(loader_iter)):
            with torch.no_grad():
                batch = next(loader_iter)
                out = self._forward(batch)
                if type(out) == tuple and len(out) == 5:
                    out = out[0]
                loss = self._compute_loss(out, batch)
                # Compute metrics
                metrics = self._compute_metrics(out, batch, metrics)
                metrics = evaluator.update("loss", loss.item(), metrics)

        return metrics

    @torch.no_grad()
    def predict(self, loader, split, results_dir="train_results", write_output=True):
        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)

        if isinstance(self.model, DistributedDataParallel):
            loader = get_dataloader(
                loader.dataset, batch_size=loader.batch_size, sampler=None
            )

        self.model.eval()
        predict, target = None, None
        ids = []
        node_level = False
        _metrics_predict = {}
        for i, batch in enumerate(loader):
            out = self._forward(batch)
            loss = self._compute_loss(out, batch)
            _metrics_predict = self._compute_metrics(out, batch, _metrics_predict)
            self._metrics_predict = self.evaluator.update(
                "loss", loss.item(), _metrics_predict
            )

            # if out is a tuple, then it's scaled data
            # if type(out) == tuple:
            #    out = out[0] * out[1].view(-1, 1).expand_as(out[0])

            batch_p = out.data
            if isinstance(self.model, DistributedDataParallel):
                batch_t = batch[self.model.module.target_attr]
            else:
                batch_t = batch[self.model.target_attr]

            # batch_ids = np.array(
            #    [item for sublist in batch.structure_id for item in sublist]
            # )
            batch_ids = batch.structure_id

            # Node level prediction
            if batch_p.shape[0] > loader.batch_size:
                node_level = True
                node_ids = batch.z.cpu().numpy()
                structure_ids = np.repeat(
                    batch_ids, batch.n_atoms.cpu().numpy(), axis=0
                )
                batch_ids = np.column_stack((structure_ids, node_ids))

            ids = batch_ids if i == 0 else np.row_stack((ids, batch_ids))
            predict = (
                batch_p if i == 0 else torch.concatenate((predict, batch_p), axis=0)
            )
            target = batch_t if i == 0 else torch.concatenate((target, batch_t), axis=0)

        if write_output:
            self.save_results(
                np.column_stack(
                    (
                        np.expand_dims(ids, axis=1),
                        target.cpu().numpy(),
                        predict.cpu().numpy(),
                    )
                ),
                results_dir,
                f"{split}_predictions.csv",
                node_level,
            )
        predict_loss = self._metrics_predict[type(self.loss_fn).__name__]["metric"]
        logging.debug("Saved {:s} error: {:.5f}".format(split, predict_loss))

        predictions = {"ids": ids, "predict": predict, "target": target}

        return predictions

    def _forward(self, batch_data):
        device = self.rank
        output = self.model(batch_data.to(device))
        return output

    def _compute_loss(self, out, batch_data):
        loss = self.loss_fn(out, batch_data)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_metrics(self, out, batch_data, metrics):
        # TODO: finish this method
        device = self.rank
        property_target = batch_data.to(device)

        metrics = self.evaluator.eval(
            out, property_target, self.loss_fn, prev_metrics=metrics
        )

        return metrics

    def _log_metrics(self, val_metrics=None, test_metrics=None):
        train_loss = self.metrics[type(self.loss_fn).__name__]["metric"]

        if not val_metrics:
            val_loss = "N/A"
        else:
            val_loss = val_metrics[type(self.loss_fn).__name__]["metric"]
        if not test_metrics:
            test_loss = "N/A"
        else:
            test_loss = test_metrics[type(self.loss_fn).__name__]["metric"]

        log_kwargs = {
            "epoch": int(self.epoch - 1),
            "lr": self.scheduler.lr,
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss != "N/A" else float("inf"),
            "test_loss": test_loss if test_loss != "N/A" else float("inf"),
            "epoch_time": self.epoch_time,
        }

        logging.info(
            "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Test Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                *log_kwargs.values()
            )
        )

        # wandb logging
        if self.use_wandb:
            wandb.log(log_kwargs)

        return {
            "lr": self.scheduler.lr,
            "train": train_loss,
            "val": val_loss,
            "time": self.epoch_time,
        }

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
