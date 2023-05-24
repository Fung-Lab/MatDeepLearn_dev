import copy
import csv
import logging
import os
import time

import torch

from matdeeplearn.common.registry import registry
# from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.trainers.property_trainer import PropertyTrainer


@registry.register_trainer("finetune")
class FinetuneTrainer(PropertyTrainer):
    def __init__(
            self,
            model,
            dataset,
            optimizer,
            sampler,
            scheduler,
            train_loader,
            val_loader,
            test_loader,
            loss,
            max_epochs,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            save_dir,
            checkpoint_dir,
            fine_tune_from
    ):
        super().__init__(
            model,
            dataset,
            optimizer,
            sampler,
            scheduler,
            train_loader,
            val_loader,
            test_loader,
            loss,
            max_epochs,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            save_dir,
            checkpoint_dir,
        )
        self.best_epoch = None
        self.fine_tune_from = fine_tune_from
        self.model = self._load_pre_trained_weights(self.model)
        self.model.to(self.device)
        self.identifier = identifier

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

        dataset = cls._load_dataset(config["dataset"])
        model = cls._load_model(config["model"], dataset["train"])
        optimizer = cls._load_optimizer(config["optim"], model)
        sampler = cls._load_sampler(config["optim"], dataset["train"])
        train_loader, val_loader, test_loader = cls._load_dataloader(
            config["optim"], config["dataset"], dataset, sampler
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
        fine_tune_from = config["model"].get("fine_tune_from", "")

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
            fine_tune_from=fine_tune_from
        )

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
            load_state = torch.load(os.path.join(checkpoints_folder, 'best_checkpoint.pt'), map_location=self.device)
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

        if training_state:
            state = {
                "epoch": self.epoch,
                "step": self.step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.scheduler.state_dict(),
                "best_val_metric": self.best_val_metric,
            }
        else:
            state = {"state_dict": self.model.state_dict(), "val_metrics": val_metrics}

        curr_checkpt_dir = os.path.join(
            self.save_dir, self.fine_tune_from, self.identifier, self.timestamp_id, "checkpoint"
        )
        os.makedirs(curr_checkpt_dir, exist_ok=True)
        filename = os.path.join(curr_checkpt_dir, checkpoint_file)

        torch.save(state, filename)
        return filename

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
            with open(os.path.join(self.save_dir, self.fine_tune_from, self.identifier, self.timestamp_id, "train.log"), "a+") as f:
                f.write(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}\n".format(
                        int(self.epoch - 1),
                        self.scheduler.lr,
                        train_loss,
                        val_loss,
                        self.epoch_time,
                    ))

    def update_best_model(self, val_metrics):
        """Updates the best val metric and model, saves the best model, and saves the best model predictions"""
        self.best_val_metric = val_metrics[type(self.loss_fn).__name__]["metric"]
        self.best_model_state = copy.deepcopy(self.model.state_dict())

        self.save_model("best_checkpoint.pt", val_metrics, False)

        logging.debug(
            f"Saving prediction results for epoch {self.epoch} to: /results/{self.timestamp_id}/"
        )
        # self.predict(self.train_loader, "train")
        # self.predict(self.val_loader, "val")
        # self.predict(self.test_loader, "test")

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

        if self.train_verbosity:
            logging.info("Starting regular training")
            logging.info(
                f"running for {end_epoch - start_epoch} epochs on {type(self.model).__name__} model"
            )

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
                batch = next(train_loader_iter).to(self.device)

                # Compute forward, loss, backward
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Compute metrics
                # TODO: revert _metrics to be empty per batch, so metrics are logged per batch, not per epoch
                #  keep option to log metrics per epoch
                _metrics = self._compute_metrics(out, batch, _metrics)
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
                        < self.best_val_metric
                ):
                    self.update_best_model(val_metrics)
                    self.best_epoch = epoch

                # step scheduler, using validation error
                self._scheduler_step()

        best_log_dir_name = os.path.join(self.save_dir, self.fine_tune_from, self.identifier)
        if not os.path.exists(best_log_dir_name):
            os.makedirs(best_log_dir_name)

        with open(os.path.join(best_log_dir_name, "best_val_metric.csv"), "a+", encoding="utf-8", newline='') as f:
            new_metric = [self.timestamp_id, self.best_val_metric, self.best_epoch]
            csv_writer = csv.writer(f)
            if not os.path.getsize(os.path.join(best_log_dir_name, "best_val_metric.csv")):
                csv_head = ["timestamp_id", "best_val_metric", "best_epoch"]
                csv_writer.writerow(csv_head)
            csv_writer.writerow(new_metric)

        return self.best_model_state
