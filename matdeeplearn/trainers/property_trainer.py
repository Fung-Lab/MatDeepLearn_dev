import logging
import time

import numpy as np
import torch

import wandb
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
        train_loader,
        val_loader,
        test_loader,
        loss,
        max_epochs,
        identifier,
        verbosity,
        save_out,
        write_output,
        save_dir,
        checkpoint_path,
        use_amp,
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
            identifier,
            verbosity,
            save_out,
            write_output,
            save_dir,
            checkpoint_path,
            use_amp,
        )

    def train(self):
        if self.train_verbosity:
            logging.info("Starting regular training")
            logging.info(
                f"running for  {self.max_epochs} epochs on {type(self.model).__name__} model"
            )

        # Start training over epochs loop
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        for epoch in range(start_epoch, self.max_epochs):
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
                #
                # # TODO: move into DOSPredict
                # # reshape s,p,d data into 1-d if for DOSPredict
                # if type(self.loss_fn).__name__ == "DOSLoss":
                #     batch.

                # Compute forward, loss, backward
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)[type(self.loss_fn).__name__]
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

                # step scheduler, using validation error
                self._scheduler_step()

        # call predict at end if didn't save throughout
        if not self.save_out:
            self.predict(self.train_loader, "train")
            self.predict(self.val_loader, "val")
            self.predict(self.test_loader, "test")

        return self.best_model_state

    def validate(self, split="val"):
        self.model.eval()
        evaluator, metrics = Evaluator(), {}

        loader_iter = (
            iter(self.val_loader) if split == "val" else iter(self.test_loader)
        )

        for i in range(0, len(loader_iter)):
            with torch.no_grad():
                batch = next(loader_iter).to(self.device)
                out = self._forward(batch.to(self.device))
                loss = self._compute_loss(out, batch)
                # Compute metrics
                metrics = self._compute_metrics(out, batch, metrics)
                metrics = evaluator.update(
                    "loss", loss[type(self.loss_fn).__name__].item(), metrics
                )

        return metrics

    @torch.no_grad()
    def predict(self, loader, split):
        # TODO: make predict method work as standalone task
        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)
        self.model.eval()

        predict, target = None, None
        ids = []
        node_level_predictions = False
        _metrics_predict = {}
        out_lst = []
        for i, batch in enumerate(loader):
            out = self._forward(batch.to(self.device))
            if type(out) != tuple:
                logging.debug(f"predict out size: {out.size()}")
                logging.debug(out)

            loss = self._compute_loss(out, batch)[type(self.loss_fn).__name__]
            _metrics_predict = self._compute_metrics(out, batch, _metrics_predict)
            self._metrics_predict = self.evaluator.update(
                "loss", loss.item(), _metrics_predict
            )

            if type(out) == tuple:
                # line up the node-level output with graphs
                scaled = out[0]
                scaling_factor = out[1]
                # TODO: change to get actual batch size
                for batch_id in range(140):
                    element_ids = (batch.batch == batch_id).nonzero(as_tuple=True)
                    scaled_out = scaled[element_ids]
                    scaling_factor_out = scaling_factor[element_ids]

                    if scaled_out.nelement() > 0:
                        out_lst.append((scaled_out, scaling_factor_out))

                # if out is a tuple, then it's scaled data, so unscale it and target
                out = out[0] * out[1].view(-1, 1).expand_as(out[0])
                target_original = batch.flat_scaled * batch.scaling_factor.view(
                    -1, 1
                ).expand_as(batch.flat_scaled)
            else:
                target_original = batch.y

            batch_p = out.data.cpu().numpy()
            batch_t = target_original.cpu().numpy()

            batch_ids = np.array(
                [item for sublist in batch.structure_id for item in sublist]
            )

            # if shape is 2D, then it has node-level predictions
            if batch_p.ndim == 2:
                node_level_predictions = True
                node_ids = batch.z.cpu().numpy()
                structure_ids = np.repeat(
                    batch_ids, batch.n_atoms.cpu().numpy(), axis=0
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

        # TODO: delete - hack to record model 2 chain output error on wandb
        if split == "model2_chain_test":
            wandb.log({"model2_predict_chain_error": predict_loss})

        if split == "model1_test":
            wandb.log({"model1_predict_error": predict_loss})

        if split == "model2_test":
            wandb.log({"model2_predict_error": predict_loss})

        return out_lst, predict_loss

    def _forward(self, batch_data):
        output = self.model(batch_data)
        return output

    def _compute_loss(self, out, batch_data):
        loss = self.loss_fn(out, batch_data.to(self.device))
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_metrics(self, out, batch_data, metrics):
        # TODO: finish this method
        property_target = batch_data.to(self.device)

        metrics = self.evaluator.eval(
            out, property_target, self.loss_fn, prev_metrics=metrics
        )

        return metrics

    def _log_metrics(self, val_metrics=None):
        if not val_metrics:
            logging.info(f"epoch: {self.epoch}, learning rate: {self.scheduler.lr}")
            logging.info(self.metrics[type(self.loss_fn).__name__]["metric"])
        else:
            train_loss = self.metrics[type(self.loss_fn).__name__]["metric"]
            val_loss = val_metrics[type(self.loss_fn).__name__]["metric"]

            log_kwargs = {
                "epoch": int(self.epoch - 1),
                "lr": self.scheduler.lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time": self.epoch_time,
            }

            logging.info(
                "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                    *log_kwargs.values()
                )
            )

            # TODO: change additional loss values we want to record to something more universal
            val_loss_dict = {}
            for key, val in val_metrics.items():
                if key != f"{type(self.loss_fn).__name__}":
                    val_loss_dict[f"val_{key}"] = val_metrics[key]["metric"]

            log_kwargs.update(val_loss_dict)

            wandb.log(log_kwargs)

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
