import copy
import logging
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.cuda.amp import autocast

from tqdm import tqdm
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
            clip_grad_norm,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            batch_tqdm,
            write_output,
            output_frequency,
            model_save_frequency,
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
            data_loader,
            loss,
            max_epochs,
            clip_grad_norm,
            max_checkpoint_epochs,
            identifier,
            verbosity,
            batch_tqdm,
            write_output,
            output_frequency,
            model_save_frequency,
            save_dir,
            checkpoint_path,
            use_amp,
        )

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
                self.metrics = self.evaluator.update("loss", loss.item(), out["output"].shape[0], _metrics)

            self.epoch = epoch + 1

            if str(self.rank) not in ("cpu", "cuda"):
                dist.barrier()

            # TODO: could add param to eval and save on increments instead of every time

            # Save current model
            torch.cuda.empty_cache()
            if str(self.rank) in ("0", "cpu", "cuda"):
                if self.model_save_frequency == 1:
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

                if epoch + 1 in [12, 25, 50, 100, 200, 300, 400]:
                    current_state = copy.deepcopy(self.model.state_dict())
                    self.model.load_state_dict(self.best_model_state)
                    self.predict(self.data_loader["test_loader"], "test")
                    self.model.load_state_dict(current_state)


                # Update best val metric and model, and save best model and predicted outputs
                if metric[type(self.loss_fn).__name__]["metric"] < self.best_metric:
                    if self.output_frequency == 0:
                        if self.model_save_frequency == 1:
                            self.update_best_model(metric, write_model=True, write_csv=False)
                        else:
                            self.update_best_model(metric, write_model=False, write_csv=False)
                    elif self.output_frequency == 1:
                        if self.model_save_frequency == 1:
                            self.update_best_model(metric, write_model=True, write_csv=True)
                        else:
                            self.update_best_model(metric, write_model=False, write_csv=True)
                # step scheduler, using validation error
                self._scheduler_step()

            torch.cuda.empty_cache()

        if self.best_model_state:
            if str(self.rank) in "0":
                self.model.module.load_state_dict(self.best_model_state)
            elif str(self.rank) in ("cpu", "cuda"):
                self.model.load_state_dict(self.best_model_state)
            # if self.data_loader.get("test_loader"):
            #    metric = self.validate("test")
            #    test_loss = metric[type(self.loss_fn).__name__]["metric"]
            # else:
            #    test_loss = "N/A"
            if self.model_save_frequency != -1:
                self.save_model("best_checkpoint.pt", metric, True)
            logging.info("Final Losses: ")
            if "train" in self.write_output:
                self.predict(self.data_loader["train_loader"], "train")
            if "val" in self.write_output and self.data_loader.get("val_loader"):
                self.predict(self.data_loader["val_loader"], "val")
            if "test" in self.write_output and self.data_loader.get("test_loader"):
                self.predict(self.data_loader["test_loader"], "test")

        return self.best_model_state

    @torch.no_grad()
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
            # print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))
            batch = next(loader_iter).to(self.rank)
            out = self._forward(batch.to(self.rank))
            loss = self._compute_loss(out, batch)
            # Compute metrics
            # print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))
            metrics = self._compute_metrics(out, batch, metrics)
            metrics = evaluator.update("loss", loss.item(), out["output"].shape[0], metrics)
            del loss, batch, out

        torch.cuda.empty_cache()

        return metrics

    @torch.no_grad()
    def predict(self, loader, split, results_dir="train_results", write_output=True, labels=True):
        self.model.eval()

        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)

        if str(self.rank) not in ("cpu", "cuda"):
            loader = get_dataloader(
                loader.dataset, batch_size=loader.batch_size, sampler=None
            )

        evaluator, metrics = Evaluator(), {}
        predict, target = None, None
        virtual_pos = None
        ids = []
        ids_pos_grad = []
        target_pos_grad = None
        ids_cell_grad = []
        target_cell_grad = None
        node_level = False
        loader_iter = iter(loader)
        for i in range(0, len(loader_iter)):
            batch = next(loader_iter).to(self.rank)
            out = self._forward(batch.to(self.rank))
            batch_p = out["output"].data.cpu().numpy()
            batch_ids = batch.structure_id

            if labels == True:
                loss = self._compute_loss(out, batch)
                metrics = self._compute_metrics(out, batch, metrics)
                metrics = evaluator.update(
                    "loss", loss.item(), out["output"].shape[0], metrics
                )
                if str(self.rank) not in ("cpu", "cuda"):
                    batch_t = batch[self.model.module.target_attr].cpu().numpy()
                else:
                    batch_t = batch[self.model.target_attr].cpu().numpy()
                    # batch_ids = np.array(
                #    [item for sublist in batch.structure_id for item in sublist]
                # )

            # Node level prediction
            if batch_p.shape[0] > loader.batch_size:
                node_level = True
                if self.model.prediction_level == "virtual":
                    virtual_mask = torch.argwhere(batch.z == 100).squeeze(1)
                    node_ids = torch.index_select(batch.z, 0, virtual_mask).cpu().numpy()
                    # node_ids = batch.z.cpu().numpy()
                    # print(batch.n_atoms.cpu().numpy())
                    virtual_batch = torch.index_select(batch.batch, 0, virtual_mask).cpu().numpy()
                    batch_virtual_pos = torch.index_select(batch.pos, 0, virtual_mask).cpu().numpy()
                    virtual_pos = batch_virtual_pos if i == 0 else np.concatenate((virtual_pos, batch_virtual_pos), axis=0)
                    structure_id_np = np.array(batch.structure_id)
                    structure_ids = structure_id_np[virtual_batch]

                else:
                    node_ids = batch.z.cpu().numpy()
                    structure_ids = np.repeat(
                        batch.structure_id, batch.n_atoms.cpu().numpy(), axis=0
                    )
                batch_ids = np.column_stack((structure_ids, node_ids))

            if out.get("pos_grad") != None:
                batch_p_pos_grad = out["pos_grad"].data.cpu().numpy()
                node_ids_pos_grad = batch.z.cpu().numpy()
                structure_ids_pos_grad = np.repeat(
                    batch.structure_id, batch.n_atoms.cpu().numpy(), axis=0
                )
                batch_ids_pos_grad = np.column_stack((structure_ids_pos_grad, node_ids_pos_grad))
                ids_pos_grad = batch_ids_pos_grad if i == 0 else np.row_stack((ids_pos_grad, batch_ids_pos_grad))
                predict_pos_grad = batch_p_pos_grad if i == 0 else np.concatenate((predict_pos_grad, batch_p_pos_grad),
                                                                                  axis=0)
                if "forces" in batch:
                    batch_t_pos_grad = batch["forces"].cpu().numpy()
                    target_pos_grad = batch_t_pos_grad if i == 0 else np.concatenate(
                        (target_pos_grad, batch_t_pos_grad), axis=0)

            if out.get("cell_grad") != None:
                batch_p_cell_grad = out["cell_grad"].data.view(out["cell_grad"].data.size(0), -1).cpu().numpy()
                batch_ids_cell_grad = batch.structure_id
                ids_cell_grad = batch_ids_cell_grad if i == 0 else np.row_stack((ids_cell_grad, batch_ids_cell_grad))
                predict_cell_grad = batch_p_cell_grad if i == 0 else np.concatenate(
                    (predict_cell_grad, batch_p_cell_grad), axis=0)
                if "stress" in batch:
                    batch_t_cell_grad = batch["stress"].view(out["cell_grad"].data.size(0), -1).cpu().numpy()
                    target_cell_grad = batch_t_cell_grad if i == 0 else np.concatenate(
                        (target_cell_grad, batch_t_cell_grad), axis=0)

            ids = batch_ids if i == 0 else np.row_stack((ids, batch_ids))
            predict = batch_p if i == 0 else np.concatenate((predict, batch_p), axis=0)
            if labels == True:
                target = batch_t if i == 0 else np.concatenate((target, batch_t), axis=0)

            if labels == True:
                del loss, batch, out
            else:
                del batch, out

        if write_output == True:
            if labels == True:
                self.save_results(
                    np.column_stack((ids,virtual_pos, target, predict)), results_dir, f"{split}_predictions.csv", node_level
                )
            else:
                self.save_results(
                    np.column_stack((ids,virtual_pos, predict)), results_dir, f"{split}_predictions.csv", node_level
                )

            # if out.get("pos_grad") != None:
            if len(ids_pos_grad) > 0:
                if isinstance(target_pos_grad, np.ndarray):
                    self.save_results(
                        np.column_stack((ids_pos_grad, target_pos_grad, predict_pos_grad)), results_dir,
                        f"{split}_predictions_pos_grad.csv", True, True
                    )
                else:
                    self.save_results(
                        np.column_stack((ids_pos_grad, predict_pos_grad)), results_dir,
                        f"{split}_predictions_pos_grad.csv", True, False
                    )
                    # if out.get("cell_grad") != None:
            if len(ids_cell_grad) > 0:
                if isinstance(target_cell_grad, np.ndarray):
                    self.save_results(
                        np.column_stack((ids_cell_grad, target_cell_grad, predict_cell_grad)), results_dir,
                        f"{split}_predictions_cell_grad.csv", False, True
                    )
                else:
                    self.save_results(
                        np.column_stack((ids_cell_grad, predict_cell_grad)), results_dir,
                        f"{split}_predictions_cell_grad.csv", False, False
                    )

        if labels == True:
            predict_loss = metrics[type(self.loss_fn).__name__]["metric"]
            logging.info("Saved {:s} error: {:.5f}".format(split, predict_loss))
            predictions = {"ids": ids, "predict": predict, "target": target}
        else:
            predictions = {"ids": ids, "predict": predict}

        torch.cuda.empty_cache()

        
        return predictions
        
    def predict_by_calculator(self, loader):        
        self.model.eval()
         
        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)
        assert len(loader) == 1, f"Predicting by calculator only allows one structure at a time, but got {len(loader)} structures."

        if str(self.rank) not in ("cpu", "cuda"):
            loader = get_dataloader(
                loader.dataset, batch_size=loader.batch_size, sampler=None
            )
            
        results = []
        loader_iter = iter(loader)
        for i in range(0, len(loader_iter)):
            batch = next(loader_iter).to(self.rank)      
            out = self._forward(batch.to(self.rank))
            
            energy = None if out.get('output') is None else out.get('output').data.cpu().numpy()
            stress = None if out.get('cell_grad') is None else out.get('cell_grad').view(-1, 3).data.cpu().numpy()
            forces = None if out.get('pos_grad') is None else out.get('pos_grad').data.cpu().numpy()
            
            results = {'energy': energy, 'stress': stress, 'forces': forces}
        return results

    def _forward(self, batch_data):
        output = self.model(batch_data)
        return output

    def _compute_loss(self, out, batch_data):
        loss = self.loss_fn(out, batch_data)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return grad_norm

    def _compute_metrics(self, out, batch_data, metrics):
        # TODO: finish this method
        property_target = batch_data.to(self.rank)

        metrics = self.evaluator.eval(
            out, property_target, self.loss_fn, prev_metrics=metrics
        )

        return metrics

    def _log_metrics(self, val_metrics=None):
        train_loss = self.metrics[type(self.loss_fn).__name__]["metric"]
        if not val_metrics:
            val_loss = "N/A"
            logging.info(
                "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {}, Time per epoch (s): {:.5f}".format(
                    int(self.epoch - 1),
                    self.scheduler.lr,
                    train_loss,
                    val_loss,
                    self.epoch_time,
                )
            )
        else:
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
