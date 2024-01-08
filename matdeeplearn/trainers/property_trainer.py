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
                    f"Running for {end_epoch - start_epoch} epochs on {type(self.model[0].module).__name__} model"
                )
            else:
                logging.info(
                    f"Running for {end_epoch - start_epoch} epochs on {type(self.model[0]).__name__} model"
                )
     
        for epoch in range(start_epoch, end_epoch):            
            epoch_start_time = time.time()
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            # skip_steps = self.step % len(self.train_loader)
            train_loader_iter = []
            for i in range(len(self.model)):
                train_loader_iter.append(iter(self.data_loader[i]["train_loader"]))
            # metrics for every epoch
            _metrics = [{} for _ in range(len(self.model))]
            
            #for i in range(skip_steps, len(self.train_loader)):
            pbar = tqdm(range(0, len(self.data_loader[0]["train_loader"])), disable=not self.batch_tqdm)
            for i in pbar:                                
                #self.epoch = epoch + (i + 1) / len(self.train_loader)
                #self.step = epoch * len(self.train_loader) + i + 1
                #print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024)) 
                batch = []
                for n, mod in enumerate(self.model):
                    mod.train()
                    batch.append(next(train_loader_iter[n]).to(self.rank))
                # Get a batch of train data
                # batch = next(train_loader_iter).to(self.rank) 
                # print(epoch, i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024), torch.sum(batch.n_atoms))          
                # Compute forward, loss, backward    
                with autocast(enabled=self.use_amp):
                    out_list = self._forward(batch)                                            
                    loss = self._compute_loss(out_list, batch) 
                #print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))                                               
                grad_norm = []
                for i in range(len(self.model)):
                    grad_norm.append(self._backward(loss[i], i))
                pbar.set_description("Batch Loss {:.4f}, grad norm {:.4f}".format(torch.mean(torch.stack(loss)).item(), torch.mean(torch.stack(grad_norm)).item()))
                # Compute metrics
                # TODO: revert _metrics to be empty per batch, so metrics are logged per batch, not per epoch
                #  keep option to log metrics per epoch  
                for n in range(len(self.model)):
                    _metrics[n] = self._compute_metrics(out_list[n], batch[n], _metrics[n])
                    self.metrics[n] = self.evaluator.update("loss", loss[n].item(), out_list[n]["output"].shape[0], _metrics[n])

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
                if self.data_loader[0].get("val_loader"):
                    metric = self.validate("val") 
                else:
                    metric = self.metrics

                # Train loop timings
                self.epoch_time = time.time() - epoch_start_time
                # Log metrics
                if epoch % self.train_verbosity == 0:
                    if self.data_loader[0].get("val_loader"):
                        self._log_metrics(metric)
                    else:
                        self._log_metrics()

                # Update best val metric and model, and save best model and predicted outputs
                for i in range(len(self.model)):
                    if metric[i][type(self.loss_fn).__name__]["metric"] < self.best_metric[i]:
                        if self.output_frequency == 0:
                            if self.model_save_frequency == 1:
                                self.update_best_model(metric[i], i, write_model=True, write_csv=False)
                            else:
                                self.update_best_model(metric[i], i, write_model=False, write_csv=False)
                        elif self.output_frequency == 1:
                            if self.model_save_frequency == 1:
                                self.update_best_model(metric[i], i, write_model=True, write_csv=True)
                            else:
                                self.update_best_model(metric[i], i, write_model=False, write_csv=True)
                    
                self._scheduler_step()
                

            torch.cuda.empty_cache()        
        
        if self.best_model_state:
            for i in range(len(self.model)):
                if str(self.rank) in "0":
                    self.model[i].module.load_state_dict(self.best_model_state[i])
                elif str(self.rank) in ("cpu", "cuda"):
                    self.model[i].load_state_dict(self.best_model_state[i])
            #if self.data_loader.get("test_loader"):
            #    metric = self.validate("test")
            #    test_loss = metric[type(self.loss_fn).__name__]["metric"]
            #else:
            #    test_loss = "N/A"             
            if self.model_save_frequency != -1:
                self.save_model("best_checkpoint.pt", index=None, metric=metric, training_state=True)
            logging.info("Final Losses: ")     
            if "train" in self.write_output:
                self.predict(self.data_loader[0]["train_loader"], "train")
            if "val" in self.write_output and self.data_loader[0].get("val_loader"):
                self.predict(self.data_loader[0]["val_loader"], "val")
            if "test" in self.write_output and self.data_loader[0].get("test_loader"):
                self.predict(self.data_loader[0]["test_loader"], "test") 
            
        return self.best_model_state
        
    @torch.no_grad()
    def validate(self, split="val"):
        for i in range(len(self.model)):
            self.model[i].eval()
        
        evaluator, metrics = Evaluator(), [{} for _ in range(len(self.model))]

        loader_iter = []
        for i in range(len(self.model)):
            if split == "val":
                loader_iter.append(iter(self.data_loader[i]["val_loader"]))
            elif split == "test":
                loader_iter.append(iter(self.data_loader[i]["test_loader"]))
            elif split == "train":
                loader_iter.append(iter(self.data_loader[i]["train_loader"]))
                
        for i in range(0, len(loader_iter[0])):
            #print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))  
            batch = []
            for i in range(len(self.model)):
                batch.append(next(loader_iter[i]).to(self.rank))
            
            out_list = self._forward(batch)
            loss = self._compute_loss(out_list, batch)
            # Compute metrics
            #print(i, torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_cached() / (1024 * 1024))          
            for n in range(len(self.model)):
                metrics[n] = self._compute_metrics(out_list[n], batch[n], metrics[n])
                metrics[n] = evaluator.update("loss", loss[n].item(), out_list[n]["output"].shape[0], metrics[n])    
            del loss, batch, out_list
        
        torch.cuda.empty_cache()
        
        return metrics

    @torch.no_grad()
    def predict(self, loader, split, results_dir="train_results", write_output=True, labels=True):        
        for mod in self.model:
            mod.eval()
         
        # assert isinstance(loader, torch.utils.data.dataloader.DataLoader)

        # TODO: make this compatible with model ensemble
        if str(self.rank) not in ("cpu", "cuda"):
            loader = get_dataloader(
                loader.dataset, batch_size=loader.batch_size, sampler=None
            )
            
        evaluator, metrics = Evaluator(), {}
        predict, target = None, None
        ids = []
        ids_pos_grad = []
        target_pos_grad = None
        ids_cell_grad = []
        target_cell_grad = None
        node_level = False 
                
        loader_iter = iter(loader)        
        for i in range(0, len(loader_iter)):
            batch = next(loader_iter).to(self.rank)
            out_list = self._forward([batch])
            
            out = {}
            out_stack={}            
            for key in out_list[0].keys():
                temp = [o[key] for o in out_list]
                if temp[0] is not None:
                    out_stack[key] = torch.stack(temp)
                    out[key] = torch.mean(out_stack[key], dim=0)
                    out[key+"_std"] = torch.std(out_stack[key], dim=0)
                else:
                    out[key] = None
                    out[key+"_std"] = None
                    
                    
            batch_p = [o["output"].data.cpu().numpy() for o in out_list]
            batch_p_mean = out["output"].cpu().numpy()   
            batch_ids = batch.structure_id
            batch_stds = out["output_std"].cpu().numpy()   

            if labels == True:
                loss = self._compute_loss(out, batch)
                metrics = self._compute_metrics(out, batch, metrics)
                metrics = evaluator.update(
                    "loss", loss.item(), out["output"].shape[0], metrics
                )
                if str(self.rank) not in ("cpu", "cuda"):
                    batch_t = batch[self.model[0].module.target_attr].cpu().numpy()
                else:
                    batch_t = batch[self.model[0].target_attr].cpu().numpy()
                        
            # Node level prediction 
            if batch_p[0].shape[0] > loader.batch_size: 
                node_level = True
                node_ids = batch.z.cpu().numpy()
                structure_ids = np.repeat(
                    batch.structure_id, batch.n_atoms.cpu().numpy(), axis=0
                )
                batch_ids = np.column_stack((structure_ids, node_ids))

            if out.get("pos_grad") != None:
                batch_p_pos_grad = out["pos_grad"].data.cpu().numpy()
                batch_p_pos_grad_std = out["pos_grad_std"].data.cpu().numpy()
                node_ids_pos_grad = batch.z.cpu().numpy()
                structure_ids_pos_grad = np.repeat(
                    batch.structure_id, batch.n_atoms.cpu().numpy(), axis=0
                )
                batch_ids_pos_grad = np.column_stack((structure_ids_pos_grad, node_ids_pos_grad)) 
                ids_pos_grad = batch_ids_pos_grad if i == 0 else np.row_stack((ids_pos_grad, batch_ids_pos_grad))            
                predict_pos_grad = batch_p_pos_grad if i == 0 else np.concatenate((predict_pos_grad, batch_p_pos_grad), axis=0)
                predict_pos_grad_std = batch_p_pos_grad_std if i == 0 else np.concatenate((predict_pos_grad_std, batch_p_pos_grad_std), axis=0)
                if "forces" in batch:
                    batch_t_pos_grad = batch["forces"].cpu().numpy()      
                    target_pos_grad = batch_t_pos_grad if i == 0 else np.concatenate((target_pos_grad, batch_t_pos_grad), axis=0)

            if out.get("cell_grad") != None:  
                batch_p_cell_grad = out["cell_grad"].data.view(out["cell_grad"].data.size(0), -1).cpu().numpy()
                batch_p_cell_grad_std = out["cell_grad_std"].data.view(out["cell_grad"].data.size(0), -1).cpu().numpy()
                batch_ids_cell_grad = batch.structure_id               
                ids_cell_grad = batch_ids_cell_grad if i == 0 else np.row_stack((ids_cell_grad, batch_ids_cell_grad))            
                predict_cell_grad = batch_p_cell_grad if i == 0 else np.concatenate((predict_cell_grad, batch_p_cell_grad), axis=0)
                predict_cell_grad_std = batch_p_cell_grad_std if i == 0 else np.concatenate((predict_cell_grad_std, batch_p_cell_grad_std), axis=0)
                if "stress" in batch:
                    batch_t_cell_grad = batch["stress"].view(out["cell_grad"].data.size(0), -1).cpu().numpy()
                    target_cell_grad = batch_t_cell_grad if i == 0 else np.concatenate((target_cell_grad, batch_t_cell_grad), axis=0)                          
            
            ids = batch_ids if i == 0 else np.row_stack((ids, batch_ids))   
            predict_mean = batch_p_mean if i == 0 else np.concatenate((predict_mean, batch_p_mean), axis=0)       
            stds = batch_stds if i == 0 else np.row_stack((stds, batch_stds))    
            if i == 0:                        
                predict = [0 for _ in range(len(self.model))]        
            for x in range(len(self.model)):
                predict[x] = batch_p[x] if i == 0 else np.concatenate((predict[x], batch_p[x]), axis=0)

            if labels == True:
                target = batch_t if i == 0 else np.concatenate((target, batch_t), axis=0)
            
            if labels == True:
                del loss, batch, out 
            else:  
                del batch, out 
                       
        if write_output == True:
            if labels == True:
                if len(self.model) > 1:    
                    self.save_results(
                        np.column_stack((ids, target, predict_mean, stds)), results_dir, f"{split}_predictions.csv", node_level, True, std=True,
                    )                                                         
                    for x in range(len(self.model)):
                        mod = str(x)                                                            
                        self.save_results(
                            np.column_stack((ids, target, predict[x])), results_dir, f"{split}_predictions_{mod}.csv", node_level, True, std=False,
                        )
                else: 
                    self.save_results(
                        np.column_stack((ids, target, predict_mean)), results_dir, f"{split}_predictions.csv", node_level, True, std=False,
                    )                                     
            else:
                if len(self.model) > 1:  
                    self.save_results(
                        np.column_stack((ids, predict_mean, stds)), results_dir, f"{split}_predictions.csv", node_level, False, std=True,
                    )                    
                    for x in range(len(self.model)):
                        mod = str(x)
                        self.save_results(
                        	np.column_stack((ids, predict[x])), results_dir, f"{split}_predictions_{mod}.csv", node_level, False, std=False,
                        )
                else: 
                    self.save_results(
                        np.column_stack((ids, predict_mean)), results_dir, f"{split}_predictions.csv", node_level, False, std=False,
                    )                                   
            #if out.get("pos_grad") != None:
            if len(ids_pos_grad) > 0:
                if isinstance(target_pos_grad, np.ndarray):
                    if len(self.model) > 1: 
                        self.save_results(
                            np.column_stack((ids_pos_grad, target_pos_grad, predict_pos_grad, predict_pos_grad_std)), results_dir, f"{split}_predictions_pos_grad.csv", True, True, std=True
                        )
                    else:
                        self.save_results(
                            np.column_stack((ids_pos_grad, target_pos_grad, predict_pos_grad)), results_dir, f"{split}_predictions_pos_grad.csv", True, True, std=False
                        )                    
                else:
                    self.save_results(
                        np.column_stack((ids_pos_grad, predict_pos_grad)), results_dir, f"{split}_predictions_pos_grad.csv", True, False, std=False
                    )
            #if out.get("cell_grad") != None:
            if len(ids_cell_grad) > 0:
                if isinstance(target_cell_grad, np.ndarray):
                    if len(self.model) > 1: 
                        self.save_results(
                            np.column_stack((ids_cell_grad, target_cell_grad, predict_cell_grad, predict_cell_grad_std)), results_dir, f"{split}_predictions_cell_grad.csv", False, True, std=True
                        )
                    else:
                        self.save_results(
                            np.column_stack((ids_cell_grad, target_cell_grad, predict_cell_grad)), results_dir, f"{split}_predictions_cell_grad.csv", False, True, std=False
                        )                    
                else:
                    self.save_results(
                        np.column_stack((ids_cell_grad, predict_cell_grad)), results_dir, f"{split}_predictions_cell_grad.csv", False, False, std=False
                    )
                                                            
        if labels == True:
            predict_loss = metrics[type(self.loss_fn).__name__]["metric"]
            logging.info("Saved {:s} error: {:.5f}".format(split, predict_loss))    
            if len(self.model) > 1:     
                predictions = {"ids":ids, "predict":predict_mean, "target":target, "std": stds}
            else:
                predictions = {"ids":ids, "predict":predict_mean, "target":target}
        else:
            if len(self.model) > 1:     
                predictions = {"ids":ids, "predict":predict_mean, "std": stds}
            else:
                predictions = {"ids":ids, "predict":predict_mean} 

        torch.cuda.empty_cache()
        
        return predictions
        
    def predict_by_calculator(self, loader):        
        for x, mod in self.model:
            mod.eval()
        
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
            out_list = self._forward(batch.to(self.rank))
            out = {}
            out_stack={}            
            for key in out_list[0].keys():
                temp = [o[key] for o in out_list]
                if temp[0] is not None:
                    out_stack[key] = torch.stack(temp)
                    out[key] = torch.mean(out_stack[key], dim=0)
                else:
                    out[key] = None

            energy = None if out.get('output') is None else out.get('output').data.cpu().numpy()
            stress = None if out.get('cell_grad') is None else out.get('cell_grad').view(-1, 3).data.cpu().numpy()
            forces = None if out.get('pos_grad') is None else out.get('pos_grad').data.cpu().numpy()
            
            results = {'energy': energy, 'stress': stress, 'forces': forces}
        
        return results

    def _forward(self, batch_data):
        if len(batch_data) > 1:
            output = []
            for i in range(len(self.model)):
                output.append(self.model[i](batch_data[i]))
        else:
            output = []
            for i in range(len(self.model)):
                output.append(self.model[i](batch_data[0]))
        return output

    def _compute_loss(self, out, batch_data):
        if isinstance(out, list):
            loss = []
            for i in range(len(out)):
                loss.append(self.loss_fn(out[i], batch_data[i]))
        else:
            loss = self.loss_fn(out, batch_data)
        return loss

    def _backward(self, loss, index=None):
        self.optimizer[index].zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model[index].parameters(),
                max_norm=self.clip_grad_norm,
            )
        self.scaler.step(self.optimizer[index])
        self.scaler.update()
            
        return grad_norm


    def _compute_metrics(self, out, batch_data, metrics):
        # TODO: finish this method
        try:
            property_target = batch_data.to(self.rank)
        except:
            property_target = batch_data

        metrics = self.evaluator.eval(
            out, property_target, self.loss_fn, prev_metrics=metrics
	    )

        return metrics

    def _log_metrics(self, val_metrics=None):
        train_loss = [torch.tensor(i[type(self.loss_fn).__name__]["metric"]) for i in self.metrics]
        train_loss = torch.mean(torch.stack(train_loss)).item()
        lr = self.scheduler[0].lr    
        if not val_metrics:
            val_loss = "N/A"
            logging.info(
                "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {}, Time per epoch (s): {:.5f}".format(
                    int(self.epoch - 1),
                    lr,
                    train_loss,
                    val_loss,
                    self.epoch_time,
                )
            )
        else:
            val_loss = [torch.tensor(i[type(self.loss_fn).__name__]["metric"]) for i in val_metrics]
            val_loss = torch.mean(torch.stack(val_loss)).item()
            lr = self.scheduler[0].lr
            logging.info(
                "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                    int(self.epoch - 1),
                    lr,
                    train_loss,
                    val_loss,
                    self.epoch_time,
                )
            )


    def _load_task(self):
        """Initializes task-specific info. Implemented by derived classes."""
        pass

    def _scheduler_step(self):
        for i in range(len(self.model)):
            if self.scheduler[i].scheduler_type == "ReduceLROnPlateau":
                self.scheduler[i].step(
                    metrics=self.metrics[i][type(self.loss_fn).__name__]["metric"]
                )
            else:
                self.scheduler[i].step()
