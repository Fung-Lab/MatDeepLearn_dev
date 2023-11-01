import time
import numpy as np
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import Batch

class PositionOptimizer(ABC):
    def __init__(
        self,
        representation_model,
        loss_fn="l1_loss",
        optimizer_config=None,
        scheduler_config=None,
        max_iterations=50,
        verbosity=1,
        **kwargs,
    ):
        self.representation_model = representation_model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.max_iterations = max_iterations
        self.verbosity = verbosity

        self.loss_fn = PositionOptimizer.load_loss_function(loss_fn)

    def get_representation(self, batch):
        self.representation_model.eval()
        out = self.representation_model(batch)["representation"]
        self.representation_model.train()
        return out

    def optimize(self, batch, target_rep, max_iterations=None):
        if max_iterations is None:
            max_iterations = self.max_iterations

        batch.pos.requires_grad_(True)
        optimizer = PositionOptimizer.load_optimizer(self.optimizer_config, batch.pos)
        scheduler = PositionOptimizer.load_scheduler(self.scheduler_config, optimizer)

        tic = time.time()
        iter_start_loss = None
        iter_end_loss = None

        for i in range(max_iterations):
            if i >= max_iterations -1:
                batch_pos = batch.pos 
                pos_old = batch.pos.clone().requires_grad_(True)
                batch.pos = pos_old
                rep_old = self.get_representation(batch)
                loss = self.loss_fn(rep_old, target_rep)

                pos_grad = torch.autograd.grad(loss, batch.pos, create_graph=True)[0]
                batch_pos = batch_pos.add(pos_grad, alpha=-1)
                batch.pos = batch_pos

                print("iter {} | grad: {}".format(i, pos_grad))

                iter_end_loss = loss.item()
            else:
                optimizer.zero_grad()
                
                # get representation for current iteration
                curr_rep = self.get_representation(batch)

                # calculate loss
                loss = self.loss_fn(curr_rep, target_rep)
                pos_grad = torch.autograd.grad(loss, batch.pos)[0]
                batch.pos.grad = pos_grad

                print("iter {} | grad: {}".format(i, pos_grad))

                # update
                # loss.backward(retain_graph=True)

                optimizer.step()
                scheduler.step(loss)

                if i == 0:
                    iter_start_loss = loss.item()

        print("iter start loss: {} | iter end loss: {}".format(iter_start_loss, iter_end_loss))
        
        return batch.pos, loss.item()

    @staticmethod
    def load_optimizer(optimizer_config, positions):
        if optimizer_config is None:
            optimizer_config = {"optimizer_type": "AdamW", "lr":0.02}

        optimizer = getattr(optim, optimizer_config["optimizer_type"])(
            [positions],
            lr=optimizer_config["lr"],
            **optimizer_config.get("optimizer_args", {})
        )

        return optimizer

    @staticmethod
    def load_scheduler(scheduler_config, optimizer):
        if scheduler_config is None:
            scheduler_config = {
                "scheduler_type": "ReduceLROnPlateau",
                "scheduler_args": {"mode":"min", "factor":0.8, "patience":10, "min_lr":0.000001, "threshold":0.0001}
            }

        scheduler_type = scheduler_config["scheduler_type"]
        scheduler_args = scheduler_config["scheduler_args"]
        scheduler = getattr(optim.lr_scheduler, scheduler_type)(
            optimizer, **scheduler_args
        )
        return scheduler

    @staticmethod
    def load_loss_function(loss_fn_name):
        return getattr(nn.functional, loss_fn_name)

def positional_encoder_loss(true_pos, opt_pos):
    """
    placeholder loss function for optimized positions and true positions
    """
    # print("true: ", true_pos.requires_grad)
    # print("opt: ", opt_pos.requires_grad)

    return torch.mean(torch.abs(true_pos - opt_pos))
