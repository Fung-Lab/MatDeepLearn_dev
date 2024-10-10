import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_poly_decay_lr(warmup_steps, total_steps, poly_power=0.5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return (1.0 - float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))) ** poly_power
    
    return lr_lambda

class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if scheduler_type == "LambdaLR":
            assert "lr_lambda" in model_parameters
            self.scheduler = LambdaLR(optimizer, lr_lambda=warmup_poly_decay_lr(**model_parameters['lr_lambda']))
        else:
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
                optimizer, **model_parameters
            )

        self.lr = self.optimizer.param_groups[0]["lr"]

    @classmethod
    def from_config(cls, optimizer, optim_config):
        scheduler_type = optim_config["scheduler_type"]
        scheduler_args = optim_config["scheduler_args"]
        return cls(optimizer, scheduler_type, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

        # update the learning rate attribute to current lr
        self.update_lr()

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]
