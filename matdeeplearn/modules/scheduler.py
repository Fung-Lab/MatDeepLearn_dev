import torch
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if self.scheduler_type == "LambdaLR":
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer,
                model_parameters["warmup_steps"],
                model_parameters["epochs"] * model_parameters["batch_size"],
            )
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


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
