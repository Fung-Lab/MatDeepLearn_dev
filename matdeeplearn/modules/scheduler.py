import torch


class LRScheduler:
    """ wrapper around torch.optim.lr_scheduler._LRScheduler"""
    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
            optimizer, **model_parameters
        )

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
                raise Exception(
                    "Validation set required for ReduceLROnPlateau."
                )
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
