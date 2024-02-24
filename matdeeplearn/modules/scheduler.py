import math
import torch


class CosineLRLambda:
    def __init__(self, scheduler_params) -> None:
        self.warmup_epochs = scheduler_params["warmup_epochs"]
        self.lr_warmup_factor = scheduler_params["warmup_factor"]
        self.max_epochs = scheduler_params["epochs"]
        self.lr_min_factor = scheduler_params["lr_min_factor"]

    def __call__(self, current_step: int):
        # `warmup_epochs` is already multiplied with the num of iterations
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            lr_scale = self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                    1 + math.cos(math.pi * (current_step / self.max_epochs))
            )
            return lr_scale


class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if self.scheduler_type == "LambdaLR":
            scheduler_lambda_fn = None
            self.lambda_type = model_parameters["lambda_type"]

            if self.lambda_type == "cosine":
                scheduler_lambda_fn = CosineLRLambda(model_parameters)
            # elif self.lambda_type == "multistep":
            #     scheduler_lambda_fn = MultistepLRLambda(self.scheduler_params)
            else:
                raise ValueError
            model_parameters["lr_lambda"] = scheduler_lambda_fn

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
