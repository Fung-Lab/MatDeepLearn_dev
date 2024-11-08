import torch


class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        if scheduler_type == "LambdaLR":
            lambda_args = {
                "warmup_steps": model_parameters["warmup_steps"],
                "total_steps": model_parameters["total_steps"],
                "warmup_factor": model_parameters["warmup_factor"],
                "power": model_parameters["power"],
            }
            model_parameters["lr_lambda"] = lambda step: warmup_polynomial_decay_lr_lambda(step, **lambda_args)
            for key in lambda_args.keys():
                del model_parameters[key]
                
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


def warmup_polynomial_decay_lr_lambda(current_step: int, **kwargs):
    print(current_step)
    if current_step <= kwargs["warmup_steps"]:
        alpha = current_step / float(kwargs["warmup_steps"])
        return kwargs["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        decay_steps = kwargs["total_steps"] - kwargs["warmup_steps"]
        decay_rate = (1 - (current_step - kwargs["warmup_steps"]) / decay_steps) ** kwargs["power"]
        return max(0.0, decay_rate)  # Ensure learning rate doesn't go negative