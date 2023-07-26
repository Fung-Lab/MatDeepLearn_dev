# Torch imports
import numpy as np
import torch
import torch.nn.functional as F


class Evaluator:
    def __init__(self, task=None):
        self.task = task
    
    def eval(self, prediction, target, loss_method, prev_metrics={}):
        metrics = prev_metrics
        res = loss_method(prediction, target)

        metrics = self.update(type(loss_method).__name__, res.item(), prediction["output"].shape[0], metrics)

        return metrics

    def update(self, key, stat, count, metrics):
        if key not in metrics:
            metrics[key] = {
                "metric": None,
                "total": 0,
                "numel": 0,
            }
        '''
        if isinstance(stat, dict):
            # If dictionary, we expect it to have `metric`, `total`, `numel`.
            metrics[key]["total"] += stat["total"]
            metrics[key]["numel"] += stat["numel"]
            metrics[key]["metric"] = metrics[key]["total"] / metrics[key]["numel"]
        elif isinstance(stat, float) or isinstance(stat, int):
            # If float or int, just add to the total and increment numel by 1.
            metrics[key]["total"] += stat
            metrics[key]["numel"] += 1
            metrics[key]["metric"] = metrics[key]["total"] / metrics[key]["numel"]
        elif torch.is_tensor(stat):
            raise NotImplementedError
        '''
        metrics[key]["total"] += stat * count
        metrics[key]["numel"] += count
        metrics[key]["metric"] = metrics[key]["total"] / metrics[key]["numel"]
            
        return metrics
