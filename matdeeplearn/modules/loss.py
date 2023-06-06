from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from matdeeplearn.common.registry import registry


@registry.register_loss("TorchLossWrapper")
class TorchLossWrapper(nn.Module):
    def __init__(self, loss_fn="l1_loss"):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)

    def forward(self, predictions: torch.Tensor, target: Batch):
        return self.loss_fn(predictions, target.y)

@registry.register_loss("NoisyNodeLoss")
class TorchLossWrapper(nn.Module):
    def __init__(self, loss_fn="l1_loss", loss_on="noise"):
        """
        loss_on: str
            select which output to perform loss fn on
            "noise": loss on noise only
            "y": loss on target property
            "all": loss on both target property and noise
        """
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)
        self.loss_on = loss_on

    def forward(self, outputs: torch.Tensor, target: Batch):
        # only works for torchmd net now
        x, noise_pred = outputs

        if self.loss_on == "noise":
            if noise_pred is None:
                raise ValueError("returned noise is None")
            return self.loss_fn(noise_pred, target.noise, reduction='sum')
        elif self.loss_on == "y":
            y_reshaped = target.y.view(-1,1)
            return self.loss_fn(x, y_reshaped)
        elif self.loss_on == "all":
            raise NotImplementedError("not implemented")

@registry.register_loss("DOSLoss")
class DOSLoss(nn.Module):
    def __init__(
        self,
        loss_fn="l1_loss",
        scaling_weight=0.05,
        cumsum_weight=0.005,
        features_weight=0.15,
    ):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_weight = scaling_weight
        self.cumsum_weight = cumsum_weight
        self.features_weight = features_weight

    def forward(self, predictions: tuple[torch.Tensor, torch.Tensor], target: Batch):
        out, scaling = predictions

        dos_loss = self.loss_fn(out, target.scaled)
        scaling_loss = self.loss_fn(scaling, target.scaling_factor)

        output_cumsum = torch.cumsum(out, axis=1)
        dos_cumsum = torch.cumsum(target.scaled, axis=1)
        dos_cumsum_loss = self.loss_fn(output_cumsum, dos_cumsum)

        scaled_dos = out * scaling.view(-1, 1).expand_as(out)
        x = torch.linspace(-10, 10, 400).to(scaled_dos)
        features = self.get_dos_features(x, scaled_dos)
        features_loss = self.loss_fn(target.features, features.to(self.device))

        loss_sum = (
            dos_loss
            + scaling_loss * self.scaling_weight
            + dos_cumsum_loss * self.cumsum_weight
            + features_loss * self.features_weight
        )

        return loss_sum

    def get_dos_features(self, x, dos):
        """get dos features"""
        dos = torch.abs(dos)
        dos_sum = torch.sum(dos, axis=1)

        center = torch.sum(x * dos, axis=1) / dos_sum
        x_offset = (
            torch.repeat_interleave(x[np.newaxis, :], dos.shape[0], axis=0)
            - center[:, None]
        )
        width = torch.diagonal(torch.mm((x_offset**2), dos.T)) / dos_sum
        skew = torch.diagonal(torch.mm((x_offset**3), dos.T)) / dos_sum / width**1.5
        kurtosis = (
            torch.diagonal(torch.mm((x_offset**4), dos.T)) / dos_sum / width**2
        )

        # find zero index (fermi level)
        zero_index = torch.abs(x - 0).argmin().long()
        ef_states = torch.sum(dos[:, zero_index - 20 : zero_index + 20], axis=1) * abs(
            x[0] - x[1]
        )
        return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)
