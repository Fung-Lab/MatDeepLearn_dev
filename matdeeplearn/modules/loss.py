from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from matdeeplearn.common.registry import registry


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
        width = torch.diagonal(torch.mm((x_offset ** 2), dos.T)) / dos_sum
        skew = torch.diagonal(torch.mm((x_offset ** 3), dos.T)) / dos_sum / width ** 1.5
        kurtosis = (
                torch.diagonal(torch.mm((x_offset ** 4), dos.T)) / dos_sum / width ** 2
        )

        # find zero index (fermi level)
        zero_index = torch.abs(x - 0).argmin().long()
        ef_states = torch.sum(dos[:, zero_index - 20: zero_index + 20], axis=1) * abs(
            x[0] - x[1]
        )
        return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)


@registry.register_loss("TorchLossWrapper")
class TorchLossWrapper(nn.Module):
    def __init__(self, loss_fn="l1_loss"):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)

    def forward(self, predictions: torch.Tensor, target: Batch):
        return self.loss_fn(predictions, target.y)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@registry.register_loss("BarlowTwinsLoss")
class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, embed_size, lambd=0.0051):
        super(BarlowTwinsLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lambd = lambd  # default=0.005
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2).to(self.device)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().to(self.device)
        off_diag = off_diagonal(c).pow_(2).sum().to(self.device)
        loss = on_diag + self.lambd * off_diag
        return loss.to(self.device)
