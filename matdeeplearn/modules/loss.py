from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from matdeeplearn.common.registry import registry


@registry.register_loss("TorchLossWrapper")
class TorchLossWrapper(nn.Module):
    def __init__(self, loss_fn="l1_loss"):
        super().__init__()
        self.loss_fn = getattr(F, loss_fn)

    def forward(self, predictions: torch.Tensor, target: Batch):
        return self.loss_fn(predictions["output"], target.y)


@registry.register_loss("ForceLoss")
class ForceLoss(nn.Module):
    def __init__(self, weight_energy=1.0, weight_force=0.1):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_force = weight_force

    def forward(self, predictions: torch.Tensor, target: Batch):  
        combined_loss = self.weight_energy*F.l1_loss(predictions["output"], target.y) + self.weight_force*F.l1_loss(predictions["pos_grad"], target.forces)
        return combined_loss


@registry.register_loss("ForceStressLoss")
class ForceStressLoss(nn.Module):
    def __init__(self, weight_energy=1.0, weight_force=0.1, weight_stress=0.1):
        super().__init__()
        self.weight_energy = weight_energy
        self.weight_force = weight_force
        self.weight_stress = weight_stress

    def forward(self, predictions: torch.Tensor, target: Batch):  
        combined_loss = self.weight_energy*F.l1_loss(predictions["output"], target.y) + self.weight_force*F.l1_loss(predictions["pos_grad"], target.forces) + self.weight_stress*F.l1_loss(predictions["cell_grad"], target.stress)
        return combined_loss
        

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


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


@registry.register_loss("NCSN")
class NCSN_version_03(torch.nn.Module):
    def __init__(self, emb_dim, sigma_begin, sigma_end, num_noise_level, anneal_power):
        super(NCSN_version_03, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.anneal_power = anneal_power

        # self.noise_type = noise_type
        self.input_distance_mlp = MultiLayerPerceptron(1, [emb_dim, 1], activation="relu")
        self.output_mlp = MultiLayerPerceptron(1 + emb_dim, [emb_dim, emb_dim // 2, 1])

        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_noise_level)),
                              dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # (num_noise_level)

        return

    def forward(self, data, node_feature, distance, debug=False):
        self.device = self.sigmas.device

        node2graph = data.batch
        edge2graph = node2graph[data.super_edge_index[0]]

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device)  # (num_graph)
        used_sigmas = self.sigmas[noise_level]  # (num_graph)
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)

        distance_noise = torch.randn_like(distance)

        # print(distance.get_device(), used_sigmas.get_device(), distance_noise.get_device())
        perturbed_distance = distance + distance_noise * used_sigmas
        distance_emb = self.input_distance_mlp(perturbed_distance)  # (num_edge, hidden)

        target = -1 / (used_sigmas ** 2) * (perturbed_distance - distance)  # (num_edge, 1)

        h_row, h_col = node_feature[data.super_edge_index[0]], node_feature[
            data.super_edge_index[1]]  # (num_edge, hidden)

        distance_feature = torch.cat([h_row + h_col, distance_emb], dim=-1)  # (num_edge, 2*hidden)
        scores = self.output_mlp(distance_feature)  # (num_edge, 1)
        scores = scores * (1. / used_sigmas)  # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        target = target.view(-1)  # (num_edge)
        scores = scores.view(-1)  # (num_edge)
        loss = 0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power)  # (num_edge)
        # print("loss", loss[:10])
        loss = scatter_add(loss, edge2graph)  # (num_graph)

        loss = loss.mean()

        if debug:
            print("distance_feature", distance_feature[:3])
            print("perturbed_distance", perturbed_distance[:10].squeeze())
            print("distance", distance[:10].squeeze())
            print("target", target[:10].squeeze())
            print("scores", scores[:10].squeeze())
        return loss
