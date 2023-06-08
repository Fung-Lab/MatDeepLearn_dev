from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from matdeeplearn.models.torchmdnet.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)
from matdeeplearn.models.torchmdnet import output_modules
from matdeeplearn.common.registry import registry

@registry.register_model("torchmd_net")
class TorchMDNet(nn.Module):
    """
    wrapper class for torchmd architecture

    adapted from https://github.com/shehzaidi/pre-training-via-denoising
    """
    def __init__(
        self,
        model_type,
        model_configs,
        output_type,
        output_noise_type,
        reduce_op="add",
        **kwargs
    ):
        super(TorchMDNet, self).__init__()
        self.model_type = model_type
        self.model_configs = model_configs
        self.output_type = output_type
        self.output_noise_type = output_noise_type
        self.reduce_op = reduce_op

        # initialize model and output heads
        self.model = self._init_model()
        self.output_head = self._init_output_head()
        self.output_noise_head = self._init_output_noise_head()

        self.reset_parameters()
        self.__dict__.update(kwargs)

    def _init_model(self):
        if self.model_type == "et" or self.model_type == "equivariant-transformer":
            from matdeeplearn.models.torchmdnet.torchmd_et import TorchMD_ET

            model = TorchMD_ET(**self.model_configs)
        else:
            raise ValueError(f"model type {self.model_type} not supported")
        
        return model

    def _init_output_head(self):
        output_head = getattr(output_modules, self.output_type)(
            self.model_configs["hidden_channels"],
            self.model_configs["activation"]
        )

        return output_head

    def _init_output_noise_head(self):
        output_noise_head = None

        if self.output_noise_type is not None:
            output_noise_head = getattr(output_modules, self.output_noise_type)(
                self.model_configs["hidden_channels"],
                self.model_configs["activation"]
            )
        
        return output_noise_head

    def reset_parameters(self):
        self.model.reset_parameters()
        self.output_head.reset_parameters()

    @property
    def target_attr(self):
        return self._target_attr

    
    def forward(self, z, pos, batch, batch_data):
        batch = torch.zeros_like(z) if batch is None else batch

        x, v, z, pos, batch = self.model(z, pos, batch, batch_data)

        # predict noise
        noise_pred = None
        if self.output_noise_head is not None:
            noise_pred = self.output_noise_head.pre_reduce(x, v, z, pos, batch)
        
        x = self.output_head.pre_reduce(x, v, z, pos, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        out = self.output_head.post_reduce(out)

        return out, noise_pred

class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)