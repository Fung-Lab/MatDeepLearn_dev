from functools import partial

import torch
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, Sequential, Sigmoid, SiLU
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel


@registry.register_model("ALIGNN_GRAPHITE")
class ALIGNN_GRAPHITE(BaseModel):
    """ALIGNN model that uses auxiliary line graph to explicitly represent and encode bond angles.
    Reference: https://www.nature.com/articles/s41524-021-00650-1.
    """

    def __init__(self, dim=64, num_interactions=4, num_species=3, cutoff=3.0):
        super().__init__()

        self.dim = dim
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embed_atm = Embedding(num_species, dim)
        self.embed_bnd = partial(bessel, start=0, end=cutoff, num_basis=dim)

        self.atm_bnd_interactions = ModuleList()
        self.bnd_ang_interactions = ModuleList()
        for _ in range(num_interactions):
            self.atm_bnd_interactions.append(EGConv(dim, dim))
            self.bnd_ang_interactions.append(EGConv(dim, dim))

        self.head = Sequential(
            Linear(dim, dim),
            SiLU(),
        )

        self.out = Sequential(
            Linear(dim, 1),
        )

        self.reset_parameters()

    @property
    def target_attr(self):
        return "y"

    def reset_parameters(self):
        self.embed_atm.reset_parameters()
        for i in range(self.num_interactions):
            self.atm_bnd_interactions[i].reset_parameters()
            self.bnd_ang_interactions[i].reset_parameters()

    def embed_ang(self, x_ang):
        cos_ang = torch.cos(x_ang)
        return gaussian(cos_ang, start=-1, end=1, num_basis=self.dim)

    def forward(self, data: Data):
        edge_index_G = data.edge_index
        edge_index_A = data.edge_index_lg
        h_atm = self.embed_atm(data.x.type(torch.long))
        h_bnd = self.embed_bnd(data.edge_attr)
        h_ang = self.embed_ang(data.edge_attr_lg)

        for i in range(self.num_interactions):
            h_bnd, h_ang = self.bnd_ang_interactions[i](h_bnd, edge_index_A, h_ang)
            h_atm, h_bnd = self.atm_bnd_interactions[i](h_atm, edge_index_G, h_bnd)

        h = scatter(h_atm, data.batch, dim=0, reduce="add")
        h = self.head(h)
        return self.out(h)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, "
            f"num_interactions={self.num_interactions}, "
            f"cutoff={self.cutoff})"
        )


class EGConv(MessagePassing):
    """Edge-gated convolution.
    This version is closer to the original formulation (without the concatenation).
    * https://arxiv.org/abs/2003.00982
    """

    def __init__(self, node_dim, edge_dim, epsilon=1e-5):
        super().__init__(aggr="add")
        self.W_src = Linear(node_dim, node_dim)
        self.W_dst = Linear(node_dim, node_dim)
        self.W_A = Linear(node_dim, edge_dim)
        self.W_B = Linear(node_dim, edge_dim)
        self.W_C = Linear(edge_dim, edge_dim)
        self.act = SiLU()
        self.sigma = Sigmoid()
        self.norm_x = LayerNorm([node_dim])
        self.norm_e = LayerNorm([edge_dim])
        self.eps = epsilon

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_src.weight)
        self.W_src.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_dst.weight)
        self.W_dst.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_A.weight)
        self.W_A.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_B.weight)
        self.W_B.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.W_C.weight)
        self.W_C.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index

        # Calculate gated edges
        sigma_e = self.sigma(edge_attr)
        e_sum = scatter(src=sigma_e, index=i, dim=0)
        e_gated = sigma_e / (e_sum[i] + self.eps)

        # Update the nodes (this utilizes the gated edges)
        out = self.propagate(edge_index, x=x, e_gated=e_gated)
        out = self.W_src(x) + out
        out = x + self.act(self.norm_x(out))

        # Update the edges
        edge_attr = edge_attr + self.act(
            self.norm_e(self.W_A(x[i]) + self.W_B(x[j]) + self.W_C(edge_attr))
        )

        return out, edge_attr

    def message(self, x_j, e_gated):
        return e_gated * self.W_dst(x_j)


def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Expand scalar features into (radial) Bessel basis function values."""
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)
    return ((2 / c) ** 0.5) * torch.sin(n * torch.pi * x / c) / x


def gaussian(x, start=0.0, end=1.0, num_basis=8):
    """Expand scalar features into Gaussian basis function values."""
    mu = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
    step = mu[1] - mu[0]
    diff = (x[..., None] - mu) / step
    # division by 1.12 so that sum of square is roughly 1
    return diff.pow(2).neg().exp().div(1.12)
