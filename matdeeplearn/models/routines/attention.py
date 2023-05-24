from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F


class MetaPathImportance(nn.Module):
    """Compute attention score for a meta-path"""

    def __init__(self, embed_dim: int, attn_size: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_size = attn_size
        self.w = nn.Linear(self.embed_dim, self.attn_size)
        self.q = nn.Parameter(torch.randn((self.attn_size, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_embed = self.w(x)
        scores = torch.tensordot(torch.tanh(projected_embed), self.q, dims=1).squeeze(
            -1
        ) / x.size(1)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights.unsqueeze(-1)


class SemanticAttention(nn.Module):
    """
    Semantic-level attention.
    Adapted from https://github.com/Jhy1993/HAN/blob/master/utils/layers.py
    """

    def __init__(self, meta_paths: list[str], embed_dim: int, attn_size: int) -> None:
        super().__init__()
        self.meta_paths = meta_paths
        # weights and attention vectors for each meta path
        self.attn = nn.ModuleDict(
            {mp: MetaPathImportance(embed_dim, attn_size) for mp in meta_paths}
        )

    def forward(self, path_embds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        meta_path_attns = defaultdict(nn.Module)
        # compute attention scores and weights for each meta path
        for mp in self.meta_paths:
            x = path_embds[mp]
            meta_path_attns[mp] = self.attn[mp](x)

        return meta_path_attns
