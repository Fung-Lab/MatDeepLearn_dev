# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.autograd.profiler as profiler

from torch_geometric.data import Batch as TorchGeoBatch

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.models.utils import GaussianSmearing
from matdeeplearn.preprocessor.pbc_transform import Batch, Data
from matdeeplearn.models.model_dev.adaptive_span import AdaptiveSpan

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
        adaptive_span_params=None
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if adaptive_span_params is not None:
            self.adaptive_span = AdaptiveSpan(
                nb_heads=num_heads,
                **adaptive_span_params
            )
        else:
            self.adaptive_span = None

    def head_reshape(self, x):
        K = self.num_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        if self.adaptive_span is not None:
            self.adaptive_span.clamp_param()
        
        n_node, n_graph, embed_dim = query.size()
        
        # with profiler.record_function("IN PROJ"):
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        
        # _shape = (-1, n_graph * self.num_heads, self.head_dim)
        # q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        # k = k.contiguous().view(_shape).transpose(0, 1)
        # v = v.contiguous().view(_shape).transpose(0, 1)

        q = self.head_reshape(q) * self.scaling
        k = self.head_reshape(k)
        v = self.head_reshape(v)
        
        # with profiler.record_function("ATTN SPAN COMPUTE"):
        if self.adaptive_span is not None:
            # [optional] trim out memory to reduce unnecessary computation
            k, v, _, attn_bias = self.adaptive_span.trim_memory(
                q, k, v, key_pe=None, attn_bias=attn_bias
            )

        # with profiler.record_function("ATTN PROBS"):
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling + attn_bias
        attn_probs = F.softmax(attn_weights, dim=-1)

        if self.adaptive_span is not None:
            # trim attention lengths according to the learned span
            attn_probs = self.adaptive_span(attn_probs) 
        
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # with profiler.record_function("APPLY ATTN"):
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        
        # with profiler.record_function("OUT PROJ"):
        attn = self.out_proj(attn)
        return attn

class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        act: str="gelu",
        adaptive_span_params: dict = None
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        
        self.act = getattr(F, act)

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            adaptive_span_params=adaptive_span_params
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.act(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, act='gelu', hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)
        self.act = getattr(F, act)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        adaptive_span_params: dict = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.force_proj1.weight)
        nn.init.xavier_uniform_(self.force_proj2.weight)
        nn.init.xavier_uniform_(self.force_proj3.weight)
    

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
    # with profiler.record_function("FORCE HEAD IN PROJ"):
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
    # with profiler.record_function("FORCE HEAD ATTN PROBS"):
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.0, self.training
        ).view(bsz, self.num_heads, n_node, n_node)

        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        
    # with profiler.record_function("FORCE HEAD APPLY ATTN"):
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        
    # with profiler.record_function("FORCE HEAD OUT PROJ"):
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


@registry.register_model("graphormer_as")
class Graphormer3D_Force_AS(BaseModel):
    def __init__(
        self,
        atom_types=20,
        n_blocks=1,
        n_layers=6,
        emb_dim=768,
        ffn_dim=768,
        n_attn_heads=48,
        input_droput=0.0,
        dropout=0.1,
        attn_dropout=0.1,
        act_dropout=0.0,
        n_kernel=128,
        act='gelu',
        **kwargs,
    ):
        super(Graphormer3D_Force_AS, self).__init__(**kwargs)
        self.atom_types = atom_types
        self.edge_types = atom_types ** 2
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.n_attn_heads = n_attn_heads
        self.dropout = dropout
        self.input_dropout = input_droput
        self.attn_dropout = attn_dropout
        self.act_dropout = act_dropout
        self.n_kernel = n_kernel
        
        adaptive_span_params = {
            "attn_span": 256,
            "adapt_span_loss": 2e-6,
            "adapt_span_ramp": 32,
            "adapt_span_init": 0.0,
            "adapt_span_cache": True
        }
        
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.emb_dim, padding_idx=0
        )
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.emb_dim,
                    self.ffn_dim,
                    num_attention_heads=self.n_attn_heads,
                    dropout=self.dropout,
                    attention_dropout=self.attn_dropout,
                    activation_dropout=self.act_dropout,
                    act=act,
                    adaptive_span_params=adaptive_span_params
                )
                for _ in range(self.n_layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.emb_dim)

        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.emb_dim, 1, act=act,
        )

        K = self.n_kernel
        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayer(K, self.edge_types)
        # self.gbf = GaussianSmearing(cutoff_lower=0.0, cutoff_upper=8.0, num_rbf=K)
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.n_attn_heads, act=act
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.emb_dim)
        self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead(
            self.emb_dim, self.n_attn_heads, adaptive_span_params=adaptive_span_params
        )
        
    @property
    def target_attr(self):
        return "y"

    def forward(self, data: TorchGeoBatch):
    
        output = {}
        out = self._forward(data)
        output["output"] = out[0]
        output["pos_grad"] = out[1]
                  
        return output 
    
    @conditional_grad(torch.enable_grad())
    def _forward(self, data: TorchGeoBatch):        
        atoms, pos, real_mask = (
            data.atoms,
            data.pos,
            data.real_mask,
        )
        padding_mask = atoms == 0

        # with profiler.record_function("DIST COMPUTATION"):
        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        # with profiler.record_function("GBF COMPUTATION"):

        edge_type = atoms.view(
            n_graph, n_node, 1
        ) * self.atom_types + atoms.view(n_graph, 1, n_node)

        gbf_feature = self.gbf(dist, edge_type)
        # gbf_feature = self.gbf(dist)
            
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = (
            self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        )
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.n_blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)

        eng_output = F.dropout(output, p=0.1, training=self.training)
        eng_output = (
            self.engergy_proj(eng_output)
        ).flatten(-2)
        output_mask = real_mask
        eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)
        
        node_output = self.node_proc(output, graph_attn_bias, delta_pos)

        node_target_mask = output_mask
        # force_output = node_output[expanded_mask.bool()].reshape(-1, 3)
        force_output = torch.cat([
            node_output[i, node_target_mask[i].expand_as(node_target_mask[i])]
        for i in range(node_target_mask.size(0))])

        return eng_output[:, None], force_output
