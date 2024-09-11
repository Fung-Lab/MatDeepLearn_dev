# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import node_rep_one_hot

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
    ):
        super().__init__()
        self.emb_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.emb_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
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
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
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
        x = F.gelu(self.fc1(x))
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
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.emb_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


@registry.register_model("graphormer3d")
class Graphormer3D(BaseModel):

    def __init__(
        self,
        atom_types,
        n_layers=2,
        n_blocks=4,
        emb_dim=128,
        ffn_dim=128,
        n_attn_heads=32,
        input_droput=0.0,
        dropout=0.1,
        attn_dropout=0.1,
        act_dropout=0.0,
        n_kernel=128,
        **kwargs,
    ):
        super(Graphormer3D, self).__init__(**kwargs)
        self.atom_types = atom_types
        self.edge_types = atom_types ** 2
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.n_attn_heads = n_attn_heads
        self.dropout = dropout
        self.input_dropout = input_droput
        self.attn_dropout = attn_dropout
        self.act_dropout = act_dropout
        self.n_kernel = n_kernel
        
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.emb_dim, padding_idx=0
        )
        self.input_dropout = self.input_dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.emb_dim,
                    self.ffn_dim,
                    num_attention_heads=self.n_attn_heads,
                    dropout=self.dropout,
                    attention_dropout=self.attn_dropout,
                    activation_dropout=self.act_dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.emb_dim)

        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.emb_dim, 1
        )
        self.energe_agg_factor: Callable[[Tensor], Tensor] = nn.Embedding(3, 1)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        K = self.n_kernel

        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayer(K, self.edge_types)
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.n_attn_heads
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.emb_dim)
        self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead(
            self.emb_dim, self.n_attn_heads
        )

    @property
    def target_attr(self):
        return "y"

    def forward(self, data):
    
        output = {}
        out = self._forward(data)
        output["output"] =  out

        if self.gradient == True and out.requires_grad == True:         
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :], torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)                        
            grad = torch.autograd.grad(
                    out,
                    [data.pos, data.displacement],
                    grad_outputs=torch.ones_like(out),
                    create_graph=self.training) 
            forces = -1 * grad[0]
            stress = grad[1]
            stress = stress / volume.view(-1, 1, 1)         

            output["pos_grad"] =  forces
            output["cell_grad"] =  stress
        else:
            output["pos_grad"] =  None
            output["cell_grad"] =  None  
                  
        return output
    
    def batch2atoms(self, data):
        # device = data.ptr.device
        # batch_size = data.ptr.shape[0] - 1
        max_atoms = max(data.n_atoms)

        # atoms = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=device)
        # cumsum_atoms = torch.cumsum(data.n_atoms, dim=0)
        # start = 0
        # for i in range(batch_size - 1):
        #     end = cumsum_atoms[i]
        #     num_atoms = end - start
        #     atoms[i, :num_atoms] = data.z[start:end]
            # start = end
            
        device = data.z.device
        batch_size = data.n_atoms.size(0)
        
        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0).expand(batch_size, -1)
        cumsum_atoms = torch.cat([torch.zeros(1, device=device), torch.cumsum(data.n_atoms, dim=0)[:-1]])
        mask = atom_indices < data.n_atoms.unsqueeze(1)
        
        # Create global indices for data.z
        global_indices = (atom_indices + cumsum_atoms.unsqueeze(1)).to(torch.long)
        
        # Create the output tensor
        atoms = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=device)
        atoms[mask] = data.z[global_indices[mask]]
        return atoms
    
    def create_distance_matrix(self, atoms, data):
        device = atoms.device
        batch_size = atoms.size(0)
        max_atoms = atoms.size(1)
        
        distances = torch.full((batch_size, max_atoms, max_atoms), self.cutoff_radius, device=device)
        
        batch_indices = torch.arange(batch_size, device=device)
        atom_indices = torch.arange(max_atoms, device=device)
        distances[batch_indices.unsqueeze(1), atom_indices, atom_indices] = 0
        
        source, target = data.edge_index
        edge_batch = data.batch[source]
        # Compute offsets for each batch
        offsets = torch.cumsum(torch.cat([torch.zeros(1, device=device), data.n_atoms[:-1]]), dim=0)
        
        source_adj = (source - offsets[edge_batch]).to(torch.long)
        target_adj = (target - offsets[edge_batch]).to(torch.long)        
        distances[edge_batch, source_adj, target_adj] = data.edge_weight
        # distances[edge_batch, target_adj, source_adj] = data.edge_weight
        
        return distances
    
    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        # print(data)
        atoms = self.batch2atoms(data)
        padding_mask = atoms.eq(0)
        # delta_pos = data.pos.unsqueeze(1) - data.pos.unsqueeze(2)
        # print(atoms.size())

        n_graph, n_node = atoms.size()
        
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
        data.edge_vec = data.edge_vec / torch.norm(data.edge_vec, dim=1).unsqueeze(1)
        
        # print(data.edge_index.size(), data.edge_index[0].unique().size())

        dist = self.create_distance_matrix(atoms, data)
        # dist = delta_pos.norm(dim=-1)
        # delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node
        )

        gbf_feature = self.gbf(dist, edge_type)
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

        graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
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
        # print(eng_output.size())
        # output_mask = (
        #     tags > 0
        # ) & real_mask  # no need to consider padding, since padding has tag 0, real_mask False

        # eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)

        # node_output = self.node_proc(output, graph_attn_bias, delta_pos)
        # print(node_output.shape)

        # node_target_mask = output_mask.unsqueeze(-1)
        return eng_output#, node_output, node_target_mask


# @register_model_architecture("graphormer3d", "graphormer3d_base")
# def base_architecture(args):
#     args.blocks = getattr(args, "blocks", 4)
#     args.layers = getattr(args, "layers", 12)
#     args.embed_dim = getattr(args, "embed_dim", 768)
#     args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
#     args.attention_heads = getattr(args, "attention_heads", 48)
#     args.input_dropout = getattr(args, "input_dropout", 0.0)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.node_loss_weight = getattr(args, "node_loss_weight", 15)
#     args.min_node_loss_weight = getattr(args, "min_node_loss_weight", 1)
#     args.eng_loss_weight = getattr(args, "eng_loss_weight", 1)
#     args.num_kernel = getattr(args, "num_kernel", 128)