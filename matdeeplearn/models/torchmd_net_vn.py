from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.nn
import matdeeplearn.models.routines.pooling as pooling
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from matdeeplearn.common.graph_data import VirtualNodeData
from matdeeplearn.common.registry import registry
from matdeeplearn.models.layers.utils import (
    CosineCutoff,
    Distance,
    NeighborEmbedding,
    act_class_mapping,
    rbf_class_mapping,
)


@registry.register_model("torchmd_net_vn")
class TorchMD_Net_ET(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
    """

    def __init__(
        self,
        data=None,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        num_post_layers=1,
        post_hidden_channels=64,
        pool="global_mean_pool",
        virtual_pool="AtomicNumberPooling",
        pool_choice="both",
        mp_pattern: list[str] = ["rv", "rr"],
        aggr="add",
        **kwargs,
    ):
        super(TorchMD_Net_ET, self).__init__()

        del kwargs, data

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        # VN specific parameters
        self.virtual_pool = virtual_pool
        self.pool_choice = pool_choice
        self.pool = pool
        self.mp_pattern = mp_pattern
        self.virtual_node_pool = (
            getattr(pooling, self.virtual_pool)(
                self.pool,
                pool_choice=self.pool_choice,
            )
            if self.virtual_pool != ""
            else None
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.pool = pool

        act_class = act_class_mapping[activation]

        # should prevent the model from crashing if the max_z is not large enough
        self.embedding = nn.Embedding(self.max_z + 1, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z + 1
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                aggr,
            ).jittable()
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.num_post_layers = num_post_layers
        self.post_hidden_channels = post_hidden_channels
        self.post_lin_list = nn.ModuleList()
        for i in range(self.num_post_layers):
            if i == 0:
                if self.virtual_pool == "AtomicNumberPooling":
                    self.post_lin_list.append(
                        nn.Linear(hidden_channels * self.max_z, post_hidden_channels)
                    )
                else:
                    self.post_lin_list.append(
                        nn.Linear(hidden_channels, post_hidden_channels)
                    )
            else:
                self.post_lin_list.append(
                    nn.Linear(post_hidden_channels, post_hidden_channels)
                )
        self.post_lin_list.append(nn.Linear(post_hidden_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self, data: VirtualNodeData
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = self.embedding(data.z)

        # edge_index_use, edge_weights_use, edge_vec_use = self.distance(data.pos, data.batch)
        # edge_attr_use = self.distance_expansion(edge_weights_use)

        # assert (
        #    edge_vec_use is not None
        # ), "Distance module did not return directional information"

        edge_index_use = torch.cat(
            [getattr(data, f"edge_index_{mp}") for mp in self.mp_pattern], dim=1
        )
        edge_weights_use = torch.cat(
            [getattr(data, f"edge_weights_{mp}") for mp in self.mp_pattern], dim=0
        )
        edge_vec_use = torch.cat(
            [getattr(data, f"edge_vec_{mp}") for mp in self.mp_pattern], dim=0
        )
        edge_attr_use = torch.cat(
            [getattr(data, f"edge_attr_{mp}") for mp in self.mp_pattern], dim=0
        )

        mask = torch.argwhere(edge_index_use[0] != edge_index_use[1]).squeeze(1)

        edge_vec_use[mask] = edge_vec_use[mask] / torch.norm(
            edge_vec_use[mask], dim=1
        ).unsqueeze(1)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(
                data.z, x, edge_index_use, edge_weights_use, edge_attr_use
            )

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for i, attn in enumerate(self.attention_layers):
            dx, dvec = attn(
                x, vec, edge_index_use, edge_weights_use, edge_attr_use, edge_vec_use
            )
            x = x + dx
            vec = vec + dvec

        x = self.out_norm(x)

        # perform modified VN pooling routine if applied
        if self.virtual_node_pool is not None:
            x = self.virtual_node_pool(data, x)
        else:
            x = getattr(torch_geometric.nn, self.pool)(x, data.batch)

        for i in range(0, len(self.post_lin_list) - 1):
            x = self.post_lin_list[i](x)
            x = getattr(F, self.activation)(x)
        x = self.post_lin_list[-1](x)

        # x = self.pool.pre_reduce(x, vec, data.z, data.pos, data.batch)
        # x = self.pool.reduce(x, data.batch)

        if x.shape[1] == 1:
            x = x.view(-1)

        return x, vec, data.z, data.pos, data.batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )

    @property
    def target_attr(self):
        return "y"


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        aggregation,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(
            aggr=aggregation, node_dim=0
        )
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
