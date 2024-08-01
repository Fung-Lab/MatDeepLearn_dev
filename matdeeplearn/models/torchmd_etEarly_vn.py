from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch_geometric.nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from matdeeplearn.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.models.torchmd_output_modules import Scalar, EquivariantScalar
from matdeeplearn.common.registry import registry
from matdeeplearn.preprocessor.helpers import node_rep_one_hot


@registry.register_model("torchmd_etEarly_vn")
class TorchMD_ET(BaseModel):
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
        self.cutoff_radius (float, optional): Upper cutoff distance for interatomic interactions.
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
            node_dim,
            edge_dim,
            output_dim,
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
            max_z=600,
            max_num_neighbors=32,
            num_post_layers=1,
            post_hidden_channels=64,
            pool="global_mean_pool",
            aggr="add",
            cutoff_radius_rn_vn=8,
            cutoff_radius_vn_vn=4,
            rn_vn_aggr="add",
            vn_vn_aggr="add",
            **kwargs
    ):
        super(TorchMD_ET, self).__init__(**kwargs)

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
        self.max_z = max_z
        self.pool = pool
        self.output_dim = output_dim
        self.cutoff_radius_rn_vn = cutoff_radius_rn_vn
        self.cutoff_radius_vn_vn = cutoff_radius_vn_vn
        cutoff_lower = 0

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            self.cutoff_radius,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, self.cutoff_radius, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, self.cutoff_radius, self.max_z
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
                self.cutoff_radius,
                "mean",
            ).jittable()
            rn_vn_layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                self.cutoff_radius,
                rn_vn_aggr,
            ).jittable()
            # vn_vn_layer = EquivariantMultiHeadAttention(
            #     hidden_channels,
            #     num_rbf,
            #     distance_influence,
            #     num_heads,
            #     act_class,
            #     attn_activation,
            #     cutoff_lower,
            #     self.cutoff_radius,
            #     vn_vn_aggr,
            # ).jittable()
            self.attention_layers.append(layer)
            self.attention_layers.append(rn_vn_layer)
            # self.attention_layers.append(vn_vn_layer)

        self.last_attention_layers = EquivariantMultiHeadAttention(
            hidden_channels,
            num_rbf,
            distance_influence,
            num_heads,
            act_class,
            attn_activation,
            cutoff_lower,
            self.cutoff_radius,
            rn_vn_aggr,
        ).jittable()
        self.out_norm = nn.LayerNorm(hidden_channels)

        self.num_post_layers = num_post_layers
        self.post_hidden_channels = post_hidden_channels
        self.post_lin_list = nn.ModuleList()
        for i in range(self.num_post_layers):
            if i == 0:
                self.post_lin_list.append(nn.Linear(hidden_channels, post_hidden_channels))
            else:
                self.post_lin_list.append(nn.Linear(post_hidden_channels, post_hidden_channels))
        self.post_lin_list.append(nn.Linear(post_hidden_channels, self.output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):

        x = self.embedding(data.z)
        x[data.z == 100] = torch.zeros(self.hidden_channels, device=x.device)

        # edge_index, edge_weight, edge_vec = self.distance(data.pos, data.batch)
        # assert (
        #    edge_vec is not None
        # ), "Distance module did not return directional information"
        if self.otf_edge_index == True:
            # data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)
            data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.generate_graph(data, self.cutoff_radius,
                                                                                            self.n_neighbors)
            edge_mask = torch.zeros_like(data.edge_index[0])
            edge_mask[(data.z[data.edge_index[0]] == 100) & (
                    data.z[data.edge_index[1]] == 100)] = 0  # virtual node to virtual node
            edge_mask[(data.z[data.edge_index[0]] != 100) & (
                    data.z[data.edge_index[1]] == 100)] = 1  # regular node to virtual node
            edge_mask[(data.z[data.edge_index[0]] == 100) & (
                    data.z[data.edge_index[1]] != 100)] = 2  # virtual node to regular node
            edge_mask[(data.z[data.edge_index[0]] != 100) & (
                    data.z[data.edge_index[1]] != 100)] = 3  # regular node to regular node

            data.edge_mask = edge_mask
        data.edge_attr = self.distance_expansion(data.edge_weight)

        # mask = data.edge_index[0] != data.edge_index[1]
        # data.edge_vec[mask] = data.edge_vec[mask] / torch.norm(data.edge_vec[mask], dim=1).unsqueeze(1)
        data.edge_vec = data.edge_vec / torch.norm(data.edge_vec, dim=1).unsqueeze(1)

        if self.otf_node_attr == True:
            data.x = node_rep_one_hot(data.z).float()

        indices_rn_to_rn = (data.edge_mask == 3) & (data.edge_weight <= self.cutoff_radius)
        indices_rn_to_vn = (data.edge_mask == 1) & (data.edge_weight <= self.cutoff_radius_rn_vn)
        indices_vn_to_vn = (data.edge_mask == 0) & (data.edge_weight <= self.cutoff_radius_vn_vn)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(
                data.z,
                x,
                data.edge_index[:, indices_rn_to_rn | indices_rn_to_vn],
                data.edge_weight[indices_rn_to_rn | indices_rn_to_vn],
                data.edge_attr[indices_rn_to_rn | indices_rn_to_vn, :]
            )

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for i, attn in enumerate(self.attention_layers):
            if i % 2 == 0:  # rn-rn
                dx, dvec = attn(x, vec, data.edge_index[:, indices_rn_to_rn], data.edge_weight[indices_rn_to_rn],
                                data.edge_attr[indices_rn_to_rn, :], data.edge_vec[indices_rn_to_rn, :])
                x = x + dx
                vec = vec + dvec
            if i % 2 == 1:  # rn-vn
                dx, dvec = attn(x, vec, data.edge_index[:, indices_rn_to_vn], data.edge_weight[indices_rn_to_vn],
                                data.edge_attr[indices_rn_to_vn, :], data.edge_vec[indices_rn_to_vn, :])
                x = x + dx
                vec = vec + dvec
            # if i % 3 == 2:  # rn-rn
            #     dx, dvec = attn(x, vec, data.edge_index[:, indices_vn_to_vn], data.edge_weight[indices_vn_to_vn],
            #                     data.edge_attr[indices_vn_to_vn, :], data.edge_vec[indices_vn_to_vn, :])
            #     x = x + dx
            #     vec = vec + dvec

        dx, dvec = self.last_attention_layers(x, vec, data.edge_index[:, indices_rn_to_vn],
                                              data.edge_weight[indices_rn_to_vn],
                                              data.edge_attr[indices_rn_to_vn, :], data.edge_vec[indices_rn_to_vn, :])
        x = x + dx
        # vec = vec + dvec
        x = self.out_norm(x)
        '''
        if self.prediction_level == "graph":
            x = getattr(torch_geometric.nn, self.pool)(x, data.batch)
            for i in range(0, len(self.post_lin_list) - 1):
                x = self.post_lin_list[i](x)
                x = getattr(F, self.activation)(x)
            x = self.post_lin_list[-1](x)
            #x = self.pool.pre_reduce(x, vec, data.z, data.pos, data.batch)
            #x = self.pool.reduce(x, data.batch)
        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list) - 1):
                x = self.post_lin_list[i](x)
                x = getattr(F, self.activation)(x)
            x = self.post_lin_list[-1](x) 
        '''
        virtual_mask = torch.argwhere(data.z == 100).squeeze(1)
        x = torch.index_select(x, 0, virtual_mask)

        for i in range(0, len(self.post_lin_list) - 1):
            x = self.post_lin_list[i](x)
            x = getattr(F, self.activation)(x)
        x = self.post_lin_list[-1](x)

        return x

    def forward(self, data):

        output = {}
        out = self._forward(data)
        output["output"] = out

        if self.gradient == True and out.requires_grad == True:
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :],
                                  torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)
            grad = torch.autograd.grad(
                out,
                [data.pos, data.displacement],
                grad_outputs=torch.ones_like(out),
                create_graph=self.training)
            forces = -1 * grad[0]
            stress = grad[1]
            stress = stress / volume.view(-1, 1, 1)

            output["pos_grad"] = forces
            output["cell_grad"] = stress
        else:
            output["pos_grad"] = None
            output["cell_grad"] = None

        return output

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
            f"self.cutoff_radius={self.self.cutoff_radius})"
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
        super(EquivariantMultiHeadAttention, self).__init__(aggr=aggregation, node_dim=0)
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
        self.layernorm2 = nn.LayerNorm(hidden_channels)
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
        vec = self.layernorm2(vec)
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