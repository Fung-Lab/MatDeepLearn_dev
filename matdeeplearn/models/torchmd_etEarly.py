from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor, nn
import torch_geometric.nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
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
from matdeeplearn.models.local_structure import DistanceNet

@registry.register_model("torchmd_etEarly")


class TorchMD_ET_Early(BaseModel):
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
        use_distill (bool): Whether model is used for distillation or not. 
            (default: :obj:`False`)
        distill_layers (string, optional): The layer features used for distillation.
        projection_head (bool): Set two linear layer for feature mapping. 
            (default: :obj:`False`)
        id_mapping (bool): Set identical mapping for feature mapping. 
            (default: :obj:`False`)
        force_linear_n2n (bool): Set one linear layer for feature mapping. 
            (default: :obj:`False`)
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
        max_z=100,
        max_num_neighbors=32,
        num_post_layers=1,
        post_hidden_channels=64,
        pool="global_mean_pool",
        aggr="add",
        use_distill=False,
        distill_layers = "",
        projection_head=False,
        id_mapping=False,
        force_linear_n2n=False,
        is_teacher=False,
        mapping_layers=1,
        **kwargs
    ):
        super(TorchMD_ET_Early, self).__init__(**kwargs)

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
        cutoff_lower = 0

        self.use_distill = use_distill
        self.projection_head = projection_head
        self.id_mapping = id_mapping
        self.force_linear_n2n = force_linear_n2n

        self.teacher_edge_dim = 1
        self.teacher_node_dim = 1
        self.teacher_vec_dim = 1
        self.is_teacher = is_teacher
        self.mapping_layers = mapping_layers

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
                aggr,
            ).jittable()
            self.attention_layers.append(layer)

        self.distance_net = DistanceNet()

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

        # set up distillation layers
        if use_distill:
            distill_layers = [
                code.lower() for code in distill_layers.split("_")
            ]
            distill_layers = [
                (code[0], int(code[1:])) for code in distill_layers
            ]
            distill_layers = sorted(
                distill_layers, key=lambda tup: (tup[1], tup[0])
            )  # Sort primarly on layer number, then on m/u
            assert all(
                code[1] > 0 for code in distill_layers
            ), "Distill layer number in torchmd net is less than 1"
            assert all(
                code[1] <= (num_layers) for code in distill_layers
            ), "Distill layer number in torchmd net is large than num_layers"
            assert all(
                code[0] in ["a"] for code in distill_layers
            ), "Distill layer type in Painn not in a(attention)"
            self.distill_layers = distill_layers
            self.num_distill_layers = len(self.distill_layers)
    
    def create_mapping(self, input_dim, output_dim, num_layers):
        layers = [torch.nn.Linear(input_dim, output_dim)]
        for _ in range(num_layers - 1):  # Subtract 1 because the first layer is already added
            layers += [torch.nn.SiLU(), torch.nn.Linear(output_dim, output_dim)]
        return torch.nn.Sequential(*layers)
    
    def setup_distillation(self):
        model_feature_dim = self.extract_feature_dimensions()

        if self.is_teacher or self.id_mapping:
            self.n2n_mapping = torch.nn.Identity()
            self.v2v_mapping = torch.nn.Identity()
            self.e2n_mapping = torch.nn.Identity()
            self.e2e_mapping = torch.nn.Identity()
        else:
            self.n2n_mapping = self.create_mapping(
                model_feature_dim["node_dim"], self.teacher_node_dim, self.mapping_layers)
            self.e2n_mapping = self.create_mapping(
                model_feature_dim["node_dim"], self.teacher_edge_dim, self.mapping_layers)
            self.e2e_mapping = self.create_mapping(
                model_feature_dim["edge_dim"], self.teacher_edge_dim, self.mapping_layers)
            self.v2v_mapping = self.create_mapping(
                model_feature_dim["vec_dim"], self.teacher_vec_dim, self.mapping_layers)


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

        #edge_index, edge_weight, edge_vec = self.distance(data.pos, data.batch)
        #assert (
        #    edge_vec is not None
        #), "Distance module did not return directional information"
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
        data.edge_attr = self.distance_expansion(data.edge_weight) 
                            
        #mask = data.edge_index[0] != data.edge_index[1]        
        #data.edge_vec[mask] = data.edge_vec[mask] / torch.norm(data.edge_vec[mask], dim=1).unsqueeze(1)
        data.edge_vec = data.edge_vec / torch.norm(data.edge_vec, dim=1).unsqueeze(1)
        
        if self.otf_node_attr == True:
            data.x = node_rep_one_hot(data.z).float()          
        
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(data.z, x, data.edge_index, data.edge_weight, data.edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for attn in self.attention_layers:
            dx, dvec = attn(x, vec, data.edge_index, data.edge_weight, data.edge_attr, data.edge_vec)
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)
        
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
                    
        return x
    
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

    @conditional_grad(torch.enable_grad())
    def extract_feature(self, data):

        output = {}

        #set distillation feature vectors
        node_feature = []
        edge_feature = []     
        vec_feature = []
        local_structure_feature = []   
        distill_layers_iter = iter(self.distill_layers)
        distill_layer = next(distill_layers_iter, (None, None))

        x = self.embedding(data.z)

        #edge_index, edge_weight, edge_vec = self.distance(data.pos, data.batch)
        #assert (
        #    edge_vec is not None
        #), "Distance module did not return directional information"
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
        data.edge_attr = self.distance_expansion(data.edge_weight) 
                            
        #mask = data.edge_index[0] != data.edge_index[1]        
        #data.edge_vec[mask] = data.edge_vec[mask] / torch.norm(data.edge_vec[mask], dim=1).unsqueeze(1)
        data.edge_vec = data.edge_vec / torch.norm(data.edge_vec, dim=1).unsqueeze(1)
        
        if self.otf_node_attr == True:
            data.x = node_rep_one_hot(data.z).float()          
        
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(data.z, x, data.edge_index, data.edge_weight, data.edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for i in range(len(self.attention_layers)):
            attn = self.attention_layers[i]
            dx, dvec = attn(x, vec, data.edge_index, data.edge_weight, data.edge_attr, data.edge_vec)
            x = x + dx
            vec = vec + dvec
            if distill_layer == ("a", i+1):
                local_structure = self.distance_net(x, data.edge_index)
                local_structure_feature.append(local_structure.clone)
                node_feature.append(x.clone())
                vec_feature.append(vec.clone())
                edge_feature.append(data.edge_attr.clone())
                distill_layer = next(distill_layers_iter, (None, None))
        x = self.out_norm(x)
        
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

        output["output"] =  x
        assert distill_layer == (None, None)
        #node_feat = torch.cat(node_feature, dim=-1)     
        #edge_feat = torch.cat(edge_feature, dim=-1)
        #vec_feat = torch.cat(vec_feature, dim=-1)
        if self.is_teacher:
            edge_to_node_feat = [scatter(feature, data.edge_index[0], dim=0, reduce='mean') for feature in edge_feature]
            output["e2n_mapping"] = [self.e2n_mapping(feature.float()) for feature in edge_to_node_feat]
        else:
            output["e2n_mapping"] = [self.e2n_mapping(feature.float()) for feature in node_feature]
        output["n2n_mapping"] = [self.n2n_mapping(feature.float()) for feature in node_feature]
        output["e2e_mapping"] = [self.e2e_mapping(feature.float()) for feature in edge_feature]
        vec_feat = [vec.reshape(vec.size(0), -1) for vec in vec_feature]
        v2v = [self.v2v_mapping(feature.float()) for feature in vec_feat]
        output["v2v_mapping"] = v2v
        output["local_structure"] = local_structure_feature
        
        if self.gradient == True and x.requires_grad == True:         
            volume = torch.einsum("zi,zi->z", data.cell[:, 0, :], torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)                        
            grad = torch.autograd.grad(
                    x,
                    [data.pos, data.displacement],
                    grad_outputs=torch.ones_like(x),
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
    
    def extract_feature_dimensions(self):
        num_distill_layer = len(self.distill_layers)
        node_feature_dim = self.hidden_channels
        edge_feature_dim =  self.num_rbf
        vec_feature_dim = 3 * self.hidden_channels
        return {"node_dim":node_feature_dim, "edge_dim":edge_feature_dim, "vec_dim":vec_feature_dim}

    def set_teacher_dim(self, teacher_dim):
        self.teacher_edge_dim = teacher_dim.get("edge_dim", 1)
        self.teacher_node_dim = teacher_dim.get("node_dim",1)
        self.teacher_vec_dim = teacher_dim.get("vec_dim",1)
 

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
        #self.layernorm_vec = EquivariantCovLayerNorm(hidden_channels)
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
        #self.layernorm_vec.reset_parameters()
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
        #vec = self.layernorm_vec(vec)
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
        
        #q1,q2,q3
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
        #sij
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        #yi, wi
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