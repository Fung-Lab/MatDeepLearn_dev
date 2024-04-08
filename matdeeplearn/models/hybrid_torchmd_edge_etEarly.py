from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch_geometric.nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_scatter import scatter_add

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
@registry.register_model("hybrid_torchmd_edge_etEarly")


class Hybrid_TorchMD_Edge_ET_Early(BaseModel):
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
        max_z=100,
        max_num_neighbors=32,
        num_post_layers=1,
        post_hidden_channels=64,
        pool="global_mean_pool",
        pool_order="late",
        aggr="add",
        **kwargs
    ):
        super(Hybrid_TorchMD_Edge_ET_Early, self).__init__(**kwargs)

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
        
        self.num_elements = kwargs.get('num_elements', 100)
        self.param_init_method = kwargs.get('param_init_method', None)
        self.param_post_fc_count = kwargs.get('param_post_fc_count', 3)
        self.use_atomic_energy = kwargs.get('use_atomic_energy', True)
        self.potential_fc_count = kwargs.get('potential_fc_count', 3)
        self.potential_dim = kwargs.get('potential_dim', 64)
        self.node_edge_ratio = kwargs.get('node_edge_ratio', (1, 1))
        
        self.potential_type = kwargs.get("potential_type", "LJ")
        print(f"{self.potential_type} applied")
        if self.potential_type == 'LJ':
            self.n_params = 2
            self.potential = self.lj_potential
        elif self.potential_type == 'Morse':
            self.potential = self.morse_potential
            self.n_params = 3
        elif self.potential_type == 'Spline':
            n_intervals = kwargs.get('n_intervals', 25)
            self.degree = kwargs.get('degree', 3)
            self.n_trail = kwargs.get('n_trail', 1)
            self.n_lead = kwargs.get('n_lead', 0)
            init_knots = torch.linspace(0, self.cutoff_radius, n_intervals + 1, device='cuda:0')
            knot_dist = init_knots[1] - init_knots[0]
            self.t = torch.cat([torch.tensor([init_knots[0] - i * knot_dist for i in range(3, 0, -1)], device='cuda:0'),
                                init_knots,
                                torch.tensor([init_knots[-1] + i * knot_dist for i in range(3)], device='cuda:0')])
            self.n_params = self.t.shape[0] - self.degree - 1
            self.potential = self.spline_potential
        elif self.potential_type == 'Sum':
            self.n_params = 1
            self.potential = self.simple_sum
            
        if self.use_atomic_energy:
            self.base_atomic_energy = nn.ParameterList([nn.Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(self.num_elements)]).to('cuda')

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
        assert pool_order in ['early', 'late'], f"{pool_order} is currently not supported"
        self.pool_order = pool_order
        self.output_dim = output_dim
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
                aggr,
            ).jittable()
            self.attention_layers.append(layer)

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
        
        if self.node_edge_ratio[1] != 0.:
            self.potential_lin_list = self._setup_potential_layers()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        
    def _setup_potential_layers(self):
        layers = torch.nn.ModuleList()
        for _ in range(self.n_params):
            param_list = []
            for i in range(self.potential_fc_count):
                if i == 0:
                    in_dim = self.hidden_channels + self.edge_dim if self.potential_type == 'Sum' else self.hidden_channels
                    layer = torch.nn.Linear(in_dim, self.potential_dim).to('cuda')
                else:
                    layer = torch.nn.Linear(self.potential_dim, self.potential_dim).to('cuda')
                param_list.append(layer)
            out_layer = torch.nn.Linear(self.potential_dim, 1).to('cuda')
            param_list.append(out_layer)
            layers.append(torch.nn.ModuleList(param_list))
        return layers
        
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
        
        node_embeddings = x
        # edge_embeddings = (node_embeddings[data.edge_index[0]] +
        #                    node_embeddings[data.edge_index[1]]) / 2
        
        params = []
        if self.node_edge_ratio[1] != 0.:
            for i in range(self.n_params):
                if self.potential_type == 'Sum':
                    param_out = torch.cat(((node_embeddings[data.edge_index[0]] +
                                node_embeddings[data.edge_index[1]]) / 2, data.edge_attr), dim=1)
                    # print(((node_embeddings[data.edge_index[0]] +
                    #             node_embeddings[data.edge_index[1]]) / 2).shape, data.edge_attr.shape)
                    # print(param_out.shape)
                else:
                    param_out = (node_embeddings[data.edge_index[0]] +
                                node_embeddings[data.edge_index[1]]) / 2
                    
                for j in range(len(self.potential_lin_list[i])):
                    if j != len(self.potential_lin_list[i]) - 1:
                        param_out = self.potential_lin_list[i][j](param_out)
                        param_out = getattr(F, self.activation)(param_out)
                    else:
                        param_out = self.potential_lin_list[i][j](param_out)
            
                if i == 0:
                    param_out = torch.clamp(param_out, min=0.5)
                elif i == 1:
                    param_out = torch.clamp(param_out, min=1, max=3)
                elif i == 2:
                    param_out = torch.clamp(param_out, min=1)
                # param_out = torch.clamp(param_out, min=0)
                # param_out = (param_out[data.edge_index[0]] +
                #             param_out[data.edge_index[1]]) / 2
                params.append(param_out)
            pot = self.potential(params, data)  
        
        if self.prediction_level == "graph":
            if self.pool_order == 'early':
                x = getattr(torch_geometric.nn, self.pool)(x, data.batch)
            for i in range(0, len(self.post_lin_list) - 1):
                x = self.post_lin_list[i](x)
                x = getattr(F, self.activation)(x)
            x = self.post_lin_list[-1](x)
            if self.pool_order == 'late':
                x = getattr(torch_geometric.nn, self.pool)(x, data.batch)
            #x = self.pool.pre_reduce(x, vec, data.z, data.pos, data.batch)
            #x = self.pool.reduce(x, data.batch)
        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list) - 1):
                x = self.post_lin_list[i](x)
                x = getattr(F, self.activation)(x)
            x = self.post_lin_list[-1](x)
           
        if self.node_edge_ratio[1] != 0.:
            return self.node_edge_ratio[0] * x + self.node_edge_ratio[1] * pot
        return self.node_edge_ratio[0] * x
        
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
    
    def simple_sum(self, params, data):
        
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        res = scatter_add(params[0].view(1, -1), index=edge_idx_to_graph, dim_size=len(data))
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
            for z in np.unique(data.z.cpu()):
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]

            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        return res.view(-1, 1) if not self.use_atomic_energy else res.view(-1, 1) + base_atomic_energy.reshape(-1, 1)
    
        
    def lj_cutoff_function(self, r, rc, ro):
        return torch.where(
            r < ro,
            1.0,
            torch.where(r < rc, (rc - r) ** 2 * (rc + 2 *
                    r - 3 * ro) / (rc - ro) ** 3, 0.0),
        )
        
    def morse_cutoff_function(self, r, rc, ro):
        s = 1.0 - (r - rc) / (ro - rc)
        return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) *
                            (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))
    
    def lj_potential(self, params, data):
        sigmas, epsilons = params[0].squeeze(), params[1].squeeze()

        edge_idx_to_graph = data.batch[data.edge_index[0]]
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda')
            for z in np.unique(data.z.cpu()):
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
        
        rc = self.cutoff_radius
        ro = 0.66 * rc
        r2 = data.edge_weight ** 2
        cutoff_fn = self.lj_cutoff_function(r2, rc**2, ro**2)
        
        c6 = (sigmas ** 2 / r2) ** 3
        c6[r2 > rc ** 2] = 0.0
        c12 = c6 ** 2
        
        pairwise_energies = 4 * epsilons * (c12 - c6)
        pairwise_energies *= cutoff_fn

        lennard_jones_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
    
        if self.use_atomic_energy:
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
               
        out = lennard_jones_out.reshape(-1, 1)
        return out if not self.use_atomic_energy else out + base_atomic_energy.reshape(-1, 1)
    
    def morse_potential(self, params, data):
        
        sigma, rm, D = params[0].squeeze(), params[1].squeeze(), params[2].squeeze()
        
        # print(sigma.mean().item(), rm.mean().item(), D.mean().item())
        
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda')
            for z in np.unique(data.z.cpu()):
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]

        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight
        fc = self.morse_cutoff_function(d, ro, rc)
        E = D * (1 - torch.exp(-sigma * (d - rm))) ** 2 - D
        
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        pairwise_energies = 0.5 * (E * fc)
        
        # print(torch.sum(data.edge_index[0] == data.edge_index[1]))
        # pairwise_energies[data.edge_index[0] == data.edge_index[1]] = 0
        
        # pairwise_energies[atoms[0] == atoms[1]] = 0
        morse_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))

        if self.use_atomic_energy:
            # base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        out = morse_out.reshape(-1, 1)
        return out if not self.use_atomic_energy else out + base_atomic_energy.reshape(-1, 1)
    
    def spline_potential(self, params, data):
        # params: list of length n_params, [n_nodes, 100]
        atoms = data.z[data.edge_index] - 1
        
        params_tensor = torch.stack(params).to('cuda:0')
        all_coefs_i = params_tensor[:, data.edge_index[0], atoms[0]]
        all_coefs_j = params_tensor[:, data.edge_index[1], atoms[1]]
        
        # all_sigmas, all_rms, all_Ds  = params[0], params[1], params[2]
        coefs = ((all_coefs_i + all_coefs_j) / 2).reshape(-1, self.n_params)

        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
            for z in np.unique(data.z.cpu()):
                if self.use_atomic_energy:
                    base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
            
        # atoms = data.z[data.edge_index] - 1
        # coef_i, coef_j = coefs[atoms[0]], coefs[atoms[1]]
        # coef = (coef_i + coef_j) / 2
        
        spline_res = torch.empty(len(data.edge_weight), device='cuda:0')

        spline_res = self.b_spline(self.degree, self.t, coefs, data.edge_weight).squeeze()
        # print(spline_res.shape)
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        spline_out = 0.5 * scatter_add(spline_res, index=edge_idx_to_graph, dim_size=len(data))
        
        if self.use_atomic_energy:
            base_atomic_energy = base_atomic_energy[data.z - 1].squeeze()  
            base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        
        res = spline_out.reshape(-1, 1)
        if self.use_atomic_energy:
            res += base_atomic_energy.reshape(-1, 1)
        return res
    
    def vectorized_deboor(self, t, x, k, interval, result):
        hh = result[:, k+1:]
        h = result
        h[:, 0] = 1.
            
        for j in range(1, k + 1):
            hh[:, :j] = h[:, :j]
            h[:, 0] = 0
            for n in range(1, j + 1):
                ind = interval + n
                xb = t[ind]
                xa = t[ind - j]
                zero_mask = xa == xb
                h[zero_mask, n] = 1.
                w = hh[:, n - 1] / (xb - xa)
                h[:, n - 1] += w * (xb - x)
                h[:, n] = w * (x - xa)
                h[zero_mask, n] = 0.
            
        return result
    
    def vec_find_intervals(self, t: torch.tensor, x: torch.tensor, k: int):
        n = t.shape[0] - k - 1
        tb = t[k]
        te = t[n]

        result = torch.zeros_like(x, dtype=int, device='cuda:0')

        nan_indices = torch.isnan(x)
        out_of_bounds_indices = torch.logical_or(x < tb, x > te)
        result[out_of_bounds_indices | nan_indices] = -1

        l = torch.full_like(x, k, dtype=int)
        while torch.any(torch.logical_and(x < t[l], l != k)):
            l = torch.where(torch.logical_and(x < t[l], l != k), l - 1, l)

        l = torch.where(l != n, l + 1, l)

        while torch.any(torch.logical_and(x >= t[l], l != n)):
            l = torch.where(torch.logical_and(x >= t[l], l != n), l + 1, l)

        result = torch.where(result != -1, l - 1, result)
        return result
    
    def evaluate_spline(self, t, c, k, xp, out):
        if out.shape[0] != xp.shape[0]:
            raise ValueError("out and xp have incompatible shapes")
        # if out.shape[1] != c.shape[1]:
        #     raise ValueError("out and c have incompatible shapes")

        # work: (N, 2k + 2)
        work = torch.empty(xp.shape[0], 2 * k + 2, dtype=torch.float, device='cuda:0')

        # intervals: (N, )
        intervals = self.vec_find_intervals(t, xp, k)
        invalid_mask = intervals < 0
        out[invalid_mask, :] = np.nan
        out[~invalid_mask, :] = 0.
        
        if invalid_mask.all():
            return out
        
        work[~invalid_mask] = self.vectorized_deboor(t, xp[~invalid_mask], k, intervals[~invalid_mask], work[~invalid_mask])
        # print(work[~invalid_mask].shape, c[intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].shape)
        
        # c = c[:, intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].squeeze(dim=2)
        indices = intervals[:, None] + torch.arange(-k, 1, device='cuda:0')

        # Index into c using advanced indexing
        c = c[torch.arange(c.size(0))[:, None], indices]
        out[~invalid_mask, :] = torch.sum(work[~invalid_mask, :k+1] * c[~invalid_mask, :], dim=1).unsqueeze(dim=-1)
        return out
    
    def b_spline(self, k, t, c, x):
        out = torch.empty((len(x), 1), dtype=c.dtype, device='cuda:0')
        res = torch.nan_to_num(self.evaluate_spline(t, c.reshape(c.shape[0], -1), k, x, out), nan=0)
        return res


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
