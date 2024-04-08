import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential, Parameter, ParameterList
from torch_geometric.nn import (
    CGConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot

@registry.register_model("Hybrid_CGCNN_Edge_Readout")
class Hybrid_CGCNN_Edge_Readout(BaseModel):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(Hybrid_CGCNN_Edge_Readout, self).__init__(**kwargs)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.gc_count = gc_count
        self.post_fc_count = post_fc_count
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.num_elements = kwargs.get('num_elements', 100)
        self.param_init_method = kwargs.get('param_init_method', None)
        self.use_atomic_energy = kwargs.get('use_atomic_energy', True)
        self.potential_fc_count = kwargs.get('potential_fc_count', 3)
        self.potential_dim = kwargs.get('potential_dim', 64)
        
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
            self.base_atomic_energy = ParameterList([Parameter(-1.5 * torch.ones(1, 1), requires_grad=True) for _ in range(self.num_elements)]).to('cuda:0')
        
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff_radius, self.edge_dim, 0.2)

        # Determine gc dimension and post_fc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.node_dim, self.node_dim
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        # setup layers
        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()
        self.potential_lin_list = self._setup_potential_layers()
        
        # set up output layer
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(self.post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not recommended to use set2set
            self.lin_out_2 = torch.nn.Linear(self.output_dim * 2, self.output_dim)

    @property
    def target_attr(self):
        return "y"

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.node_dim, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):
            conv = CGConv(
                self.gc_dim, self.edge_dim, aggr="mean", batch_norm=False
            )
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats
                )
                bn_list.append(bn)

        return conv_list, bn_list

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(self.post_fc_dim * 2, self.dim2)
                    else:
                        lin = torch.nn.Linear(self.post_fc_dim, self.dim2)
                else:
                    lin = torch.nn.Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = torch.nn.Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = torch.nn.Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = torch.nn.Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out
    
    def _setup_potential_layers(self):
        all_layers = torch.nn.ModuleList()
        for _ in range(self.n_params):
            layers = torch.nn.ModuleList()
            for i in range(self.potential_fc_count):
                if i == 0:
                    layer = torch.nn.Linear(self.post_fc_dim, self.potential_dim).to('cuda')
                else:
                    layer = torch.nn.Linear(self.potential_dim, self.potential_dim).to('cuda')
                layers.append(layer)
                # bn_layer = torch.nn.BatchNorm1d(self.potential_dim).to('cuda')
                # layers.append(bn_layer)
            out_layer = torch.nn.Linear(self.potential_dim, 1).to('cuda')
            layers.append(out_layer)
            all_layers.append(layers)
        return all_layers

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, _, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
            if self.otf_edge_attr == True:
                data.edge_attr = self.distance_expansion(data.edge_weight)
            else:
                logging.warning("Edge attributes should be re-computed for otf edge indices.")
                
        if self.otf_edge_index == False:
            if self.otf_edge_attr == True:
                data.edge_attr = self.distance_expansion(data.edge_weight) 
                
        if self.otf_node_attr == True:
            data.x = node_rep_one_hot(data.z).float()        
            
        # Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm:
                    out = self.conv_list[i](
                        data.x, data.edge_index, data.edge_attr
                    )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        data.x, data.edge_index, data.edge_attr
                    )
            else:
                if self.batch_norm:  
                    out = self.conv_list[i](
                        out, data.edge_index, data.edge_attr
                    )
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](
                        out, data.edge_index, data.edge_attr
                    )
                    # out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            
        
        node_embeddings = out
        # edge_embeddings = (node_embeddings[data.edge_index[0]] +
        #                    node_embeddings[data.edge_index[1]]) / 2
        
        # print(edge_embeddings.shape, data.edge_index.shape)
        
        # param layers
        params = []
        for i in range(self.n_params):
            param_out = (node_embeddings[data.edge_index[0]] +
                        node_embeddings[data.edge_index[1]]) / 2
            for j in range(len(self.potential_lin_list[i])):
                if j != len(self.potential_lin_list[i]) - 1:
                    param_out = self.potential_lin_list[i][j](param_out)
                    param_out = getattr(F, self.act)(param_out)
                else:
                    param_out = self.potential_lin_list[i][j](param_out)
            
            # if i == 0:
            #     param_out = torch.clamp(param_out, min=0.5)
            # elif i == 1:
            #     param_out = torch.clamp(param_out, min=1, max=3)
            # elif i == 2:
            #     param_out = torch.clamp(param_out, min=1)
            params.append(param_out)
        
        # Post-GNN dense layers
        if self.prediction_level == "graph":
            if self.pool_order == "early":
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)
    
            elif self.pool_order == "late":
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                    out = self.lin_out_2(out)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                    
        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)                
        
        # print(param[0][:5, 13], param[1][:5, 13])
        pot = self.potential(params, data)  
        return pot + out   
        
        
    def forward(self, data):
        
        output = {}
        out = self._forward(data)
        output["output"] = out

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
    
    def simple_sum(self, params, data):
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        res = 0.5 * scatter_add(params[0].view(1, -1), index=edge_idx_to_graph, dim_size=len(data))
        
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
        atoms = data.z[data.edge_index] - 1
        
        all_sigmas, all_epsilons = params[0], params[1]
        
        # print(all_sigmas.shape, all_epsilons.shape)
        all_edge_sigmas_i, all_edge_sigmas_j = all_sigmas[data.edge_index[0], atoms[0]], all_sigmas[data.edge_index[1], atoms[1]]
        all_edge_epsilons_i, all_edge_epsilons_j = all_epsilons[data.edge_index[0], atoms[0]], all_epsilons[data.edge_index[1], atoms[1]]
        
        sigmas = (all_edge_sigmas_i + all_edge_sigmas_j) / 2
        epsilons = (all_edge_epsilons_i + all_edge_epsilons_j) / 2

        edge_idx_to_graph = data.batch[data.edge_index[0]]
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
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
        
        sigma = params[0].squeeze()
        rm = params[1].squeeze()
        D = params[2].squeeze()
        
        # print(sigma.shape, rm.shape, D.shape)
        
        # print(sigma[:5], rm[:5], D[:5])
        
        if self.use_atomic_energy:
            base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')
            for z in np.unique(data.z.cpu()):
                base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]

        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight
        fc = self.morse_cutoff_function(d, ro, rc)
        E = D * (1 - torch.exp(-sigma * (d - rm))) ** 2 - D
        
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        pairwise_energies = 0.5 * (E * fc)
        
        pairwise_energies[data.edge_index[0] == data.edge_index[1]] = 0
        morse_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))

        if self.use_atomic_energy:
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
        # print(coefs.shape, data.edge_weight.shape)
        
        # print(all_sigmas.shape, all_epsilons.shape)
        # all_edge_sigmas_i, all_edge_sigmas_j = all_sigmas[data.edge_index[0], atoms[0]], all_sigmas[data.edge_index[1], atoms[1]]
        # all_edge_rms_i, all_edge_rms_j = all_rms[data.edge_index[0], atoms[0]], all_rms[data.edge_index[1], atoms[1]]
        # all_edge_ds_i, all_edge_ds_j = all_Ds[data.edge_index[0], atoms[0]], all_Ds[data.edge_index[1], atoms[1]]
        
        # sigma = (all_edge_sigmas_i + all_edge_sigmas_j) / 2
        # rm = (all_edge_rms_i + all_edge_rms_j) / 2
        # D = (all_edge_ds_i + all_edge_ds_j) / 2
        
        # coefs = torch.zeros((len(self.coefs), self.n)).to('cuda:0')
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
