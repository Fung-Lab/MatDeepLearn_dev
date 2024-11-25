import numpy as np
import torch

from torch import nn
from torch.nn import Parameter, ParameterList
from torch_scatter import scatter_add

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad


def cutoff_function(r, rc, ro):
    """
    Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.
    Ported from JuLIP.jl, https://github.com/JuliaMolSim/JuLIP.jl

    Parameters
    ----------
    rc - inner cutoff radius
    ro - outder cutoff radius
    """""
    s = 1.0 - (r - rc) / (ro - rc)
    return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) *
                        (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))

class MorsePotential(nn.Module):
    def __init__(self, cutoff_radius):
        super(MorsePotential, self).__init__()
        self.cutoff_radius = cutoff_radius
        
    def forward(self, data, D, rm, alpha):
        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight
        fc = cutoff_function(d, ro, rc)
        E = D * (1 - torch.exp(-alpha * (d - rm))) ** 2 - D
        
        pairwise_energies = 0.5 * (E * fc)
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        morse_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
        return morse_out.reshape(-1, 1)
    
class EAMEmbedding(nn.Module):
    def __init__(self, cutoff_radius, n_exp_basis):
        super(EAMEmbedding, self).__init__()
        self.n_exp_basis = n_exp_basis
        self.cutoff_radius = cutoff_radius

    def electron_density(self, r, A, beta):
        # print(beta[:, 0].shape)
        return (A * torch.exp(-beta * r.unsqueeze(-1))).sum(dim=1)
    
    def embedding_function(self, rho, B, rho0):
        return B * (rho - rho0)**2
        
    def forward(self, data, A, beta, B, rho0):
        rc = self.cutoff_radius
        ro = 0.66 * rc
        d = data.edge_weight
        fc = cutoff_function(d, ro, rc)        
        rho_ij = self.electron_density(d, A, beta) * fc        
        rho_i = scatter_add(rho_ij, index=data.edge_index[0], dim_size=data.num_nodes)
        E_embed = self.embedding_function(rho_i, B, rho0)
        eam_out = scatter_add(E_embed, index=data.batch, dim_size=len(data))        
        return eam_out.reshape(-1, 1)

@registry.register_model("EAM_Basis")
class EAM_Basis(BaseModel):
    def __init__(
        self,
        **kwargs
    ):
        super(EAM_Basis, self).__init__(**kwargs)
        self.combination_method = kwargs.get('combination_method', 'average')
        self.with_coefs = kwargs.get("with_coefs", False)
        self.n_exp_basis = kwargs.get("n_exp_basis", 10)
        
        param_init = kwargs.get("param_init", {})
        rm_init = param_init.get("rm", 1.0)
        alphas_init = param_init.get("alphas", 1.5)
        D_init = param_init.get("D", 1.0)
        A_init = param_init.get("A", 1.0)
        beta_init = param_init.get("beta", 3.0)
        B_init = param_init.get("B", 1.0)
        rho0_init = param_init.get("rho0", 0.1)
        # base_atomic_energy_init = param_init.get("base_atomic_energy", -1.5)

        self.rm = ParameterList([Parameter(rm_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0') 
        self.alphas = ParameterList([Parameter(alphas_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.D = ParameterList([Parameter(D_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.A = ParameterList([Parameter(A_init * torch.ones(self.n_exp_basis,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.beta = ParameterList([Parameter(beta_init * torch.ones(self.n_exp_basis,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.B = ParameterList([Parameter(B_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        self.rho0 = ParameterList([Parameter(rho0_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0')
        # self.base_atomic_energy = ParameterList([Parameter(base_atomic_energy_init * torch.ones(1,), requires_grad=True) for _ in range(100)]).to('cuda:0')

        self.morse = MorsePotential(self.cutoff_radius)
        self.eam_density = EAMEmbedding(self.cutoff_radius, self.n_exp_basis)

    @property
    def target_attr(self):
        return "y"

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, _, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
        pot = self.eam(data).view(-1, 1)
        return pot
    
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
    
    def eam(self, data):
        atoms = data.z[data.edge_index] - 1
        
        atomic_rm = torch.zeros((len(self.rm), 1)).to('cuda:0')
        atomic_D = torch.zeros((len(self.D), 1)).to('cuda:0')
        atomic_alphas = torch.zeros((len(self.alphas), 1)).to('cuda:0')
        atomic_A = torch.zeros((len(self.A), self.n_exp_basis)).to('cuda:0')
        atomic_beta = torch.zeros((len(self.beta), self.n_exp_basis)).to('cuda:0')
        atomic_B = torch.zeros((len(self.B), 1)).to('cuda:0')
        atomic_rho0 = torch.zeros((len(self.rho0), 1)).to('cuda:0')
        # base_atomic_energy = torch.zeros((len(self.base_atomic_energy), 1)).to('cuda:0')

        for z in np.unique(data.z.cpu()):
            atomic_alphas[z - 1] = self.alphas[z - 1]
            atomic_rm[z - 1] = self.rm[z - 1]
            atomic_D[z - 1] = self.D[z - 1]
            atomic_A[z - 1] = self.A[z - 1]
            atomic_beta[z - 1] = self.beta[z - 1]
            atomic_B[z - 1] = self.B[z - 1]
            atomic_rho0[z - 1] = self.rho0[z - 1]
            # base_atomic_energy[z - 1] = self.base_atomic_energy[z - 1]
        
        rm_i, rm_j = atomic_rm[atoms[0]], atomic_rm[atoms[1]]
        sigma_i, sigma_j = atomic_alphas[atoms[0]], atomic_alphas[atoms[1]]
        D_i, D_j = atomic_D[atoms[0]], atomic_D[atoms[1]]
        A_i, A_j = atomic_A[atoms[0]], atomic_A[atoms[1]]
        beta_i, beta_j = atomic_beta[atoms[0]], atomic_beta[atoms[1]]

        B = atomic_B[data.z - 1].squeeze()
        rho0 = atomic_rho0[data.z - 1].squeeze()

        rm = (rm_i + rm_j).squeeze() / 2
        sigma = (sigma_i + sigma_j).squeeze() / 2
        D = (D_i + D_j).squeeze() / 2
        A = (A_i + A_j).squeeze() / 2
        beta = (beta_i + beta_j).squeeze() / 2
        
        morse = self.morse(data, D, rm, sigma)
        eam = self.eam_density(data, A, beta, B, rho0)
        return morse + eam