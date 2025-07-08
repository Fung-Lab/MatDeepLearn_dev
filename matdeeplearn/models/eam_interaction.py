import logging
from itertools import combinations
from typing import Any, Literal

import mendeleev as mdv
import numpy as np
import torch

from torch import nn
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
    

class EAMParameters(nn.Module):
    def element2number(self, element):
        return mdv.element(element).atomic_number
    
    def __init__(
        self,
        atom_types: list[str],
        n_basis: int,
        init_mode: Literal['constant', 'covalent_radius', 'custom'] = 'constant',
        init_args: dict[str, Any] | None = None,
        freeze_params: list[str] | None = None,
        device='cuda:0'
    ):
        """
        Initialize interaction parameters with user-provided values
        
        Args:
            atom_types: List of atomic numbers or element types
            init_values: Dictionary with format {(type1, type2): {'param_name': value}}
            default_value: Default value for unspecified parameters
        """
        super().__init__()
        
        self.atom_types = atom_types
        self.type_to_idx = {
            atom_type: self.element2number(atom_type) for atom_type in atom_types
        }
        self.device = device
        
        interactions = list(combinations(self.atom_types, 2))
        interactions.extend([(atom_type, atom_type) for atom_type in self.atom_types])
        
        self.interactions = [
            (self.type_to_idx[type1], self.type_to_idx[type2])
            for type1, type2 in interactions
        ]
        max_element = max(self.type_to_idx.values())
        
        # Map (z1, z2) to index of parameter in the parameter list
        self.pair2idx = {
            interaction: idx for idx, interaction in enumerate(self.interactions)
        }
        self.interaction_lookup = torch.zeros(
            max_element + 1, max_element + 1, dtype=torch.long, device=device
        )
        for (i, j), idx in self.pair2idx.items():
            self.interaction_lookup[i, j] = idx
            self.interaction_lookup[j, i] = idx
        self.interaction_lookup.requires_grad = False
        
        # Map z to index of parameter in the parameter list
        self.z2idx = {
            z: idx for idx, z in enumerate(self.type_to_idx.values())
        }
        self.atom_lookup = torch.zeros(
            max_element + 1, dtype=torch.long, device=device
        )
        for z, idx in self.z2idx.items():
            self.atom_lookup[z] = idx
        self.atom_lookup.requires_grad = False
        
        self.param_dict = nn.ModuleDict()
        dimensions = {
            'D': 1, 'rm': 1, 'alpha': 1, 'base_atomic_energy': 1,
            'A': n_basis, 'beta': n_basis, 'B': 1, 'rho0': 1
        }
        
        # Interaction parameters
        for param in ('D', 'rm', 'alpha', 'A', 'beta'):
            params = [
                torch.nn.Parameter(
                    torch.full(
                        (dimensions[param],), 1, dtype=torch.float32, device=device
                    )
                ) for _ in range(len(self.interactions))
            ]
            self.param_dict[param] = nn.ParameterList(params)
            
        # Atomic parameters
        for param in ('base_atomic_energy', 'B', 'rho0'):
            params = [
                torch.nn.Parameter(
                    torch.full(
                        (dimensions[param],), 1, dtype=torch.float32, device=device
                    )
                ) for _ in range(len(atom_types))
            ]
            self.param_dict[param] = nn.ParameterList(params)
            
        self.init_parameters(init_mode, init_args)
        # Freeze parameters
        if freeze_params is not None:
            for param_name in freeze_params:
                for param in self.param_dict[param_name]:
                    assert isinstance(param, nn.Parameter)
                    param.requires_grad = False
            
    def init_parameters(
        self,
        init_mode: Literal['constant', 'covalent_radius', 'custom'],
        init_args: dict[str, Any] | None
    ):
        """
        init_values as dict:
        {
            "Mn-O": {"rm": 1.0, "D": 1.0, "alpha": 1.0} for interaction parameter
            "Mn": {"base_atomic_energy": 1.0} for atomic parameter
        }
        """
        if init_mode == 'constant':
            init_val = init_args.get('constant_value', 1.0)
            for params in self.param_dict.values():
                # params: nn.ParameterList
                for p in params:
                    assert isinstance(p, nn.Parameter)
                    p.data.fill_(init_val)
        elif init_mode == 'custom':
            assert init_args is not None
            for key, val_dict in init_args.items():
                assert isinstance(val_dict, dict)
                if '-' in key:
                    type1, type2 = key.split('-')
                    z1, z2 = self.type_to_idx[type1], self.type_to_idx[type2]
                    param_idx = self.pair2idx[(z1, z2)]
                    valid_params = ('D', 'rm', 'alpha', 'A', 'beta')
                else:
                    z = self.type_to_idx[key]
                    param_idx = self.z2idx[z]
                    valid_params = ('base_atomic_energy', 'B', 'rho0')
                
                for param_name, value in val_dict.items():
                    assert param_name in valid_params
                    param = self.param_dict[param_name][param_idx]
                    assert isinstance(param, nn.Parameter)
                    param.data.fill_(value)
        elif init_mode == 'covalent_radius':
            scale = init_args.get('scale', 1.0)
            init_vals = {
                idx: (
                    mdv.element(z1).covalent_radius
                    + mdv.element(z2).covalent_radius
                ) * 0.01 * scale
                for (z1, z2), idx in self.pair2idx.items()
            }
            for idx, init_val in init_vals.items():
                param = self.param_dict['rm'][idx]
                assert isinstance(param, nn.Parameter)
                param.data.fill_(init_val)
        else:
            raise ValueError(f"Invalid init_mode: {init_mode}")               
    
    def query_batch(self, query: torch.Tensor, param_name: str) -> torch.Tensor:
        param_tensor = torch.zeros(
            (len(self.param_dict[param_name]), len(self.param_dict[param_name][0])),
            dtype=torch.float32
        ).to(self.device)
        for idx in range(len(param_tensor)):
            param_tensor[idx] = self.param_dict[param_name][idx]
        
        if query.ndim == 2: # Interaction query
            atom1 = query[0, :]
            atom2 = query[1, :]
            indices = self.interaction_lookup[atom1, atom2]
        else: # Atomic query
            indices = self.atom_lookup[query]
        return param_tensor[indices].squeeze()    

class EAMEmbedding(nn.Module):
    def __init__(self, cutoff_radius):
        super(EAMEmbedding, self).__init__()
        self.cutoff_radius = cutoff_radius
    
    def embedding_function(self, rho, B, rho0):
        return B * (rho - rho0)**2
        
    def forward(self, data, B, rho0, A, beta):
        rc = self.cutoff_radius
        ro = 0.66 * rc
        d = data.edge_weight
        fc = cutoff_function(d, ro, rc)        
        rho_ij = (A * torch.exp(-beta * d.unsqueeze(-1))).sum(dim=1) * fc
        rho_i = scatter_add(rho_ij, index=data.edge_index[0], dim_size=data.num_nodes)
        E_embed = self.embedding_function(rho_i, B, rho0)
        eam_out = scatter_add(E_embed, index=data.batch, dim_size=len(data))        
        return eam_out.reshape(-1, 1)


@registry.register_model("EAM_Interaction")
class EAM_Interaction(BaseModel):
    def __init__(
        self,
        atom_types: list[str],
        n_basis: int = 10,
        init_mode: Literal['constant', 'covalent_radius', 'custom'] = 'constant',
        init_args: dict[str, Any] | None = None,
        freeze_params: list[str] | None = None,
        **kwargs
    ):
        super(EAM_Interaction, self).__init__(**kwargs)
        if init_args is None:
            init_args = {}
        self.eam_params = EAMParameters(
            atom_types, n_basis, init_mode, init_args, freeze_params
        )
        self.eam = EAMEmbedding(self.cutoff_radius)
        
    @property
    def target_attr(self):
        return "y"

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        if self.otf_edge_index == True:
            #data.edge_index, edge_weight, data.edge_vec, cell_offsets, offset_distance, neighbors = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)   
            data.edge_index, data.edge_weight, _, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)  
    
        pot = self.potential(data)
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
    
    def potential(self, data):
        atoms = data.z[data.edge_index]
        
        D = self.eam_params.query_batch(atoms, 'D')
        rm = self.eam_params.query_batch(atoms, 'rm')
        alpha = self.eam_params.query_batch(atoms, 'alpha')
        
        A = self.eam_params.query_batch(atoms, 'A')
        beta = self.eam_params.query_batch(atoms, 'beta')
        B = self.eam_params.query_batch(data.z, 'B')
        rho0 = self.eam_params.query_batch(data.z, 'rho0')
        
        base_atomic_energy = self.eam_params.query_batch(data.z, 'base_atomic_energy')
        
        rc = self.cutoff_radius
        ro = 0.66 * rc

        d = data.edge_weight
        fc = cutoff_function(d, ro, rc)
        
        E = D * (1 - torch.exp(-alpha * (d - rm))) ** 2 - D
        eam_out = self.eam(data, B, rho0, A, beta)
        
        pairwise_energies = E * fc
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        morse_out = 0.5 * scatter_add(pairwise_energies, index=edge_idx_to_graph, dim_size=len(data))
    
        base_atomic_energy = scatter_add(base_atomic_energy, index=data.batch)
        return morse_out.reshape(-1, 1)\
            + base_atomic_energy.reshape(-1, 1)\
            + eam_out
        # return morse_out.reshape(-1, 1) + base_atomic_energy.reshape(-1, 1)