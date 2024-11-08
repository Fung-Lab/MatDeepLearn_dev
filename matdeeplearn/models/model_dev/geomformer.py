import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch as TorchGeoBatch

from matdeeplearn.common.registry import registry
from matdeeplearn.models.model_dev.graphormer import GaussianLayer
from matdeeplearn.preprocessor.pbc_transform import Batch
from matdeeplearn.models.base_model import conditional_grad, BaseModel
from matdeeplearn.models.utils import GaussianSmearing


import torch
import torch.nn as nn

class EquivariantLayerNorm(nn.Module):
    def __init__(self, emb_e):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_e))
        self.bias = nn.Parameter(torch.zeros(emb_e))
        self.eps = 1e-5

    def forward(self, x):
        # x shape: (batch, n, 3, emb_e)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, n, 3, 1)
        
        # Compute covariance matrix
        x_centered = x - mean
        cov = torch.einsum('bnid,bnjd->bnij', x_centered, x_centered) / x.shape[-1]
        
        # Compute U (inverse square root of covariance matrix)
        eye = torch.eye(3, device=x.device).unsqueeze(0).unsqueeze(0)
        U = torch.linalg.pinv((cov + self.eps * eye) ** 0.5)
        # U = torch.matrix_power(U, 1/2)  # Square root of the inverse
        
        # Apply Equ-LN: U(z_i^E - μ1^T) ⊙ γ
        x_normalized = torch.einsum('bnij,bnjd->bnid', U, x_centered)
        
        return x_normalized * self.scale + self.bias

class GeomformerBlock(nn.Module):
    def __init__(self, emb_n, emb_e, n_heads) -> None:
        super(GeomformerBlock, self).__init__()
        
        self.emb_n = emb_n
        self.emb_e = emb_e
        self.n_heads = n_heads
        self.inv_head_dim = emb_n // n_heads
        self.equ_head_dim = emb_e // n_heads
        
        assert (
            self.inv_head_dim * n_heads == emb_n
        ), "Embedding dimension must be divisible by number of heads"
        assert (
            self.equ_head_dim * n_heads == emb_e
        ), "Embedding dimension must be divisible by number of heads"
        
        self.inv_self_attn = nn.ModuleDict({
            "W_Q_I": nn.Linear(emb_n, emb_n),
            "W_K_I": nn.Linear(emb_n, emb_n),
            "W_V_I": nn.Linear(emb_n, emb_n),
        })
        self.equ_self_attn = nn.ModuleDict({
            "W_Q_E": nn.Linear(emb_e, emb_e),
            "W_K_E": nn.Linear(emb_e, emb_e),
            "W_V_E": nn.Linear(emb_e, emb_e),
        })
        self.inv_cross_attn = nn.ModuleDict({
            "W_K_IE1": nn.Linear(emb_e, emb_n),
            "W_K_IE2": nn.Linear(emb_e, emb_n),
            "W_V_IE1": nn.Linear(emb_e, emb_n),
            "W_V_IE2": nn.Linear(emb_e, emb_n),
        })
        self.equ_cross_attn = nn.ModuleDict({
            "W_K_EI1": nn.Linear(emb_e, emb_e),
            "W_K_EI2": nn.Linear(emb_n, emb_e),
            "W_V_EI1": nn.Linear(emb_e, emb_e),
            "W_V_EI2": nn.Linear(emb_n, emb_e),
        })
        self.inv_ffn = nn.Sequential(
            nn.Linear(emb_n, emb_n),
            nn.GELU(),
            nn.Linear(emb_n, emb_n),
        )
        self.equ_ffn = nn.ModuleDict({
            "W_inv": nn.Sequential(
                nn.Linear(emb_n, emb_e),
                nn.GELU(),
                nn.Linear(emb_e, emb_e),
            ),
            "W_equ": nn.Linear(emb_e, emb_e),
        })
        
        self.inv_layer_norms = nn.ModuleList([nn.LayerNorm(emb_n) for _ in range(3)])
        self.equ_layer_norms = nn.ModuleList([nn.LayerNorm(emb_e) for _ in range(3)])
        self.out_proj_inv_self_attn = nn.Linear(emb_n, emb_n)
        self.out_proj_inv_cross_attn = nn.Linear(emb_n, emb_n)
        self.out_proj_equ_self_attn = nn.Linear(emb_e, emb_e)
        self.out_proj_equ_cross_attn = nn.Linear(emb_e, emb_e)


    def dot_product(self, X, Y):
        # X, Y shape: (batch, n, 3, d)
        # Output shape: (batch, n, d)
        return torch.sum(X * Y, dim=2)
    
    def scalar_product(self, X, Y):
        # X shape: (batch, n, 3, d)
        # Y shape: (batch, n, d)
        # Output shape: (batch, n, 3, d)
        return X * Y.unsqueeze(-2)
    
    def inv_attention(self, bsz, n, Q, K, V, bias=None):
        Q = Q.view(bsz, n, self.n_heads, self.inv_head_dim).transpose(1, 2)  # (batch, n_heads, n, inv_head_dim)
        K = K.view(bsz, n, self.n_heads, self.inv_head_dim).transpose(1, 2)  # (batch, n_heads, n, inv_head_dim)
        V = V.view(bsz, n, self.n_heads, self.inv_head_dim).transpose(1, 2)  # (batch, n_heads, n, inv_head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.inv_head_dim ** 0.5)  # (batch, n_heads, n, n)

        if bias is not None:
            scores += bias
        
        attn_probs = F.softmax(scores, dim=-1)  # (batch, n_heads, n, n)        
        attn_output = torch.matmul(attn_probs, V)  # (batch, n_heads, n, inv_head_dim)        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, n, self.emb_n)  # (batch, n, emb_dim)
        
        return attn_output, attn_probs
    
    def equ_attention(self, bsz, n, Q, K, V, bias=None):
        Q = Q.view(bsz, n * 3, self.n_heads, self.equ_head_dim).transpose(1, 2)  # (batch, num_heads, n*3, head_dim)
        K = K.view(bsz, n * 3, self.n_heads, self.equ_head_dim).transpose(1, 2)  # (batch, num_heads, n*3, head_dim)
        V = V.view(bsz, n * 3, self.n_heads, self.equ_head_dim).transpose(1, 2)  # (batch, num_heads, n*3, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.equ_head_dim ** 0.5)  # (batch, num_heads, n*3, n*3)
        if bias is not None:
            scores += bias
        
        attn_probs = F.softmax(scores, dim=-1)  # (batch, num_heads, n*3, n*3)        
        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, n*3, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, n * 3, self.emb_e)  # (batch, n*3, emb_dim)

        # (batch, n*3, emb_dim)        
        attn_output = attn_output.view(bsz, n, 3, self.emb_e)
        
        return attn_output, attn_probs
        
    def forward(self, Z_inv, Z_equ, bias=None):
        batch_size, n = Z_inv.shape[:2]
        
        Z_inv_norm = self.inv_layer_norms[0](Z_inv)
        Z_equ_norm = self.equ_layer_norms[0](Z_equ)

        # Compute Q, K, V for the first attention
        Q_I, K_I, V_I = (
            self.inv_self_attn["W_Q_I"](Z_inv_norm),
            self.inv_self_attn["W_K_I"](Z_inv_norm),
            self.inv_self_attn["W_V_I"](Z_inv_norm),
        )
        Q_E, K_E, V_E = (
            self.equ_self_attn["W_Q_E"](Z_equ_norm),
            self.equ_self_attn["W_K_E"](Z_equ_norm),
            self.equ_self_attn["W_V_E"](Z_equ_norm),
        )
        
        # Invariant
        Z_inv_1 = Z_inv + self.out_proj_inv_self_attn(
            self.inv_attention(batch_size, n, Q_I, K_I, V_I, bias)[0]
        )
        Z_inv_1 = self.inv_layer_norms[1](Z_inv_1)
        
        K_IE = self.dot_product(self.inv_cross_attn["W_K_IE1"](Z_equ), self.inv_cross_attn["W_K_IE2"](Z_equ))
        V_IE = self.dot_product(self.inv_cross_attn["W_V_IE1"](Z_equ), self.inv_cross_attn["W_V_IE2"](Z_equ))
        
        Z_inv_2 = Z_inv_1 + self.out_proj_inv_cross_attn(
            self.inv_attention(batch_size, n, Q_I, K_IE, V_IE, bias)[0] 
        )       
        Z_inv_2_norm = self.inv_layer_norms[2](Z_inv_2)
        Z_inv_out = Z_inv_2 + self.inv_ffn(Z_inv_2_norm)

        # Equivariant
        Z_equ_1 = Z_equ + self.out_proj_equ_self_attn(
            self.equ_attention(batch_size, n, Q_E, K_E, V_E)[0]
        )
        Z_equ_1 = self.equ_layer_norms[1](Z_equ_1) 

        K_EI = self.scalar_product(self.equ_cross_attn["W_K_EI1"](Z_equ), self.equ_cross_attn["W_K_EI2"](Z_inv))
        V_EI = self.scalar_product(self.equ_cross_attn["W_V_EI1"](Z_equ), self.equ_cross_attn["W_V_EI2"](Z_inv))
        
        Z_equ_2 = Z_equ_1 + self.out_proj_equ_cross_attn(
            self.equ_attention(batch_size, n, Q_E, K_EI, V_EI)[0]
        )
        Z_equ_2_norm = self.equ_layer_norms[2](Z_equ_2)
        Z_equ_out = Z_equ_2 + self.scalar_product(self.equ_ffn["W_equ"](Z_equ_2_norm), self.equ_ffn["W_inv"](Z_inv_2_norm))

        return Z_inv_out, Z_equ_out


@registry.register_model("geomformer")
class Geomformer(BaseModel):
    def __init__(
        self,
        atom_types=20,
        emb_n=768,
        emb_e=128,
        n_heads=32,
        n_layers=4,
        
        **kwargs,
    ) -> None:
        super(Geomformer, self).__init__(**kwargs)
        
        self.atom_types = atom_types
        self.emb_n = emb_n
        self.emb_e = emb_e
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        self.blocks = nn.ModuleList(
            [GeomformerBlock(emb_n, emb_e, n_heads) for _ in range(n_layers)]
        )
        self.gbf = GaussianSmearing(cutoff_upper=8., num_rbf=128)
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.emb_n, padding_idx=0
        )
        self.energy_ln = nn.Linear(emb_n, 1)
        self.energy_norm = nn.LayerNorm(emb_n)
        self.force_ln = nn.Linear(emb_e, 1)
        self.force_norm = nn.LayerNorm(emb_e)
        
    def forward(self, data: TorchGeoBatch):
        output = {}
        out = self._forward(data)
        output["output"] = out[0]
        output["pos_grad"] = out[1]
                  
        return output 
    
    @property
    def target_attr(self):
        return "y"
    
    @conditional_grad(torch.enable_grad())
    def _forward(self, data: TorchGeoBatch):
        device = data.pos.device
        batch: Batch = Batch.from_batch(data).to(device)
        data = data.to(device)
                
        atoms, pos, real_mask = (
            batch.atoms,
            batch.pos,
            batch.real_mask,
        )
        padding_mask = atoms == 0
        
        n_graph, n_node = atoms.size()
        
        mean_pos = pos.mean(dim=1, keepdim=True)
        r_prime = pos - mean_pos
        # delta_pos = r_prime.unsqueeze(1) - r_prime.unsqueeze(2)
        dist = r_prime.norm(dim=-1)
        r_prime_normalized = r_prime / (dist.unsqueeze(-1) + 1e-5)

        # edge_type = atoms.view(
        #     n_graph, n_node, 1
        # ) * self.atom_types + atoms.view(n_graph, 1, n_node)

        gbf_feature = self.gbf(dist)
        Z_equ = r_prime_normalized.unsqueeze(-1) * gbf_feature.unsqueeze(-2)
        Z_inv = self.atom_encoder(atoms)
        
        for block in self.blocks:
            Z_inv, Z_equ = block(Z_inv, Z_equ)
        
        eng_output = self.energy_ln(self.energy_norm(Z_inv)).flatten(-2)
        output_mask = real_mask  # no need to consider padding, since padding has tag 0, real_mask False

        eng_output = eng_output * output_mask
        energy = eng_output.sum(dim=-1, keepdim=True)
        
        force_output = self.force_ln(self.force_norm(Z_equ)).squeeze(-1)
        node_target_mask = output_mask
        expanded_mask = node_target_mask.unsqueeze(-1).expand_as(force_output)
        force = force_output[expanded_mask.bool()].reshape(-1, 3)
        
        return energy, force