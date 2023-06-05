"""Implementation based on the template of ALIGNN."""

from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from matdeeplearn.models.matformer.utils import RBFExpansion
from matdeeplearn.models.matformer.utils import angle_emb_mp
from torch_scatter import scatter
from matdeeplearn.models.matformer.transformer import MatformerConv
from matdeeplearn.models.base_model import BaseModel, conditional_grad

from matdeeplearn.common.registry import registry

@registry.register_model("Matformer")
class Matformer(BaseModel):
    """att pyg implementation."""

    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        conv_layers: int = 5,
        edge_layers: int = 0,
        atom_input_features: int = 92,
        edge_features: int = 128,
        triplet_input_features: int = 40,
        node_features: int = 128,
        fc_layers: int = 1,
        fc_features: int = 128,
        output_features: int = 1,
        node_layer_head: int = 4,
        edge_layer_head: int = 4,
        nn_based: bool = False,
        link: Literal["identity", "log", "logit"] = "identity",
        zero_inflated: bool = False,
        use_angle: bool = False,
        angle_lattice: bool = False,
        classification: bool = False,
        **kwargs
    ):
        """Set up att modules."""
        super().__init__()
        self.classification = classification
        self.use_angle = use_angle
        self.atom_embedding = nn.Linear(
            node_dim, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_dim,
            ),
            nn.Linear((edge_dim), node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )
        self.angle_lattice = angle_lattice
        if self.angle_lattice: ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=edge_dim,
                ),
                nn.Linear((edge_dim), node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=triplet_input_features,
                ),
                nn.Linear(triplet_input_features, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(node_features * 6, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(node_features * 2, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )


        self.edge_init = nn.Sequential( ## module not used
            nn.Linear(3 * node_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features)
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0) ## module not used

        self.angle_init_layers = nn.Sequential( ## module not used
            nn.Linear(120, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features)
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=node_features, out_channels=node_features, heads=node_layer_head, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )
        
        self.edge_update_layers = nn.ModuleList( ## module not used
            [
                MatformerConv(in_channels=node_features, out_channels=node_features, heads=edge_layer_head, edge_dim=node_features)
                for _ in range(edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(node_features, fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                fc_features, output_dim
            )

        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif link == "logit":
            self.link = torch.sigmoid

    @conditional_grad(torch.enable_grad())
    def _forward(self, data) -> torch.Tensor:
        if self.otf_edge == True:
            data.edge_index, data.edge_weight, data.edge_vec, _, _, _ = self.generate_graph(data, self.cutoff_radius, self.n_neighbors)
        #data, ldata, lattice = data
        data.x = data.x.to(dtype=torch.float)
        # initial node features: atom feature network...
        lattice = None
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
        if self.angle_lattice: ## module not used
            lattice_len = torch.norm(lattice, dim=-1) # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128) # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,1,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,1,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,1,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,1,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))
        
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)


        # crystal-level readout
        if self.prediction_level == "graph":
            features = scatter(node_features, data.batch, dim=0, reduce="add")
        features = node_features

        if self.angle_lattice:
            # features *= F.sigmoid(lattice_emb)
            features += lattice_emb
        
        # features = F.softplus(features)
        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)
        if self.classification:
            out = self.softmax(out)

        return torch.squeeze(out)
    @property
    def target_attr(self):
        return "y"
    
    def forward(self, data):
    
        output = {}
        out = self._forward(data)
        output["output"] =  out

        if self.gradient == True and out.requires_grad == True:         
            if self.gradient_method == "conventional":
                volume = torch.einsum("zi,zi->z", data.cell[:, 0, :], torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1)).unsqueeze(-1)                        
                grad = torch.autograd.grad(
                        out,
                        [data.pos, data.cell],
                        grad_outputs=torch.ones_like(out),
                        create_graph=self.training) 
                forces = -1 * grad[0]
                stress = grad[1] 
                stress = stress / volume.view(-1, 1, 1)
            #For calculation of stress, see https://github.com/mir-group/nequip/blob/main/nequip/nn/_grad_output.py
            #Originally from: https://github.com/atomistic-machine-learning/schnetpack/issues/165                              
            elif self.gradient_method == "nequip":
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

