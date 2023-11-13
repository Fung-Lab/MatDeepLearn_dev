from __future__ import print_function, division

import torch
import numpy as np
import torch.nn as nn
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.common.registry import registry
from torch_scatter import scatter, segment_coo
from torch_geometric.nn import (
    global_mean_pool,
    MessagePassing,
)
import torch.nn.functional as F
import torch_geometric

import warnings

warnings.filterwarnings("ignore")

#in comparison to my2, here we use softmax rather than sigmoid
#we also have the choice to use max pool rather than mean pool
class ConvLayer(MessagePassing):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len, k=3):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__(aggr="add", node_dim=0)
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.k = k
        self.fc_fulls=nn.ModuleList([nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len+self.nbr_fea_len) for i in range(k)])
        self.fc_fs=nn.ModuleList([nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 self.atom_fea_len) for i in range(k)])
        self.fc_ss=nn.ModuleList([nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 self.atom_fea_len) for i in range(k)])
        self.softmax= nn.Softmax(dim=1)
        self.softmax2= nn.Softmax(dim=2)
        self.softmax3= nn.Softmax(dim=2)
        self.softplus1 = nn.ReLU()
        self.bn1s = nn.ModuleList([nn.BatchNorm1d(2*self.atom_fea_len+self.nbr_fea_len) for i in range(k)])
        self.bn2s =  nn.ModuleList([nn.BatchNorm1d(self.atom_fea_len) for i in range(k)])
        self.atom_fc = nn.Linear(self.k , 2*self.k) 
        self.nbr_fc = nn.Linear(self.k , 2*self.k) 
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, distances):
        self.new_attr = []
        self.outs = []
        self.currInd = 0
        for i in range(self.k):
            self.outs.append(x)
            self.new_attr.append(distances)
        for i in range(self.k):
            aggregatedMessages = self.propagate(edge_index, x=x, distances=distances)
            aggregatedMessages = self.bn2s[i](aggregatedMessages)
            self.outs[i] = aggregatedMessages + x
        out=torch.stack(self.outs, dim=2)
        new_nbr=torch.stack(self.new_attr, dim=2)
        
        out_gated=self.atom_fc(out)
        new_nbr_gated=self.nbr_fc(new_nbr)
       
        out_core, out_filter = out_gated.split([self.k, self.k], dim=2) 
        new_nbr_core, new_nbr_filter = new_nbr_gated.split([self.k, self.k], dim=2)
        out_filter=self.softmax2(out_filter)
        new_nbr_filter=self.softmax3(new_nbr_filter)
        out = torch.sum(out_core * out_filter, dim=2)
        new_nbr = torch.sum(new_nbr_core* new_nbr_filter, dim=2)
        return out, new_nbr


        
    def message(self, x_i, x_j, distances):
        z = torch.cat([x_i, x_j, distances], dim=-1)
        total_gated_fea = self.fc_fulls[self.currInd](z)
        total_gated_fea = self.bn1s[self.currInd](total_gated_fea)
        nbr_filter, nbr_core, new_edge_attrs = total_gated_fea.split([self.atom_fea_len,self.atom_fea_len,self.nbr_fea_len], dim=1)
        self.new_attr[self.currInd] += new_edge_attrs
        self.currInd += 1
        return self.softmax(self.fc_fs[self.currInd-1](z)) * self.softplus1(self.fc_ss[self.currInd-1](z)) 

            


@registry.register_model("CrystalGraphMulti")
class CrystalGraphConvNet(BaseModel):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, node_dim, edge_dim, output_dim,
                 dim1=128, n_conv=4, dim2=128, n_h=1,k=3, pool="global_mean_pool",
                pool_order="early", act="relu", classification=False, **kwargs):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        node_dim: int
          Number of atom features in the input.
        edge_dim: int
          Number of bond features.
        dim1: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        dim2: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(node_dim, dim1)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=dim1,
                                    nbr_fea_len=edge_dim,k=k)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(dim1, dim2)
        self.conv_to_fc_softplus = nn.ReLU()
        self.output_softplus= nn.ReLU()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(dim2, dim2)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.ReLU()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(dim2, 2)
        else:
            self.fc_out = nn.Linear(dim2, output_dim)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        
        self.pool = pool
        self.act = act
        self.pool_order = pool_order

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
    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        """
        Forward pass

        data: graph features

        Parameters
        ----------

        data:
          data.x: node features
            shape = (N, node_dim)
          data.edge_index: list of edges
            shape = (2, E)
          data.edge_attr: edge attributes (distances)
            shape = (E, edge_dim)
          data.batch: crystal id for each node
            shape = (N, )

        Returns
        -------

        prediction: graph predictions
          shape = (batch_size, )
        """
        if self.otf_edge_index == True:
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

        
        atom_fea = data.x
        edge_index = data.edge_index
        distances = data.edge_attr
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea, distances = conv_func(atom_fea, edge_index, distances)
        
        if self.prediction_level == "graph":
            if self.pool_order == "early":
                crys_fea = getattr(torch_geometric.nn, self.pool)(atom_fea, data.batch)
                crys_fea = self.conv_to_fc(getattr(F, self.act)(crys_fea))
                crys_fea = getattr(F, self.act)(crys_fea)
                if hasattr(self, 'fcs'):
                    for fc in self.fcs:
                        crys_fea = getattr(F, self.act)(fc(crys_fea))
                out = self.fc_out(crys_fea)    
            elif self.pool_order == "late":
                crys_fea = self.conv_to_fc(getattr(F, self.act)(crys_fea))
                crys_fea = getattr(F, self.act)(crys_fea)
                if hasattr(self, 'fcs'):
                    for fc in self.fcs:
                        crys_fea = getattr(F, self.act)(fc(crys_fea))
                out = self.fc_out(crys_fea) 
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                    
        elif self.prediction_level == "node":
            crys_fea = getattr(torch_geometric.nn, self.pool)(atom_fea, data.batch)
            crys_fea = self.conv_to_fc(getattr(F, self.act)(crys_fea))
            crys_fea = getattr(F, self.act)(crys_fea)
            if hasattr(self, 'fcs'):
                for fc in self.fcs:
                    crys_fea = getattr(F, self.act)(fc(crys_fea))
            out = self.fc_out(crys_fea)
        return out
        
    @property
    def target_attr(self):
        return "y"

