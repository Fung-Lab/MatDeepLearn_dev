from __future__ import print_function, division

import torch
import numpy as np
import torch.nn as nn
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.common.registry import registry
from torch_scatter import scatter, segment_coo
from torch_geometric.nn import (
    global_mean_pool,
    MessagePassing,
)
import warnings

warnings.filterwarnings("ignore")

#in comparison to my2, here we use softmax rather than sigmoid
#we also have the choice to use max pool rather than mean pool
class ConvLayer(MessagePassing):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, edge_fea_len):
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
        self.edge_fea_len = edge_fea_len
        self.fc_full=nn.Linear(2*self.atom_fea_len+self.edge_fea_len,
                                 2*self.atom_fea_len+self.edge_fea_len)
        self.fc_f = nn.Linear(2*self.atom_fea_len+self.edge_fea_len, self.atom_fea_len)
        self.fc_s = nn.Linear(2*self.atom_fea_len+self.edge_fea_len, self.atom_fea_len)
        #self.fc_e = nn.Linear(2*self.atom_fea_len+self.edge_fea_len, self.edge_fea_len)
        self.softmax= nn.Softmax(dim=1)
        self.softmax2= nn.Softmax(dim=2)
        self.softmax3= nn.Softmax(dim=3)
        self.softplus1 = nn.ReLU()
        self.softplus2 = nn.ReLU()
        self.softplus3 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len+self.edge_fea_len)
        self.bn2 =  nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2s = nn.ReLU()
        self.softplus3s = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, distances):
        self.edge_attrs = distances
        aggregatedMessages = self.propagate(edge_index, x=x, distances=distances)
        aggregatedMessages = self.bn2(aggregatedMessages)
        out = aggregatedMessages + x
        return out, self.edge_attrs


        
    def message(self, x_i, x_j, distances):
        #concatenate atom features, bond features, and bond distances
        z = torch.cat([x_i, x_j, distances], dim=-1)
        #fully connected layer
        total_gated_fea = self.fc_full(z)
        total_gated_fea = self.bn1(total_gated_fea)
        #split into atom features, bond features, and bond distances and apply functions
        nbr_filter, nbr_core, new_edge_attrs = total_gated_fea.split([self.atom_fea_len,self.atom_fea_len,self.edge_fea_len], dim=1)
        #nbr_filter = self.softmax(nbr_filter)
        #nbr_core = self.softplus1(nbr_core)
        #aggregate and return
        #nbr_sumed = nbr_filter * nbr_core
        self.edge_attrs += new_edge_attrs
        #self.edge_attrs += self.fc_e(z)
        #return nbr_sumed
        return self.softmax(self.fc_f(z)) * self.softplus1(self.fc_s(z))
    

            


@registry.register_model("CrystalGraphGeo")
class CrystalGraphConvNet(BaseModel):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, data, dim1=128, n_conv=4, dim2=128, n_h=3,
                 classification=False, **kwargs):
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
        if isinstance(data, torch.utils.data.Subset):
            data = data.dataset
        node_dim = data.num_features
        edge_dim = data.num_edge_features
        output_dim=1
        self.classification = classification
        self.embedding = nn.Linear(node_dim, dim1)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=dim1,
                                    edge_fea_len=edge_dim)
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
    def Fakeforward(self, data):
    
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
    def forward(self, data):
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
        #initialize variables
        atom_fea = data.x
        edge_index = data.edge_index
        distances = data.edge_attr
        #distances = data.distances.unsqueeze(1).expand(len(data.distances), 64) / 8.0
        #embed atom features
        atom_fea = self.embedding(atom_fea)
        #convolutional layers
        for conv_func in self.convs:
            atom_fea, distances = conv_func(atom_fea, edge_index, distances)
        
        #pooling
        crys_fea = global_mean_pool(atom_fea, data.batch)
        #fully connected layers
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        #output
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
        return out
    @property
    def target_attr(self):
        return "y"
