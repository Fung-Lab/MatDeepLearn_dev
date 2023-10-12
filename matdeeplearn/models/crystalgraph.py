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
        self.softmax= nn.Softmax(dim=1)
        self.softmax2= nn.Softmax(dim=2)
        self.softmax3= nn.Softmax(dim=3)
        self.softplus1 = nn.ReLU()
        self.softplus2 = nn.ReLU()
        self.softplus3 = nn.ReLU()
        self.bn1s = nn.ModuleList([nn.BatchNorm1d(2*self.atom_fea_len+self.nbr_fea_len) for i in range(k)])
        self.bn2s =  nn.ModuleList([nn.BatchNorm1d(self.atom_fea_len) for i in range(k)])
        self.softplus2s = nn.ModuleList([nn.ReLU() for i in range(k)])
        self.softplus3s = nn.ModuleList([nn.ReLU() for i in range(k)])
        self.atom_fc = nn.Linear(self.k , 2*self.k) 
        self.nbr_fc = nn.Linear(self.k , 2*self.k) 
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, distances):
        self.new_nbrs = []
        #self.outs = []
        for i in range(self.k):
            #self.outs.append(torch.zeros(x.size()).to(x.get_device()))
            self.new_nbrs.append(distances)
        self.nbr_fea = distances
        out = self.propagate(edge_index, x=x, distances=distances)
        #out=torch.stack(self.outs, dim=2)
        new_nbr=torch.stack(self.new_nbrs, dim=3)
        
        #out_gated=self.atom_fc(out)
        new_nbr_gated=self.nbr_fc(new_nbr)
       
        #out_core, out_filter = out_gated.split([self.k, self.k], dim=2) 
        new_nbr_core, new_nbr_filter = new_nbr_gated.split([self.k, self.k], dim=2)
        #out_filter=self.softmax2(out_filter)
        new_nbr_filter=self.softmax3(new_nbr_filter)
        #out = torch.sum(out_core * out_filter, dim=2)
        new_nbr = torch.sum(new_nbr_core* new_nbr_filter, dim=3)
        return out, new_nbr


        
    def message(self, x_i, x_j, distances):
        z = torch.cat([x_i, x_j, distances], dim=-1)
        #temp = torch.zeros(distances.size()[0], self.atom_fea_len).to(x_i.get_device())
        for fc, bn1, bn2, softplus2, softplus3, i in zip(self.fc_fulls, self.bn1s, self.bn2s, self.softplus2s, self.softplus3s, self.k) :
            total_gated_fea = fc(z)
            total_gated_fea = bn1(total_gated_fea)#.view(-1, self.atom_fea_len*2+self.nbr_fea_len)).view(self.N, self.M, self.atom_fea_len*2+self.nbr_fea_len)
            nbr_filter, nbr_core, new_nbr = total_gated_fea.split([self.atom_fea_len,self.atom_fea_len,self.nbr_fea_len], dim=1)
            nbr_filter = self.softmax(nbr_filter)
            nbr_core = self.softplus1(nbr_core) 
            #nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
            nbr_sumed = nbr_filter * nbr_core
            #temp = temp + nbr_sumed
            self.outs[i] += nbr_sumed
            self.new_nbrs[i] += new_nbr
            #self.nbr_fea += new_nbr
        return x_i
    

            


@registry.register_model("CrystalGraphGeo")
class CrystalGraphConvNet(BaseModel):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, node_dim, edge_dim, output_dim,
                 dim1=128, n_conv=9, dim2=128, n_h=1,k=3,
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
    def _forward(self, data):#atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = data.x
        edge_index = data.edge_index
        distances = data.edge_attr
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea, distances = conv_func(atom_fea, edge_index, distances)
            #print('IN FORWARD, atom_fea size', atom_fea.size())
        #crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = global_mean_pool(atom_fea, data.batch)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out
    @property
    def target_attr(self):
        return "y"

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        #print('In POOLING', atom_fea.data.shape[0])
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        #summed_fea = [torch.max(atom_fea[idx_map],dim=0, keepdim=True)[0] for idx_map in crystal_atom_idx]
        #print('In POOLING, summed_fea ', len(summed_fea))
        #print('In POOLING, crystal_atom_idx', len(crystal_atom_idx))
        #a = torch.Tensor(summed_fea)
        a = torch.cat(summed_fea, dim=0)
        #print('return tensor size', a.size())
        return a
