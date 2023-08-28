from __future__ import print_function, division

import torch
import torch.nn as nn

#in comparison to my2, here we use softmax rather than sigmoid
#we also have the choice to use max pool rather than mean pool
class ConvLayer(nn.Module):
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
        super(ConvLayer, self).__init__()
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
    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        #print('N,M', N, M)
        # convolution
        #print('atom_in_fea size', atom_in_fea.size())
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        #print('atom_nbr_fea size', atom_nbr_fea.size())
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        #print('total_nbr_fea size', total_nbr_fea.size())
        total_gated_feas = [fc(total_nbr_fea) for fc in self.fc_fulls]
       
        outs =[]
        new_nbrs=[]
        for total_gated_fea, bn1, bn2, softplus2, softplus3 in zip(total_gated_feas, self.bn1s, self.bn2s, self.softplus2s, self.softplus3s) :
            total_gated_fea = bn1(total_gated_fea.view(-1, self.atom_fea_len*2+self.nbr_fea_len)).view(N, M, self.atom_fea_len*2+self.nbr_fea_len)
            nbr_filter, nbr_core, new_nbr = total_gated_fea.split([self.atom_fea_len,self.atom_fea_len,self.nbr_fea_len], dim=2)
            nbr_filter = self.softmax(nbr_filter)
            nbr_core = self.softplus1(nbr_core) 
            #print('nbr_filter size', nbr_filter.size())
            #print('nbr_core size', nbr_core.size())
            nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
            nbr_sumed = bn2(nbr_sumed)
            out = atom_in_fea + nbr_sumed
            new_nbr = new_nbr + nbr_fea
            outs.append(out)
            new_nbrs.append(new_nbr)
           
        out=torch.stack(outs, dim=2)
        new_nbr=torch.stack(new_nbrs, dim=3)
        
        out_gated=self.atom_fc(out)
        new_nbr_gated=self.nbr_fc(new_nbr)
       
        out_core, out_filter = out_gated.split([self.k, self.k], dim=2) 
        new_nbr_core, new_nbr_filter = new_nbr_gated.split([self.k, self.k], dim=3)
        out_filter=self.softmax2(out_filter)
        new_nbr_filter=self.softmax3(new_nbr_filter)
        out = torch.sum(out_core * out_filter, dim=2)
        new_nbr = torch.sum(new_nbr_core* new_nbr_filter, dim=3)
        return out, new_nbr


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,k=3,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len,k=k)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.ReLU()
        self.output_softplus= nn.ReLU()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.ReLU()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
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
        print("test")
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea, nbr_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            #print('IN FORWARD, atom_fea size', atom_fea.size())
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
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
        return out, []

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
