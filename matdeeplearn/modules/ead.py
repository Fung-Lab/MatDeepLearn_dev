import ase
import torch
import math
import itertools
import numpy as np
from itertools import combinations, combinations_with_replacement

from torch_scatter import scatter

from matdeeplearn.modules.descriptor_utils import *

class EmbeddedAtom:
    '''
    See
        1. Zhang, Y., et al. (2019). The Journal of Physical Chemistry Letters, 10(17), 4962-4967.

    Implementation closely follows pyXtal_ff (https://github.com/MaterSim/PyXtal_FF)
    '''
    def __init__(
        self,
        cutoff=20.,
        all_neighbors=True,
        offset_count=3,
        L=0,
        eta=[1,20,90],
        Rs=None,
        derivative=False,
        stress=False,
    ) -> None:
        self.cutoff = cutoff
        self.all_neighbors = all_neighbors
        self.offset_count = offset_count
        self.L = L
        self.eta = eta
        self.derivative = derivative
        self.stress = stress

        if Rs is None:
            self.Rs = np.arange(0, 20, step=0.1)

        # check GPU availability & set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dsize = (self.L + 1) * len(self.eta) * len(self.Rs)

    def __str__(self):
        return 'ead'

    def get_features(self, positions, cell, atomic_numbers, offset_count=None, device=None):
        if offset_count is None:
            offset_count = self.offset_count
        if device is None:
            device = self.device
        if isinstance(cell, ase.cell.Cell):
            cell = np.array(cell)
        
        offsets = get_pbc_offsets(cell, offset_count, device)
        distances, rij = get_all_distances(positions, offsets, device)
        
        return self.get_ead_features(distances, rij, cell, atomic_numbers, positions)

    def get_batch_features(self, data_batch, offset_count=None, device=None):
        num_samples = data_batch.ptr.size(0) - 1

        features_list = []

        # Split the DataBatch into individual samples
        for i in range(num_samples):
            sample = data_batch[i]
            positions = sample.pos
            cell = torch.squeeze(sample.cell)
            atomic_numbers = sample.z

            features = self.get_features(positions, cell, atomic_numbers)
            features = features.view(1, -1)
            features_list.append(features)
        
        # concat features
        batch_features = torch.cat(features_list, dim=0)
        return batch_features

    def get_volume(self, cell):
        cell_vec1, cell_vec2, cell_vec3 = cell
        if not isinstance(cell, torch.Tensor):
            cell_vec1 = torch.tensor(cell_vec1, device=self.device, dtype=torch.float)
            cell_vec2 = torch.tensor(cell_vec2, device=self.device, dtype=torch.float)
            cell_vec3 = torch.tensor(cell_vec3, device=self.device, dtype=torch.float)

        vol = torch.dot(torch.cross(cell_vec1, cell_vec2), cell_vec3)

        return vol

    def pooling(self, x, topk=None):
        m, n = x.shape

        if topk is None:
            out = torch.zeros([2*n], dtype=torch.float, device=self.device)
            minn, _ = torch.min(x, dim=0)
            maxx, _ = torch.max(x, dim=0)
            # mean = torch.mean(x, dim=0)

            out[:n] = minn
            out[n:] = maxx
            # out[2*n:] = mean
        else:
            out = torch.zeros([2*topk*n], dtype=torch.float, device=self.device)
            minn, _ = torch.topk(x, topk, largest=False, dim=0)
            maxx, _ = torch.topk(x, topk, dim=0)
            out[:n*topk] = minn.flatten()
            out[n*topk:] = maxx.flatten()

        return out

    def get_ead_features(self, distances, rij, cell, atomic_numbers, positions):
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float, device=self.device)

        n_atoms = len(atomic_numbers)
        vol = self.get_volume(cell).item()

        # temp save of original atomic number
        # an = torch.tensor(atomic_numbers, dtype=int, device=self.device)
        # atomic_numbers = torch.tensor(atomic_numbers, dtype=int)
        an = atomic_numbers.clone().detach()
        atomic_numbers = atomic_numbers.view(1, -1).expand(n_atoms, -1)
        atomic_numbers = atomic_numbers.cpu().reshape(n_atoms, -1).numpy()

        unique_N = 50

        x = torch.zeros((n_atoms, self.dsize), dtype=torch.float, device=self.device)
        if self.derivative:
            dxdr = torch.zeros((unique_N, self.dsize, 3), dtype=torch.float, device=self.device)
            seq = torch.zeros((unique_N, 2), dtype=torch.int, device=self.device)
        if self.stress:
            rdxdr = torch.zeros((unique_N, self.dsize, 3, 3), dtype=torch.float, device=self.device)

        seq_count = 0

        for i in range(n_atoms):
            Rc = self.cutoff
            dist = distances[i]
            mask = (dist <= Rc) & (dist != 0.0)
            
            Dij = distances[i, mask]
            Rij = rij[i, mask]

            atomic_numbers_mat = an.view(1,-1,1).expand(len(an), len(an), distances.shape[-1])
            # IDs = atomic_numbers[i, mask.cpu().numpy()]
            IDs = torch.argwhere(mask.squeeze())
            IDs = IDs[:,0]
            Z = atomic_numbers_mat[i, mask]
            # Z = Z.cpu().detach().numpy()

            _x, _dxdr, _rdxdr, _seq = self.calculate_eamd(i, n_atoms, positions[i], Rij, Dij, Z, IDs)

            x[i] = _x
            if self.derivative:
                n_seq = len(_seq)
                dxdr[seq_count:seq_count+n_seq] = _dxdr
                seq[seq_count:seq_count+n_seq] = _seq
            
            if self.stress:
                rdxdr[seq_count:seq_count+n_seq] = _rdxdr/vol
            
            if self.derivative:
                seq_count += n_seq
        
        # out = scatter(x, an.to(x.device), dim=0, dim_size=100, reduce="mean").flatten()

        out = self.pooling(x)
        return out

    def calculate_eamd(self, i, n_atoms, ri, rij, dij, Z, IDs):
        l_index = [1, 4, 10, 20]
        Rc = self.cutoff
        Rs = torch.tensor(self.Rs, dtype=torch.float, device=self.device)
        eta = torch.tensor(self.eta, dtype=torch.float, device=self.device)
        L = self.L
        d1, d2, d3, j = len(Rs), len(eta), L+1, len(dij)

        ij_list = i * torch.ones((len(Z), 2), dtype=torch.int, device=self.device)
        ij_list[:, 1] = IDs

        unique_js = torch.unique(IDs)

        if i not in unique_js:
            i_tensor = torch.tensor([i], dtype=torch.int, device=self.device)
            unique_js = torch.cat((unique_js, i_tensor))
        unique_js, _ = torch.sort(unique_js)

        # seq
        seq = i * torch.ones((len(unique_js), 2), dtype=torch.int, device=self.device)
        seq[:, 1] = unique_js
        uN = len(unique_js)
        _i = torch.where(unique_js==i)[0].item()

        term1, d_term1, i_d_term1, j_d_term1 = self.get_xyz(unique_js, rij, ij_list, L)

        d0 = dij - Rs.view(-1, 1)
        d02 = d0**2

        fc = self.cosine_cutoff(dij, Rc)
        cj_cutoff = Z * fc

        term2_1 = torch.exp(torch.einsum('i,jk->ijk', -eta, d02)) # [d2, d1, j]
        term2 = torch.einsum('ijk,k->ijk', term2_1, cj_cutoff) # [d2, d1, j]

        term = torch.einsum('ij,kli->jkl', term1, term2) # [D3, d2, d1]

        if self.derivative:
            dterm0 = torch.einsum('k, ijk->ijk', Z, term2_1).view([d1*d2, j]) # [d2*d1, j]
            dterm11 = torch.einsum('ij, j->ij', dterm0, fc).view([d1*d2, j]) # [d2*d1, j]
            dterm1 = torch.einsum('ij, jklm->jmilk', dterm11, d_term1) # [j, D3, d2*d1, 3, uN]
            i_dterm1 = torch.einsum('ij, jklm->jmilk', dterm11, i_d_term1) 
            j_dterm1 = torch.einsum('ij, jklm->jmilk', dterm11, j_d_term1)

            dterm20 = torch.einsum('ij, ki->jki', term1, dterm0) # [D3, d2*d1, j]
            dterm21 = self.cosine_cutoff_derivative(dij, Rc) # [j]
            _dterm22 = torch.einsum('ij,j->ij', d0, fc) # [d1, j]
            dterm22 = 2 * torch.einsum('i,jk->ijk', eta, _dterm22) # [d2, d1, j]
            dterm23 = (dterm21 - dterm22).reshape([d2*d1, j]) # [d2*d1, j]
            dterm24 = torch.einsum('ijk, jk->ijk', dterm20, dterm23) # [D3, d2*d1, j]

            dRij_dRm = torch.zeros([j, 3, uN], dtype=torch.float, device=self.device)
            i_dRij_dRm = torch.zeros([j, 3, uN], dtype=torch.float, device=self.device)
            j_dRij_dRm = torch.zeros([j, 3, uN], dtype=torch.float, device=self.device)

            for mm, _m in enumerate(unique_js):
                mm_list = _m * torch.ones([j, 1], dtype=int, device=self.device)
                dRij_dRm[:,:,mm], i_dRij_dRm[:,:,mm], j_dRij_dRm[:,:,mm] = \
                        self.dRij_dRm_norm(rij, torch.cat((ij_list, mm_list), dim=1)) # [j, 3, uN]

            dterm2 = torch.einsum('ijk, klm->kijlm', dterm24, dRij_dRm) # [j, D3, d2*d1, 3, uN]
            i_dterm2 = torch.einsum('ijk, klm->kijlm', dterm24, i_dRij_dRm) # [j, D3, d2*d1, 3, uN]
            j_dterm2 = torch.einsum('ijk, klm->kijlm', dterm24, j_dRij_dRm) # [j, D3, d2*d1, 3, uN]
            
            dphi_dRm = dterm1 + dterm2 # [j, D3, d2*d1, 3, uN]
            i_dphi_dRm = i_dterm1 + i_dterm2 
            j_dphi_dRm = j_dterm1 + j_dterm2 
            
            dterm = torch.einsum('ij, hijkl->ijkl', term.view([term.shape[0], d2*d1]), dphi_dRm) # [D3, d2*d1, 3, uN]

            if self.stress:
                _RDXDR = torch.zeros([term.shape[0], d2*d1, 3, uN, 3], dtype=torch.float, device=self.device)
                for count, ij in enumerate(ij_list):
                    _j = torch.where(unique_js==ij[1])[0][0]
                    i_tmp = i_dphi_dRm[count, :, :, :]
                    j_tmp = j_dphi_dRm[count, :, :, :]
                    _RDXDR[:, :, :, _i, :] += torch.einsum('ijk,l->ijkl', i_tmp[:,:,:,_i], ri)
                    _RDXDR[:, :, :, _j, :] += torch.einsum('ijk,l->ijkl', j_tmp[:,:,:,_j], rij[count]+ri)
                sterm = torch.einsum('ij, ijklm->ijklm', term.view([term.shape[0], d2*d1]), _RDXDR)

        count = 0
        x = torch.zeros([d3*d2*d1], dtype=torch.float, device=self.device)
        dxdr, rdxdr = None, None
        if self.derivative:
            dxdr = torch.zeros([uN, d1*d2*d3, 3], dtype=torch.float, device=self.device) # [uN, d3*d2*d1, 3]
        if self.stress:
            rdxdr = torch.zeros([uN, d1*d2*d3, 3, 3], dtype=torch.float, device=self.device)
        
        for l in range(L+1):
            Rc2l = Rc**(2*l)
            L_fac = math.factorial(l)
            x[count:count+d1*d2] = L_fac * torch.einsum('ijk->jk', term[:l_index[l]] ** 2).ravel() / Rc2l
            if self.derivative:
                dxdr[:, count:count+d1*d2, :] = 2 * L_fac * torch.einsum('ijkl->ljk', dterm[:l_index[l]]) / Rc2l
            if self.stress:
                rdxdr[:, count:count+d1*d2, :, :] = 2 * L_fac * torch.einsum('ijklm->ljkm', sterm[:l_index[l]]) / Rc2l

            count += d1*d2

        return x, dxdr, rdxdr, seq

    def get_xyz(self, unique_js, rij, ij_list, L):
        normalize = 1 / np.sqrt(np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2,     # 1 / sqrt(lx! ly! lz!)
                                      1, 2, 2, 2, 2, 2, 2, 6, 6, 6]))

        L_list = [[[0], [0]],                 # lx = 1, ly = 0, lz = 0; L = 1
                [[1], [1]],                   # lx = 0, ly = 1, lz = 0; L = 1
                [[2], [2]],                   # lx = 0, ly = 0, lz = 1; L = 1
                [[0,1], [0,1]],               # lx = 1, ly = 1, lz = 0; L = 2
                [[0,2], [0,2]],               # lx = 1, ly = 0, lz = 1; L = 2
                [[1,2], [1,2]],               # lx = 0, ly = 1, lz = 1; L = 2
                [[0], [3]],                   # lx = 2, ly = 0, lz = 0; L = 2
                [[1], [4]],                   # lx = 0, ly = 2, lz = 0; L = 2
                [[2], [5]],                   # lx = 0, ly = 0, lz = 2; L = 2
                [[0,1,2], [0,1,2]],           # lx = 1, ly = 1, lz = 1; L = 3
                [[1,2], [1,5]],               # lx = 0, ly = 1, lz = 2; L = 3
                [[1,2], [4,2]],               # lx = 0, ly = 2, lz = 1; L = 3
                [[0,2], [0,5]],               # lx = 1, ly = 0, lz = 2; L = 3
                [[0,1], [0,4]],               # lx = 1, ly = 2, lz = 0; L = 3
                [[0,1], [3,1]],               # lx = 2, ly = 1, lz = 0; L = 3
                [[0,2], [3,2]],               # lx = 2, ly = 0, lz = 1; L = 3
                [[0], [6]],                   # lx = 3, ly = 0, lz = 0; L = 3
                [[1], [7]],                   # lx = 0, ly = 3, lz = 0; L = 3
                [[2], [8]]                    # lx = 0, ly = 0, lz = 3; L = 3
                ]
        
        normalize = torch.tensor(normalize, dtype=torch.float, device=self.device)
        uN = len(unique_js)

        l = 1
        RIJ = torch.zeros((len(rij), 9), dtype=torch.float, device=self.device)
        dRIJ = torch.zeros((len(rij), 9), dtype=torch.float, device=self.device)

        if L == 1:
            l = 4
            RIJ[:, :3] = rij
            if self.derivative:
                dRIJ[:, :3] += 1
        elif L == 2:
            l = 10
            RIJ[:, :3] = rij
            RIJ[:, 3:6] = rij*rij
            if self.derivative:
                dRIJ[:, :3] += 1
                dRIJ[:, 3:6] = 2*rij
        elif L == 3:
            l = 20
            RIJ[:, :3] = rij
            RIJ[:, 3:6] = rij*rij
            RIJ[:, 6:9] = (rij*rij)*rij

            if self.derivative:
                dRIJ[:, :3] += 1
                dRIJ[:, 3:6] = 2*rij
                dRIJ[:, 6:9] = 3*RIJ[:, 3:6] 
        
        xyz = torch.ones((len(rij), 3, l), dtype=torch.float, device=self.device)
        if self.derivative:
            dxyz = torch.zeros([len(rij), uN, 3, l], dtype=torch.float, device=self.device)
            i_dxyz = torch.zeros([len(rij), uN, 3, l], dtype=torch.float, device=self.device)
            j_dxyz = torch.zeros([len(rij), uN, 3, l], dtype=torch.float, device=self.device)

        dij_dmlist, i_dij_dmlist, j_dij_dmlist = self.dij_dm_list(unique_js, ij_list)

        for i in range(1, l):
            idx_t1 = torch.tensor(L_list[i-1][0], dtype=torch.long, device=self.device)
            idx_t2 = torch.tensor(L_list[i-1][1], dtype=torch.long, device=self.device)
            xyz[:, idx_t1, i] = RIJ[:, idx_t2]
            if self.derivative:
                dxyz[:, :, idx_t1, i] = torch.einsum('ij,ik->ijk', dij_dmlist, dRIJ[:, idx_t2]) 
                i_dxyz[:, :, idx_t1, i] = torch.einsum('ij,ik->ijk', i_dij_dmlist, dRIJ[:, idx_t2])
                j_dxyz[:, :, idx_t1, i] = torch.einsum('ij,ik->ijk', j_dij_dmlist, dRIJ[:, idx_t2])

        result = xyz[:, 0, :] * xyz[:, 1, :] * xyz[:, 2, :] * normalize[:l]

        if self.derivative:
            d_result = torch.zeros_like(dxyz) # [j, uN, 3, l]
            d_result[:, :, 0, :] = torch.einsum('ijk,ik->ijk', dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
            d_result[:, :, 1, :] = torch.einsum('ijk,ik->ijk', dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
            d_result[:, :, 2, :] = torch.einsum('ijk,ik->ijk', dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
            d_result = torch.einsum('ijkl,l->ijkl', d_result, normalize[:l])

            i_d_result = torch.zeros_like(dxyz) # [j, uN, 3, l]
            i_d_result[:, :, 0, :] = torch.einsum('ijk,ik->ijk', i_dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
            i_d_result[:, :, 1, :] = torch.einsum('ijk,ik->ijk', i_dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
            i_d_result[:, :, 2, :] = torch.einsum('ijk,ik->ijk', i_dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
            i_d_result = torch.einsum('ijkl,l->ijkl', i_d_result, normalize[:l])

            j_d_result = torch.zeros_like(dxyz) # [j, uN, 3, l]
            j_d_result[:, :, 0, :] = torch.einsum('ijk,ik->ijk', j_dxyz[:, :, 0, :], xyz[:, 1, :]*xyz[:, 2, :])
            j_d_result[:, :, 1, :] = torch.einsum('ijk,ik->ijk', j_dxyz[:, :, 1, :], xyz[:, 0, :]*xyz[:, 2, :])
            j_d_result[:, :, 2, :] = torch.einsum('ijk,ik->ijk', j_dxyz[:, :, 2, :], xyz[:, 0, :]*xyz[:, 1, :])
            j_d_result = torch.einsum('ijkl,l->ijkl', j_d_result, normalize[:l])

            return result, d_result, i_d_result, j_d_result
        else:
            return result, None, None, None

    def dij_dm_list(self, unique_js, ij_list):
        uN = len(unique_js)
        result = torch.zeros([len(ij_list), uN], dtype=torch.float, device=self.device)
        i_result = torch.zeros([len(ij_list), uN], dtype=torch.float, device=self.device)
        j_result = torch.zeros([len(ij_list), uN], dtype=torch.float, device=self.device)

        ijm_list = torch.zeros([len(ij_list), 3, uN], dtype=torch.int, device=self.device)
        ijm_list[:, -1, :] = unique_js
        ijm_list[:, :2, :] = torch.broadcast_to(ij_list[...,None], ij_list.shape+(uN,))
        
        arr = (ijm_list[:, 2, :] == ijm_list[:, 0, :])
        result[arr] = -1
        i_result[arr] = -1

        arr = (ijm_list[:, 2, :] == ijm_list[:, 1, :])
        result[arr] = 1
        j_result[arr] = 1

        arr = (ijm_list[:, 0, :] == ijm_list[:, 1, :])  
        result[arr] = 0

        return result, i_result, j_result

    def dRij_dRm_norm(self, Rij, ijm_list):
        dRij_m = torch.zeros([len(Rij), 3], dtype=torch.float, device=self.device)
        i_dRij_m = torch.zeros([len(Rij), 3], dtype=torch.float, device=self.device)
        j_dRij_m = torch.zeros([len(Rij), 3], dtype=torch.float, device=self.device)
        R1ij = torch.linalg.norm(Rij, axis=1).view([len(Rij),1])
        
        l1 = (ijm_list[:,2]==ijm_list[:,0])
        dRij_m[l1, :] = -Rij[l1]/R1ij[l1]
        i_dRij_m[l1, :] = -Rij[l1]/R1ij[l1]

        l2 = (ijm_list[:,2]==ijm_list[:,1])
        dRij_m[l2, :] = Rij[l2]/R1ij[l2]
        j_dRij_m[l2, :] = Rij[l2]/R1ij[l2]

        l3 = (ijm_list[:,0]==ijm_list[:,1])
        dRij_m[l3, :] = 0

        return dRij_m, i_dRij_m, j_dRij_m

    def cosine_cutoff(self, Rij, Rc):
        '''
        Cosine cutoff function
        See Behler, J. Chem. Phys. (2011) Eqn (4)

            Parameters:
                Rij (torch.Tensor): distance between atom i and j
                Rc (float): cutoff radius
            
            Returns:
                out (torch.Tensor): cosine cutoff
        '''
        
        out = 0.5 * (torch.cos(np.pi * Rij / Rc) + 1.)
        out[out > Rc] = 0.
        return out

    def cosine_cutoff_derivative(self, Rij, Rc):
        # Rij is the norm
        ids = (Rij > Rc)
        result = -0.5 * torch.pi / Rc * torch.sin(torch.pi * Rij / Rc)
        result[ids] = 0
        return result