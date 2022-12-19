import os
import torch
import numpy as np
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from matdeeplearn.preprocessor.helpers import compute_bond_angles, triplets
from scipy.spatial.distance import cdist

'''
here resides the transform classes needed for data processing

From PyG:
    Transform: A function/transform that takes in an torch_geometric.data.Data
    object and returns a transformed version. 
    The data object will be transformed before every access.
'''


class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data


class NumNodeTransform(object):
    '''
    Adds the number of nodes to the data object
    '''

    def __call__(self, data):
        data.num_nodes = data.x.shape[0]
        return data


class LineGraphMod(object):
    '''
    Adds line graph attributes to the data object
    '''

    def __call__(self, data):
        # CODE FROM PYG LINEGRAPH TRANSFORM (DIRECTED)
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes)
        cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)

        cols = [
            i[cumsum[col[j]]:cumsum[col[j] + 1]]
            for j in range(col.size(0))
        ]
        rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]

        row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

        data.edge_index_lg = torch.stack([row, col], dim=0)
        data.x_lg = data.edge_attr
        data.num_nodes_lg = edge_index.size(1)

        # CUSTOM CODE FOR CALCULATING EDGE ATTRIBUTES
        edge_attr_lg = torch.zeros(
            (data.edge_index_lg.shape[1], 1), device='cuda')

        # compute bond angles
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes)
        triplet_pairs = torch.stack([idx_kj, idx_ji], dim=0)

        # move triplets and edges to CPU for sklearn based calculation
        match_indices = torch.Tensor(
            np.where(cdist(data.edge_index_lg.T.cpu(), triplet_pairs.T.cpu()) == 0)[
                0].reshape(-1, 1)
        ).type(torch.long)

        # assign bond angles to edge attributes
        edge_attr_lg[match_indices.squeeze(-1)] = angles.reshape(-1, 1)

        data.edge_attr_lg = edge_attr_lg
        
        return data

class ToFloat(object):
    '''
    Convert non-int attributes to float
    '''
    def __call__(self, data):
        data.x = data.x.float()
        data.x_lg = data.x_lg.float()
        
        data.distances = data.distances.float()
        data.pos = data.pos.float()

        data.edge_attr = data.edge_attr.float()
        data.edge_attr_lg = data.edge_attr_lg.float()

        return data