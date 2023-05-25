import os

import torch

"""
here resides the transform classes needed for data processing

From PyG:
    Transform: A function/transform that takes in an torch_geometric.data.Data
    object and returns a transformed version.
    The data object will be transformed before every access.
"""


class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]

        # add in noise for DOS
        weight = 0.1
        rand_t = torch.rand(data.scaled.shape)
        rand_t_weighted = rand_t * weight
        data.scaled = torch.mul(data.scaled, rand_t_weighted)

        return data
