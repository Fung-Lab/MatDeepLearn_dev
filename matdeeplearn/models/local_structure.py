import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class DistanceNet(MessagePassing):
    def __init__(self):
        super(DistanceNet, self).__init__()

    def forward(self, x, edge_index):
        # compute gassian weight
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        edge_weights = self.message(x_i, x_j)
        
        edge_weights_softmax = softmax(edge_weights, edge_index[0])
        
        return edge_weights_softmax

    def message(self, x_i, x_j):
        diff = torch.abs(x_i - x_j)
        edge_weights = torch.exp((-1.0 / 100) * torch.sum(diff, dim=1))
        return edge_weights