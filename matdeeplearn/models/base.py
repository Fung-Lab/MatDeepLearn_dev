import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def generate_graph(self):
        pass