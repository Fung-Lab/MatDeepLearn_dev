"""
    A class implementing the molecular similarity graph (MSG) by
    taking data from the MoleculeProcessor class as well as the model
    embeddings. Again, it should follow the steps in the notebook
    and have methods that perform the following
        - constructs an adjacency matrix from the tanimoto coefficients
        subject to a cutoff,
        - takes model embeddings and adjacency matrix and converts them
        into PyTorch Geometric graph data.
        - (optional) networkx visualization of molecular similarity graph
"""
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import networkx as nx
import matplotlib.pyplot as plt


class MolecularSimilarityGraph:
    def __init__(
            self,
            similarityMap,
            tanimotoThreshold,
            embeddings):
        self.initialAdj = similarityMap
        self.tanimotoThreshold = tanimotoThreshold
        self.filteredInitialAdj = np.dot(
            self.initialAdj, (self.initialAdj >= self.tanimotoThreshold).T)
        self.embeddings = embeddings
        self.graphData = None

    def toGraphData(self):
        # Convert adjacency matrix to edge_index format
        edge_index = torch.transpose(torch.tensor(
            np.argwhere(self.filteredInitialAdj > 0), dtype=torch.long), 0, 1)

        # Extract edge weights from the adjacency matrix
        edge_weights = torch.tensor(
            self.filteredInitialAdj[self.filteredInitialAdj > 0], dtype=torch.float)

        # Convert node embeddings to PyTorch tensor
        x = torch.tensor(self.embeddings, dtype=torch.float)

        # Create a PyTorch Geometric Data object with edge weights
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

        self.graphData = data

    def visualize(self):
        pass
