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
        self.initialAdj = similarityMap.squeeze()
        self.tanimotoThreshold = tanimotoThreshold

        # filter elements under the threshold
        self.initialAdj[self.initialAdj < self.tanimotoThreshold] = 0
        np.fill_diagonal(self.initialAdj, 0)  # remove self-edges

        self.embeddings = embeddings
        self.graphData = None

    def toGraphData(self):
        # Convert adjacency matrix to edge_index format
        edge_index = torch.transpose(torch.tensor(
            np.argwhere(self.initialAdj > 0), dtype=torch.long), 0, 1)

        # Extract edge weights from the adjacency matrix
        edge_weights = torch.tensor(
            self.initialAdj, dtype=torch.float)

        # Convert node embeddings to PyTorch tensor
        x = torch.tensor(self.embeddings, dtype=torch.float)

        # Create a PyTorch Geometric Data object with edge weights
        self.graphData = Data(x=x, edge_index=edge_index,
                              edge_attr=edge_weights)
        # data = Data(x=x, edge_index=edge_index)

    def visualize(self, n):
        g = pyg_utils.to_networkx(self.graphData, to_undirected=True)

        # Draw the graph using NetworkX and matplotlib
        plt.figure(3, figsize=(12, 12))
        nx.draw(g, node_size=100)
        plt.savefig(f'sim_graph_networkx{n}.png')
