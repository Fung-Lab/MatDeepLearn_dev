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

class MolecularSimilarityGraph:
    def __init__(self, moleculeData):
        pass
    
    def constructAdjacencyMatrix(self):
        pass
    
    def toGraphData(self):
        pass
    
    def visualize(self):
        pass