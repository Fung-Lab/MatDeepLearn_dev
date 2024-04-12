"""
    Run our GNN-GSL model.
        - Much of this code can be taken from main.py, but
        we will have to modify the running process so that it
        takes the 'readout' output from the trainer and feeds
        it into the GSL pipeline.
"""

import json

from xyz_processor import *
from molecular_similarity_graph import *
import numpy as np


data = []
processor = None
n = 700

# testing molecule processing
with open('data/QM9.json') as f:
    data = json.load(f)
    data = data[:n]
    processor = MoleculeProcessor(molList=data)
    processor.processMolObjects()
    processor.computeMetrics()

    # print("PROCESSED MOLS: ", processor.processedMols)
    # print(processor.similarityMap)
    # print(type(processor.similarityMap))
    f.close()

# Testing molecular similarity graph generation

# Generate random embeddings for testing purposes.
# The jth vector in this array corresponds to the
# embedding of molecule j.
embeddings = np.array([np.random.standard_normal(5) for i in range(n)])
threshold = 0.14
simGraph = MolecularSimilarityGraph(
    similarityMap=processor.similarityMap,
    tanimotoThreshold=threshold,
    embeddings=embeddings)

simGraph.toGraphData()
print(simGraph.initialAdj)
# print(len(simGraph.initialAdj[0]))
# print(len(simGraph.initialAdj.T[0]))
simGraph.visualize(n)
