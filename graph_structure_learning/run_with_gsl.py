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
from graph_structure_learner import GraphLearner, Model, ModelHandler


data = []
processor = None
n = 700

"""
    Testing molecule data processing
"""

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

"""
    Testing molecular similarity graph generation 
"""

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

"""
    Testing Graph Structure Learning. 
    
    - Current issues:
        - There are a bunch of util functions being used that still need to be grabbed from the IDGL codebase.

        - Some of the code currently does not fit our purposes (i.e. some hyperparameters must be
        changed. 
            - The YAML file needs to be changed.
            - One specific issue: they perform the adjacency matrix epsilon-sparcification in 'GraphLearner.forward()'
              using PyTorch functions, but we already have that implemented in 'MolecularSimilarityGraph.__init__()' 
              using just Numpy, then the adjacency matrix is converted into a Torch tensor.
                - So, maybe it is better to keep it the way we did it, which would likely allows us to simplify some of
                the GSL code, as we'd just pass it a ready initial adjacency matrix.
                - OR, maybe it's better to do it their way. We need to determine this.
        
        - Also, remember that we have a bit of difference with the IDGL code: the node features are the embeddings of a
          previously-run GNN. This may also require we change some things.
            - In particular, we will have to run the code differently than the way the IDGL people run their code from 
              main. For example, where they call 'ModelHandler(config)', we will probably do something to the effect of
              'ModelHandler(config, embeddings)', where we modify ModelHandler to also call the code in
              'molecular_similarity_graph.py'.
              
            - I.e., running the code will look something like:
                - TorchMD is run pretty much identically to scripts/main.py,
                - the embeddings output by TorchMD are taken and fed into a
                  ModelHandler() object along with the config containing all
                  of the hyperparameter settings,
                - ModelHandler() computes the similarity metrics (by calling 
                  the code in 'xyz_processor.py'), constructs the MSG, and
                  invokes the graph structure learner on it (as done in IDGL).
"""
