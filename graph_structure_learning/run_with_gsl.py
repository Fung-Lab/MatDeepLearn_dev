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


data = []
processor = None

# testing molecule processing
with open('data/QM9.json') as f:
    data = json.load(f)
    data = data[:1000]
    processor = MoleculeProcessor(molList=data)
    processor.processMolObjects()
    processor.computeMetrics()

    print("PROCESSED MOLS: ", processor.processedMols)
    print(processor.similarityMap)
    print(type(processor.similarityMap))
    f.close()

# testing molecular similarity graph generation
simGraph = None
