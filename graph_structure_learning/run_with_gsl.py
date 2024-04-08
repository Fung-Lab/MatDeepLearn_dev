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


processor = MoleculeProcessor()
# testing
with open('../data/QM9.json') as f:
    data = json.load(f)
    processor.rawMols = data[:1000]
    processor.processMolObjects()
    print(processor.rawMols == None)
    print("PROCESSED MOLS: ", processor.processedMols)
    processor.computeMetrics()
    print(processor.similarityMap)
