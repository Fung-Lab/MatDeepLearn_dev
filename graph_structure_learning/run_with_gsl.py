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


# testing
with open('data/QM9.json') as f:
    data = json.load(f)
    processor = MoleculeProcessor(molList=data)
    print(data[:100])
    processor.processMolObjects()
    print(processor.rawMols == None)
    print(processor.processedMols)
    processor.computeMetrics()
    print(processor.similarityMap)
