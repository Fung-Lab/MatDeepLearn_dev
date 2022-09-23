import numpy as np
import torch
from matdeeplearn.process import *

root_path = './data/test_data/'
target_file_path = './data/test_data/targets.csv'
r = 8.0
n_neighbors = 12
edge_steps = 50

processor = DataProcessor(
    root_path=root_path, 
    target_file_path=target_file_path, 
    r=r, 
    n_neighbors=n_neighbors, 
    edge_steps=edge_steps
)

# generate grpah structures
data = processor.process()

print(data[0])
