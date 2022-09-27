from matdeeplearn.process import *

root_path = './data/dup_test_data/'
target_file_path = './data/dup_test_data/targets.csv'
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

processor.process()
dataset = get_dataset(root_path, 0)

print(dataset[0])