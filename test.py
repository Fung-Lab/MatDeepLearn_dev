from matdeeplearn.process import DataProcessor

root_path = '/Users/shuyijia/Documents/GitHub/Fung-Lab/MatDeepLearn/data/dup_test_data/'
targets_path = '/Users/shuyijia/Documents/GitHub/Fung-Lab/MatDeepLearn/data/dup_test_data/targets.csv'
r = 8.
n_neighbors = 12
edge_steps = 50

processor = DataProcessor(root_path, targets_path, r, n_neighbors, edge_steps)
processor.process()