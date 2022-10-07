from matdeeplearn.utils.data import *
from matdeeplearn.preprocessor import *

# root_path = '/Users/shuyijia/Github/Fung-Lab/MatDeepLearn/data/dup_test_data/'
# targets_path = '/Users/shuyijia/Github/Fung-Lab/MatDeepLearn/data/dup_test_data/targets.csv'
root_path = '/Users/shuyijia/Github/Fung-Lab/MatDeepLearn/data/bulk_data_500/'
targets_path = '/Users/shuyijia/Github/Fung-Lab/MatDeepLearn/data/bulk_data_500/targets.csv'
r = 8.
n_neighbors = 12
edge_steps = 50

processor = DataProcessor(root_path, targets_path, r, n_neighbors, edge_steps)
processor.process()

# ds = get_dataset(root_path)
# train, val, test = dataset_split(ds)
# train_loader = get_dataloader(train, 5)

# for i, batch in enumerate(train_loader):
#     print(batch)
#     break

# edge_index, cell_offsets, neighbors = radius_graph_pbc(batch, r, n_neighbors)

# print(edge_index.shape)
# print(cell_offsets.shape)
# print(neighbors)