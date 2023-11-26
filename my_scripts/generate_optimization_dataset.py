import pandas as pd
import json
from tqdm import tqdm
import os


def load_structure_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df['structure_id'].to_list()


def load_structures_from_json(json_path):
    f = open(json_path, 'r')
    data = json.load(f)
    f.close()
    return data


if __name__ == '__main__':
    data_together = load_structures_from_json('../../../Shared/Materials_datasets/MP_data_forces/raw/data_relaxed_unrelaxed_together.json')
    
    data_relaxed = load_structures_from_json('../../../Shared/Materials_datasets/MP_data_forces/raw/data_relaxed.json')
    data_unrelaxed = load_structures_from_json('../../../Shared/Materials_datasets/MP_data_forces/raw/data_unrelaxed.json')
    
    test_ids = load_structure_ids_from_csv('results/CGCNN_LJ_v4/train_results/test_predictions.csv')
    
    relaxed_names = [d['structure_id'] for d in data_relaxed]
    unrelaxed_names = [d['structure_id'] for d in data_unrelaxed]
    
    test_relaxed, test_unrelaxed = set(), set()
    
    final_dataset = []
    
    print("Classifying relaxed/unrelaxed data...")
    for id in tqdm(test_ids):
        truncated_id = id.split('-')[0] + '-' + id.split('-')[1]
        if id in relaxed_names:
            test_relaxed.add(truncated_id)
        elif id in unrelaxed_names:
            test_unrelaxed.add(truncated_id)
          
    relax, unrelax = 0, 0
    print('Building dataset...')  
    for structure in tqdm(data_together):
        if structure['structure_id'] in test_relaxed:
            structure['type'] = 'relaxed_in_test'
            relax += 1
            final_dataset.append(structure)
        elif structure['structure_id'] in test_unrelaxed:
            structure['type'] = 'unrelaxed_in_test'
            unrelax += 1
            final_dataset.append(structure)
    
    save_dir = 'results/optim_data'
    os.makedirs(save_dir, exist_ok=True)
    
    json_file_path = save_dir + '/data.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(final_dataset, json_file)
    
    print(f'Dataset saved to {json_file_path}')
    print(f'Relaxed structures: {relax}, Unrelaxed structures: {unrelax}')
