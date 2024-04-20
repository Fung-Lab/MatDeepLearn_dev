import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def get_structures_idx(test_pred_csv_path, data_pt_path):
    df = pd.read_csv(test_pred_csv_path)
    structure_ids = df['structure_id'].tolist()
    structure_ids = set(map(lambda x: str(x), structure_ids))
    
    data = torch.load(data_pt_path)[0]
    selected_idxs = []

    for i in range(len(data.structure_id)):
        if data.structure_id[i][0] in structure_ids:
            selected_idxs.append(i)
    return selected_idxs

if __name__ == '__main__':
    json_path = 'data/Silica_data/raw/data.json'
    forces = []
    
    data = torch.load('data/Silica_data/data.pt')[0]
    test_pred_csv_path = 'results/2024-04-06-10-48-11-973-cgcnn_sio2/train_results/test_predictions.csv'
    val_pred_csv_path = 'results/2024-04-06-10-48-11-973-cgcnn_sio2/train_results/val_predictions.csv'
    silica_dataset = 'data/Silica_data/data.pt'
    
    test_indices = set(get_structures_idx(test_pred_csv_path, silica_dataset))
    val_indices = set(get_structures_idx(val_pred_csv_path, silica_dataset))
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):
        if i not in test_indices and i not in val_indices:
            forces.append(data[i]['forces'])
    
    print(len(forces))
    forces = np.vstack(forces)
    force_magnitute = np.linalg.norm(forces, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(force_magnitute, bins=100, color='blue', alpha=0.7)
    plt.title('Distribution of Force Magnitude')
    plt.xlabel('Force Magnitude')
    plt.xlim(left=0, right=20)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print(force_magnitute.shape)
    print(f"Mean force magnitude: {np.mean(force_magnitute):.4f}")
    print(f"Force magnitude std: {np.std(force_magnitute):.4f}")
    
    # print(f"Mean force magnitude on each dimension: {np.mean(forces, axis=0)}")
    print(f"Force magnitude std on each dimension: {np.std(forces, axis=0)}")
    
    