import pandas as pd
import json
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('train_outs/test/unrelax_energy.csv')
    id_list = df['strucure_id'].to_list()
    
    json_path = 'data/optimization_data/data.json'
    f = open(json_path, 'r')
    data = json.load(f)
    f.close()
    
    count = 0
    
    relaxed_energy_list = []
    for id in id_list:
        for s in data:
            if s['structure_id'] == id:
                # print(id, s['unrelaxed_energy'], s['relaxed_energy'])
                relaxed_energy_list.append(s['relaxed_energy'])
                break
    
    df['true_relaxed_energy'] = np.array(relaxed_energy_list).reshape(-1, 1)
    df.to_csv('train_outs/test/unrelax_energy_analysis.csv', index=False)