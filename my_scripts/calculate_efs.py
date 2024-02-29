import os

import pandas as pd
import torch
import torch.nn.functional as F


def get_efs_error(folder_name, ratio=(0.01, 50, 50), split='test'):
    assert len(ratio) == 3
    files = [f'{split}_predictions.csv', f'{split}_predictions_cell_grad.csv', f'{split}_predictions_pos_grad.csv']
    assert all(f in os.listdir(folder_name) for f in files)

    result = {}

    # energy loss
    df = pd.read_csv(os.path.join(folder_name, f'{split}_predictions.csv'))
    preds = torch.tensor(df[['prediction']].values)
    target = torch.tensor(df[['target']].values)
    e_error = F.l1_loss(preds, target)
    result.update({"energy_error": e_error})

    # force loss
    df = pd.read_csv(os.path.join(folder_name, f'{split}_predictions_pos_grad.csv'))
    preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
    target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
    f_error = F.l1_loss(preds, target)
    result.update({"force_error": f_error})

    # stress loss
    s_error = 0.
    result.update({"stress_error": 0.})
    if ratio[2] != 0.:
        df = pd.read_csv(os.path.join(folder_name, f'{split}_predictions_cell_grad.csv'))
        preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
        target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
        s_error = F.l1_loss(preds, target)
        result.update({"stress_error": s_error})

    result.update({"EFS error": ratio[0] * e_error + ratio[1] * f_error + ratio[2] * s_error})
    return result

if __name__ == '__main__':
    #result_path = './results/silica_tests/2024-01-25-13-56-30-293-lj_sio2/train_results'
    result_path = './results/2024-02-27-14-41-51-090-pt_sl1t1_25_no_atomic_e_sps/train_results'
    result = get_efs_error(result_path, ratio=(0.01, 50, 0), split='test')
    
    for key in result.keys():
        print(f"{key}: {result[key]:.6f}")