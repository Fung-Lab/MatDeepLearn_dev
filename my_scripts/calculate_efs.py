import pandas as pd
import torch.nn.functional as F
import torch
import os
import numpy as np


def get_efs_error(folder_name, ratio=(0.01, 50, 50)):
    assert len(ratio) == 3
    files = ['test_predictions.csv', 'test_predictions_cell_grad.csv', 'test_predictions_pos_grad.csv']
    assert all(f in os.listdir(folder_name) for f in files)

    result = {}

    # energy loss
    df = pd.read_csv(os.path.join(folder_name, 'val_predictions.csv'))
    preds = torch.tensor(df[['prediction']].values)
    target = torch.tensor(df[['target']].values)
    e_error = F.l1_loss(preds, target)
    result.update({"energy_error": e_error})

    # force loss
    df = pd.read_csv(os.path.join(folder_name, 'val_predictions_pos_grad.csv'))
    preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
    target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
    f_error = F.l1_loss(preds, target)
    result.update({"force_error": f_error})

    # stress loss
    df = pd.read_csv(os.path.join(folder_name, 'val_predictions_cell_grad.csv'))
    preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
    target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
    s_error = F.l1_loss(preds, target)
    result.update({"stress_error": s_error})

    result.update({"EFS error": ratio[0] * e_error + ratio[1] * f_error + ratio[2] * s_error})
    return result

if __name__ == '__main__':
    #result_path = './results/2023-12-14-07-07-37-147-torchmd_lj_v2//train_results'
    result_path = './results/2023-12-08-08-34-21-195-torchmd_morse_with_coef/train_results'
    result = get_efs_error(result_path)
    
    for key in result.keys():
        print(f"{key}: {result[key]:.6f}")