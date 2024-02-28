import pandas as pd
import torch.nn.functional as F
import torch
import os

def get_efs_error(folder_name, ratio=(0.01, 50, 50)):
    files = os.listdir(folder_name)
    result = {}

    # energy loss
    energy_file = 'test_predictions.csv'
    if energy_file in files:
        df = pd.read_csv(os.path.join(folder_name, energy_file))
        preds = torch.tensor(df[['prediction']].values)
        target = torch.tensor(df[['target']].values)
        e_error = F.l1_loss(preds, target)
        result.update({"energy_error": e_error})
    else:
        e_error = 0

    # force loss
    force_file = 'test_predictions_pos_grad.csv'
    if force_file in files:
        df = pd.read_csv(os.path.join(folder_name, force_file))
        preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
        target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
        f_error = F.l1_loss(preds, target)
        f_error_mse = F.mse_loss(preds, target)  # Calculate MSE for force
        result.update({"force_error": f_error})
        result.update({"force_error_MSE": f_error_mse})
    else:
        f_error = 0

    # stress loss
    stress_file = 'test_predictions_cell_grad.csv'
    if stress_file in files:
        df = pd.read_csv(os.path.join(folder_name, stress_file))
        preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values)
        target = torch.tensor(df[['target', 'target.1', 'target.2']].values)
        s_error = F.l1_loss(preds, target)
        result.update({"stress_error": s_error})
    else:
        s_error = 0

    # Overall EFS error
    result.update({"EFS_error": ratio[0] * e_error + ratio[1] * f_error + ratio[2] * s_error})

    return result

if __name__ == '__main__':
    result = get_efs_error('./results/2024-02-26-23-21-53-973-cgcnn_distill/train_results')
    for key in result.keys():
        print(f"{key}: {result[key]:.4f}")
