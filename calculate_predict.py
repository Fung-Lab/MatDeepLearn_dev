import pandas as pd
import torch.nn.functional as F
import torch
import os

def get_efs_error(folder_name, ratio=(0.01, 50, 50)):
    files = os.listdir(folder_name)
    result = {}

    # energy loss
    energy_file = 'predict_predictions.csv'
    if energy_file in files:
        df = pd.read_csv(os.path.join(folder_name, energy_file))
        preds = torch.tensor(df[['prediction']].values)
        target = torch.tensor(df[['target']].values)
        e_error = F.l1_loss(preds, target)
        result.update({"energy_error": e_error})
    else:
        e_error = 0

    # force loss
    force_file = 'predict_predictions_pos_grad.csv'
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

    stress_file = 'predict_predictions_cell_grad.csv'
    if stress_file in files:
        df = pd.read_csv(os.path.join(folder_name, stress_file))
        target_cols = ['target', 'target.1', 'target.2']
        if all(col in df.columns for col in target_cols):
            preds = torch.tensor(df[['prediction', 'prediction.1', 'prediction.2']].values, dtype=torch.float32)
            target = torch.tensor(df[target_cols].values, dtype=torch.float32)
            s_error = F.l1_loss(preds, target)
        else:
            s_error = torch.tensor(0.0)
        result.update({"stress_error": s_error})
    else:
        s_error = torch.tensor(0.0)

    # Overall EFS error, ensure e_error, f_error, and s_error are tensors for consistent operations
    result.update({"EFS_error": ratio[0] * e_error + ratio[1] * f_error + ratio[2] * s_error})

    return result

if __name__ == '__main__':
    result = get_efs_error('./results/2024-05-19-22-53-08-597-torch_md_predict/train_results')
    for key in result.keys():
        print(f"{key}: {result[key]:.4f}")