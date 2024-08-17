import time
import subprocess
import os
import random
import shutil
import yaml
from tqdm import tqdm

# 运行次数
num_runs = 5
train_nums = [50, 100, 300, 500]
sample_nums = [10, 50, 200, 500, 1000, 5000]

def split_files(train_num):

    original_data_folder = "../data_qm9_qmc/data_qm9_qmc/"

    shutil.rmtree(os.path.join(original_data_folder, "train_alb"))
    os.makedirs(os.path.join(original_data_folder, "train_alb"))

    dirs = [d for d in os.listdir(os.path.join(original_data_folder, "train"))
            if os.path.isdir(os.path.join(original_data_folder, "train", d)) and "singlet" in d]
    random.seed(20)
    random.shuffle(dirs)
    full_train_size = len(dirs)

    # train_ratio = 0.9
    val_ratio = 0.05
    test_ratio = 0.05

    for i in range(train_num):
        dir_name = dirs[i]
        shutil.copytree(os.path.join(original_data_folder, "train", dir_name),
                        os.path.join(original_data_folder, "train_alb", dir_name))

    # for i, dir_name in enumerate(tqdm(dirs)):
    #     if "singlet" in dir_name:
    #         if i < train_len:
    #             shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "train", dir_name))
            # elif i < train_len + valid_len:
            #     shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "val", dir_name))
            # elif i < train_len + valid_len + test_len:
            #     shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "test", dir_name))

def change_sample(sample_num):
    with open("configs/config_torchmd_rs.yml", "r") as file:
        data = yaml.safe_load(file)

    # Modify value
    # data["dataset"]["preprocess_params"]["virtual_params"]["num_samples"] = sample_num
    data["task"]["identifier"] = "qm9_train_num_" + str(sample_num)

    # Write updated data back to YAML file
    with open("configs/config_torchmd_rs.yml", "w") as file:
        yaml.dump(data, file)

if __name__ == '__main__':

    # folder = "../data_qm9_qmc/data_qm9_qmc/"
    # dirs = [d for d in os.listdir(os.path.join(folder, "test")) if os.path.isdir(os.path.join(folder, "test", d))]
    #
    # for dr in dirs:
    #     change_sample(dr)
    #     command = "python scripts/main.py --run_mode=predict --config_path=configs/config_torchmd_rs.yml"
    #     subprocess.run(command, shell=True, check=True)

    # for sample_n in sample_nums:
    #     change_sample(sample_n)
    #
    #     command = "python scripts/main.py --run_mode=train --config_path=configs/config_torchmd_rs.yml"
    #     subprocess.run(command, shell=True, check=True)

    for t_num in train_nums:
        change_sample(t_num)
        split_files(t_num)
        command = "python scripts/main.py --run_mode=train --config_path=configs/config_torchmd_rs.yml"
        subprocess.run(command, shell=True, check=True)


