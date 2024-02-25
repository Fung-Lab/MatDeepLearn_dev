import time
import subprocess
import os
import random
import shutil
from tqdm import tqdm

# 运行次数
num_runs = 5
train_nums = [50, 100, 300, 500, -1]

def split_files(train_num):

    original_data_folder = "../data_qm9_qmc/"

    shutil.rmtree(os.path.join(original_data_folder, "train"))
    os.makedirs(os.path.join(original_data_folder, "train"))

    dirs = [d for d in os.listdir(original_data_folder) if os.path.isdir(os.path.join(original_data_folder, d)) and "singlet" in d]
    random.seed(20)
    random.shuffle(dirs)
    dataset_size = len(dirs)

    # train_ratio = 0.9
    val_ratio = 0.05
    test_ratio = 0.05

    train_len = train_num if train_num != -1 else 0.9 * dataset_size  # int(train_ratio * dataset_size)
    valid_len = int(val_ratio * dataset_size)
    test_len = int(test_ratio * dataset_size)
    unused_len = dataset_size - train_len - valid_len - test_len


    for i, dir_name in enumerate(tqdm(dirs)):
        if "singlet" in dir_name:
            if i < train_len:
                shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "train", dir_name))
            # elif i < train_len + valid_len:
            #     shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "val", dir_name))
            # elif i < train_len + valid_len + test_len:
            #     shutil.copytree(os.path.join(original_data_folder, dir_name), os.path.join(original_data_folder, "test", dir_name))


if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):
        split_files(train_nums[i])

        command = "python scripts/main.py --run_mode=train --config_path=configs/config_torchmd.yml"
        subprocess.run(command, shell=True, check=True)


