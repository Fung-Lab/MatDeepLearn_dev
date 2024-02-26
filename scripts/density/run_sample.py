import time
import subprocess
import os
import random
import shutil
from tqdm import tqdm
import yaml

# 运行次数
num_runs = 5
num_samples = [200, 500, 1000, 5000, 10000]


def change_sample(num_sample):
    with open("configs/config_torchmd_test.yml", "r") as file:
        data = yaml.safe_load(file)

    # Modify value
    data["dataset"]["preprocess_params"]["virtual_params"]["num_samples"] = num_sample

    # Write updated data back to YAML file
    with open("configs/config_torchmd_test.yml", "w") as file:
        yaml.dump(data, file)

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):

        change_sample(num_samples[i])

        command = "python scripts/main.py --run_mode=train --config_path=configs/config_torchmd_test.yml"
        subprocess.run(command, shell=True, check=True)


