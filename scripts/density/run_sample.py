import time
import subprocess
import os
import random
import shutil
from tqdm import tqdm

# 运行次数
num_runs = 5
num_samples = [200, 500, 1000, 5000, 10000]

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):

        command = "python scripts/main.py --run_mode=train --config_path=configs/config_torchmd.yml --num_samples " \
                  + str(num_samples[i])
        subprocess.run(command, shell=True, check=True)


