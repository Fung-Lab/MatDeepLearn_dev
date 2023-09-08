import time
import subprocess
import os
import random

# 运行次数
num_runs = 5
seeds = [123, 456, 789, 987, 543, 321, 876, 654, 234, 345]

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):
        seed = str(seeds[i])

        command = "python scripts/main.py --run_mode=train --config_path=configs/config.yml"
        subprocess.run(command, shell=True)
