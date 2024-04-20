import time
import subprocess
import os
import random

# 运行次数
num_runs = 3
seeds = [123, 456, 789, 987, 543, 321, 876, 654, 234, 345]

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):
        seed = str(seeds[i])

        # command = "python scripts/main.py --config_path=configs/finetune/config_2D.yml --seed=" + seed
        # subprocess.run(command, shell=True)
        #
        # command = "python scripts/main.py --config_path=configs/finetune/config_2D_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_MOF.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_MOF_scratch.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_Dielectric.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_Dielectric_scratch.yml --seed=" + seed
        subprocess.run(command, shell=True)

        # command = "python scripts/main.py --config_path=configs/finetune/config_Surface.yml --seed=" + seed
        # subprocess.run(command, shell=True)
        #
        # command = "python scripts/main.py --config_path=configs/finetune/config_Surface_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        # 记录本次运行的开始时间
        # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f"Run {i + 1} completed at {current_time}")