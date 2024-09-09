import time
import subprocess
import os
import random

num_runs = 3
seeds = [123, 456, 789, 987, 543, 321, 876, 654, 234, 345]

if __name__ == '__main__':

    for i in range(num_runs):
        seed = str(seeds[i])

        # command = "python scripts/main.py --config_path=configs/finetune/config_JDFT_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        # command = "python scripts/main.py --config_path=configs/finetune/config_Dielectric_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        # command = "python scripts/main.py --config_path=configs/finetune/config_Perovskites_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)
        #
        # command = "python scripts/main.py --config_path=configs/finetune/config_Phonons_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        # if i == 0:
        #     command = "python scripts/main.py --config_path=configs/finetune/config_GVRH_scratch.yml --seed=" + seed
        #     subprocess.run(command, shell=True)

        # command = "python scripts/main.py --config_path=configs/finetune/config_KVRH_scratch.yml --seed=" + seed
        # subprocess.run(command, shell=True)

    for seed in [456, 789]:
        command = "python scripts/main.py --config_path=configs/finetune/config_KVRH_scratch.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_Surface_scratch.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_Dielectric_scratch.yml --seed=" + seed
        subprocess.run(command, shell=True)