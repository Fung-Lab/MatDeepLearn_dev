import time
import subprocess
import os
import random

num_runs = 3
seeds = [123, 456, 789, 987, 543, 321, 876, 654, 234, 345]

if __name__ == '__main__':

    for i in range(num_runs):
        seed = str(seeds[i])

        # command = "python scripts/main.py --config_path=configs/finetune/config_JDFT.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_Dielectric.yml --seed=" + seed
        subprocess.run(command, shell=True)

        # command = "python scripts/main.py --config_path=configs/finetune/config_Perovskites.yml --seed=" + seed
        # subprocess.run(command, shell=True)
        #
        # command = "python scripts/main.py --config_path=configs/finetune/config_Phonons.yml --seed=" + seed
        # subprocess.run(command, shell=True)

        if i == 0:
            command = "python scripts/main.py --config_path=configs/finetune/config_GVRH.yml --seed=" + seed
            subprocess.run(command, shell=True)

        command = "python scripts/main.py --config_path=configs/finetune/config_KVRH.yml --seed=" + seed
        subprocess.run(command, shell=True)
