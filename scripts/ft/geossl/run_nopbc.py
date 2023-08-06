import time
import subprocess
import os
import random

# 运行次数
num_runs = 5
seeds = [42, 123, 9876, 5678, 8888]

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):
        seed = seeds[i]

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_JDFT_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_Dielectric_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_Perovskites_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_Phonons_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_GVRH_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_nopbc/config_KVRH_ct_ft.yml --seed=" + seed
        subprocess.run(command, shell=True)

        # 记录本次运行的开始时间
        # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f"Run {i + 1} completed at {current_time}")
