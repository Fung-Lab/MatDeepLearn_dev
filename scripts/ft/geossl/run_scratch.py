import time
import subprocess
import os

# 运行次数
num_runs = 10
seeds = [123, 456, 789, 987, 543, 321, 876, 654, 234, 345]

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):
        seed = str(seeds[i])

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_JDFT.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_Dielectric.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_Perovskites.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_GVRH.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_KVRH.yml --seed=" + seed
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/GeoSSL_qm9/config_Phonons.yml --seed=" + seed
        subprocess.run(command, shell=True)



        # 记录本次运行的开始时间
        # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f"Run {i + 1} completed at {current_time}")
