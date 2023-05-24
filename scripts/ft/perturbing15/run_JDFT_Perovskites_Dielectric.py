import time
import subprocess
import os

# 运行次数
num_runs = 10

if __name__ == '__main__':

    # 循环执行
    for i in range(num_runs):

        # 执行命令行命令
        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_JDFT.yml"
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_JDFT_ct_ft.yml"
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_Dielectric.yml"
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_Dielectric_ct_ft.yml"
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_Perovskites.yml"
        subprocess.run(command, shell=True)

        command = "python scripts/main.py --run_mode=train --config_path=configs/ct_ft/perturbing15/config_Perovskites_ct_ft.yml"
        subprocess.run(command, shell=True)

        # 记录本次运行的开始时间
        # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f"Run {i + 1} completed at {current_time}")
