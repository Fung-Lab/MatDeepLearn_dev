#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -J cgcnn-vn
#SBATCH --mail-user=sidnbaskaran@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 6:00:00
#SBATCH -A m3641_g

#OpenMP settings:
export OMP_NUM_THREADS=1

/global/homes/s/sidnb13/.conda/envs/matdeeplearn/bin/python /global/cfs/projectdirs/m3641/Sidharth/MatDeepLearn_dev/scripts/main.py \
    --config_path="/global/cfs/projectdirs/m3641/Sidharth/MatDeepLearn_dev/configs/examples/cgcnn_vn/config_cgcnn_novn_ocp.yml" \
    --run_mode="train" \
    --use_wandb=True \
