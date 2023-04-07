#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J cgcnn-vn
#SBATCH --mail-user=sidnbaskaran@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00
#SBATCH -A m3641_g

#OpenMP settings:
export OMP_NUM_THREADS=1

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  \
    /global/homes/s/sidnb13/.conda/envs/matdeeplearn/bin/python /global/cfs/projectdirs/m3641/Sidharth/MatDeepLearn_dev/scripts/main.py \
    --config_path="/global/cfs/projectdirs/m3641/Sidharth/MatDeepLearn_dev/configs/examples/cgcnn_vn_hg/config_cgcnn_vn_hg.yml" \
    --run_mode="train" --use_wandb=True
