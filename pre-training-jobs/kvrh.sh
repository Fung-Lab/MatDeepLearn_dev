#!/bin/bash
#SBATCH -A m3641_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G

export SLURM_CPU_BIND="cores"
module load python
conda activate mdl
cd /global/homes/s/shuyijia/github/MatDeepLearn_dev/
python scripts/main.py --run_mode=train --config_path=pre-training-jobs/cgcnn_kvrh.yml