#!/bin/bash

#SBATCH -J denoising-torchmdnet-carbon
#SBATCH -A gts-vfung3-paid
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --mem-per-gpu=64G
#SBATCH -t 60:00:00
#SBATCH -q inferno
#SBATCH -o denoising-torchmdnet-carbon.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shuyijia@gatech.edu

module load anaconda3
conda activate mdl-llm

python scripts/main.py \
    --run_mode train \
    --config_path configs/torchmd_carbon.yml