#!/bin/bash
#SBATCH -Jcgcnn_hg                                  # Job name
#SBATCH -Agts-vfung3                                # Charge account
#SBATCH -N1 --gres=gpu:A100:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=80G                           # Memory per gpu
#SBATCH -CA100-80GB                                 # GPU type       
#SBATCH -t12:00:00                                  # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                   # QOS name
#SBATCH -ocgcnn_job-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=sidharth.baskaran@gatech.edu    # e-mail address for notifications

cd /storage/home/hcoda1/9/sbaskaran31/p-vfung3-0/MatDeepLearn_dev/scripts
conda activate matdeeplearn

~/.conda/envs/matdeeplearn/bin/python main.py --config_path="/storage/home/hcoda1/9/sbaskaran31/p-vfung3-0/MatDeepLearn_dev/configs/examples/cgcnn_vn_hg/config_cgcnn_vn_hg_1.yml" \
    --run_mode="train" \
    --use_wandb=True