#!/bin/bash
#SBATCH -Jcgcnn_novn                                # Job name
#SBATCH -Agts-vfung3                                # Charge account
#SBATCH -N1 --gres=gpu:A100:1                       # Number of nodes and GPUs required
#SBATCH --mem=0
#SBATCH -CA100-40GB                                 # GPU type       
#SBATCH -t03:00:00                                  # Duration of the job (Ex: 15 mins) D-HH:MM:SS
#SBATCH -qinferno                                   # QOS name
#SBATCH -ocgcnn_job-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=sidharth.baskaran@gatech.edu    # e-mail address for notifications

cd /storage/home/hcoda1/9/sbaskaran31/p-vfung3-0/MatDeepLearn_dev/scripts

~/.conda/envs/matdeeplearn/bin/python main.py --config_path="/storage/home/hcoda1/9/sbaskaran31/p-vfung3-0/MatDeepLearn_dev/configs/examples/cgcnn_vn/config_cgcnn_novn_ocp.yml" \
    --run_mode="train" \
    --use_wandb=True
