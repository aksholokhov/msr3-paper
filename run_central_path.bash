#!/bin/bash

#SBATCH --job-name="central_path"
#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:55:00
#SBATCH --array=0-3999
#SBATCH --output=logs/slurm_scaling/%A_%a.out
#SBATCH --mem=8G

ic="jones_bic"
source ~/.bashrc
conda activate compute
cd ~/storage/repos/msr3-paper

time python central_path.py  --trial ${SLURM_ARRAY_TASK_ID} --ic $ic --experiment_name "central_path_$ic"


