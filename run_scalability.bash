#!/bin/bash

#SBATCH --job-name="scaling"
#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:55:00
#SBATCH --array=1-1400%100
#SBATCH --output=logs/slurm_scaling/%A_%a.out
#SBATCH --mem=8G

experiment_name="scaling"

source ~/.bashrc
conda activate compute
cd ~/storage/repos/msr3-paper

time python generate_all.py \
  --experiment_name $experiment_name \
  --trial ${SLURM_ARRAY_TASK_ID} \
