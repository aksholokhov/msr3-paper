#!/bin/bash

#SBATCH --job-name="ps3_5"
#SBATCH --account=amath
#SBATCH --partition=gpu_rtx6k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-23:00:00
#SBATCH --array=1-20%4
#SBATCH --output=logs/slurm_logs/%A_%a.out
#SBATCH --mem=16G


experiment_name="scaling_5"
trial_num=${SLURM_ARRAY_TASK_ID}
trials_from=$trial_num
trials_to=$((trial_num + 1))
experiments="L0,L1,ALASSO,SCAD"
models="PGD,MSR3"
groups_sizes="50,75,20,40,15,25,90,45,30"
num_covariates=99

source ~/.bashrc
conda activate compute
cd ~/storage/repos/msr3-paper

time python generate_all.py \
  --experiment_name $experiment_name \
  --experiments $experiments \
  --trials_from $trials_from \
  --trials_to $trials_to \
  --use_dask 0 \
  --models $models\
  --groups_sizes $groups_sizes \
  --num_covariates $num_covariates \
  --worker_number ${SLURM_ARRAY_TASK_ID} \
