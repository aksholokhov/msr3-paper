#!/bin/bash

#SBATCH --job-name="ps3"
#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-23:00:00
#SBATCH --array=0-100%20
#SBATCH --output=logs/slurm_logs/%A_%a.out
#SBATCH --mem=8G


trial_num=${SLURM_ARRAY_TASK_ID}
trials_from=$trial_num
trials_to=$((trial_num + 1))
if [ $trial_num -eq 0 ]
then
  experiments="intuition,bullying"
else
  experiments="L0,L1,ALASSO,SCAD"
fi
source ~/.bashrc
conda activate compute
cd ~/storage/repos/msr3-paper

time python generate_all.py \
  --experiments $experiments \
  --trials_from $trials_from \
  --trials_to $trials_to \
  --worker_number ${SLURM_ARRAY_TASK_ID} \
  --experiment_name "${SLURM_ARRAY_JOB_ID}" \


