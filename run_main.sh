#!/bin/bash

# Set the task name and other hyperparameters for this specific run

task_name="slcp_distractors"
theta_dim=5
y_dim=100
num_timesteps=500
learning_rate=0.001
batch_size=128
num_epochs=1000
layer_sizes="512 512 512"
#simulation_budgets="10000"
simulation_budgets="10000 20000 30000"
num_runs=5
export beta_schedule="quadratic"
export model_type="linear"




# Check if the script is running within a SLURM job
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running within SLURM, so submit the job

    # Ensure logs directory exists
    mkdir -p logs

    # Submit this script to SLURM with dynamically set job name and output files
    sbatch \
      -A name_of_your_project \
      --job-name=diffusion_${task_name} \
      --output=logs/diffusion_${task_name}_%j.txt \
      --error=logs/diffusion_error_${task_name}_%j.txt \
      --gpus-per-node \
      --time=24:00:00 \
      "$0"
    exit 0
fi



# Load required modules
module purge  # Clear any loaded modules first
module load cuda/12.2
module load Python/3.10.8-GCCcore-12.2.0
module load virtualenv/20.23.1-GCCcore-12.2.0
module load matplotlib/3.7.2-gfbf-2022b
module load SciPy-bundle/2023.02-gfbf-2022b
module load h5py/3.8.0-foss-2022b

# Activate the virtual environment
source env/bin/activate


srun python /diffusion/main.py \
    --task_name $task_name \
    --theta_dim $theta_dim \
    --y_dim $y_dim \
    --num_timesteps $num_timesteps \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --layer_sizes $layer_sizes \
    --simulation_budgets $simulation_budgets \
    --num_runs $num_runs \
    --beta_schedule $beta_schedule \
    --model_type $model_type
