#!/bin/bash

# Check if the script is running within a SLURM job
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running within SLURM, so submit the job

    # Ensure logs directory exists
    mkdir -p logs

    # Submit this script to SLURM
    sbatch \
      -A name \
      --job-name=diffusion_vilar \
      --output=logs/diffusion_vilar_%j.txt \
      --error=logs/diffusion_error_vilar_%j.txt \
      --gpus-per-node= \
      --time=24:00:00 \
      "$0"
    exit 0
fi

# Below this point, the script runs within the SLURM job

# Load required modules
module purge  # Clear any loaded modules first
module load cuda/12.2
module load Python/3.10.8-GCCcore-12.2.0
module load virtualenv/20.23.1-GCCcore-12.2.0
module load matplotlib/3.7.2-gfbf-2022b
module load SciPy-bundle/2023.02-gfbf-2022b
module load h5py/3.8.0-foss-2022b

# Activate the virtual environment
source env activate

# Run the Python script
cd /diffusion/vilar
python vilar_main.py
