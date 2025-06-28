#!/bin/bash
#SBATCH -A project_name
#SBATCH --job-name=metric_study      # Job name
#SBATCH --output=logs/metrics_%j.txt # Output file
#SBATCH --error=logs/metrics_error_%j.txt  # Error file
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --time=24:00:00        # Time limit hrs:min:sec

# Ensure logs directory exists
mkdir -p logs

# Load your CUDA and Python environments
module load cuda/12.2
module load anaconda
module load virtualenv/20.23.1-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load h5py/3.9.0-foss-2023a

# Activate your conda environment
source env/bin/activate

srun python/diffusion/metrics.py
