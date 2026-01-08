#!/bin/bash
#SBATCH --job-name=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-TW3740TU
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

module purge
module load 2025
module load python
module load cuda/12.9
module load openmpi

source .venv/bin/activate

which python

srun python -u example.py
