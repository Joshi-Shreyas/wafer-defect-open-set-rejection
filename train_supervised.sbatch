#!/bin/bash
#SBATCH --job-name=wafer_supervised
#SBATCH --partition=gpu
#SBATCH --account=eece5640.202630
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/joshi.shreyas/wafer_results/supervised/slurm_%j.out
#SBATCH --error=/scratch/joshi.shreyas/wafer_results/supervised/slurm_%j.err

# Make sure the output directory exists before SLURM tries to write logs there
mkdir -p /scratch/joshi.shreyas/wafer_results/supervised

# Load conda and activate the environment
source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/joshi.shreyas/conda/envs/wafer_env

cd ~/wafer-defect-open-set-rejection

echo "Job started on $(hostname) at $(date)"
nvidia-smi

python train_supervised.py --epochs 30 --batch_size 128 --lr 0.001

echo "Job finished at $(date)"
