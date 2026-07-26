#!/bin/bash
#SBATCH --job-name=wafer_finetune
#SBATCH --partition=gpu
#SBATCH --account=eece5640.202630
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/joshi.shreyas/wafer_results/finetuned/slurm_%j.out
#SBATCH --error=/scratch/joshi.shreyas/wafer_results/finetuned/slurm_%j.err

mkdir -p /scratch/joshi.shreyas/wafer_results/finetuned

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/joshi.shreyas/conda/envs/wafer_env

cd ~/wafer-defect-open-set-rejection

echo "Job started on $(hostname) at $(date)"
nvidia-smi

python finetune.py --epochs 30 --batch_size 128 --lr 0.0005

echo "Job finished at $(date)"
