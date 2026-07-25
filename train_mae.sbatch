#!/bin/bash
#SBATCH --job-name=wafer_mae
#SBATCH --partition=gpu
#SBATCH --account=eece5640.202630
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/joshi.shreyas/wafer_results/mae/slurm_%j.out
#SBATCH --error=/scratch/joshi.shreyas/wafer_results/mae/slurm_%j.err

mkdir -p /scratch/joshi.shreyas/wafer_results/mae

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/joshi.shreyas/conda/envs/wafer_env

cd ~/wafer-defect-open-set-rejection

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# --resume: if this job gets killed by the time limit before finishing, resubmitting
# with the same command will pick up from the last saved checkpoint instead of restarting.
python train_mae.py --epochs 20 --batch_size 256 --mask_ratio 0.6 --resume

echo "Job finished at $(date)"
