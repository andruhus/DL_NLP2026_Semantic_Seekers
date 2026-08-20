#!/bin/bash
#SBATCH --job-name=train-bart-detection
#SBATCH --time=00:20:00
#SBATCH --partition=grete:shared
#SBATCH --gpus=A100:1
#SBATCH --mem-per-gpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

set -euo pipefail

source activate dnlp
cd "${SLURM_SUBMIT_DIR}"

EPOCHS="${1:-5}"
LOSS_MODE="${2:-compare}"

echo "Submission directory: ${SLURM_SUBMIT_DIR}"
echo "Node: ${SLURM_NODELIST}"
echo "Epochs: ${EPOCHS}"
echo "Loss mode: ${LOSS_MODE}"

python --version
python -m torch.utils.collect_env 2> /dev/null

module load git
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest commit: $(git rev-parse --short HEAD)"
echo "Uncommitted changes: $(git status --porcelain | wc -l)"

python -u bart_detection.py \
    --use_gpu \
    --loss_mode "${LOSS_MODE}" \
    --epochs "${EPOCHS}"
