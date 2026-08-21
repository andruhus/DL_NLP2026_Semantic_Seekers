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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
NODE_NAME="${SLURM_NODELIST:-$(hostname)}"
cd "${WORKING_DIR}"

USE_GPU=true
POSITIONAL_ARGS=()
for argument in "$@"; do
    case "${argument}" in
        --use_gpu)
            USE_GPU=true
            ;;
        --no-use_gpu)
            USE_GPU=false
            ;;
        *)
            POSITIONAL_ARGS+=("${argument}")
            ;;
    esac
done

if (( ${#POSITIONAL_ARGS[@]} > 2 )); then
    echo "Usage: sbatch $0 [epochs] [loss_mode] [--use_gpu|--no-use_gpu]" >&2
    exit 2
fi

EPOCHS="${POSITIONAL_ARGS[0]:-5}"
LOSS_MODE="${POSITIONAL_ARGS[1]:-compare}"

echo "Working directory: ${WORKING_DIR}"
echo "Node: ${NODE_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Loss mode: ${LOSS_MODE}"
echo "Use GPU: ${USE_GPU}"

python --version
python -m torch.utils.collect_env 2> /dev/null

if command -v module > /dev/null 2>&1; then
    module load git
fi
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest commit: $(git rev-parse --short HEAD)"
echo "Uncommitted changes: $(git status --porcelain | wc -l)"

BART_ARGS=(
    --loss_mode "${LOSS_MODE}"
    --epochs "${EPOCHS}"
)
if [[ "${USE_GPU}" == true ]]; then
    BART_ARGS+=(--use_gpu)
fi

python -u bart_detection.py "${BART_ARGS[@]}"
