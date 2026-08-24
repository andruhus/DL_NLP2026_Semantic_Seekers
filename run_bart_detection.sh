#!/bin/bash
#SBATCH --job-name=train-bart-detection
#SBATCH --time=00:20:00
#SBATCH --partition=grete:shared
#SBATCH --gpus=A100:1
#SBATCH --mem-per-gpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=andrii.demydenko@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/paraphrase_detection_%j.out
#SBATCH --error=./slurm_files/paraphrase_detection_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
NODE_NAME="${SLURM_NODELIST:-$(hostname)}"
cd "${WORKING_DIR}"

USE_GPU=true
BATCH_SIZE=16
POSITIONAL_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --use_gpu)
            USE_GPU=true
            shift
            ;;
        --no-use_gpu)
            USE_GPU=false
            shift
            ;;
        --batch_size)
            if (( $# < 2 )); then
                echo "--batch_size requires a value" >&2
                exit 2
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL_ARGS[@]} > 2 )); then
    echo "Usage: sbatch $0 [epochs] [loss_mode] [--batch_size N] [--use_gpu|--no-use_gpu]" >&2
    exit 2
fi

EPOCHS="${POSITIONAL_ARGS[0]:-5}"
LOSS_MODE="${POSITIONAL_ARGS[1]:-compare}"

case "${LOSS_MODE}" in
    unweighted|weighted|compare)
        ;;
    *)
        echo "loss_mode must be one of: unweighted, weighted, compare" >&2
        exit 2
        ;;
esac

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_DATE="$(date +%Y-%m-%d_%H-%M-%S)"
    LOG_DIRECTORY="${WORKING_DIR}/slurm_files"
    LOG_STEM="${LOG_DIRECTORY}/paraphrase_detection_${RUN_DATE}_${SLURM_JOB_ID}_${LOSS_MODE}"
    exec > "${LOG_STEM}.out" 2> "${LOG_STEM}.err"
    rm -f \
        "${LOG_DIRECTORY}/paraphrase_detection_${SLURM_JOB_ID}.out" \
        "${LOG_DIRECTORY}/paraphrase_detection_${SLURM_JOB_ID}.err"
fi

source activate dnlp
set -u
export TOKENIZERS_PARALLELISM=false

echo "Working directory: ${WORKING_DIR}"
echo "Node: ${NODE_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Loss mode: ${LOSS_MODE}"
echo "Batch size: ${BATCH_SIZE}"
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
    --batch_size "${BATCH_SIZE}"
)
if [[ "${USE_GPU}" == true ]]; then
    BART_ARGS+=(--use_gpu)
fi

python -u bart_detection.py "${BART_ARGS[@]}"
