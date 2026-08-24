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
FOCAL_GAMMAS=()
COMPARE_BCE_ONLY=false
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
        --focal_gamma)
            if (( $# < 2 )); then
                echo "--focal_gamma requires a value" >&2
                exit 2
            fi
            FOCAL_GAMMAS+=("$2")
            shift 2
            ;;
        --compare_bce_only)
            COMPARE_BCE_ONLY=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL_ARGS[@]} > 2 )); then
    echo "Usage: sbatch $0 [epochs] [loss_mode] [--batch_size N] [--focal_gamma G ...] [--compare_bce_only] [--use_gpu|--no-use_gpu]" >&2
    exit 2
fi

EPOCHS="${POSITIONAL_ARGS[0]:-5}"
LOSS_MODE="${POSITIONAL_ARGS[1]:-compare}"
if (( ${#FOCAL_GAMMAS[@]} == 0 )); then
    FOCAL_GAMMAS=(2.0)
fi

case "${LOSS_MODE}" in
    unweighted|weighted|focal|compare)
        ;;
    *)
        echo "loss_mode must be one of: unweighted, weighted, focal, compare" >&2
        exit 2
        ;;
esac

if [[ "${COMPARE_BCE_ONLY}" == true && "${LOSS_MODE}" != compare ]]; then
    echo "--compare_bce_only requires loss_mode=compare" >&2
    exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_DATE="$(date +%Y-%m-%d)"
    RUN_TIME="$(date +%H-%M-%S)"
    INITIAL_LOG_DIRECTORY="${WORKING_DIR}/slurm_files"
    LOG_DIRECTORY="${INITIAL_LOG_DIRECTORY}/${RUN_DATE}/${RUN_TIME}"
    mkdir -p "${LOG_DIRECTORY}"
    LOG_STEM="${LOG_DIRECTORY}/paraphrase_detection_${LOSS_MODE}_${EPOCHS}_${SLURM_JOB_ID}"
    exec > "${LOG_STEM}.out" 2> "${LOG_STEM}.err"
    rm -f \
        "${INITIAL_LOG_DIRECTORY}/paraphrase_detection_${SLURM_JOB_ID}.out" \
        "${INITIAL_LOG_DIRECTORY}/paraphrase_detection_${SLURM_JOB_ID}.err"
fi

source activate dnlp
set -u
export TOKENIZERS_PARALLELISM=false

echo "Working directory: ${WORKING_DIR}"
echo "Node: ${NODE_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Loss mode: ${LOSS_MODE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Focal gammas: ${FOCAL_GAMMAS[*]}"
echo "Compare BCE only: ${COMPARE_BCE_ONLY}"
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
if [[ "${COMPARE_BCE_ONLY}" == true ]]; then
    BART_ARGS+=(--compare_bce_only)
fi
for focal_gamma in "${FOCAL_GAMMAS[@]}"; do
    BART_ARGS+=(--focal_gamma "${focal_gamma}")
done
if [[ "${USE_GPU}" == true ]]; then
    BART_ARGS+=(--use_gpu)
fi

python -u bart_detection.py "${BART_ARGS[@]}"
