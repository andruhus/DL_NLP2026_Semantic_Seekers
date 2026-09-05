#!/bin/bash
#SBATCH --job-name=train-bart-generation
#SBATCH --time=02:00:00
#SBATCH --partition=grete:shared
#SBATCH --gpus=A100:1
#SBATCH --mem-per-gpu=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=andrii.demydenko@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/paraphrase_generation_%j.out
#SBATCH --error=./slurm_files/paraphrase_generation_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
NODE_NAME="${SLURM_NODELIST:-$(hostname)}"
cd "${WORKING_DIR}"

USE_GPU=true
BATCH_SIZE=8
LEARNING_RATES=(2e-5)
MIN_LRS=(0.0)
STEP_DECAY_EPOCHS=(1)
STEP_GAMMAS=(0.5)
WARMUP_STEPS=(100)
METRIC_FACTORS=(0.5)
METRIC_PATIENCES=(1)
METRIC_THRESHOLDS=(1e-8)
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
        --learning_rate|--min_lr|--step_decay_epochs|--step_gamma|--warmup_steps|--metric_factor|--metric_patience|--metric_threshold)
            OPTION="$1"
            VALUES=()
            shift
            while (( $# > 0 )) && [[ "$1" != --* ]]; do
                VALUES+=("$1")
                shift
            done
            if (( ${#VALUES[@]} == 0 )); then
                echo "${OPTION} requires at least one value" >&2
                exit 2
            fi
            case "${OPTION}" in
                --learning_rate) LEARNING_RATES=("${VALUES[@]}") ;;
                --min_lr) MIN_LRS=("${VALUES[@]}") ;;
                --step_decay_epochs) STEP_DECAY_EPOCHS=("${VALUES[@]}") ;;
                --step_gamma) STEP_GAMMAS=("${VALUES[@]}") ;;
                --warmup_steps) WARMUP_STEPS=("${VALUES[@]}") ;;
                --metric_factor) METRIC_FACTORS=("${VALUES[@]}") ;;
                --metric_patience) METRIC_PATIENCES=("${VALUES[@]}") ;;
                --metric_threshold) METRIC_THRESHOLDS=("${VALUES[@]}") ;;
            esac
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL_ARGS[@]} > 2 )); then
    echo "Usage: sbatch $0 [epochs] [scheduler_mode] [options]" >&2
    echo "Scheduler options accept one or more values, e.g. --step_gamma 0.3 0.5." >&2
    exit 2
fi

EPOCHS="${POSITIONAL_ARGS[0]:-5}"
SCHEDULER_MODE="${POSITIONAL_ARGS[1]:-compare}"
case "${SCHEDULER_MODE}" in
    constant|step|cosine|linear|inverse_sqrt|metric|compare)
        ;;
    *)
        echo "scheduler_mode must be one of: constant, step, cosine, linear, inverse_sqrt, metric, compare" >&2
        exit 2
        ;;
esac

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_DATE="$(date +%Y-%m-%d)"
    RUN_TIME="$(date +%H-%M-%S)"
    INITIAL_LOG_DIRECTORY="${WORKING_DIR}/slurm_files"
    LOG_DIRECTORY="${INITIAL_LOG_DIRECTORY}/${RUN_DATE}/${RUN_TIME}"
    mkdir -p "${LOG_DIRECTORY}"
    LOG_STEM="${LOG_DIRECTORY}/paraphrase_generation_${SCHEDULER_MODE}_${EPOCHS}_${SLURM_JOB_ID}"
    exec > "${LOG_STEM}.out" 2> "${LOG_STEM}.err"
    rm -f \
        "${INITIAL_LOG_DIRECTORY}/paraphrase_generation_${SLURM_JOB_ID}.out" \
        "${INITIAL_LOG_DIRECTORY}/paraphrase_generation_${SLURM_JOB_ID}.err"
fi

source activate dnlp
set -u
export TOKENIZERS_PARALLELISM=false

echo "Working directory: ${WORKING_DIR}"
echo "Node: ${NODE_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Scheduler mode: ${SCHEDULER_MODE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Initial learning rates: ${LEARNING_RATES[*]}"
echo "Minimum learning rates: ${MIN_LRS[*]}"
echo "Step-decay intervals (epochs): ${STEP_DECAY_EPOCHS[*]}"
echo "Step-decay gammas: ${STEP_GAMMAS[*]}"
echo "Inverse-sqrt warmup steps: ${WARMUP_STEPS[*]}"
echo "Metric factors: ${METRIC_FACTORS[*]}"
echo "Metric patiences: ${METRIC_PATIENCES[*]}"
echo "Metric thresholds: ${METRIC_THRESHOLDS[*]}"
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
    --epochs "${EPOCHS}"
    --scheduler_mode "${SCHEDULER_MODE}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATES[@]}"
    --min_lr "${MIN_LRS[@]}"
    --step_decay_epochs "${STEP_DECAY_EPOCHS[@]}"
    --step_gamma "${STEP_GAMMAS[@]}"
    --warmup_steps "${WARMUP_STEPS[@]}"
    --metric_factor "${METRIC_FACTORS[@]}"
    --metric_patience "${METRIC_PATIENCES[@]}"
    --metric_threshold "${METRIC_THRESHOLDS[@]}"
)
if [[ "${USE_GPU}" == true ]]; then
    BART_ARGS+=(--use_gpu)
fi

python -u bart_generation.py "${BART_ARGS[@]}"
