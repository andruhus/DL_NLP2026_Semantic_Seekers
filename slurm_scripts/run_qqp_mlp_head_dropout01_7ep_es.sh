#!/bin/bash
#SBATCH --job-name=qqp_mlp_d01_7es
#SBATCH --partition=grete:shared
#SBATCH -G 1g.10gb
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_files/slurm-%x-%j.out
#SBATCH --error=slurm_files/slurm-%x-%j.err

source ~/.bashrc
conda activate dnlp

cd "$SLURM_SUBMIT_DIR"

mkdir -p models predictions/bert slurm_files
mkdir -p experiments/qqp_mlp_head/dropout01_7ep_es_predictions
mkdir -p experiments/qqp_mlp_head/dropout01_7ep_es_checkpoints

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Working directory after cd: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
echo "Git status:"
git status --short

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 'none')"

echo "Cleaning old QQP outputs..."
rm -f predictions/bert/quora-paraphrase-dev-output.csv
rm -f predictions/bert/quora-paraphrase-test-output.csv
rm -f models/finetune-7-1e-05-qqp.pt

python multitask_classifier.py \
  --task qqp \
  --seed 11711 \
  --option finetune \
  --use_gpu \
  --local_files_only \
  --epochs 7 \
  --batch_size 8 \
  --lr 1e-5 \
  --hidden_dropout_prob 0.1 \
  --early_stopping \
  --patience 2

echo "Training finished at: $(date)"

EXP_PRED_DIR="experiments/qqp_mlp_head/dropout01_7ep_es_predictions"
EXP_CKPT_DIR="experiments/qqp_mlp_head/dropout01_7ep_es_checkpoints"

cp predictions/bert/quora-paraphrase-dev-output.csv \
  "$EXP_PRED_DIR/quora_mlp_head_dropout01_7ep_es_job${SLURM_JOB_ID}_dev.csv"

cp predictions/bert/quora-paraphrase-test-output.csv \
  "$EXP_PRED_DIR/quora_mlp_head_dropout01_7ep_es_job${SLURM_JOB_ID}_test.csv"

if [ -f models/finetune-7-1e-05-qqp.pt ]; then
  cp models/finetune-7-1e-05-qqp.pt \
    "$EXP_CKPT_DIR/qqp_mlp_head_dropout01_7ep_es_job${SLURM_JOB_ID}.pt"
fi

echo "Artifacts copied at: $(date)"
echo "Prediction files:"
ls -lh "$EXP_PRED_DIR"

echo "Checkpoint files:"
ls -lh "$EXP_CKPT_DIR"

echo "Job ended at: $(date)"
