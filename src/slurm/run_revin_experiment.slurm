#!/bin/bash
#SBATCH --job-name=revin_experiment
#SBATCH --gres=gpu:1
#SBATCH --output=logs/revin_experiment_%j.out
#SBATCH --error=logs/revin_experiment_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src"

if [ "$TEST_MODE" = true ]; then
  DATASETS=(electricity solar)
  SETTINGS=(168:24 720:168)
  SEEDS=(1 2)
  MODELS=(patchtst)
else
  DATASETS=(etth1 electricity traffic solar weather exchange_rate)
  SETTINGS=(168:24 168:168 504:24 504:168 505:504 720:168 720:720)
  SEEDS=(1 2 3 4 5)
  MODELS=(dlinear patchtst)
fi

METHODS=(none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse revin_last_nmse revin_arcsinh_nmse)
EPOCHS="${EPOCHS:-10000}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/revin_experiment}"
SEEDS_CSV="$(IFS=,; echo "${SEEDS[*]}")"

method_args() {
  case "$1" in
    none_mse) ARGS=(normalization.name=none '~normalization.kwargs.affine' training.loss=mse) ;;
    standard_mse) ARGS=(normalization.name=standard '~normalization.kwargs.affine' training.loss=mse) ;;
    instance_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=mse) ;;
    instance_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=nmse) ;;
    revin_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=mse) ;;
    revin_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=nmse) ;;
    revin_last_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=last training.loss=nmse) ;;
    revin_arcsinh_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.transform=arcsinh training.loss=nmse) ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    L="${setting%%:*}"; H="${setting##*:}"
    for model in "${MODELS[@]}"; do
      for method in "${METHODS[@]}"; do
        method_args "$method"
        echo "$(date -Is) dataset=$dataset setting=$L-$H model=$model method=$method seeds=$SEEDS_CSV"
        srun python -m scripts.experiment \
          data.root="$DATA_ROOT" data.name="$dataset" data.eval_stride="$H" \
          task.lags="$L" task.horizon="$H" model.name="$model" \
          training.batch_size=256 training.lr=1e-5 training.epochs="$EPOCHS" \
          training.valid_eval_freq="$VALID_EVAL_FREQ" \
          training.logging_eval_freq="$LOGGING_EVAL_FREQ" \
          "${ARGS[@]}" seeds="[$SEEDS_CSV]" \
          output.dir="$OUT_ROOT/$dataset/${L}_${H}" \
          output.name="${model}_${method}"
      done
    done
  done
done

DATASET_ARG="$(IFS=,; echo "${DATASETS[*]}")"
for model in "${MODELS[@]}"; do
  TABLE_METHODS=()
  for method in "${METHODS[@]}"; do TABLE_METHODS+=("${model}_${method}"); done
  METHOD_ARG="$(IFS=,; echo "${TABLE_METHODS[*]}")"
  for split in test1 test2; do
    srun python -m utils.results "$OUT_ROOT" \
      --split "$split" --metric mse --datasets "$DATASET_ARG" \
      --methods "$METHOD_ARG" --show-std --decimals 2 \
      --output "$OUT_ROOT/results_${model}_${split}_mse.tex"
  done
done
