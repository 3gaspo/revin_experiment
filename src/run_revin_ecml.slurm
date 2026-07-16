#!/bin/bash
#SBATCH --job-name=revin_ecml
#SBATCH --gres=gpu:1
#SBATCH --output=revin_ecml_%j.out
#SBATCH --error=revin_ecml_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src"

DATASETS=(electricity solar traffic synthetic)
MODELS=(patchtst dlinear)
SEEDS=(1 2 3 4 5)
METHODS=(none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse revin_last_nmse revin_arcsinh_nmse)
EPOCHS=1200
SHOW_STD=true
DATA_ROOT="$ROOT/datasets"
OUT_ROOT="$ROOT/outputs/revin_ecml"

settings() {
  if [ "$1" = synthetic ]; then
    echo "40 10;100 20;100 100"
  else
    echo "168 24;504 24;504 168;504 504"
  fi
}

method_args() {
  case "$1" in
    none_mse) ARGS=(normalization.name=none "~normalization.kwargs.affine" training.loss=mse) ;;
    standard_mse) ARGS=(normalization.name=standard "~normalization.kwargs.affine" training.loss=mse) ;;
    instance_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=mse) ;;
    instance_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=nmse) ;;
    revin_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=mse) ;;
    revin_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=nmse) ;;
    revin_last_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=last training.loss=nmse) ;;
    revin_arcsinh_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.transform=arcsinh training.loss=nmse) ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  IFS=';' read -ra DATASET_SETTINGS <<< "$(settings "$dataset")"
  for setting in "${DATASET_SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for model in "${MODELS[@]}"; do
      for method in "${METHODS[@]}"; do
        method_args "$method"
        for seed in "${SEEDS[@]}"; do
          echo "$(date -Is) dataset=$dataset setting=$L-$H model=$model method=$method seed=$seed"
          srun python src/scripts/experiment.py \
            data.root="$DATA_ROOT" \
            data.name="$dataset" \
            data.eval_stride="$H" \
            task.lags="$L" \
            task.horizon="$H" \
            model.name="$model" \
            training.epochs="$EPOCHS" \
            "${ARGS[@]}" \
            seed="$seed" \
            output.dir="$OUT_ROOT/$dataset/${L}_${H}" \
            output.name="${model}_${method}/seed_${seed}"
        done
      done
    done
  done
done

DATASET_ARG="$(IFS=,; echo "${DATASETS[*]}")"
TABLE_METHODS=()
for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    TABLE_METHODS+=("${model}_${method}")
  done
done
METHOD_ARG="$(IFS=,; echo "${TABLE_METHODS[*]}")"
STD_ARG=()
[ "$SHOW_STD" = true ] && STD_ARG=(--show-std)

for split in test1 test2; do
  srun python src/utils/results.py "$OUT_ROOT" \
    --split "$split" \
    --metric mse \
    --datasets "$DATASET_ARG" \
    --methods "$METHOD_ARG" \
    "${STD_ARG[@]}" \
    --output "$OUT_ROOT/results_${split}_mse.tex"
done

echo "$(date -Is) finished: $OUT_ROOT"
