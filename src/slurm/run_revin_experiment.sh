#!/bin/bash
# Enumerate RevIN experiment configurations, run training, and build result tables.
# Submit ../../revin.slurm; source this implementation only for local debugging.
set -euo pipefail

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
log_section() { printf '\n%s %s\n' "$(date -Is)" "$*"; }
log_error() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

TEST_MODE="${TEST_MODE:-true}"
RUN_MODE="${RUN_MODE:-both}" # train, tables, or both
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

VENV_ACTIVATE="${VENV_ACTIVATE:-$ROOT/.venv/bin/activate}"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
elif [ -z "${VIRTUAL_ENV:-}" ]; then
  log_error "no active environment and $VENV_ACTIVATE does not exist"
  exit 1
fi
export PYTHONPATH="$ROOT/src"

if [ "$TEST_MODE" = true ]; then
  DEFAULT_DATASETS="electricity solar"
  DEFAULT_SETTINGS="168:24 720:168"
  DEFAULT_SEEDS="1 2"
  DEFAULT_MODELS="patchtst"
  DEFAULT_EPOCHS=20
  DEFAULT_VALID_EVAL_FREQ=10
  DEFAULT_LOGGING_EVAL_FREQ=10
  DEFAULT_OUT_ROOT="$ROOT/outputs/revin_experiment_test"
else
  DEFAULT_DATASETS="etth1 electricity traffic solar weather exchange_rate"
  DEFAULT_SETTINGS="168:24 168:168 504:24 504:168 504:504 720:168 720:720"
  DEFAULT_SEEDS="1 2 3 4 5"
  DEFAULT_MODELS="dlinear patchtst"
  DEFAULT_EPOCHS=10000
  DEFAULT_VALID_EVAL_FREQ=1000
  DEFAULT_LOGGING_EVAL_FREQ=1000
  DEFAULT_OUT_ROOT="$ROOT/outputs/revin_experiment"
fi

DEFAULT_METHODS="none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse revin_last_nmse revin_arcsinh_nmse"
DATASETS_SPEC="${DATASETS:-$DEFAULT_DATASETS}"
SETTINGS_SPEC="${SETTINGS:-$DEFAULT_SETTINGS}"
SEEDS_SPEC="${SEEDS:-$DEFAULT_SEEDS}"
MODELS_SPEC="${MODELS:-$DEFAULT_MODELS}"
METHODS_SPEC="${METHODS:-$DEFAULT_METHODS}"
SUMMARY_METHODS_SPEC="${SUMMARY_METHODS:-standard_mse instance_nmse revin_nmse}"
ORACLE_METHODS_SPEC="${ORACLE_METHODS:-standard_mse revin_nmse}"

# Lists may be space- or comma-separated. Hydra receives seeds as one list.
read -r -a DATASET_LIST <<< "${DATASETS_SPEC//,/ }"
read -r -a SETTING_LIST <<< "${SETTINGS_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEEDS_SPEC//,/ }"
read -r -a MODEL_LIST <<< "${MODELS_SPEC//,/ }"
read -r -a METHOD_LIST <<< "${METHODS_SPEC//,/ }"
read -r -a SUMMARY_SUFFIX_LIST <<< "${SUMMARY_METHODS_SPEC//,/ }"
read -r -a ORACLE_SUFFIX_LIST <<< "${ORACLE_METHODS_SPEC//,/ }"

EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-$DEFAULT_VALID_EVAL_FREQ}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-$DEFAULT_LOGGING_EVAL_FREQ}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EVAL_STRIDE="${EVAL_STRIDE:-horizon}"
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
SKIP_COMPLETED="${SKIP_COMPLETED:-false}"
GENERATE_SUMMARY="${GENERATE_SUMMARY:-true}"
STRICT_SUMMARY="${STRICT_SUMMARY:-true}"
BASELINE_METHOD="${BASELINE_METHOD:-standard_mse}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SEEDS_CSV="$(IFS=,; echo "${SEED_LIST[*]}")"

case "$RUN_MODE" in
  train|tables|both) ;;
  *) log_error "RUN_MODE must be train, tables, or both (got $RUN_MODE)"; exit 2 ;;
esac
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  log_error "SHARD_COUNT must be a positive integer"
  exit 2
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
  log_error "SHARD_INDEX must be in [0, SHARD_COUNT)"
  exit 2
fi
if [ "$RUN_MODE" = both ] && [ "$SHARD_COUNT" -gt 1 ]; then
  log_error "use RUN_MODE=train for a sharded array, then a dependent RUN_MODE=tables job"
  exit 2
fi
if [ "$EVAL_STRIDE" != horizon ] && ! [[ "$EVAL_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  log_error "EVAL_STRIDE must be 'horizon' or a positive integer"
  exit 2
fi

# On another machine, set DATA_ROOT to the available dataset directory or edit
# these candidates. DLinear and PatchTST do not use pretrained weights.
DATA_ROOT_SPEC="${DATA_ROOT:-}"
resolve_data_root() {
  local dataset="$1"
  local candidate
  if [ -n "$DATA_ROOT_SPEC" ]; then
    if [ -f "$DATA_ROOT_SPEC/$dataset/$dataset.csv" ]; then
      printf '%s\n' "$DATA_ROOT_SPEC"
      return
    fi
    log_error "missing $DATA_ROOT_SPEC/$dataset/$dataset.csv"
    return 1
  fi
  for candidate in "$ROOT/datasets" "$ROOT/../datasets" "$ROOT/../../../datasets"; do
    if [ -f "$candidate/$dataset/$dataset.csv" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  log_error "cannot find $dataset/$dataset.csv; set DATA_ROOT explicitly"
  return 1
}

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
    *) log_error "unknown method=$1"; return 2 ;;
  esac
}

validate_setting() {
  if ! [[ "$1" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    log_error "setting must have L:H form (got $1)"
    return 2
  fi
}

configuration_complete() {
  local output="$1"
  local seed
  for seed in "${SEED_LIST[@]}"; do
    [ -f "$output/seed_$seed/results.json" ] || return 1
  done
}

run_training() {
  local configuration_index=0
  local dataset data_root setting L H stride model method output
  for dataset in "${DATASET_LIST[@]}"; do
    data_root="$(resolve_data_root "$dataset")"
    for setting in "${SETTING_LIST[@]}"; do
      validate_setting "$setting"
      L="${setting%%:*}"
      H="${setting##*:}"
      stride="$H"
      if [ "$EVAL_STRIDE" != horizon ]; then stride="$EVAL_STRIDE"; fi
      for model in "${MODEL_LIST[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
          if [ $((configuration_index % SHARD_COUNT)) -eq "$SHARD_INDEX" ]; then
            method_args "$method"
            output="$OUT_ROOT/$dataset/${L}_${H}/${model}_${method}"
            if [ "$SKIP_COMPLETED" = true ] && configuration_complete "$output"; then
              log "skip complete dataset=$dataset lags=$L horizon=$H model=$model method=$method"
            else
              printf '\n%s configuration=%s dataset=%s lags=%s horizon=%s model=%s method=%s seeds=%s batch_size=%s learning_rate=%s epochs=%s eval_stride=%s valid_eval_frequency=%s logging_frequency=%s overrides=%s\n' \
                "$(date -Is)" "$configuration_index" "$dataset" "$L" "$H" "$model" "$method" "$SEEDS_CSV" \
                "$BATCH_SIZE" "$LEARNING_RATE" "$EPOCHS" "$stride" "$VALID_EVAL_FREQ" "$LOGGING_EVAL_FREQ" "${ARGS[*]}"
              srun --ntasks=1 python -m scripts.experiment \
                data.root="$data_root" data.name="$dataset" data.eval_stride="$stride" \
                task.lags="$L" task.horizon="$H" model.name="$model" \
                training.batch_size="$BATCH_SIZE" training.lr="$LEARNING_RATE" \
                training.epochs="$EPOCHS" \
                training.valid_eval_freq="$VALID_EVAL_FREQ" \
                training.logging_eval_freq="$LOGGING_EVAL_FREQ" \
                "${ARGS[@]}" seeds="[$SEEDS_CSV]" \
                output.dir="$OUT_ROOT/$dataset/${L}_${H}" \
                output.name="${model}_${method}"
            fi
          fi
          configuration_index=$((configuration_index + 1))
        done
      done
    done
  done
}

contains_method() {
  local wanted="$1"
  local method
  for method in "${METHOD_LIST[@]}"; do
    [ "$method" = "$wanted" ] && return 0
  done
  return 1
}

run_tables() {
  local dataset_arg setting_arg method_arg summary_arg oracle_arg model split suffix
  local -a setting_dirs table_methods summary_methods oracle_methods result_args
  setting_dirs=()
  for setting in "${SETTING_LIST[@]}"; do
    validate_setting "$setting"
    setting_dirs+=("${setting/:/_}")
  done
  dataset_arg="$(IFS=,; echo "${DATASET_LIST[*]}")"
  setting_arg="$(IFS=,; echo "${setting_dirs[*]}")"

  for model in "${MODEL_LIST[@]}"; do
    table_methods=()
    for suffix in "${METHOD_LIST[@]}"; do table_methods+=("${model}_${suffix}"); done
    method_arg="$(IFS=,; echo "${table_methods[*]}")"

    summary_methods=()
    for suffix in "${SUMMARY_SUFFIX_LIST[@]}"; do
      if contains_method "$suffix"; then summary_methods+=("${model}_${suffix}"); fi
    done
    oracle_methods=()
    for suffix in "${ORACLE_SUFFIX_LIST[@]}"; do
      if contains_method "$suffix"; then oracle_methods+=("${model}_${suffix}"); fi
    done
    summary_arg="$(IFS=,; echo "${summary_methods[*]}")"
    oracle_arg="$(IFS=,; echo "${oracle_methods[*]}")"

    for split in test1 test2; do
      result_args=(
        "$OUT_ROOT" --split "$split" --metric mse
        --datasets "$dataset_arg" --settings "$setting_arg"
        --methods "$method_arg" --show-std --decimals 2
        --output "$OUT_ROOT/results_${model}_${split}_mse.tex"
      )
      if [ "$GENERATE_SUMMARY" = true ] && [ "${#summary_methods[@]}" -gt 0 ]; then
        result_args+=(
          --summary-output "$OUT_ROOT/summary_${model}_${split}_mse.json"
          --summary-methods "$summary_arg" --oracle-methods "$oracle_arg"
          --baseline-method "${model}_${BASELINE_METHOD}"
          --expected-seeds "$SEEDS_CSV"
        )
        if [ "$STRICT_SUMMARY" = true ]; then result_args+=(--strict-summary); fi
      fi
      printf '\n%s table model=%s split=%s metric=mse output=%s\n' \
        "$(date -Is)" "$model" "$split" "$OUT_ROOT/results_${model}_${split}_mse.tex"
      srun --ntasks=1 python -m utils.results "${result_args[@]}"
    done
  done
}

log_section "job start kind=revin run_mode=$RUN_MODE test_mode=$TEST_MODE datasets=$DATASETS_SPEC settings=$SETTINGS_SPEC models=$MODELS_SPEC methods=$METHODS_SPEC seeds=$SEEDS_SPEC shard=$SHARD_INDEX/$SHARD_COUNT"
if [ "$RUN_MODE" = train ] || [ "$RUN_MODE" = both ]; then run_training; fi
if [ "$RUN_MODE" = tables ] || [ "$RUN_MODE" = both ]; then run_tables; fi
log_section "job done kind=revin output=$OUT_ROOT"
