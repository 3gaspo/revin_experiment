#!/bin/bash
# Configure and orchestrate the complete RevIN workflow.
# Submit ../../revin.slurm; source this implementation only for local debugging.
set -euo pipefail

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
log_section() { printf '\n%s %s\n' "$(date -Is)" "$*"; }
log_error() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
STAGES_SPEC="${STAGES:-train,tables}"
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

DEFAULT_SMALL_SETTINGS="168:24 336:48 504:168"
DEFAULT_FULL_SETTINGS="$DEFAULT_SMALL_SETTINGS 336:96 336:720"
DEFAULT_SETTINGS="$DEFAULT_FULL_SETTINGS"
DEFAULT_SEEDS="1 2 3"
DEFAULT_EPOCHS=10000
DEFAULT_STEPS=10000
DEFAULT_VALID_EVAL_FREQ=1000
DEFAULT_LOGGING_EVAL_FREQ=1000
DEFAULT_OUT_ROOT="$ROOT/outputs/revin_experiment"
DEFAULT_SKIP_COMPLETED=true

# The staged publication matrix isolates the questions supported by the current
# implementation. Core methods cross the four original forward-normalization
# choices with data-space and normalized-space MSE. Component methods add the
# independently removable statistics, robust statistics, and global MIN.
# Full/ultra additionally test centering/transform variants under both losses.
CORE_METHODS="none_mse none_nmse standard_mse standard_nmse instance_mse instance_nmse revin_mse revin_nmse"
COMPONENT_METHODS="$CORE_METHODS mean_only_mse mean_only_nmse scale_only_mse scale_only_nmse median_mad_mse median_mad_nmse min_mse min_nmse"
EXTENDED_METHODS="$COMPONENT_METHODS revin_last_mse revin_last_nmse revin_arcsinh_mse revin_arcsinh_nmse"
CORE_TABLE_METHODS="$CORE_METHODS"
COMPONENT_TABLE_METHODS="none_mse none_nmse mean_only_mse mean_only_nmse scale_only_mse scale_only_nmse instance_mse instance_nmse median_mad_mse median_mad_nmse"
MODULATION_TABLE_METHODS="instance_mse instance_nmse revin_mse revin_nmse min_mse min_nmse"
TRANSFORM_TABLE_METHODS="instance_mse instance_nmse revin_last_mse revin_last_nmse revin_arcsinh_mse revin_arcsinh_nmse"

case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_DATASETS="electricity"
    DEFAULT_SETTINGS="504:168"
    DEFAULT_SEEDS="1"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="standard_mse standard_nmse instance_mse instance_nmse mean_only_nmse scale_only_nmse median_mad_nmse min_nmse"
    DEFAULT_EPOCHS=2000
    DEFAULT_STEPS=2000
    DEFAULT_VALID_EVAL_FREQ=200
    DEFAULT_LOGGING_EVAL_FREQ=200
    DEFAULT_OUT_ROOT="$ROOT/outputs/revin_experiment_test"
    ;;
  small)
    DEFAULT_DATASETS="traffic electricity solar weather exchange_rate"
    DEFAULT_SETTINGS="$DEFAULT_SMALL_SETTINGS"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="$COMPONENT_METHODS"
    ;;
  full)
    DEFAULT_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 traffic electricity solar weather exchange_rate"
    DEFAULT_SETTINGS="$DEFAULT_FULL_SETTINGS"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="$EXTENDED_METHODS"
    ;;
  ultra)
    DEFAULT_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 traffic electricity solar weather exchange_rate"
    DEFAULT_SETTINGS="$DEFAULT_FULL_SETTINGS"
    DEFAULT_MODELS="dlinear patchtst"
    DEFAULT_METHODS="$EXTENDED_METHODS"
    ;;
  *) log_error "EXPERIMENT_MODE must be test, small, full, or ultra (got $EXPERIMENT_MODE)"; exit 2 ;;
esac
DATASETS_SPEC="${DATASETS:-$DEFAULT_DATASETS}"
SETTINGS_SPEC="${SETTINGS:-$DEFAULT_SETTINGS}"
SEEDS_SPEC="${SEEDS:-$DEFAULT_SEEDS}"
MODELS_SPEC="${MODELS:-$DEFAULT_MODELS}"
METHODS_SPEC="${METHODS:-$DEFAULT_METHODS}"
# By default, fixed-method summaries and validation/test selection include every
# scheduled method. Narrow these explicitly for a paper-specific comparison.
SUMMARY_METHODS_SPEC="${SUMMARY_METHODS:-$METHODS_SPEC}"
ORACLE_METHODS_SPEC="${ORACLE_METHODS:-$SUMMARY_METHODS_SPEC}"

# Lists may be space- or comma-separated. Hydra receives seeds as one list.
read -r -a DATASET_LIST <<< "${DATASETS_SPEC//,/ }"
read -r -a SETTING_LIST <<< "${SETTINGS_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEEDS_SPEC//,/ }"
read -r -a MODEL_LIST <<< "${MODELS_SPEC//,/ }"
read -r -a METHOD_LIST <<< "${METHODS_SPEC//,/ }"
read -r -a SUMMARY_SUFFIX_LIST <<< "${SUMMARY_METHODS_SPEC//,/ }"
read -r -a ORACLE_SUFFIX_LIST <<< "${ORACLE_METHODS_SPEC//,/ }"
read -r -a STAGE_LIST <<< "${STAGES_SPEC//,/ }"
TOTAL_CONFIGURATIONS=$((${#DATASET_LIST[@]} * ${#SETTING_LIST[@]} * ${#MODEL_LIST[@]} * ${#METHOD_LIST[@]}))
TOTAL_SEED_RUNS=$((TOTAL_CONFIGURATIONS * ${#SEED_LIST[@]}))

EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
STEPS="${STEPS:-$DEFAULT_STEPS}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-$DEFAULT_VALID_EVAL_FREQ}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-$DEFAULT_LOGGING_EVAL_FREQ}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EVAL_STRIDE="${EVAL_STRIDE:-horizon}"
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
SKIP_COMPLETED="${SKIP_COMPLETED:-$DEFAULT_SKIP_COMPLETED}"
GENERATE_SUMMARY="${GENERATE_SUMMARY:-true}"
STRICT_SUMMARY="${STRICT_SUMMARY:-true}"
BASELINE_METHOD="${BASELINE_METHOD:-standard_mse}"
SEEDS_CSV="$(IFS=,; echo "${SEED_LIST[*]}")"

stage_requested() {
  local wanted="$1" stage
  for stage in "${STAGE_LIST[@]}"; do
    [ "$stage" = "$wanted" ] && return 0
  done
  return 1
}
for stage in "${STAGE_LIST[@]}"; do
  case "$stage" in
    train|tables) ;;
    *) log_error "STAGES must contain only train,tables (got $STAGES_SPEC)"; exit 2 ;;
  esac
done
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
    none_nmse) ARGS=(normalization.name=none '~normalization.kwargs.affine' training.loss=nmse) ;;
    standard_mse) ARGS=(normalization.name=standard '~normalization.kwargs.affine' training.loss=mse) ;;
    standard_nmse) ARGS=(normalization.name=standard '~normalization.kwargs.affine' training.loss=nmse) ;;
    instance_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=mse) ;;
    instance_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false training.loss=nmse) ;;
    revin_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=mse) ;;
    revin_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=true training.loss=nmse) ;;
    mean_only_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.scale=none training.loss=mse) ;;
    mean_only_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.scale=none training.loss=nmse) ;;
    scale_only_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=none training.loss=mse) ;;
    scale_only_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=none training.loss=nmse) ;;
    median_mad_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=median +normalization.kwargs.scale=mad training.loss=mse) ;;
    median_mad_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=median +normalization.kwargs.scale=mad training.loss=nmse) ;;
    min_mse) ARGS=(normalization.name=min normalization.kwargs.affine=true training.loss=mse) ;;
    min_nmse) ARGS=(normalization.name=min normalization.kwargs.affine=true training.loss=nmse) ;;
    revin_last_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=last training.loss=mse) ;;
    revin_last_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.center=last training.loss=nmse) ;;
    revin_arcsinh_mse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.transform=arcsinh training.loss=mse) ;;
    revin_arcsinh_nmse) ARGS=(normalization.name=revin normalization.kwargs.affine=false +normalization.kwargs.transform=arcsinh training.loss=nmse) ;;
    *) log_error "unknown method=$1"; return 2 ;;
  esac
}

configuration_signature() {
  local dataset="$1" lags="$2" horizon="$3" model="$4" method="$5" stride="$6"
  RUN_SIGNATURE="v1|dataset=$dataset|lags=$lags|horizon=$horizon|model=$model|method=$method|epochs=$EPOCHS|steps=$STEPS|batch_size=$BATCH_SIZE|learning_rate=$LEARNING_RATE|eval_stride=$stride|valid_eval_freq=$VALID_EVAL_FREQ|logging_eval_freq=$LOGGING_EVAL_FREQ"
}

validate_setting() {
  if ! [[ "$1" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    log_error "setting must have L:H form (got $1)"
    return 2
  fi
}

pending_seeds() {
  local output="$1"
  local source_config="$2"
  local signature="$3"
  local seed
  local seed_root
  PENDING_SEED_LIST=()
  for seed in "${SEED_LIST[@]}"; do
    seed_root="$output/seed_$seed"
    if [ ! -s "$seed_root/results.json" ] ||
      [ ! -s "$seed_root/config.yaml" ] ||
      [ ! -s "$seed_root/dataset_config.json" ] ||
      [ ! -s "$seed_root/run.complete" ] ||
      [ "$(head -n 1 "$seed_root/run.complete" 2>/dev/null || true)" != "$signature|seed=$seed" ] ||
      { ! grep -Fq '"valid1"' "$seed_root/results.json" && [ ! -s "$seed_root/history.pt" ]; } ||
      { ! grep -Fq '"valid2"' "$seed_root/results.json" && [ ! -s "$seed_root/history.pt" ]; } ||
      ! grep -Fq '"window_anchor": "query_t"' "$seed_root/dataset_config.json" ||
      ! grep -Eq "^[[:space:]]+steps: $STEPS$" "$seed_root/config.yaml" ||
      { [ -f "$source_config" ] && [ "$source_config" -nt "$seed_root/results.json" ]; }; then
      PENDING_SEED_LIST+=("$seed")
    fi
  done
}

run_training() {
  local configuration_index=0
  local dataset data_root dataset_config setting L H stride model method output run_seeds_csv signature seed
  local dataset_args=()
  for dataset in "${DATASET_LIST[@]}"; do
    data_root="$(resolve_data_root "$dataset")"
    dataset_config="$data_root/$dataset/config.json"
    dataset_args=()
    for setting in "${SETTING_LIST[@]}"; do
      validate_setting "$setting"
      L="${setting%%:*}"
      H="${setting##*:}"
      stride="$H"
      if [ "$EVAL_STRIDE" != horizon ]; then stride="$EVAL_STRIDE"; fi
      for model in "${MODEL_LIST[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
          method_args "$method"
          output="$OUT_ROOT/$dataset/${L}_${H}/${model}_${method}"
          configuration_signature "$dataset" "$L" "$H" "$model" "$method" "$stride"
          signature="$RUN_SIGNATURE"
          PENDING_SEED_LIST=("${SEED_LIST[@]}")
          if [ "$SKIP_COMPLETED" = true ]; then pending_seeds "$output" "$dataset_config" "$signature"; fi
          if [ "${#PENDING_SEED_LIST[@]}" -eq 0 ]; then
            log "skip complete dataset=$dataset lags=$L horizon=$H model=$model method=$method seeds=$SEEDS_CSV"
          else
            run_seeds_csv="$(IFS=,; echo "${PENDING_SEED_LIST[*]}")"
            printf '\n%s configuration=%s dataset=%s lags=%s horizon=%s model=%s method=%s requested_seeds=%s run_seeds=%s batch_size=%s learning_rate=%s epochs=%s steps=%s eval_stride=%s valid_eval_frequency=%s logging_frequency=%s overrides=%s\n' \
              "$(date -Is)" "$((configuration_index + 1))/$TOTAL_CONFIGURATIONS" "$dataset" "$L" "$H" "$model" "$method" "$SEEDS_CSV" \
              "$run_seeds_csv" "$BATCH_SIZE" "$LEARNING_RATE" "$EPOCHS" "$STEPS" "$stride" "$VALID_EVAL_FREQ" "$LOGGING_EVAL_FREQ" "${ARGS[*]}"
            srun --ntasks=1 python -m scripts.experiment \
              data.root="$data_root" data.name="$dataset" data.eval_stride="$stride" \
              "${dataset_args[@]}" \
              task.lags="$L" task.horizon="$H" model.name="$model" \
              training.batch_size="$BATCH_SIZE" training.lr="$LEARNING_RATE" \
              training.epochs="$EPOCHS" \
              training.steps="$STEPS" \
              training.valid_eval_freq="$VALID_EVAL_FREQ" \
              training.logging_eval_freq="$LOGGING_EVAL_FREQ" \
              "${ARGS[@]}" seeds="[$run_seeds_csv]" \
              output.dir="$OUT_ROOT/$dataset/${L}_${H}" \
              output.name="${model}_${method}"
            for seed in "${PENDING_SEED_LIST[@]}"; do
              if [ ! -s "$output/seed_$seed/results.json" ]; then
                log_error "training completed without required result $output/seed_$seed/results.json"
                exit 1
              fi
              printf '%s\n' "$signature|seed=$seed" > "$output/seed_$seed/run.complete"
            done
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

filter_method_spec() {
  local spec="$1"
  local suffix
  local -a candidates
  FILTERED_SUFFIXES=()
  read -r -a candidates <<< "${spec//,/ }"
  for suffix in "${candidates[@]}"; do
    if contains_method "$suffix"; then FILTERED_SUFFIXES+=("$suffix"); fi
  done
}

run_tables() {
  local dataset_arg setting_arg method_arg summary_arg oracle_arg model split suffix
  local group group_index output table_spec summary_baseline
  local -a setting_dirs table_methods summary_methods oracle_methods result_args
  local -a table_groups table_specs
  setting_dirs=()
  for setting in "${SETTING_LIST[@]}"; do
    validate_setting "$setting"
    setting_dirs+=("${setting/:/_}")
  done
  dataset_arg="$(IFS=,; echo "${DATASET_LIST[*]}")"
  setting_arg="$(IFS=,; echo "${setting_dirs[*]}")"

  for model in "${MODEL_LIST[@]}"; do
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
    summary_baseline="${model}_${BASELINE_METHOD}"
    if ! contains_method "$BASELINE_METHOD" && [ "${#summary_methods[@]}" -gt 0 ]; then
      summary_baseline="${summary_methods[0]}"
    fi

    table_groups=(core components modulation transforms)
    table_specs=(
      "$CORE_TABLE_METHODS"
      "$COMPONENT_TABLE_METHODS"
      "$MODULATION_TABLE_METHODS"
      "$TRANSFORM_TABLE_METHODS"
    )

    for split in test1 test2; do
      for group_index in "${!table_groups[@]}"; do
        group="${table_groups[$group_index]}"
        table_spec="${table_specs[$group_index]}"
        if [ "$group" = components ] &&
          ! contains_method mean_only_mse && ! contains_method mean_only_nmse; then
          continue
        fi
        if [ "$group" = modulation ] &&
          ! contains_method min_mse && ! contains_method min_nmse; then
          continue
        fi
        if [ "$group" = transforms ] &&
          ! contains_method revin_last_mse && ! contains_method revin_last_nmse &&
          ! contains_method revin_arcsinh_mse && ! contains_method revin_arcsinh_nmse; then
          continue
        fi
        filter_method_spec "$table_spec"
        if [ "${#FILTERED_SUFFIXES[@]}" -eq 0 ]; then continue; fi
        table_methods=()
        for suffix in "${FILTERED_SUFFIXES[@]}"; do
          table_methods+=("${model}_${suffix}")
        done
        method_arg="$(IFS=,; echo "${table_methods[*]}")"
        output="$OUT_ROOT/results_${model}_${split}_${group}_mse.tex"
        if [ "$group" = core ]; then
          output="$OUT_ROOT/results_${model}_${split}_mse.tex"
        fi
        result_args=(
          "$OUT_ROOT" --split "$split" --metric mse
          --datasets "$dataset_arg" --settings "$setting_arg"
          --methods "$method_arg" --show-std --decimals 2
          --output "$output"
        )
        if [ "$group" = core ] && [ "${#oracle_methods[@]}" -gt 0 ]; then
          result_args+=(--selection-methods "$oracle_arg")
        fi
        if [ "$group" = core ] && [ "$GENERATE_SUMMARY" = true ] &&
          [ "${#summary_methods[@]}" -gt 0 ]; then
          result_args+=(
            --summary-output "$OUT_ROOT/summary_${model}_${split}_mse.json"
            --summary-methods "$summary_arg" --oracle-methods "$oracle_arg"
            --baseline-method "$summary_baseline"
            --expected-seeds "$SEEDS_CSV"
          )
          if [ "$STRICT_SUMMARY" = true ]; then result_args+=(--strict-summary); fi
        fi
        printf '\n%s table model=%s split=%s group=%s metric=mse output=%s\n' \
          "$(date -Is)" "$model" "$split" "$group" "$output"
        srun --ntasks=1 python -m utils.results "${result_args[@]}"
      done
    done
  done
}

verify_table_inputs() {
  local dataset data_root dataset_config setting L H stride model method seed seed_root
  for dataset in "${DATASET_LIST[@]}"; do
    data_root="$(resolve_data_root "$dataset")"
    dataset_config="$data_root/$dataset/config.json"
    for setting in "${SETTING_LIST[@]}"; do
      L="${setting%%:*}"
      H="${setting##*:}"
      stride="$EVAL_STRIDE"
      [ "$stride" != horizon ] || stride="$H"
      for model in "${MODEL_LIST[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
          configuration_signature "$dataset" "$L" "$H" "$model" "$method" "$stride"
          for seed in "${SEED_LIST[@]}"; do
            seed_root="$OUT_ROOT/$dataset/${L}_${H}/${model}_${method}/seed_$seed"
            if [ ! -s "$seed_root/results.json" ] || [ ! -s "$seed_root/run.complete" ] ||
              [ "$(head -n 1 "$seed_root/run.complete" 2>/dev/null || true)" != "$RUN_SIGNATURE|seed=$seed" ] ||
              { [ -f "$dataset_config" ] && [ "$dataset_config" -nt "$seed_root/results.json" ]; }; then
              log_error "missing completed input $seed_root"
              return 1
            fi
          done
        done
      done
    done
  done
  return 0
}

WORKFLOW_STATE_DIR="$OUT_ROOT/.workflow"
TABLE_INPUT_NAME=run.complete
TRAIN_STAGE_SIGNATURE="v1|mode=$EXPERIMENT_MODE|datasets=$DATASETS_SPEC|settings=$SETTINGS_SPEC|models=$MODELS_SPEC|methods=$METHODS_SPEC|seeds=$SEEDS_SPEC|epochs=$EPOCHS|steps=$STEPS|batch_size=$BATCH_SIZE|learning_rate=$LEARNING_RATE|eval_stride=$EVAL_STRIDE|valid_eval_freq=$VALID_EVAL_FREQ|logging_eval_freq=$LOGGING_EVAL_FREQ"
TABLE_STAGE_SIGNATURE="v2|mode=$EXPERIMENT_MODE|datasets=$DATASETS_SPEC|settings=$SETTINGS_SPEC|models=$MODELS_SPEC|methods=$METHODS_SPEC|seeds=$SEEDS_SPEC|summary_methods=$SUMMARY_METHODS_SPEC|oracle_methods=$ORACLE_METHODS_SPEC|baseline=$BASELINE_METHOD|strict=$STRICT_SUMMARY"
TABLE_REQUIRED_OUTPUTS=()
for model in "${MODEL_LIST[@]}"; do
  for split in test1 test2; do
    filter_method_spec "$CORE_TABLE_METHODS"
    if [ "${#FILTERED_SUFFIXES[@]}" -gt 0 ]; then
      TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_${split}_mse.tex")
      if [ "$GENERATE_SUMMARY" = true ]; then
        filter_method_spec "$SUMMARY_METHODS_SPEC"
        if [ "${#FILTERED_SUFFIXES[@]}" -gt 0 ]; then
          TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/summary_${model}_${split}_mse.json")
          TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/summary_${model}_${split}_mse.tex")
        fi
      fi
    fi
    if contains_method mean_only_mse || contains_method mean_only_nmse; then
      TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_${split}_components_mse.tex")
    fi
    if contains_method min_mse || contains_method min_nmse; then
      TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_${split}_modulation_mse.tex")
    fi
    if contains_method revin_last_mse || contains_method revin_last_nmse ||
      contains_method revin_arcsinh_mse || contains_method revin_arcsinh_nmse; then
      TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_${split}_transforms_mse.tex")
    fi
  done
done

log_section "workflow start kind=revin experiment_mode=$EXPERIMENT_MODE stages=$STAGES_SPEC skip_completed=$SKIP_COMPLETED configurations=$TOTAL_CONFIGURATIONS seed_runs=$TOTAL_SEED_RUNS datasets=$DATASETS_SPEC settings=$SETTINGS_SPEC models=$MODELS_SPEC methods=$METHODS_SPEC seeds=$SEEDS_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done kind=revin output=$OUT_ROOT"
