#!/bin/bash
# Configure and orchestrate the complete RevIN workflow.
# Submit one root family front; source this implementation only for local debugging.
set -euo pipefail

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
log_section() { printf '\n%s %s\n' "$(date -Is)" "$*"; }
log_error() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-core}"
STAGES_SPEC="${STAGES:-train,tables}"
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
LOGS_ROOT="${LOGS_ROOT:-$ROOT/logs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT/outputs}"
mkdir -p "$LOGS_ROOT" "$OUTPUTS_ROOT"

VENV_ACTIVATE="${VENV_ACTIVATE:-$ROOT/.venv/bin/activate}"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
elif [ -z "${VIRTUAL_ENV:-}" ]; then
  log_error "no active environment and $VENV_ACTIVATE does not exist"
  exit 1
fi
export PYTHONPATH="$ROOT/src"

DEFAULT_SETTINGS="168:24 336:48 504:168 336:96 336:720"
DEFAULT_SEEDS="1 2 3"
DEFAULT_EPOCHS=10000
DEFAULT_STEPS=10000
DEFAULT_VALID_EVAL_FREQ=1000
DEFAULT_LOGGING_EVAL_FREQ=1000
DEFAULT_OUT_ROOT="$OUTPUTS_ROOT/$EXPERIMENT_FAMILY"
DEFAULT_SKIP_COMPLETED=true

# Publication methods are split into independent fronts. They share the same
# run directory and per-seed completion contract, but keep family-specific
# workflow markers and tables.
CORE_METHODS="none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse"
NMSE_METHODS="none_nmse standard_nmse"
EXOTIC_METHODS="mean_only_mse mean_only_nmse scale_only_mse scale_only_nmse median_mad_mse median_mad_nmse revin_last_mse revin_last_nmse revin_arcsinh_mse revin_arcsinh_nmse"
MIN_METHODS="min_mse min_nmse"
TEST_METHODS="standard_mse instance_mse instance_nmse min_nmse"

case "$EXPERIMENT_FAMILY" in
  core) FAMILY_METHODS="$CORE_METHODS" ;;
  nmse) FAMILY_METHODS="$NMSE_METHODS" ;;
  exotic) FAMILY_METHODS="$EXOTIC_METHODS" ;;
  min) FAMILY_METHODS="$MIN_METHODS" ;;
  *) log_error "EXPERIMENT_FAMILY must be core, nmse, exotic, or min (got $EXPERIMENT_FAMILY)"; exit 2 ;;
esac

case "$EXPERIMENT_MODE" in
  test)
    if [ "$EXPERIMENT_FAMILY" != core ]; then
      log_error "test mode is available only through the core revin.slurm front"
      exit 2
    fi
    DEFAULT_DATASETS="electricity"
    DEFAULT_SETTINGS="504:168"
    DEFAULT_SEEDS="1"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="$TEST_METHODS"
    DEFAULT_EPOCHS=2000
    DEFAULT_STEPS=2000
    DEFAULT_VALID_EVAL_FREQ=200
    DEFAULT_LOGGING_EVAL_FREQ=200
    ;;
  small)
    DEFAULT_DATASETS="traffic electricity solar"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="$FAMILY_METHODS"
    ;;
  full)
    DEFAULT_DATASETS="traffic electricity solar weather exchange_rate ETTh1 ETTh2 ETTm1 ETTm2"
    DEFAULT_MODELS="patchtst"
    DEFAULT_METHODS="$FAMILY_METHODS"
    ;;
  ultra)
    DEFAULT_DATASETS="traffic electricity solar weather exchange_rate ETTh1 ETTh2 ETTm1 ETTm2"
    DEFAULT_MODELS="dlinear patchtst"
    DEFAULT_METHODS="$FAMILY_METHODS"
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
TABLE_OUTPUT_ROOT="${TABLE_OUTPUT_ROOT:-$OUTPUTS_ROOT/reports/$EXPERIMENT_FAMILY/$EXPERIMENT_MODE}"
SKIP_COMPLETED="${SKIP_COMPLETED:-$DEFAULT_SKIP_COMPLETED}"
RUN_CONFLICT_POLICY="${RUN_CONFLICT_POLICY:-overwrite_exact}"
FORCE_RUN="${FORCE_RUN:-false}"
TABLE_CONFIG_POLICY="${TABLE_CONFIG_POLICY:-distinct}"
TABLE_REPEAT_POLICY="${TABLE_REPEAT_POLICY:-selected}"
if [ "$EXPERIMENT_MODE" = test ]; then TABLE_PURPOSE="${TABLE_PURPOSE:-smoke}"; else TABLE_PURPOSE="${TABLE_PURPOSE:-publication}"; fi
EXPERIMENT_LAUNCH_ID="${EXPERIMENT_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export EXPERIMENT_LAUNCH_ID
ACTIVE_STAGE=""
ACTIVE_TASK=""

stage_start() {
  ACTIVE_STAGE="$1"
  log_section "stage $ACTIVE_STAGE started"
}

stage_complete() {
  log_section "stage $ACTIVE_STAGE completed status=success"
  ACTIVE_STAGE=""
}

task_start() {
  ACTIVE_TASK="$*"
  log "task $ACTIVE_TASK started"
}

task_complete() {
  local status="$1"
  log "task $ACTIVE_TASK completed status=$status"
  ACTIVE_TASK=""
}

revin_on_exit() {
  local status=$?
  trap - EXIT
  if [ -n "$ACTIVE_TASK" ]; then
    log_error "task $ACTIVE_TASK completed status=failed exit_code=$status"
  fi
  if [ -n "$ACTIVE_STAGE" ]; then
    log_error "stage $ACTIVE_STAGE completed status=failed exit_code=$status"
  fi
  if [ "$status" -ne 0 ]; then
    python -m pipeline.runs interrupt-launch --root "$OUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" || true
  elif python -m pipeline.runs complete-launch --root "$OUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null; then
    :
  else
    status=$?
  fi
  if [ "$status" -eq 0 ]; then
    log_section "workflow completed status=success exit_code=0"
  else
    log_error "workflow completed status=failed exit_code=$status"
  fi
  exit "$status"
}
trap revin_on_exit EXIT
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

validate_setting() {
  if ! [[ "$1" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    log_error "setting must have L:H form (got $1)"
    return 2
  fi
}

run_training() {
  local configuration_index=0
  local dataset data_root dataset_config setting L H stride model method run_seeds_csv seed normalization loss
  local identity_root run_dir run_action run_signature purpose
  local -a allocation_args pending required_artifacts
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
          loss="${method##*_}"
          normalization="${method%_$loss}"
          task_start "$((configuration_index + 1))/$TOTAL_CONFIGURATIONS configuration dataset=$dataset setting=$setting model=$model method=$method"
          identity_root="$OUT_ROOT/$dataset/${L}_${H}/${model,,}/${normalization,,}/${loss,,}"
          if [ "$EXPERIMENT_MODE" = test ]; then purpose=smoke; else purpose=publication; fi
          allocation_args=(
            --identity-root "$identity_root" --project revin_experiment --workflow "$EXPERIMENT_FAMILY"
            --dataset "$dataset" --lookback "$L" --horizon "$H" --backbone "$model"
            --model-config-order normalization,loss --model-config "normalization=$normalization" --model-config "loss=$loss"
            --pipeline-config "training.epochs=$EPOCHS" --pipeline-config "training.steps=$STEPS"
            --pipeline-config "training.batch_size=$BATCH_SIZE" --pipeline-config "training.learning_rate=$LEARNING_RATE"
            --pipeline-config "evaluation.stride=$stride" --pipeline-config "training.valid_eval_freq=$VALID_EVAL_FREQ"
            --pipeline-config "training.logging_eval_freq=$LOGGING_EVAL_FREQ" --pipeline-config "hydra_overrides=${ARGS[*]}"
            --runtime-config training.device=gpu --runtime-config "slurm.job_id=${SLURM_JOB_ID:-}"
            --purpose "$purpose" --mode "$EXPERIMENT_MODE" --display-name "${model}_${method}"
            --row-config normalization --column-config loss
            --policy "$RUN_CONFLICT_POLICY" --skip-completed "$SKIP_COMPLETED" --force "$FORCE_RUN"
            --launch-id "$EXPERIMENT_LAUNCH_ID"
          )
          if [ "${dataset,,}" = weather ]; then
            allocation_args+=(--pipeline-config "data.missing_values=zero")
          fi
          for seed in "${SEED_LIST[@]}"; do allocation_args+=(--seed "$seed"); done
          if [ -f "$dataset_config" ]; then allocation_args+=(--input "dataset_config=$dataset_config"); fi
          if [ -n "${RUN_INDEX:-}" ]; then allocation_args+=(--run-index "$RUN_INDEX"); fi
          IFS=$'\t' read -r run_dir run_action run_signature < <(python -m pipeline.runs allocate "${allocation_args[@]}")
          if [ "$run_action" = skip ]; then
            log "skip complete dataset=$dataset lags=$L horizon=$H model=$model method=$method run=$run_dir"
            task_complete skipped
          else
            run_seeds_csv="$(python -m pipeline.runs pending-seeds --run-dir "$run_dir")"
            IFS=, read -ra pending <<< "$run_seeds_csv"
            for seed in "${pending[@]}"; do python -m pipeline.runs status --run-dir "$run_dir" --status running --seed "$seed"; done
            printf '\n%s configuration=%s dataset=%s lags=%s horizon=%s model=%s normalization=%s loss=%s requested_seeds=%s run_seeds=%s run=%s computation_signature=%s batch_size=%s learning_rate=%s epochs=%s steps=%s eval_stride=%s valid_eval_frequency=%s logging_frequency=%s overrides=%s\n' \
              "$(date -Is)" "$((configuration_index + 1))/$TOTAL_CONFIGURATIONS" "$dataset" "$L" "$H" "$model" "$normalization" "$loss" \
              "$SEEDS_CSV" "$run_seeds_csv" "$run_dir" "$run_signature" "$BATCH_SIZE" "$LEARNING_RATE" "$EPOCHS" "$STEPS" "$stride" "$VALID_EVAL_FREQ" "$LOGGING_EVAL_FREQ" "${ARGS[*]}"
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
              output.dir="$run_dir" output.name= \
              hydra.run.dir="$run_dir/hydra/$EXPERIMENT_LAUNCH_ID"
            required_artifacts=()
            for seed in "${pending[@]}"; do
              if [ ! -s "$run_dir/seed_$seed/results.json" ] || [ ! -s "$run_dir/seed_$seed/config.yaml" ] || [ ! -s "$run_dir/seed_$seed/dataset_config.json" ]; then
                log_error "training completed without required results in $run_dir/seed_$seed"
                exit 1
              fi
              python -m pipeline.runs status --run-dir "$run_dir" --status ready --seed "$seed" \
                --artifact "seed_$seed/results.json" --artifact "seed_$seed/config.yaml" --artifact "seed_$seed/dataset_config.json"
            done
            for seed in "${SEED_LIST[@]}"; do
              required_artifacts+=(--artifact "seed_$seed/results.json" --artifact "seed_$seed/config.yaml" --artifact "seed_$seed/dataset_config.json")
            done
            python -m pipeline.runs ready --run-dir "$run_dir" "${required_artifacts[@]}"
            python -m pipeline.runs complete --run-dir "$run_dir" --launch-id "$EXPERIMENT_LAUNCH_ID"
            task_complete success
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
  local dataset_arg setting_arg method_arg summary_arg oracle_arg model split suffix pair
  local output summary_baseline
  local current=0
  local total=$((${#MODEL_LIST[@]} * 2))
  local -a setting_dirs table_methods summary_methods oracle_methods result_args
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

    for split in test1 test2; do
      table_methods=()
      for suffix in "${METHOD_LIST[@]}"; do
        table_methods+=("${model}_${suffix}")
      done
      method_arg="$(IFS=,; echo "${table_methods[*]}")"
      output="$TABLE_OUTPUT_ROOT/results_${model}_${split}_mse.tex"
      result_args=(
        "$OUT_ROOT" --split "$split" --metric mse
        --datasets "$dataset_arg" --settings "$setting_arg"
        --methods "$method_arg" --show-std --decimals 2
        --config-policy "$TABLE_CONFIG_POLICY" --repeat-policy "$TABLE_REPEAT_POLICY"
        --purpose "$TABLE_PURPOSE"
        --output "$output"
      )
      if [ -n "${TABLE_PIPELINE_CONFIGS:-}" ]; then
        for pair in ${TABLE_PIPELINE_CONFIGS}; do result_args+=(--pipeline-config "$pair"); done
      fi
      if [ -n "${TABLE_PURPOSE:-}" ]; then result_args+=(--purpose "$TABLE_PURPOSE"); fi
      if [ "${#oracle_methods[@]}" -gt 0 ]; then
        result_args+=(--selection-methods "$oracle_arg")
      fi
      if [ "$GENERATE_SUMMARY" = true ] && [ "${#summary_methods[@]}" -gt 0 ]; then
        result_args+=(
          --summary-output "$TABLE_OUTPUT_ROOT/summary_${model}_${split}_mse.json"
          --summary-methods "$summary_arg" --oracle-methods "$oracle_arg"
          --baseline-method "$summary_baseline"
          --expected-seeds "$SEEDS_CSV"
        )
        if [ "$STRICT_SUMMARY" = true ]; then result_args+=(--strict-summary); fi
      fi
      printf '\n%s table family=%s model=%s split=%s metric=mse output=%s\n' \
        "$(date -Is)" "$EXPERIMENT_FAMILY" "$model" "$split" "$output"
      current=$((current + 1))
      task_start "$current/$total table model=$model split=$split metric=mse"
      srun --ntasks=1 python -m scripts.report "${result_args[@]}"
      task_complete success
    done
  done
}

TABLE_REQUIRED_OUTPUTS=()
for model in "${MODEL_LIST[@]}"; do
  for split in test1 test2; do
    TABLE_REQUIRED_OUTPUTS+=("$TABLE_OUTPUT_ROOT/results_${model}_${split}_mse.tex")
    if [ "$GENERATE_SUMMARY" = true ]; then
      TABLE_REQUIRED_OUTPUTS+=("$TABLE_OUTPUT_ROOT/summary_${model}_${split}_mse.json")
      TABLE_REQUIRED_OUTPUTS+=("$TABLE_OUTPUT_ROOT/summary_${model}_${split}_mse.tex")
    fi
  done
done

log_section "workflow start kind=revin family=$EXPERIMENT_FAMILY experiment_mode=$EXPERIMENT_MODE stages=$STAGES_SPEC skip_completed=$SKIP_COMPLETED configurations=$TOTAL_CONFIGURATIONS seed_runs=$TOTAL_SEED_RUNS datasets=$DATASETS_SPEC settings=$SETTINGS_SPEC models=$MODELS_SPEC methods=$METHODS_SPEC seeds=$SEEDS_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done kind=revin family=$EXPERIMENT_FAMILY output=$OUT_ROOT"
