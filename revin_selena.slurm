#!/bin/bash
#SBATCH --job-name=s_revin
#SBATCH --gres=gpu:1
#SBATCH --output=logs_selena/%x_%j.out
#SBATCH --error=logs_selena/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --wckey=P12CU:DATASCIENCE

# Complete resumable workflow. Select test, small, full, or ultra here or at submission.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY=core
STAGES="${STAGES:-train,tables}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"
LOGS_ROOT="$PROJECT_ROOT/logs_selena"
EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY STAGES OUTPUTS_ROOT LOGS_ROOT EXPERIMENT_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"

