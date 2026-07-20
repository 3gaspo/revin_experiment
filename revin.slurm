#!/bin/bash
#SBATCH --job-name=revin
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

# Submit from the project root. Select test, small, or large here or at submission.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-small}"
RUN_MODE="${RUN_MODE:-both}" # train, tables, or both

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE RUN_MODE
source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"
