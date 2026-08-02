#!/bin/bash
#SBATCH --job-name=exotic
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

EXPERIMENT_MODE="${EXPERIMENT_MODE:-small}"
EXPERIMENT_FAMILY=exotic
STAGES="${STAGES:-train,tables}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY STAGES
source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"
