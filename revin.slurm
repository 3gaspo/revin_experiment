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

# Submit this file from the project root. Edit resources and the smoke/full switch here.
TEST_MODE="${TEST_MODE:-true}"
RUN_MODE="${RUN_MODE:-both}" # train, tables, or both

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TEST_MODE RUN_MODE
source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"
