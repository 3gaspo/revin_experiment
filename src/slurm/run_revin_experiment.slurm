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

# Slurm front end: edit scheduler resources and the smoke/full switch here.
TEST_MODE="${TEST_MODE:-true}"
RUN_MODE="${RUN_MODE:-both}" # train, tables, or both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_MODE RUN_MODE
source "$SCRIPT_DIR/run_revin_experiment.sh"
