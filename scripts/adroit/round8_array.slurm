#!/bin/bash
# Round 8 as a Slurm job array: one task per (cell, fold, seed).
#
# The point of moving this to a cluster is width, not speed per run. Each run is
# ~5 minutes on 2 cores, so 50 tasks running concurrently finishes the whole
# 5-fold x 5-seed x 2-cell design in about the time one run takes -- which makes
# the "more seeds" question (the actual weakness of round 4's result: a 0.013
# margin on a 0.173 seed spread) cheap to settle properly.
#
# Submit from the repo root on a login node, after scripts/adroit/setup_env.sh:
#     sbatch scripts/adroit/round8_array.slurm
#     squeue -u $USER
#
# Array size must match CELLS x FOLDS x SEEDS below: 2 x 5 x 5 = 50 -> 0-49.
# Change SEEDS and the --array range together, or tasks silently go unrun.
#
#SBATCH --job-name=rl-round8
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/round8_%A_%a.out
#SBATCH --error=slurm_logs/round8_%A_%a.out
#
# Deliberately no --partition / --account: those differ per cluster and per
# allocation, so this uses the default. Check `sinfo -s` and add
#   #SBATCH --partition=<name>
# if the default queue is wrong for your account. No --gres=gpu on purpose (see
# setup_env.sh: a GPU does not help this workload).

# Not 'set -u': this sources .venv/bin/activate, whose own code touches unset
# variables (PS1) in a non-interactive shell -- under 'set -u' that would abort
# every array task before training starts. Explicit :- defaults are used below
# wherever a Slurm variable might be missing.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p slurm_logs

CELLS=(baseline bootstrap)
FOLDS=(fold1_test2020 fold2_test2021 fold3_test2022 fold4_test2023 fold5_test2024)
# Override at submit time; sbatch exports the submitting environment by default.
# The --array range must match 2 x 5 x len(SEEDS):
#     SEEDS="0 1 2 3 4"                     -> sbatch --array=0-49  (the default)
#     SEEDS="0 1 2 3 4 5 6 7 8 9"           -> sbatch --array=0-99
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-0 1 2 3 4})
CONFIG=configs/attention_frozen_excess_lowlr.yaml

n_seeds=${#SEEDS[@]}
n_folds=${#FOLDS[@]}
n_total=$(( ${#CELLS[@]} * n_folds * n_seeds ))

i="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$i" -ge "$n_total" ]; then
  echo "task $i is outside the ${n_total}-task design; nothing to do"
  exit 0
fi

seed=${SEEDS[$(( i % n_seeds ))]}
rest=$(( i / n_seeds ))
fold=${FOLDS[$(( rest % n_folds ))]}
cell=${CELLS[$(( rest / n_folds ))]}

# One task, two cores: keep BLAS from oversubscribing the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== task $i/$((n_total - 1)): cell=$cell fold=$fold seed=$seed on $(hostname)"
echo "=== started $(date '+%F %T')"

# Each task writes only its own runs_*/<cell>_<fold>_seed<n>/ directory, so
# tasks never collide on results. They DO all rewrite the same
# <cell>_*_summary.json, and those writes race -- harmless, because every
# summary is derivable from the per-run metrics.json files. Rebuild them after
# the array finishes with:
#     python scripts/aggregate_results.py --write-summaries
if [ "$cell" = "baseline" ]; then
  python scripts/run_rolling_validation.py --config "$CONFIG" --seeds "$seed" --folds "$fold"
else
  python scripts/run_bootstrap_ablation.py --config "$CONFIG" --seeds "$seed" --folds "$fold"
fi
status=$?

echo "=== finished $(date '+%F %T') with status $status"
exit $status
