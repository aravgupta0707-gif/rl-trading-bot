#!/bin/bash
# Round 8: confirm (or kill) round 4's block-bootstrap result.
#
# Round 4 found the only benchmark crossing in the project -- bootstrap-
# augmented training data took the 2022 bear fold from -0.564 to -0.423 past
# its -0.436 benchmark -- but only on 2 of 5 folds, 3 seeds, with a seed
# spread (0.173) an order of magnitude wider than the margin (0.013). This
# round runs the same cell with and without augmentation across ALL 5 folds so
# the comparison is regime-wide rather than two hand-picked folds.
#
# Both cells are launched in parallel (4 cores here, 2 threads each). Runs that
# already have a metrics.json are skipped by the Python runners, so this script
# is safe to re-run and self-resumes if a process is killed mid-round -- this
# project has a history of long background jobs dying unpredictably.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/attention_frozen_excess_lowlr.yaml
NAME=attention_frozen_excess_lowlr
FOLDS="fold1_test2020 fold2_test2021 fold3_test2022 fold4_test2023 fold5_test2024"
SEEDS="${SEEDS:-0 1 2}"
EXPECTED_PER_CELL=$(( $(echo "$FOLDS" | wc -w) * $(echo "$SEEDS" | wc -w) ))
TARGET=$(( EXPECTED_PER_CELL * 2 ))

LOG=round8.log
BASE_LOG=round8_baseline.log
BOOT_LOG=round8_bootstrap.log
MAX_HOURS="${MAX_HOURS:-10}"
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

# 4 cores, two concurrent trainers.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

count_done() {
  local n=0
  n=$(( n + $(ls runs_rolling/${NAME}_fold*_seed*/metrics.json 2>/dev/null | wc -l) ))
  n=$(( n + $(ls runs_bootstrap/${NAME}_bootstrap_fold*_seed*/metrics.json 2>/dev/null | wc -l) ))
  echo "$n"
}

echo "$(date '+%F %T') round 8 starting: target ${TARGET} runs (${EXPECTED_PER_CELL}/cell), seeds: ${SEEDS}" >> "$LOG"

while true; do
  done_count=$(count_done)
  echo "$(date '+%F %T') progress ${done_count}/${TARGET}" >> "$LOG"
  if [ "$done_count" -ge "$TARGET" ]; then
    echo "$(date '+%F %T') COMPLETE" >> "$LOG"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date '+%F %T') DEADLINE EXCEEDED (${MAX_HOURS}h), stopping at ${done_count}/${TARGET}" >> "$LOG"
    break
  fi

  echo "$(date '+%F %T') launching both cells" >> "$LOG"
  python scripts/run_rolling_validation.py --config "$CONFIG" --seeds $SEEDS \
      --folds $FOLDS >> "$BASE_LOG" 2>&1 &
  base_pid=$!
  python scripts/run_bootstrap_ablation.py --config "$CONFIG" --seeds $SEEDS \
      --folds $FOLDS >> "$BOOT_LOG" 2>&1 &
  boot_pid=$!
  wait $base_pid; echo "$(date '+%F %T') baseline exited ($?)" >> "$LOG"
  wait $boot_pid; echo "$(date '+%F %T') bootstrap exited ($?)" >> "$LOG"
done

# Benchmarks must be recomputed on the current data vintage: this container
# re-derived cache/ from the committed cache_widectx parquet, which runs 6
# trading days later than the cache the committed fold5 numbers were produced
# on (364 vs 358 test days). Folds 1-4 have fixed test_end dates and are
# unaffected; fold 5 runs to the end of available data, so its benchmark has
# to come from the same vintage as the models it judges.
if [ "$(count_done)" -ge "$TARGET" ]; then
  python scripts/rolling_benchmark.py --config "$CONFIG" \
      --out runs_rolling/equal_weight_benchmark_round8.json >> "$LOG" 2>&1
  echo "$(date '+%F %T') ROUND 8 FINISHED" >> "$LOG"
else
  echo "$(date '+%F %T') ROUND 8 INCOMPLETE" >> "$LOG"
fi
