#!/bin/bash
# Round 2: attention_frozen[_excess] rolling validation finished clean (30/30
# runs, no errors) but neither reward variant robustly beats the per-fold
# equal-weight benchmark (wins 1/5 and 0/5 folds respectively), and training
# diagnostics show entropy/policy-std staying flat for the full 200k
# timesteps -- the policy never sharpens. DESIGN.md's original working prior
# was that linear/PCA should be competitive-or-better given how little data
# this is (~13 assets, a few thousand daily rows) -- that was never tested
# on the regime-diverse rolling folds, only on one static split. This runs
# linear_frozen and linear_frozen_excess across the same 5 folds x 3 seeds
# for a real comparison, then rebuilds the full comparison + ensemble tables.
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run2.log
MAX_HOURS=10
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

count_done() {
  ls runs_rolling/"$1"_fold*/metrics.json 2>/dev/null | wc -l
}

run_until_complete() {
  local config="$1"
  local name
  name=$(basename "$config" .yaml)
  while true; do
    local done_count
    done_count=$(count_done "$name")
    echo "$(date '+%F %T') [$name] $done_count/15 folds complete" >> "$LOG"
    if [ "$done_count" -ge 15 ]; then
      echo "$(date '+%F %T') [$name] COMPLETE" >> "$LOG"
      return 0
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "$(date '+%F %T') [$name] DEADLINE EXCEEDED (${MAX_HOURS}h), stopping" >> "$LOG"
      return 1
    fi
    echo "$(date '+%F %T') [$name] launching (attempt)" >> "$LOG"
    python scripts/run_rolling_validation.py --config "$config" --seeds 0 1 2 >> "$LOG" 2>&1
    echo "$(date '+%F %T') [$name] process exited (code $?), re-checking completion" >> "$LOG"
  done
}

echo "$(date '+%F %T') overnight wrapper 2 (linear encoder rolling test) starting" >> "$LOG"

run_until_complete configs/linear_frozen.yaml
BASE_OK=$?

run_until_complete configs/linear_frozen_excess.yaml
EXCESS_OK=$?

if [ "$BASE_OK" -eq 0 ] && [ "$EXCESS_OK" -eq 0 ]; then
  python scripts/rolling_summary_compare.py \
    --cells attention_frozen_excess attention_frozen linear_frozen_excess linear_frozen \
    --benchmark runs_rolling/equal_weight_benchmark.json \
    --out runs_rolling/comparison_summary_v2.txt >> "$LOG" 2>&1
  python scripts/ensemble_backtest.py --cells linear_frozen_excess linear_frozen \
    --out runs_rolling/ensemble_summary_linear.json >> "$LOG" 2>&1
  echo "$(date '+%F %T') ROUND 2 COMPLETE - see runs_rolling/comparison_summary_v2.txt and runs_rolling/ensemble_summary_linear.json" >> "$LOG"
else
  echo "$(date '+%F %T') ROUND 2 INCOMPLETE (base_ok=$BASE_OK excess_ok=$EXCESS_OK) - check $LOG" >> "$LOG"
fi
