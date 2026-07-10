#!/bin/bash
# Round 3: optimization ablation. Two full rounds (reward shaping, then
# encoder architecture) came back negative -- 0/20 architecture x reward x
# fold combinations robustly beat the equal-weight benchmark. The one
# unexplained diagnostic from round 1 is that entropy_loss and policy std
# stay essentially flat for the full 200k-timestep run in every cell (the
# policy never sharpens into a confident allocation), which could mean the
# problem is PPO not converging rather than there being no signal to find.
# This tests that directly on attention_frozen_excess (best cell so far),
# restricted to fold2 (2022 bear, worst result) and fold5 (2025+, most
# recent/decision-relevant), 3 seeds each:
#   - longtrain: total_timesteps 200k -> 600k
#   - bighead:   policy/value net_arch [32] -> [64,64]
#   - lowlr:     learning_rate 3e-4 -> 1e-4
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run3.log
MAX_HOURS=10
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
FOLDS_SUBSET="fold2_test2021 fold5_test2024"

count_done() {
  local name="$1"
  local total=0
  for fold in $FOLDS_SUBSET; do
    total=$(( total + $(ls runs_rolling/"${name}"_"${fold}"_seed*/metrics.json 2>/dev/null | wc -l) ))
  done
  echo "$total"
}

run_until_complete() {
  local config="$1"
  local name
  name=$(basename "$config" .yaml)
  while true; do
    local done_count
    done_count=$(count_done "$name")
    echo "$(date '+%F %T') [$name] $done_count/6 runs complete" >> "$LOG"
    if [ "$done_count" -ge 6 ]; then
      echo "$(date '+%F %T') [$name] COMPLETE" >> "$LOG"
      return 0
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "$(date '+%F %T') [$name] DEADLINE EXCEEDED (${MAX_HOURS}h), stopping" >> "$LOG"
      return 1
    fi
    echo "$(date '+%F %T') [$name] launching (attempt)" >> "$LOG"
    python scripts/run_rolling_validation.py --config "$config" --seeds 0 1 2 --folds $FOLDS_SUBSET >> "$LOG" 2>&1
    echo "$(date '+%F %T') [$name] process exited (code $?), re-checking completion" >> "$LOG"
  done
}

echo "$(date '+%F %T') overnight wrapper 3 (PPO optimization ablation) starting" >> "$LOG"

run_until_complete configs/attention_frozen_excess_bighead.yaml
BIGHEAD_OK=$?

run_until_complete configs/attention_frozen_excess_lowlr.yaml
LOWLR_OK=$?

run_until_complete configs/attention_frozen_excess_longtrain.yaml
LONGTRAIN_OK=$?

if [ "$BIGHEAD_OK" -eq 0 ] && [ "$LOWLR_OK" -eq 0 ] && [ "$LONGTRAIN_OK" -eq 0 ]; then
  echo "$(date '+%F %T') ROUND 3 COMPLETE - all ablation variants trained" >> "$LOG"
else
  echo "$(date '+%F %T') ROUND 3 INCOMPLETE (bighead_ok=$BIGHEAD_OK lowlr_ok=$LOWLR_OK longtrain_ok=$LONGTRAIN_OK) - check $LOG" >> "$LOG"
fi
