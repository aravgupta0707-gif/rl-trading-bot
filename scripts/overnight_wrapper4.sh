#!/bin/bash
# Round 4: block-bootstrap data augmentation. Round 3 (PPO optimization
# ablation) showed longtrain genuinely converges the policy (entropy halves,
# std drops 1.0->0.51) but that convergence barely moves Sharpe -- ruling out
# "PPO hasn't had enough steps" as the bottleneck. This tests the remaining
# hypothesis: not enough distinct real training history to learn a real edge
# from. Doubles attention_frozen_excess_lowlr's (best variant so far) real
# training data with a block-bootstrapped synthetic block (see
# features/bootstrap.py), on the same fold2 (2022 bear)/fold5 (2025+) x 3
# seeds used throughout round 3, for direct comparison against those numbers.
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run4.log
MAX_HOURS=6
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
CONFIG=configs/attention_frozen_excess_lowlr.yaml
NAME=$(basename "$CONFIG" .yaml)

count_done() {
  local total=0
  for fold in fold2_test2021 fold5_test2024; do
    total=$(( total + $(ls runs_bootstrap/"${NAME}"_bootstrap_"${fold}"_seed*/metrics.json 2>/dev/null | wc -l) ))
  done
  echo "$total"
}

echo "$(date '+%F %T') overnight wrapper 4 (bootstrap augmentation) starting" >> "$LOG"

while true; do
  done_count=$(count_done)
  echo "$(date '+%F %T') $done_count/6 runs complete" >> "$LOG"
  if [ "$done_count" -ge 6 ]; then
    echo "$(date '+%F %T') COMPLETE" >> "$LOG"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date '+%F %T') DEADLINE EXCEEDED (${MAX_HOURS}h), stopping" >> "$LOG"
    break
  fi
  echo "$(date '+%F %T') launching (attempt)" >> "$LOG"
  python scripts/run_bootstrap_ablation.py --config "$CONFIG" --seeds 0 1 2 \
    --folds fold2_test2021 fold5_test2024 --augment-multiplier 1.0 --block-len 21 >> "$LOG" 2>&1
  echo "$(date '+%F %T') process exited (code $?), re-checking completion" >> "$LOG"
done

echo "$(date '+%F %T') ROUND 4 FINISHED - see runs_bootstrap/${NAME}_bootstrap_summary.json" >> "$LOG"
