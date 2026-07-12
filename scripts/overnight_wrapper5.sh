#!/bin/bash
# Round 5: wider observed context. Rounds 1-4 all worked with the exact same
# 70-column feature panel (13 tradeable ETFs + 5 macro series) -- none of
# them changed what the model could actually see, only how it trained on
# that same information. This widens the encoder's input (credit spreads,
# yield-curve shape, dollar, oil, leading index, plus HYG/LQD/IEF/SHY/EEM/UUP
# as observed-only context assets) while leaving the tradeable action space
# untouched -- see configs/attention_frozen_excess_widectx.yaml and
# features/panel.py's context_tickers param. Tests on the same fold2
# (2022 bear)/fold5 (2025+) x 3 seeds used throughout round 3, for direct
# comparison against the existing attention_frozen_excess numbers
# (fold2=-0.760, fold5=1.231) and the benchmark (fold2=-0.436, fold5=1.409).
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run5.log
MAX_HOURS=6
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
CONFIG=configs/attention_frozen_excess_widectx.yaml
NAME=$(basename "$CONFIG" .yaml)

count_done() {
  local total=0
  for fold in fold2_test2021 fold5_test2024; do
    total=$(( total + $(ls runs_rolling/"${NAME}"_"${fold}"_seed*/metrics.json 2>/dev/null | wc -l) ))
  done
  echo "$total"
}

echo "$(date '+%F %T') overnight wrapper 5 (wider observed context) starting" >> "$LOG"

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
  python scripts/run_rolling_validation.py --config "$CONFIG" --seeds 0 1 2 \
    --folds fold2_test2021 fold5_test2024 >> "$LOG" 2>&1
  echo "$(date '+%F %T') process exited (code $?), re-checking completion" >> "$LOG"
done

echo "$(date '+%F %T') ROUND 5 FINISHED - see runs_rolling/${NAME}_fold*/metrics.json" >> "$LOG"
