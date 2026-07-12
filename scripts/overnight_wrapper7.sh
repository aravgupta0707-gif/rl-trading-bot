#!/bin/bash
# Round 7: basket built from the broad ~67-ticker universe screen
# (scripts/screen_etf_universe.py) combined with the earlier revealed-
# preference analysis. XLK/XLI/XLF (twice-validated top sectors), SMH
# (concentrated tech alternative), LQD/HYG (credit -- the biggest gap
# identified), BIL (yield-bearing safe harbor, addresses the "never goes
# defensive" finding), GLD/EFA (validated diversifiers). Same fold2
# (2022 bear)/fold5 (2025+) x 3 seeds used throughout rounds 3-6, plus its
# own fresh equal-weight benchmark.
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run7.log
MAX_HOURS=6
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
CONFIG=configs/attention_frozen_excess_basket_universe.yaml
NAME=$(basename "$CONFIG" .yaml)

count_done() {
  local total=0
  for fold in fold2_test2021 fold5_test2024; do
    total=$(( total + $(ls runs_rolling/"${NAME}"_"${fold}"_seed*/metrics.json 2>/dev/null | wc -l) ))
  done
  echo "$total"
}

echo "$(date '+%F %T') overnight wrapper 7 (universe-derived basket) starting" >> "$LOG"

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

if [ "$done_count" -ge 6 ]; then
  python scripts/rolling_benchmark.py --config "$CONFIG" \
    --out runs_rolling/equal_weight_benchmark_basket_universe.json >> "$LOG" 2>&1
  echo "$(date '+%F %T') ROUND 7 FINISHED" >> "$LOG"
else
  echo "$(date '+%F %T') ROUND 7 INCOMPLETE" >> "$LOG"
fi
