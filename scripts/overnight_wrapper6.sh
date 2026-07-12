#!/bin/bash
# Round 6: narrowed baskets. Two candidates from the asset-promise analysis
# (scripts/asset_promise.py) -- basket_div keeps the strongest performer
# from each exposure bucket (XLK/XLI/XLF/XLU + EFA/GLD), basket_sharpe is
# pure top-6 by leakage-free pre-2020 Sharpe (all correlated US equity
# sectors, no diversifiers). Both narrow the actual tradeable/action
# universe from 13 to 6 assets, unlike round 5's wider-context experiment
# which only widened what's observed. Each basket gets its own equal-weight
# benchmark (an "equal-weight basket" means something different at 6 assets
# than 13) via scripts/rolling_benchmark.py. Same fold2 (2022 bear)/fold5
# (2025+) x 3 seeds used throughout rounds 3-5.
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run6.log
MAX_HOURS=6
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

count_done() {
  local name="$1"
  local total=0
  for fold in fold2_test2021 fold5_test2024; do
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
    python scripts/run_rolling_validation.py --config "$config" --seeds 0 1 2 \
      --folds fold2_test2021 fold5_test2024 >> "$LOG" 2>&1
    echo "$(date '+%F %T') [$name] process exited (code $?), re-checking completion" >> "$LOG"
  done
}

echo "$(date '+%F %T') overnight wrapper 6 (narrowed baskets) starting" >> "$LOG"

run_until_complete configs/attention_frozen_excess_basket_div.yaml
DIV_OK=$?
run_until_complete configs/attention_frozen_excess_basket_sharpe.yaml
SHARPE_OK=$?

if [ "$DIV_OK" -eq 0 ] && [ "$SHARPE_OK" -eq 0 ]; then
  python scripts/rolling_benchmark.py --config configs/attention_frozen_excess_basket_div.yaml \
    --out runs_rolling/equal_weight_benchmark_basket_div.json >> "$LOG" 2>&1
  python scripts/rolling_benchmark.py --config configs/attention_frozen_excess_basket_sharpe.yaml \
    --out runs_rolling/equal_weight_benchmark_basket_sharpe.json >> "$LOG" 2>&1
  echo "$(date '+%F %T') ROUND 6 FINISHED" >> "$LOG"
else
  echo "$(date '+%F %T') ROUND 6 INCOMPLETE (div_ok=$DIV_OK sharpe_ok=$SHARPE_OK)" >> "$LOG"
fi
