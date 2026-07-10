#!/bin/bash
# Resilient overnight runner: the attention_frozen_excess rolling validation
# has died silently at least once already this session (last progress at
# 02:04, no process alive since) -- this wrapper relaunches it in a loop
# until all 15 fold/seed cells are done (safe: run_rolling_validation.py
# skips any cell whose metrics.json already exists), then runs the
# attention_frozen (log-return reward) baseline across the same 5 folds for
# a fair regime-level comparison, then writes the combined summary.
set -uo pipefail
cd "C:\Users\h2010\Documents\RL trading bot"

LOG=overnight_run.log
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

echo "$(date '+%F %T') overnight wrapper starting" >> "$LOG"

run_until_complete configs/attention_frozen_excess.yaml
EXCESS_OK=$?

run_until_complete configs/attention_frozen.yaml
BASELINE_OK=$?

if [ "$EXCESS_OK" -eq 0 ] && [ "$BASELINE_OK" -eq 0 ]; then
  python scripts/rolling_benchmark.py --config configs/attention_frozen_excess.yaml >> "$LOG" 2>&1
  python scripts/rolling_summary_compare.py --cells attention_frozen_excess attention_frozen \
    --benchmark runs_rolling/equal_weight_benchmark.json >> "$LOG" 2>&1
  echo "$(date '+%F %T') OVERNIGHT RUN COMPLETE - see runs_rolling/comparison_summary.txt" >> "$LOG"
else
  echo "$(date '+%F %T') OVERNIGHT RUN INCOMPLETE (excess_ok=$EXCESS_OK baseline_ok=$BASELINE_OK) - check $LOG" >> "$LOG"
fi
