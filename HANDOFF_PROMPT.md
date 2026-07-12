You're picking up an RL ETF portfolio-allocation research project cold. Read
these three files in order before doing anything else: `DESIGN.md` (original
architecture rationale), `HANDOFF.md` (bug-fix + reward-shaping history),
`HANDOFF2.md` (everything since — this is the current source of truth).

## Where things stand in one paragraph

This is a PPO agent allocating daily across a basket of ETFs, benchmarked
against a naive equal-weight buy-and-hold of whatever basket is in play.
Across 7 rounds of experiments (~110 trained models) covering reward
shaping, encoder architecture (linear vs. attention), a PPO-convergence
ablation, block-bootstrap data augmentation, wider observed context, and
basket narrowing/rebuilding, **no configuration has robustly beaten its
benchmark across regime-diverse test folds**. The one genuinely promising,
still-unconfirmed lead is round 4 (block-bootstrap-augmented training data),
which crossed the benchmark on the hardest fold (2022 bear: −0.423 vs.
−0.436) with only 3 seeds tested — needs more seeds and all 5 folds before
it's trustworthy. Full numbers and reasoning for every round are in
`HANDOFF2.md`.

## Immediate state

- **No background jobs are currently running.** Confirmed via `ps -ef` before
  this handoff — safe to start fresh.
- **56 files are uncommitted** (`git status --short`): the RSI bug fix,
  `HANDOFF2.md` itself, ~5 new config files, `features/bootstrap.py`,
  3 new analysis scripts, and all the `runs_rolling/`/`runs_bootstrap/`
  result directories from rounds 3-7. The user was asked whether to commit
  now vs. wait for something specific and **has not yet answered** — don't
  assume either way; ask, or use judgement based on what they say when you
  start.
- Repo is a private GitHub repo: `aravgupta0707-gif/rl-trading-bot`, `gh`
  CLI already installed and authenticated on this machine.

## Conventions and gotchas you need before touching anything

1. **Use `python`, not `python3`, in Bash commands.** `python3` resolves to
   a different, incomplete interpreter (no `pandas_datareader`) on this
   machine. This cost real time to diagnose once already.
2. **`fetch_etf_prices`/`fetch_macro` cache to one file per `cache_dir`,
   regardless of which tickers/series were requested.** Every distinct
   ticker/macro-series set needs its own `cache_dir` in its config, or it
   will silently reuse stale cached data missing whatever's new. Follow the
   existing `cache_basket_*`/`cache_widectx` naming pattern.
3. **A narrowed basket needs its own equal-weight benchmark.** "Equal-weight
   basket" means something different at 6 assets than at 13 — never compare
   a narrowed basket's model against `runs_rolling/equal_weight_benchmark.json`
   (the original 13-asset one). Use `scripts/rolling_benchmark.py --config
   <that basket's config>` and its own output file.
4. **Any asset-promise/Sharpe screening must use pre-2020-only data**
   (or more generally, only data through whichever fold's train_end is
   earliest among the folds you'll evaluate on). Screening on data that
   overlaps a test window you'll later evaluate against is leakage — this
   was caught and fixed once already in round 6-7 (`HANDOFF2.md` has the
   details of exactly what flipped when it was corrected).
5. **Near-zero-volatility instruments can silently truncate the whole
   panel.** `features/panel.py`'s RSI calculation now handles `loss==0`
   correctly (fixed this session), but if you add anything even lower-vol
   than T-bills, or touch that function, re-verify no NaN survives
   `build_panel()`'s dropna() before trusting results — check
   `panel.isna().any().any()` and spot-check row counts per fold, the way
   the RSI bug was actually caught (a fold5 result had 59 days instead of
   the expected ~360, which was the tell).
6. **Every background training job in this session used the same pattern**:
   a `scripts/overnight_wrapperN.sh` that loops `run_rolling_validation.py`
   until all expected `metrics.json` files exist (auto-resumes if the
   process dies mid-run — this machine has a history of background
   processes getting killed unpredictably, see `HANDOFF.md`), launched via
   `nohup ... & disown`, with a `Monitor` tool watch on the log file for
   completion. Reuse this pattern for anything multi-hour; don't block
   synchronously on long training runs.
7. **Verify numbers from files on disk, not from truncated/interleaved
   background-task notification text.** Notifications can arrive
   interleaved from multiple concurrent monitors and get truncated — this
   session caught a real bug specifically by cross-checking disk state
   rather than trusting notification output at face value.

## What to actually do next

The user paused mid-session to get a clean handoff — there's no single
mandated next action. `HANDOFF2.md`'s "Immediate next steps" section lists
several open threads (resolve the correlation-vs-economic-sense basket
tension, push bootstrap augmentation further with more seeds/folds, address
the "model never goes defensive" finding directly, commit the pending work).
Ask the user which they want to pick up, or if they just say "continue,"
the bootstrap-augmentation thread (round 4) is the strongest untouched lead
and the most defensible default.
