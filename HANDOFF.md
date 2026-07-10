# Handoff — RL ETF Trading Bot

Snapshot of where this stands, for whoever (human or Claude) picks this up next.
See `DESIGN.md` for the original architecture rationale — this doc is the
current-state/what-happened-since layer on top of it.

## TL;DR

- Pipeline is built and working: data → feature panel → encoder (linear/MLP/
  attention) → PPO portfolio-allocation policy → backtest.
- Found and fixed a real bug where stable-baselines3 silently discarded
  pretrained/frozen encoder weights (see "Bug fixed" below) — the first full
  grid's `frozen`/`finetune` results were invalid until this was caught.
- The winning cell (`attention` + `frozen`) beat every other grid cell but
  **did not clearly beat a naive equal-weight benchmark** — that finding
  changed the whole direction of the work since.
- Basket expansion (10 → 13 assets) didn't close the gap. Reward shaping did:
  switching from an absolute log-return reward to an **excess-return-vs-
  benchmark reward** raised `attention_frozen`'s test Sharpe from 1.22 to a
  3-seed mean of **1.259** (std 0.044, much tighter than before) — narrowing
  the gap to the 1.395 benchmark from ~0.18 to ~0.14.
- **Currently running in the background:** a 5-fold rolling walk-forward
  validation of the reward-shaped winner (`attention_frozen_excess`) across
  regime-diverse historical windows (2020 COVID, 2021 bull, 2022 bear, 2023
  recovery, 2024+), to check the edge is stable over time rather than an
  artifact of the one 2024–present test window used everywhere above.
- **No broker/paper-trading integration exists anywhere in this repo.** That's
  unbuilt work, not a config flag, and shouldn't be started until a cell
  clearly and robustly beats the benchmark.

## Chronological findings (read in order if picking this up cold)

1. **Initial grid (10-asset basket, log-return reward).** 9 cells (linear/
   mlp/attention × frozen/finetune/e2e) × 3 seeds. First full run looked
   plausible but was corrupted:
2. **Bug found:** `finetune` and `e2e` produced byte-identical metrics for
   linear and MLP encoders. Root cause: SB3's `ActorCriticPolicy._build()`
   re-applies orthogonal init to the *entire* features extractor — including
   our pretrained/frozen encoder — right after `PPO(...)` construction,
   silently wiping `prepare_encoder()`'s work. True since the original smoke
   test. **Fixed** in `training/train_ppo.py::train()`: snapshot the
   encoder's `state_dict()` before `PPO(...)` construction, restore it after.
3. **Grid rerun (post-fix).** `attention_frozen` won: test Sharpe 1.15, and
   turnover less than half of every other cell. `frozen` legitimately beat
   `finetune`/`e2e` for all three encoders once the bug was gone (the
   opposite of what an undetected bug would produce — good sign the fix was
   real). Full diagnostics published as an HTML artifact (equity curves,
   drawdown, allocation, seed variability, the bug writeup) —
   ask the user if they still have the link, it was shared mid-session.
4. **Benchmark check (the pivotal finding):** a naive daily-rebalanced
   equal-weight basket scored Sharpe **1.28** over the same test window —
   *beating* the winning RL cell. The model's only clear edge was turnover.
5. **Basket expansion (10 → 13 assets):** added TLT (long treasuries), GLD
   (gold), EFA (international equities) — sector SPDRs alone are mostly
   undifferentiated equity beta, these give genuine cross-asset dispersion.
   Result: **the benchmark's Sharpe rose even more than any RL cell's** (to
   1.395), so the gap didn't close. Confirmed this wasn't a cost-fairness
   artifact either — a cost-matched and a true buy-and-hold benchmark variant
   both still beat every RL cell (1.388 and 1.36 respectively).
6. **Root-cause rethink:** the reward (`log(1 + return − cost)`) trains the
   policy to maximize its own return, with no incentive to beat a diversified
   basket. If the model can't find a strong rotation signal, that objective's
   natural equilibrium is "diversify like the benchmark, but pay extra
   turnover getting there."
7. **Reward shaping (the fix that worked):** added `excess_return_net_cost`
   as a new `env.reward_type` in `envs/portfolio_env.py::step()` — reward =
   portfolio net return **minus that day's equal-weight-basket return**, net
   of cost. (Backtest metrics still track real portfolio value/net_return;
   only the training signal changed.) Result: `attention_frozen_excess`
   3-seed mean Sharpe 1.259, std 0.044 (vs. 0.089 pre-fix-era grid) — higher
   *and* more consistent across seeds. `mlp_frozen_excess` was worse and
   noisier (mean 1.168, std 0.115, ~2-4x the turnover) — attention remains
   the architecture of choice.
8. **In progress:** rolling walk-forward validation of `attention_frozen_excess`
   across 5 regime-diverse folds (see next section).

## What's running right now

`scripts/run_rolling_validation.py --config configs/attention_frozen_excess.yaml --seeds 0 1 2`
— 5 folds × 3 seeds = 15 training runs, each ~7-10 min (attention encoder
pace) → roughly 2-2.5 hours total. Output goes to `runs_rolling/`, summary to
`runs_rolling/attention_frozen_excess_rolling_summary.json`. Log: `rolling_run.log`.

**As of this writing: 4 of 15 folds complete**, fold2/seed1 in progress. Fold
boundaries are hardcoded in `scripts/run_rolling_validation.py::FOLDS`:

| fold | train through | val | test |
|---|---|---|---|
| 1 | 2019-12-31 | 2020 | 2021 |
| 2 | 2020-12-31 | 2021 | 2022 |
| 3 | 2021-12-31 | 2022 | 2023 |
| 4 | 2022-12-31 | 2023 | 2024 |
| 5 | 2023-12-31 | 2024 | 2025–present |

**Known issue: this background process keeps getting killed unpredictably**
(happened repeatedly throughout this session, roughly every 30-90 min of
wall-clock). Investigated once: it was AC sleep-after-idle (fixed via
`powercfg /change standby-timeout-ac 0` — confirm this is still in effect if
kills resume, via `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE`,
should read `0x00000000` for "Current AC Power Setting Index"). It has also
been killed at least once *after* that fix with no sleep/wake event recorded
(`powercfg /lastwake` showed zero) and no reboot in between — cause
undetermined, possibly something in the harness itself. **The run is safe to
resume**: each of the 15 cells writes `runs_rolling/<name>/metrics.json` only
on completion, so nothing is lost on a kill — just relaunch the same command
and it'll retrain whichever cells don't have a `metrics.json` yet (the script
checks `metrics_path.exists()` per cell, no `--skip-existing` flag needed
here, unlike `run_grid.py`).

## Repo/directory guide (multiple `runs*` dirs exist — don't confuse them)

- `runs/` — the original 9-cell × 3-seed grid, **10-asset basket, log-return
  reward, post-bug-fix**. Still the historical record of the architecture
  search. `runs/grid_summary.json` has the aggregate.
- `runs_expanded_check/` — quick single-seed recheck of `linear/mlp/attention
  _frozen` on the **13-asset basket**, log-return reward. Used to confirm
  attention still made sense as the architecture before the reward change.
- `runs_reward_check/` — `attention_frozen_excess` (3 seeds) and
  `mlp_frozen_excess` (3 seeds), 13-asset basket. This is where the reward-
  shaping comparison lives.
- `runs_rolling/` — the in-progress 5-fold rolling validation, currently
  being written to.
- `cache/etf_prices.parquet`, `cache/macro.parquet` — cached data for the
  **current** `configs/default.yaml` basket (13 assets). If the basket
  config changes again, re-fetch with `refresh=True` or delete these first —
  fetch functions silently reuse whatever's cached regardless of what the
  config currently asks for.
- Root has ~20MB of `grid_run*.log` / `rolling_run.log` files from repeated
  background-task kills/resumes across this session — safe to delete once
  the current rolling validation finishes and its log is captured/summarized,
  not cleaned up yet.
- **Not a git repo.** No version control on any of this. Worth initializing
  one before much more work piles up.

## Key files added/changed this session

- `envs/portfolio_env.py` — added `excess_return_net_cost` reward type.
- `training/train_ppo.py` — the SB3 orthogonal-init bug fix (state_dict
  snapshot/restore around `PPO(...)` construction).
- `features/splits.py` — added `rolling_walk_forward_split()` (explicit
  `test_end`, for bounded-length folds).
- `configs/default.yaml` — basket expanded to 13 tickers.
- `configs/{attention,mlp}_frozen_excess.yaml` — reward-shaped configs.
- `scripts/run_grid.py` — runs the encoder×regime×seed grid in one process
  (builds panel once); `--skip-existing` to resume after a kill.
- `scripts/run_rolling_validation.py` — the 5-fold regime-robustness runner.
- `scripts/diagnostics.py` — full daily trajectory (equity/weights/turnover)
  for one trained run, not just summary stats.

## Immediate next steps

1. **Let the rolling validation finish** (resume on kill, see above). Look at
   whether `attention_frozen_excess` beats its fold-specific benchmark
   consistently, or only in some regimes (e.g. does it hold up in the 2022
   bear-market fold?).
2. If robust: revisit the benchmark comparison once more at the *rolling*
   level (each fold's own equal-weight Sharpe, not just the single static
   1.395) before calling this settled.
3. Only after that: point-in-time macro data, and actually building broker
   integration (Alpaca was discussed as a candidate but not decided/built).
4. Housekeeping, whenever convenient: git-init the repo, clean up the root
   log files, consider consolidating the `runs*` directories now that the
   basket/reward have both changed since the original grid.
