# RL ETF Portfolio Allocation

A research codebase for a reinforcement-learning portfolio allocator: a PPO
agent that sets daily target weights across a basket of ETFs plus cash, with a
pretrained representation encoder (linear / MLP / cross-asset attention) in
front of the policy. Backtest-first, walk-forward validated across market
regimes, and benchmarked honestly against the thing it has to beat — a naive
equal-weight buy-and-hold of the same basket.

**This is a research log, not a trading system.** There is no broker or
paper-trading integration, deliberately: no configuration has robustly beaten
its benchmark yet, and connecting a strategy that loses to buy-and-hold would
just be a faster way to lose money.

---

## Headline finding

Across **8 rounds of experiments and ~175 trained models**, covering reward
shaping, encoder architecture, PPO convergence, data augmentation, wider
macro context, and four different tradeable baskets:

> **No configuration robustly beats equal-weight buy-and-hold on
> regime-diverse test folds.** Of 20 fold-level comparisons in the main
> architecture sweep, exactly one came out ahead — by 0.008 Sharpe, well
> inside one seed's noise.

The one lever that has survived scrutiny is **block-bootstrap data
augmentation**. Tested across all 5 folds in round 8, it cuts the mean gap to
benchmark from −0.089 to **−0.016**, and beats the benchmark on 3 of 5 folds
where the un-augmented baseline manages 1. But no individual fold's margin
exceeds its own seed spread, and augmentation *widens* that spread wherever it
helps — so the honest verdict is a real effect of unproven size. Settling it
needs seeds, not another lever; see [`scripts/adroit/`](scripts/adroit/).

Why this negative result is worth reading: the interesting part isn't that the
agent loses, it's *why* every obvious fix fails to help. See
[RESULTS.md](RESULTS.md) for the full experiment log and
[NOTES.md](NOTES.md) for the methodology traps found along the way — several
of which silently invalidate results if you don't know about them.

---

## How it works

```
ETF prices (yfinance) ─┐
                       ├─→ feature panel ─→ train/val/test split ─→ encoder ─→ PPO policy ─→ backtest
FRED macro series ─────┘    (88 cols)        (walk-forward,          (pretrained,   (softmax →     (Sharpe, DD,
                                              train-only norm)       frozen)        long-only      turnover)
                                                                                    weights)
```

- **Observation** — one cross-sectional row per day (88 columns for the
  13-asset basket): per-ETF returns (1/5/20d), 20d volatility, 14d RSI, 20d
  volume z-score, plus macro z-scores and changes.
  Every ETF's row carries every other ETF's features, so cross-asset effects
  are visible to the encoder. No column is a future value.
- **Encoder** — trained unsupervised (reconstruction MSE), then frozen.
  Explicitly *not* supervised on future returns; see [DESIGN.md](DESIGN.md)
  for why that constraint was chosen and what it ruled out.
- **Action** — logits → softmax → long-only target weights over ETFs + cash,
  rebalanced daily, charged 5 bps of turnover.
- **Reward** — either `log_return_net_cost` (absolute) or
  `excess_return_net_cost` (return minus the same-day equal-weight basket
  return). Backtest metrics always report real portfolio value regardless of
  which training signal was used.

---

## Results

All numbers are **test-split Sharpe, mean over 3 seeds**, from the JSON in
`runs_rolling/` and `runs_bootstrap/`. Folds are expanding-window
walk-forward; see [NOTES.md](NOTES.md#fold-naming) for the fold table and a
naming quirk worth knowing before you read directory names.

### Main sweep: reward shaping × encoder (13-asset basket, all 5 folds)

| cell | 2021 | 2022 bear | 2023 | 2024 | 2025+ |
|---|---|---|---|---|---|
| **equal-weight benchmark** | **2.225** | **−0.436** | **1.140** | **1.404** | **1.409** |
| `attention_frozen` | 1.829 | −0.764 | 0.897 | 1.295 | 1.217 |
| `attention_frozen_excess` | 1.805 | −0.760 | _1.148_ ✅ | 1.209 | 1.231 |
| `linear_frozen` | 1.796 | −0.700 | 1.030 | 1.192 | 1.053 |
| `linear_frozen_excess` | 1.784 | −0.737 | 0.934 | 1.152 | 1.118 |

One win in 20 (✅), by 0.008 on a seed std of 0.212. Reward shaping helps in
some folds and hurts in others — a wash, not a fix. Linear and attention reach
comparable Sharpe, but linear carries roughly double the turnover and higher
seed variance, so attention stays the default.

### Ablations (fold 2 = 2022 bear, fold 5 = 2025+; the hardest and the most decision-relevant)

| variant | 2022 bear | 2025+ | verdict |
|---|---|---|---|
| equal-weight benchmark | −0.436 | 1.409 | the bar |
| `attention_frozen_excess` (baseline) | −0.760 | 1.231 | — |
| `_bighead` (`net_arch` `[32]`→`[64,64]`) | −0.758 | 0.992 | ruled out |
| `_longtrain` (200k→600k steps) | −0.582 | 1.308 | converges, doesn't help |
| `_lowlr` (lr 3e-4→1e-4) | −0.564 | 1.351 | best single lever |
| `_widectx` (+6 FRED series, +6 observed ETFs) | −0.652 | 1.296 | real, modest lift |
| `_lowlr` + **block bootstrap** | **−0.423** ✅ | 1.319 | extended to all folds in round 8 |

### Round 8: bootstrap augmentation across all 5 folds (3 seeds, single vintage)

| fold (test year) | baseline `_lowlr` | + bootstrap | benchmark | baseline gap | bootstrap gap |
|---|---|---|---|---|---|
| 1 (2021) | 2.001 | 2.134 | 2.225 | −0.223 | −0.091 |
| 2 (2022 bear) | −0.564 | **−0.423** | −0.436 | −0.128 | **+0.013** ✅ |
| 3 (2023) | **1.181** | 1.151 | 1.140 | **+0.041** ✅ | +0.011 ✅ |
| 4 (2024) | 1.326 | **1.525** | 1.404 | −0.078 | **+0.120** ✅ |
| 5 (2025+) | 1.289 | 1.211 | 1.344 | −0.055 | −0.132 |
| **mean gap** | | | | **−0.089** | **−0.016** |

Fold 5's benchmark is 1.344 rather than 1.409 here because its test window
tracks the data vintage — see the vintage note in
[RESULTS.md](RESULTS.md#round-8).

### Basket studies — each against *its own* equal-weight benchmark

| basket | 2022 bear: model / bench (gap) | 2025+: model / bench (gap) |
|---|---|---|
| 13-asset (default) | −0.760 / −0.436 (**−0.324**) | 1.231 / 1.409 (**−0.178**) |
| `basket_div` — 6 assets, diversification-preserving | −0.774 / −0.404 (−0.370) | **1.446** / 1.638 (−0.192) |
| `basket_sharpe` — 6 assets, pure top-Sharpe | −0.825 / −0.578 (−0.247) | 0.992 / 1.166 (−0.174) |
| `basket_universe` — 9 assets, + credit + T-bills | −0.790 / −0.722 (**−0.068**) | 1.583 / 1.688 (**−0.105**) |

**The basket-quality trap.** `basket_div` posted the best absolute Sharpe in
the entire project (1.446) — and still lost, because its own benchmark rose
further (1.638). Concentrating into stronger assets lifts the model and the
naive benchmark in lockstep, so picking better assets cannot close the gap by
itself. The model has to do something equal-weight structurally can't.

---

## What's been ruled out, and what's still open

**Ruled out as the bottleneck** (each tested directly, not argued):

- *Optimization budget.* Given 3× the steps, entropy and policy std genuinely
  converge (−19.9 → −9.1, 1.0 → 0.49) and Sharpe barely moves. PPO is not
  under-trained.
- *Policy/value network capacity.* A bigger head made things worse.
- *Reward specification.* Excess-return-vs-benchmark shaping is a wash across
  folds.
- *Encoder architecture.* Linear ≈ attention on Sharpe; the 3×3
  encoder × regime grid was settled early (`frozen` beats `finetune`/`e2e`
  for all three encoders).
- *Asset selection by trailing quality.* See the basket-quality trap above.

**Open leads**, in the order they seem worth pursuing:

1. **More seeds on the round-8 design.** All 5 folds are now done at 3 seeds and
   the effect is real but not sized: fold 4 read +0.188 at 2 seeds and +0.120 at
   3, against a spread of 0.157. 10 seeds × 5 folds × 2 cells settles it, and as
   a Slurm array that is one submission — [`scripts/adroit/`](scripts/adroit/).
2. **The policy barely differentiates, and not because it can't.** The action
   space does impose a cap — `Box(low=-1, high=1)` softmaxed over 14 slots
   confines every weight to **[1.0%, 36.2%]** — but measured against trained
   rollouts (`scripts/check_action_space.py`) that cap is **not binding**:
   realized weights stay within **1.6%–15.5%**, cash sits at 2–6% where 36% is
   available, and no day comes within 90% of the cap. The learned portfolio is a
   *mildly tilted equal-weight basket by choice*, which explains the tight
   benchmark tracking and why no lever moves Sharpe much. The question worth
   attacking is why the policy stays near-uniform — candidates: the
   excess-return reward is ~1e-4 per day, `ent_coef=0` leaves nothing pushing
   exploration, and advantage scale may simply be too small to move the
   Gaussian's mean off its initialization. Widening the action bounds is *not*
   the fix; the agent doesn't use the range it already has.
3. **The model never goes defensive — and this is now confirmed as a learning
   failure, not a specification limit.** Average cash weight in the 2022 bear
   fold (0.027) is no higher than in the best fold (0.042). Since the agent
   *could* hold up to 36% cash and holds 2–6%, the capacity to de-risk exists
   and goes unused, which makes this a reward/exploration problem rather than a
   plumbing one. `basket_universe`'s `BIL` sleeve was a first probe at giving
   defensiveness an actual payoff.
4. **Correlation-optimal vs. economically-sensible baskets.** Pre-2020
   correlations show `LQD` (investment-grade credit) is ~uncorrelated with
   equities (0.05) while `HYG` (high-yield) is not (0.66) — junk bonds carry
   equity risk, investment grade doesn't. Pure greedy min-correlation
   selection picks mathematically diverse but odd baskets
   (Brazil / Hong Kong / ARKK). A properly constrained selection — diversify
   across meaningful buckets, best Sharpe within each — is still unbuilt.
5. **Combining levers.** Bootstrap + wide context + a diversified basket have
   only ever been tested in isolation.

Deliberately untouched until something clears the bar: point-in-time macro
vintages, and broker integration.

---

## Quickstart

```bash
pip install -r requirements.txt

# single experiment: fetch → panel → train → backtest → runs/<name>/
python scripts/run_experiment.py --config configs/attention_frozen_excess.yaml

# regime robustness: same cell retrained across 5 walk-forward folds
python scripts/run_rolling_validation.py \
    --config configs/attention_frozen_excess.yaml --seeds 0 1 2

# the benchmark a model must beat (per basket — see NOTES.md)
python scripts/rolling_benchmark.py --config configs/attention_frozen_excess.yaml

# the current lead: bootstrap-augmented training data
python scripts/run_bootstrap_ablation.py \
    --config configs/attention_frozen_excess_lowlr.yaml --seeds 0 1 2
```

Use `python`, not `python3`, and read [NOTES.md](NOTES.md) before changing a
basket or a feature — three of the gotchas there silently corrupt results
rather than failing loudly. First run downloads from Yahoo Finance and FRED
(no API keys needed) and caches to parquet.

---

## Repo layout

```
data/fetch.py          cache-first downloaders: fetch_etf_prices(), fetch_macro()
features/panel.py      build_panel() — the cross-sectional feature panel (88 cols at 13 assets)
features/splits.py     walk-forward splits + train-only normalization
features/bootstrap.py  generate_synthetic_panel() — block-resampled synthetic history
encoders/              linear (PCA-equivalent) | mlp | attention (cross-ETF), one interface
envs/portfolio_env.py  gym env: weights action, cost-aware reward, both reward types
training/              unsupervised encoder pretraining; PPO wiring (frozen/finetune/e2e)
eval/                  Sharpe / drawdown / turnover metrics; deterministic backtest rollout
configs/               default.yaml + one small override file per experiment cell
scripts/               experiment runners, benchmarks, basket screens, diagnostics
```

Results directories, which are committed so findings are reproducible without
retraining:

| dir | what's in it |
|---|---|
| `runs/` | original 9-cell encoder × regime grid (10-asset basket, log-return reward) |
| `runs_expanded_check/` | single-seed recheck of the three `frozen` cells on the 13-asset basket |
| `runs_reward_check/` | the reward-shaping comparison (attention & mlp, 3 seeds each) |
| `runs_rolling/` | all 5-fold rolling validations, per-basket benchmarks, basket screens |
| `runs_bootstrap/` | block-bootstrap augmentation runs |
| `cache_*/` | per-experiment data caches (one per distinct ticker/macro set — see NOTES.md) |

## Docs

| file | what it's for |
|---|---|
| [DESIGN.md](DESIGN.md) | architecture decisions and the reasoning behind them |
| [RESULTS.md](RESULTS.md) | the experiment log, round by round, with verified numbers |
| [NOTES.md](NOTES.md) | conventions, gotchas, and the bugs found — read before changing anything |
