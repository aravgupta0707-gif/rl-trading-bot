# RL ETF Portfolio Allocation

A research codebase for a reinforcement-learning portfolio allocator: a PPO
agent that sets daily target weights across a basket of ETFs plus cash, with a
pretrained representation encoder (linear / MLP / cross-asset attention) in
front of the policy. Backtest-first, walk-forward validated across market
regimes, and benchmarked honestly against the thing it has to beat — a naive
equal-weight buy-and-hold of the same basket.

**This is a research log, not a trading system**, and it is **concluded**. There
is no broker or paper-trading integration, deliberately: nothing here beat
buy-and-hold, and wiring up a strategy that ties its benchmark at best would
only convert a research negative into a financial one. What the project produced
instead is a clear answer, an audited pipeline, and a set of transferable lessons
about measuring this kind of question — see
[Conclusions](RESULTS.md#conclusions).

---

## Headline finding

Across **8 rounds of experiments and ~240 trained models**, covering reward
shaping, encoder architecture, PPO convergence, data augmentation, wider
macro context, and four different tradeable baskets:

> **No configuration robustly beats equal-weight buy-and-hold on
> regime-diverse test folds.** Of 20 fold-level comparisons in the main
> architecture sweep, exactly one came out ahead — by 0.008 Sharpe, well
> inside one seed's noise.

The one lever that has survived scrutiny is **block-bootstrap data
augmentation**, and round 8 measured exactly what it is worth: 100 runs, 10
seeds × 5 folds × 2 cells. It does **not** beat the benchmark anywhere. What it
does is close the gap to a statistical tie — the un-augmented baseline is
significantly *worse* than equal-weight on 3 of 5 folds (t = −4.2, −4.2, −2.9),
while the augmented cell is not significantly worse on any fold, nor better on
any. Its one real fold-level effect is on the 2022 bear market (+0.146,
t = 2.10), where resampled blocks give the policy more falling-market history
than the real 1–4 years contain.

Round 8 also killed the earlier 3-seed result it was testing: the bear-fold
"crossing" of −0.423 vs −0.436 became **−0.510** at 10 seeds. Every crossing
found before round 8 was a small-sample artifact — see
[the methodological lesson](RESULTS.md#round-8).

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

All numbers are **test-split Sharpe**, from the JSON in `runs_rolling/` and
`runs_bootstrap/` — mean over 3 seeds except round 8, which is 10. Folds are
expanding-window walk-forward; see [NOTES.md](NOTES.md#fold-naming) for the
fold table and a naming quirk worth knowing before you read directory names.
**Treat any 3-seed margin below ~0.15 as unresolved** — round 8 showed why
([lesson](RESULTS.md#round-8)).

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
| `_lowlr` + **block bootstrap** | ~~−0.423~~ | ~~1.319~~ | superseded — 10 seeds give −0.510 / 1.293, see round 8 |

### Round 8: bootstrap augmentation, 10 seeds × 5 folds (100 runs)

| fold (test year) | baseline `_lowlr` | + bootstrap | difference | t | verdict |
|---|---|---|---|---|---|
| 1 (2021) | 2.057 ± 0.126 | 2.116 ± 0.217 | +0.059 | 0.74 | noise |
| 2 (2022 bear) | −0.656 ± 0.166 | **−0.510** ± 0.144 | **+0.146** | **2.10** | **bootstrap better** |
| 3 (2023) | 1.198 ± 0.109 | 1.146 ± 0.158 | −0.053 | −0.87 | noise |
| 4 (2024) | 1.410 ± 0.147 | 1.418 ± 0.161 | +0.008 | 0.12 | noise |
| 5 (2025+) | 1.250 ± 0.102 | 1.293 ± 0.140 | +0.043 | 0.79 | noise |

And against the benchmark, with each cell's own standard error:

| fold | benchmark | baseline gap (t) | bootstrap gap (t) |
|---|---|---|---|
| 1 (2021) | 2.225 | −0.168 (**−4.22**) | −0.109 (−1.59) |
| 2 (2022 bear) | −0.436 | −0.220 (**−4.18**) | −0.074 (−1.61) |
| 3 (2023) | 1.140 | +0.058 (1.70) | +0.005 (0.10) |
| 4 (2024) | 1.404 | +0.005 (0.12) | +0.014 (0.27) |
| 5 (2025+) | 1.344 | −0.094 (**−2.90**) | −0.050 (−1.14) |

Neither cell beats the benchmark on any fold. Augmentation's contribution is to
stop losing to it. Fold 5's benchmark is 1.344 rather than 1.409 because its
test window tracks the data vintage — see [RESULTS.md](RESULTS.md#round-8).
Reproduce with `python scripts/compare_cells.py attention_frozen_excess_lowlr
attention_frozen_excess_lowlr_bootstrap --benchmark
runs_rolling/equal_weight_benchmark_round8.json`.

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

## What's been ruled out

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

## What we learned

Four things about the problem:

1. **The benchmark is the hard part, not the model.** Equal weight is not a weak
   baseline — it is the rational allocation under no predictive information, it
   trades almost nothing, and it earns the same risk premia the model reaches
   for. Beating it needs an edge; no amount of optimization substitutes for one.
2. **When the baseline is built from your inputs, better inputs don't help.**
   The basket-quality trap: concentrating into stronger assets lifted the model
   *and* its benchmark together. `basket_div` posted the project's best absolute
   Sharpe (1.446) and lost to its own benchmark (1.638) by more than the
   original gap. This generalizes past finance — any relative objective whose
   baseline is a function of the same inputs is immune to input quality.
3. **A near-uniform policy is the right answer to no signal, not a broken one.**
   Trained rollouts hold weights in 1.6–15.5% around a 7.14% uniform, and 2–6%
   cash where 36% was reachable — never within 90% of the action-space bound on
   any of 753 days. The agent could concentrate and chose not to. That also
   explains why every lever moved Sharpe so little: each was resizing a small
   tilt.
4. **The room a learned policy has is where fixed weights can't adapt.** The only
   statistically real fold effect in the project is augmentation on the 2022 bear
   market (+0.146, t = 2.10) — the one window where the benchmark loses money.
   Equal weight cannot de-risk; a policy can.

And five about doing the research, which transfer further:

5. **Measure your precision before running experiments.** Seed sd here is
   0.10–0.25 Sharpe, so a 3-seed standard error (0.06–0.14) is larger than every
   margin this project ever claimed. Eight rounds ran at a precision that could
   not resolve what they tested; two "findings" evaporated at 10 seeds.
6. **One test window is a hypothesis, not a finding.** Reward shaping looked like
   a real gain on a single window and won 1 of 20 comparisons across regimes.
7. **Research bugs return plausible numbers, not errors.** Both found here were
   silent: SB3 overwriting the pretrained encoder (caught because two cells
   agreed to the last decimal) and one `NaN` RSI column deleting 81% of the panel
   (caught because a fold reported 59 days instead of 360). Validate invariants,
   not just outputs.
8. **Leakage comes through side doors.** Features were clean from day one; the
   near-miss was in *asset screening* — ranking assets on test-window data, then
   evaluating the chosen basket on those same windows.
9. **Derived artifacts must match what they judge.** A six-day-newer data cache
   moved a benchmark by 0.066 (larger than most effects here), and a reporting
   script silently scored 6-asset baskets against the 13-asset bar, inventing
   three crossings. Enforce it in code, don't remember it.

Full reasoning and numbers: [Conclusions](RESULTS.md#conclusions).

## What would justify resuming

**Not another optimizer lever.** The untested knobs (`ent_coef`, reward scale,
`turnover_penalty`) would make the policy tilt harder; with no signal to tilt on,
the likeliest outcome is an unchanged mean at higher variance. Three things would
genuinely change the odds:

1. **A different objective.** Drawdown-constrained or volatility-targeted
   allocation, scored on Calmar or max drawdown rather than Sharpe-vs-equal-weight.
   This competes where a policy has structural advantage rather than requiring it
   to out-predict the market, and it is where finding 4 above points. Reuses
   nearly all of this code: a new `reward_type`, new metrics.
2. **Different information.** Point-in-time macro vintages, intraday bars,
   options-implied volatility, higher-frequency credit, cross-sectional
   fundamentals. Daily adjusted closes on the most liquid ETFs in existence,
   with features standard since the 1980s, is the most heavily mined dataset in
   finance — the null result is what that prior predicts.
3. **A less efficient market.** A different asset class, at the cost of more
   noise and more regime risk.

Broker integration stays unbuilt, which was the right call throughout.

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
