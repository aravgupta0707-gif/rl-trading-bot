# Design decisions

Why the system is built the way it is. This document is deliberately about
*reasoning*, not results — findings live in [RESULTS.md](RESULTS.md), current
status in [README.md](README.md), and operational traps in
[NOTES.md](NOTES.md).

## Goal

An RL trading agent over a basket of ETFs, with a learned representation layer
in front of the policy. Daily bars, backtest-first, architected so it could
eventually connect to a broker — but only if it earns that.

## Scope: portfolio allocation, not single-asset timing

The action space is a **target weight vector across the whole basket plus
cash**, rebalanced daily — not a long/flat/short decision on one instrument.
Allocation is the setting where cross-asset information actually matters, and
it makes the benchmark unambiguous: an equal-weight basket of the same assets.

## Cross-asset and macro effects are a first-class requirement

Every ETF's features should influence how any other ETF is traded (rate moves
hit rate-sensitive and cyclical sectors differently), and macro series feed in
directly. This ruled out a per-ETF model ensemble: the representation layer has
to see the **whole basket plus macro at every timestep**, which is why each
panel row carries every asset's features rather than one asset's own history.

## The encoder must not be supervised on future returns

The early framing was "signal generator" (a supervised model predicting
next-period return, whose prediction enters the RL state) versus "feature
compressor" (unsupervised dimensionality reduction). Those two converge in
shape — both map raw features to a smaller vector — and differ only in what
supervises them.

The decision was explicitly **against** the signal-generator framing: no
component in this pipeline is trained against a known future return. A
return-predicting front end makes leakage nearly impossible to reason about,
and it quietly turns the RL agent into a thin wrapper around a forecast.

That ruled out regression-on-returns as the preprocessor and left two honest
ways for the encoder to learn anything:

- **End-to-end** — the encoder is just the front of the PPO policy/value
  network, shaped only by reward gradients.
- **Unsupervised pretraining** — a reconstruction objective on the
  cross-sectional panel, with no reference to what happens next, then frozen
  or fine-tuned during RL training.

Reward is the only place future information legitimately enters:
`compute_forward_returns()` derives real next-day returns from raw prices for
the reward alone, and they are never an input feature.

## Build every variant, let backtests choose

Rather than settle linear-vs-neural or frozen-vs-finetune-vs-end-to-end by
argument, all combinations were implemented behind one interface and selected
empirically on held-out data, averaged over seeds:

|  | frozen | finetune | end-to-end |
|---|---|---|---|
| **linear** (PCA-equivalent) | ✓ | ✓ | ✓ |
| **MLP** (nonlinear autoencoder) | ✓ | ✓ | ✓ |
| **attention** (cross-ETF self-attention) | ✓ | ✓ | ✓ |

Since a cell is a config flag rather than a separate codebase, the cost of
building all nine was low — but the shared plumbing (data → panel → env →
reward → PPO) was validated on the cheapest cell first, because **a bug in
shared code corrupts every cell identically and is invisible from the
comparison alone.** That caution was justified; see
[NOTES.md](NOTES.md#bug-1-sb3-silently-discarded-the-pretrained-encoder) for the
bug that did exactly this.

**The working prior**, recorded before running anything: linear/PCA probably
wins, because ~13 assets and a few thousand daily rows is not enough data for a
nonlinear encoder to beat overfitting noise; if a neural encoder does win it
should be attention, which models "how much does every other ETF influence this
one" directly rather than making an MLP infer that structure from a flattened
vector.

Outcome: attention won, and `frozen` beat both `finetune` and `e2e` for all
three encoders — the pretraining objective does real work. Linear was close
enough on Sharpe to matter, but carried ~2× the turnover and higher seed
variance. See [RESULTS.md](RESULTS.md#round-0).

## Benchmark choice: equal-weight buy-and-hold

A daily-rebalanced equal-weight basket of the same assets. It is the honest bar
for an allocator — cheap, obvious, and what a practitioner would actually do
instead — and it was checked for fairness: cost-matched and true buy-and-hold
variants both beat the model too, so its edge isn't an artifact of free
rebalancing.

Two consequences that shape everything else:

- A **narrowed basket needs its own benchmark**, since equal-weight means
  something different at 6 assets than at 13.
- Because the benchmark is built from the same assets, **improving the assets
  improves the benchmark too** — the basket-quality trap in
  [RESULTS.md](RESULTS.md#round-67).

## Data

- **ETF prices** — Yahoo Finance via `yfinance`, daily split/dividend-adjusted
  OHLCV. Default basket: 10 sector SPDRs (XLK, XLF, XLE, XLV, XLY, XLP, XLI,
  XLU, XLB, XLRE) plus TLT (long treasuries), GLD (gold) and EFA (developed
  ex-US equities). The three non-equity additions exist because all-sector-SPDR
  is mostly undifferentiated equity beta — with nothing but correlated
  exposures, there is no rotation for the agent to find.
- **Macro** — FRED via `pandas_datareader`, no API key: fed funds rate,
  10Y-2Y curve, 10Y breakeven inflation, VIX, unemployment. The `widectx`
  experiment adds credit spreads, 10Y-3M curve, jobless claims, the dollar,
  oil and a leading index.
- **Features** — per asset: 1/5/20-day returns, 20-day volatility, 14-day RSI,
  20-day volume z-score; per macro series: rolling z-score and 5-day change.
  88 columns for the 13-asset basket (70 for the original 10-asset one). No
  column is a future value.
- **Splits** — walk-forward by date, with normalizer statistics fit on the
  train split only. Fitting on the full panel would leak test-period
  distribution information into the features the agent trains on.
- **Caching** — both fetchers cache to parquet per `cache_dir`; see
  [NOTES.md](NOTES.md#1-every-distinct-tickermacro-set-needs-its-own-cache_dir),
  since the cache key is the directory alone.

**Known limitation:** macro series are used as currently reported, not as they
would have appeared in real time before revisions. Acceptable for research;
would need point-in-time vintages before any live use.

## Why there is no broker integration

Nothing here beat equal-weight buy-and-hold across regime-diverse folds; the
best configuration reached a statistical tie. Wiring that into a broker would
convert a research negative into a financial one, so it stayed unbuilt — and
with the project concluded, that remains the right call. See
[Conclusions](RESULTS.md#conclusions) for what the evidence supports and what
would have to change first.
