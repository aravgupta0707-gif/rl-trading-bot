# RL ETF Trading Bot — Design Notes

## Goal

An RL-based trading bot over a basket of ETFs, with a NN or linear
preprocessing layer in front of the RL agent. Daily bars, backtest-first,
architected so it can eventually connect to a broker for paper/live trading.

## Key design decisions (and why)

### Scope: multi-ETF portfolio allocation, not single-asset timing
The action space is a target weight vector across a basket of ETFs + cash,
rebalanced daily — not a long/flat/short call on one instrument.

### Cross-asset + macro effects are a first-class requirement
Prices of other ETFs in the basket should affect how any single ETF is
traded (e.g. rate moves affecting rate-sensitive vs. cyclical sectors
differently), and macro series should feed in directly. This ruled out
building one isolated per-ETF model — the preprocessing layer has to see the
whole basket + macro at every timestep, not just one ETF's own history.

### The preprocessing layer must NOT be supervised with a future-return label
Early on, the natural framing was "signal generator" (supervised model
predicting next-period return, feeding the prediction into the RL state) vs.
"feature compressor" (unsupervised dimensionality reduction). Those converge
in shape — both map raw features to a smaller vector — and differ only in
what supervises them. The decision here was explicitly **against** the
signal-generator framing: no component in the pipeline is trained against a
known future return. This ruled out ridge/linear regression-on-returns as
the preprocessor and left two options for how the encoder actually learns
anything useful:

- **End-to-end**: the encoder is just the front end of the PPO policy/value
  network, shaped purely by reward gradients, no separate loss.
- **Unsupervised pretraining**: an autoencoder/PCA-style reconstruction
  objective on the cross-sectional feature panel, with no reference to what
  happens next — frozen or fine-tuned before/during RL training.

### Don't pick one design a priori — build all variants, compare empirically
Rather than deciding linear-vs-NN or frozen-vs-finetune-vs-end-to-end by
argument, the decision was to implement all combinations behind one shared
interface (config-driven) and let backtested performance (Sharpe, max
drawdown, turnover on a held-out test split, averaged over seeds) pick the
winner. This gives a 3×3 grid:

|                | frozen | finetune | end-to-end |
|----------------|--------|----------|------------|
| **linear** (PCA-equivalent) | | | |
| **MLP** (nonlinear autoencoder) | | | |
| **attention** (cross-ETF self-attention) | | | |

Practical sequencing: build the code for all 9 cells (cheap — it's a config
flag, not separate codebases), but validate the shared plumbing (data →
panel → env → reward → PPO loop) on the single cheapest cell first
(linear + frozen) before spending compute running the full grid — a bug in
shared code corrupts every cell identically and would be invisible from the
comparison alone.

**Working prior** (not yet confirmed empirically): linear/PCA likely wins
given the data size (~10 ETFs, a few thousand daily rows) — not enough data
for a nonlinear encoder to reliably beat overfitting noise. If an NN encoder
does win, it's expected to be the attention variant, since attention
explicitly models "how much does every other ETF influence this one,"
matching the cross-asset requirement directly, rather than an MLP that has
to learn that structure implicitly from a flattened input.

## Data sources

- **ETF prices**: Yahoo Finance via `yfinance` — daily split/dividend-adjusted
  OHLCV, one request per ticker. Basket: 10 sector SPDRs (XLK, XLF, XLE, XLV,
  XLY, XLP, XLI, XLU, XLB, XLRE) + TLT (long treasuries), GLD (gold), EFA
  (developed-market ex-US equities) — added after the initial grid showed the
  winning cell barely beating a naive equal-weight benchmark. All-sector-SPDR
  is mostly undifferentiated equity beta; TLT/GLD/EFA give the model assets
  with genuinely different macro sensitivities to actually rotate across.
- **Macro series**: FRED via `pandas_datareader`, no API key required — fed
  funds rate, 10Y-2Y yield curve spread, 10Y breakeven inflation, VIX,
  unemployment rate.
- Both cache to local parquet (`cache/etf_prices.parquet`,
  `cache/macro.parquet`) so repeat runs don't hit the network; `refresh=True`
  forces a re-download.
- Known limitation: no point-in-time vintage for macro data (FRED values are
  used as currently reported, not as they would've appeared in real time
  before revisions) — acceptable for research/backtesting, would need fixing
  before live trading.

## Repo layout

```
data/fetch.py            fetch_etf_prices(), fetch_macro() — cache-first downloaders
features/panel.py        build_panel() — cross-sectional feature panel (70 cols):
                          per ETF: 1d/5d/20d returns, 20d vol, 14d RSI, 20d volume z-score;
                          per macro series: rolling z-score + 5d change.
                          Every ETF's row also carries every other ETF's features.
                          No column is a future value.
features/splits.py       walk_forward_split() (train/val/test by date),
                          rolling_walk_forward_split() (same, with an explicit test_end
                          so multiple folds get comparable-length windows),
                          fit_normalizer()/apply_normalizer() (train-only z-score)
encoders/base.py          Encoder interface: forward(x)->latent, reconstruct(x), reconstruction_target(x)
encoders/linear.py        LinearEncoder — single unbiased linear layer (PCA-equivalent subspace)
encoders/mlp.py           MLPEncoder — input->64->latent->64->input autoencoder
encoders/attention.py     AttentionEncoder — each ETF as a token, self-attention across the
                          basket + a macro token, per-ETF contextualized output preserved
envs/portfolio_env.py     PortfolioEnv(gym.Env) — obs = normalized feature row + current weights;
                          action = logits -> softmax -> long-only weights (ETFs + cash);
                          reward = log(1 + portfolio return - transaction cost);
                          compute_forward_returns() derives real next-day returns from raw
                          prices for the reward only (never used as an input feature)
training/pretrain_encoder.py  Unsupervised reconstruction-MSE training loop + freeze()
training/train_ppo.py     EncoderFeaturesExtractor (custom SB3 feature extractor wiring the
                          encoder into PPO's policy/value network); prepare_encoder() implements
                          the frozen/finetune/e2e regimes
eval/metrics.py           Sharpe, annualized return, max drawdown, avg daily turnover
eval/backtest.py          run_backtest() — deterministic rollout of a trained model over a split
configs/default.yaml      ETF list, macro series, date ranges, split boundaries, env costs, PPO hyperparams
configs/{enc}_{regime}.yaml   9 grid configs, each overriding just encoder.type/encoder.regime
                          (deep-merged onto default.yaml by configs/loader.py)
scripts/run_experiment.py CLI: fetch -> panel -> split/normalize -> train -> backtest val+test ->
                          runs/<name>/{model.zip, config.json, metrics.json}
scripts/run_grid.py       Runs the full encoder x regime x seed grid in one process (panel built
                          once, reused across cells); --skip-existing to resume
scripts/diagnostics.py    Full daily trajectory (equity, weights, turnover) for one trained run,
                          not just summary stats -- runs/<name>/diagnostics.json
scripts/run_rolling_validation.py  Retrains/backtests one cell across 5 sequential folds spanning
                          different regimes (2020 COVID, 2021 bull, 2022 bear, 2023 recovery,
                          2024+) instead of one static split -- runs_rolling/
```

## Status

**Bug found and fixed (2026-07-09):** the first full 9-cell grid run produced
byte-identical results for `finetune` and `e2e` on the linear and MLP
encoders. Root cause: stable-baselines3's `ActorCriticPolicy._build()`
re-applies its default orthogonal init to every `nn.Linear`/`nn.Conv2d` in
the *entire* features extractor — including our custom encoder — right after
PPO construction, silently discarding `prepare_encoder()`'s pretrained/frozen
weights. This had been true since the original smoke test. Fixed in
`training/train_ppo.py::train()` by snapshotting the encoder's `state_dict()`
before `PPO(...)` construction and restoring it after. Verified at smoke
scale (all three regimes now diverge for all three encoder types) before
rerunning the full grid.

**Grid result (10-ETF basket, post-fix):** `attention_frozen` wins on test
Sharpe (1.15) and by far the lowest turnover (0.15 vs 0.28-0.42 for every
other cell), consistent with this project's stated prior that attention
should win if any NN encoder does. `frozen` beat `finetune`/`e2e` for all
three encoder types once the bug was fixed, i.e. the pretraining objective is
doing real work now. **Caveat that matters:** `attention_frozen`'s Sharpe
(1.15) did *not* clearly beat a naive daily-rebalanced equal-weight
buy-and-hold benchmark over the same test window (Sharpe 1.28, similar
drawdown) — the model's only clear edge was turnover, not risk-adjusted
return. Full diagnostics (equity curves, drawdown, allocation, seed
variability) were written up as an HTML report.

## Next steps

- **In progress:** basket expanded from 10 sector SPDRs to 13 assets (see
  Data sources) for more genuine cross-sectional dispersion, plus rolling
  walk-forward validation (`scripts/run_rolling_validation.py`) across 5
  regime-diverse folds to check whether the winning cell's edge is stable
  over time rather than an artifact of one test window.
- Re-confirm which encoder/regime wins on the expanded basket (architecture
  ranking may shift with more, more-differentiated assets).
- Revisit the benchmark comparison to be cost-matched (the equal-weight
  baseline currently assumes free daily rebalancing, which understates its
  real-world cost relative to the model).
- Only after a cell clearly clears the benchmark: consider point-in-time
  macro data and broker-API integration for paper trading (none exists yet).
