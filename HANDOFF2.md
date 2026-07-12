# Handoff 2 — RL ETF Trading Bot

Continuation of `HANDOFF.md`, which stopped mid-way through the first 5-fold
rolling validation (4/15 cells done). That validation finished, and a lot
happened after — this doc picks up from there. Read `DESIGN.md` for original
architecture rationale, `HANDOFF.md` for the bug-fix/reward-shaping history,
this doc for everything since. The [research notebook artifact](https://claude.ai/code/artifact/69d4ed51-bcc5-4850-be5f-f48b616f1d27)
has a visual walkthrough of rounds 1-3 below (published mid-session, not
updated with rounds 4-7 — numbers here are the current source of truth).

## TL;DR

- **The repo is now a git repo**, pushed to a **private GitHub repo**:
  `aravgupta0707-gif/rl-trading-bot`. `gh` CLI installed + authenticated on
  this machine. Root log files (`*.log`) and `cache/` are gitignored;
  everything else (code, configs, and all `runs*/` results including
  `model.zip` checkpoints) is committed. **Nothing has been committed since
  the initial commit** — nine rounds of work below are uncommitted.
- **Core finding across rounds 1-2 (60 models, 4 cells × 5 folds): reward
  shaping and encoder architecture (linear vs. attention) are not the
  bottleneck.** Only 1 of 20 fold results beat its benchmark, and that one
  (fold 3, +0.008) is inside a single seed's noise. This overturned the
  single-window result `HANDOFF.md` ended on.
- **Round 3 (PPO optimization ablation) settled the "is PPO even
  converging" question: no.** Given 3× the training steps, entropy/policy
  std genuinely converge (entropy −19.9→−9.1, std 1.0→0.49) — but Sharpe
  barely moves. The bottleneck isn't optimization budget.
- **Round 4 (block-bootstrap data augmentation) is the first real positive
  signal**: doubling effective training history via block-resampled
  synthetic data moved the worst fold (2022 bear) from −0.564 to **−0.423,
  fractionally past its −0.436 benchmark** — the first of 30+
  configurations tested to cross the line. Seed spread still wide, not
  proven yet.
- **Round 5 (wider observed context — credit spreads, curve shape, dollar,
  oil, leading index, 6 observed-only ETFs) gave a real but modest lift**
  on top of the plain baseline, without touching the tradeable universe.
- **Round 6-7 (narrowing/rebuilding the tradeable basket) is the most
  important methodological finding of this half of the session**: raising
  basket quality raises the naive benchmark *at least* as much as it raises
  the model, so the gap doesn't close by picking better assets alone — see
  "The basket-quality trap" below. Also surfaced and fixed a **real latent
  bug**: near-zero-volatility instruments (T-bills) broke the RSI feature
  and silently truncated the whole panel.
- **No broker/paper-trading integration exists anywhere in this repo.**
  Still correctly unbuilt — no cell has robustly cleared its benchmark yet.

## Chronological findings, rounds 1-7 (read in order if picking this up cold)

### Round 1-2: reward shaping × architecture, across 5 regime folds

`HANDOFF.md` ended believing reward shaping had narrowed the gap on one
static window (Sharpe 1.259 vs. 1.395 benchmark). Retested both reward
types (`log_return_net_cost` / `excess_return_net_cost`) × both encoders
that had shown promise (`attention`, `linear`) across the 5 rolling folds
(2021 bull, 2022 bear, 2023 recovery, 2024, 2025+), 3 seeds each — 60
models. Full table in `runs_rolling/comparison_summary_v2.txt` and the
research notebook artifact. **Result: 1 real win out of 20 fold-level
comparisons**, and it's noise (fold 3, attention+excess, +0.008 on a std of
0.173). Reward shaping helps in some folds and hurts in others vs. the
plain baseline — a wash, not a fix. Linear vs. attention: comparable
Sharpe, but linear has meaningfully higher seed variance and ~2x the
turnover — attention remains the better-behaved choice even without a
clear Sharpe edge.

### Round 3: is PPO even converging?

Every cell above trained for 200k timesteps with entropy loss and policy
std staying almost flat — the policy visibly never sharpens. Isolated the
question on `attention_frozen_excess` (best cell), fold 2 (2022 bear,
worst result) + fold 5 (2025+, most decision-relevant), 3 seeds:
- **Bigger policy/value head** (`net_arch` `[32]`→`[64,64]`): no help on
  fold 2, made fold 5 *worse* (0.992 vs. 1.231 baseline). Ruled out.
- **Lower learning rate** (3e-4→1e-4): best result of the ablation on both
  folds (fold2 −0.564, fold5 1.351), without much entropy movement — a
  side effect of smaller/more conservative gradient steps, not "the policy
  converged further."
- **3× longer training** (200k→600k timesteps): the decisive test. Entropy
  and std **do** genuinely converge (see the notebook's line charts) — but
  fold2/fold5 Sharpe only nudge to −0.582/1.308. **Given the policy is
  demonstrably capable of converging, and convergence doesn't close the
  gap, the bottleneck isn't optimization** — see round 4/5.

New optional PPO config keys added to support this (backward-compatible,
default to prior behavior if absent): `ppo.net_arch`, `ppo.learning_rate`
(`training/train_ppo.py::train()`). New `--folds` filter added to
`scripts/run_rolling_validation.py` so ablations don't need to retrain all
5 folds.

### Round 4: block-bootstrap data augmentation

If PPO converges fine but finds no edge, the likely culprit is too little
distinct real history — each fold has only 1-4 years of daily data. Built
`features/bootstrap.py::generate_synthetic_panel()`: resamples ~21-day
blocks of real historical returns/volume/macro *together* (same block
indices across all series, preserving real cross-asset co-movement and fat
tails — deliberately not parametric GBM, which assumes Gaussian returns
and would smooth over the exact regime shifts being tested against).
Doubled `attention_frozen_excess_lowlr`'s training data this way via
`scripts/run_bootstrap_ablation.py`, evaluated only on real data (same
protocol as everywhere else — no leakage). **Result: fold2 mean Sharpe
−0.564 → −0.423, past that fold's −0.436 benchmark** — first real crossing
in the whole search. Fold5 stayed flat (1.319 vs. 1.351, no regression).
3-seed spread on fold2 is still wide (−0.26 to −0.60) — needs more seeds
and all 5 folds before trusting it. **This is the strongest untouched
follow-up thread.**

### Round 5: wider observed context, same tradeable universe

Everything above used the exact same 70-column feature panel. Widened what
the encoder *observes* without touching the *tradeable* action space: added
6 FRED series (`BAA10Y` credit spread, `T10Y3M` curve, `ICSA` jobless
claims, `DTWEXBGS` dollar, `DCOILWTICO` oil, `USSLIND` leading index) and 6
observed-only ETFs (`HYG`/`LQD` credit, `IEF`/`SHY` curve shape, `EEM`
emerging markets, `UUP` dollar) as extra `macro__ctx_*` columns —
`features/panel.py::build_panel()` grew an optional `context_tickers`
param that folds these into the existing macro block, so every encoder
picks them up with **zero changes to `encoders/*.py` or the action space**
(config: `configs/attention_frozen_excess_widectx.yaml`). Result: real,
modest lift over baseline on both test folds (fold2 −0.760→−0.652, fold5
1.231→1.296) — still losing to benchmark, but a consistent positive
direction from genuinely new information rather than more training.

### Round 6-7: narrowing/choosing the tradeable basket — the basket-quality trap

Built `scripts/asset_promise.py`: two independent, zero-retrain-needed
views on "which individual assets show promise" — (a) revealed preference
(average weight the already-trained models actually allocate, per asset)
and (b) a model-free cross-check (plain buy-and-hold Sharpe per asset).
**Important methodology fix made along the way**: the first pass computed
buy-and-hold Sharpe aggregated across all 5 folds' *test* windows, which
would leak if used to pick a basket then evaluated on those same folds
(caught before it shipped). Redid it using only pre-2020 data (the one
cutoff legitimately out-of-sample for every fold) — this flips several
rankings, e.g. XLE goes from "4th best" (leaky, dominated by the 2022 oil
spike) to dead last, TLT flips from worst to mid-pack (its collapse was the
2022+ regime, invisible pre-2020). **Any future asset screening must use
this pre-2020-only methodology, not the aggregate-test-window one.**

Also found: **cash allocation shows no regime-adaptive behavior at all** —
average cash weight in the 2022 bear fold (0.0266) is not meaningfully
different from the best fold's (0.0424 in fold1, actually *lower* than
several good folds). The model never learned to get defensive. This is a
distinct, additional explanation for why it loses badly specifically in
bad regimes, separate from "no rotation signal."

Widened the candidate universe with `scripts/screen_etf_universe.py`: ~67
liquid, non-leveraged, pre-2016-inception ETFs across every major category
(style factors, single countries, credit/duration tiers, commodities,
currencies, sub-sectors) — **not** a literal "top 100 by volume" (checked:
that's dominated by leveraged/inverse/single-stock-daily-target products
with decay mechanics, e.g. Yahoo Finance's most-active list was headed by
`SOXS`/`TZA`/`MSTU`/`TQQQ`; excluded categorically). Found LQD/HYG (credit)
and `USMV` (min-vol factor) rank near the top; `BIL` (T-bills) tops the
list but its Sharpe is a near-zero-volatility artifact, not a real
"opportunity" per se.

**Then built `scripts/build_diversified_basket.py`** because a basket that
*sounds* diversified (different sector names) can still be highly
correlated. Computed the actual pre-2020 pairwise correlation matrix.
Confirmed: `XLK`/`XLI`/`XLF`/`SMH` are 0.61-0.86 correlated with each
other, and even `EFA` (international) is 0.68-0.75 correlated with US
equity sectors — a basket built from these "sounds diversified" but mostly
isn't. Real finding: **`LQD` (investment-grade credit) has ~0 correlation
with equities (0.05); `HYG` (high-yield) is highly correlated (0.66)** —
junk bonds carry equity-like risk, investment-grade doesn't. A pure
greedy min-correlation selection (no economic-sense constraint) picks
odd-but-genuinely-uncorrelated things like Brazil/Hong Kong single-country
funds and `ARKK` — mathematically diverse, not obviously a good basket
either. **This tension (correlation-optimal vs. economically-sensible) is
unresolved — worth a proper constrained selection next, not another ad hoc
pick.**

Tested two narrowed baskets on fold2/fold5 (both against their own
freshly-computed 6-asset equal-weight benchmarks, since "equal-weight"
means something different at 6 assets than 13):

| | model fold2 | own benchmark | model fold5 | own benchmark |
|---|---|---|---|---|
| `basket_div` (XLK,XLI,XLF,XLU,EFA,GLD) | −0.774 | −0.404 | **1.446** (best absolute Sharpe seen anywhere) | 1.638 |
| `basket_sharpe` (XLK,XLU,XLY,XLI,XLF,XLV, no diversifiers) | −0.824 | −0.578 | 0.992 | 1.166 |

**The basket-quality trap**: `basket_div`'s model achieved the best
absolute Sharpe of the entire project (1.446) — but its own benchmark rose
even more (1.638), so the *gap* is actually slightly wider than the
original 13-asset baseline's. This is the exact same pattern `HANDOFF.md`
documented when the basket first expanded 10→13 assets: concentrating into
stronger performers raises the model and the naive benchmark together,
roughly in lockstep. **Picking "better" assets by trailing quality doesn't
by itself close the gap** — the model needs to be doing something the
equal-weight benchmark structurally can't, not just holding better stuff.

**Bug found and fixed**: `basket_universe` (XLK,XLI,XLF,SMH,LQD,HYG,BIL,GLD,EFA,
combining the twice-validated top sectors + credit + a `BIL` safe-harbor
pick) initially produced a fold5 result with only 59 days of test data
instead of ~360. Root cause: `BIL`'s price is so stable that its 14-day
RSI window frequently has **zero down-days**, making `loss=0` →
`gain/0=NaN` → that single feature column silently poisons
`build_panel()`'s global `dropna()`, wiping ~81% of recent rows across
*every* asset in the basket. None of the original 13 tickers ever
triggered this (normal volatility always has some down-days) — any future
near-zero-volatility instrument (short bond funds, cash-equivalents) would
hit the same bug. **Fixed** in `features/panel.py::_etf_features()`:
standard RSI convention is `loss==0 → RSI=100`, not `NaN`
(`rsi.where(loss != 0, 100.0)`). Verified: 99→512 rows recovered, zero NaN
remaining, and the rerun's `n_days` came back correct (251/364, matching
every other fold). **Treat any pre-fix `basket_universe` numbers (fold5
`n_days=59`) as invalid** — superseded by the numbers below.

**`basket_universe` rerun result** (own benchmark, not the 13-asset one):

| | model | own benchmark | gap |
|---|---|---|---|
| fold2 (bear) | −0.790 | −0.722 | −0.068 |
| fold5 (recent) | 1.583 | 1.688 | −0.105 |

Still losing both folds, but **the smallest gap of any basket-narrowing
variant tested** — tighter than `basket_div`'s (−0.370 / −0.192) or
`basket_sharpe`'s. The credit/safe-harbor additions (LQD, HYG, BIL) pull
their weight even though the basket doesn't clear the bar. Also notably:
`basket_universe`'s own benchmark (fold2 −0.722) is *worse* than
`basket_div`'s (−0.404) despite basket_universe including genuine
diversifiers (LQD, GLD, BIL) — a reminder that the correlation-vs-quality
tension flagged above is still unresolved, not that diversifiers don't
help; this basket's non-diversifier picks (XLK/XLI/XLF/SMH, all highly
correlated per the matrix above) likely dominate its benchmark's behavior.

## Repo/directory guide (additions since HANDOFF.md)

- **This is now a git repo**, private GitHub remote `aravgupta0707-gif/rl-trading-bot`,
  `gh` CLI authenticated. One commit so far (everything through the end of
  the original `HANDOFF.md`). **Commit the work described in this doc.**
- `runs_bootstrap/` — round 4's bootstrap-augmented training runs.
- `cache_widectx/`, `cache_basket_div/`, `cache_basket_sharpe/`,
  `cache_basket_universe/`, `cache_universe_screen/` — separate cache dirs
  per experiment, since `fetch_etf_prices`/`fetch_macro` cache to one file
  per `cache_dir` regardless of which tickers were requested (same gotcha
  `HANDOFF.md` flagged — each new ticker/macro set needs its own cache_dir,
  not a shared one).
- `runs_rolling/equal_weight_benchmark*.json` — per-basket equal-weight
  benchmarks (one for the original 13-asset universe, one each for
  `basket_div`/`basket_sharpe`/`basket_universe`). **Don't compare a
  narrowed basket's model against the 13-asset benchmark** — always use
  the basket's own.
- `runs_rolling/etf_universe_screen.json` — the ~67-ticker pre-2020 Sharpe
  screen.
- `runs_rolling/diversified_basket_selection.json` — the pure
  correlation-greedy basket selection (the one with Brazil/Hong
  Kong/ARKK — see round 6-7 notes, not currently used as an actual config).
- `runs_rolling/attention_frozen_excess_asset_promise.json` — revealed-
  preference + buy-and-hold Sharpe per asset, original 13-asset basket.
- Root still has `overnight_run[1-7].log` from this session's background
  jobs — same "safe to delete once captured" status as before, not done
  yet.

## Key files added/changed this session (on top of HANDOFF.md's list)

- `training/train_ppo.py` — optional `ppo.net_arch`/`ppo.learning_rate`
  config overrides (round 3).
- `scripts/run_rolling_validation.py` — `--folds` filter.
- `scripts/rolling_benchmark.py`, `scripts/rolling_summary_compare.py` —
  per-fold benchmark computation + multi-cell comparison table.
- `scripts/ensemble_backtest.py` — averages the 3 trained seeds' action
  logits at each timestep; small, consistent, real improvement (mostly
  lower turnover/variance) but not enough to flip any conclusion — worth
  keeping as standard practice going forward regardless.
- `features/bootstrap.py`, `scripts/run_bootstrap_ablation.py` — round 4.
- `features/panel.py::build_panel()` — optional `context_tickers` param
  (round 5); RSI zero-loss fix (round 6-7, see above).
- `scripts/asset_promise.py`, `scripts/screen_etf_universe.py`,
  `scripts/build_diversified_basket.py` — round 6-7 tooling.
- `configs/attention_frozen_excess_{bighead,lowlr,longtrain}.yaml` (round 3),
  `configs/attention_frozen_excess_widectx.yaml` (round 5),
  `configs/attention_frozen_excess_basket_{div,sharpe,universe}.yaml`,
  `configs/linear_frozen_excess.yaml` (round 6-7).

## Immediate next steps

1. **Let the `basket_universe` rerun finish**, read its real fold2/fold5
   numbers against its own benchmark.
2. **Resolve the correlation-optimal-vs-economically-sensible basket
   tension** flagged in round 6-7 — a proper constrained selection
   (diversify across meaningful buckets, best Sharpe within each) rather
   than either pure category intuition or pure greedy correlation
   minimization.
3. **Bootstrap augmentation (round 4) is still the strongest untouched
   lead** — more seeds, all 5 folds, and worth combining with round 5's
   wider context and/or a genuinely diversified basket rather than testing
   levers in isolation.
4. **The "never goes defensive" finding is unaddressed** — cash allocation
   doesn't rise in the bear fold. Worth investigating directly (e.g. does
   giving the model a yield-bearing safe-harbor asset like `BIL`, now that
   the RSI bug is fixed, actually change this behavior?) rather than
   inferring it only through aggregate Sharpe.
5. **Commit and push everything in this doc to git** — nothing since the
   initial commit has been pushed.
6. Still correctly untouched: point-in-time macro data, broker integration
   — no cell has robustly cleared its benchmark yet.
