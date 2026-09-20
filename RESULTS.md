# Experiment log

Every round of experiments in order, with what it was testing, what came out,
and what it changed about the plan. All Sharpe figures are test-split, mean
over seeds, recomputed from the committed JSON in `runs*/` rather than copied
from earlier notes.

Rounds 0–2 were run on a single static test window; everything from round 1
onward uses 5-fold rolling walk-forward validation, because the static-window
conclusion turned out to be wrong.

**Contents**

- [Round 0 — architecture grid, and the benchmark that changed the project](#round-0)
- [Round 1–2 — reward shaping × architecture, across 5 regimes](#round-12)
- [Round 3 — is PPO even converging?](#round-3)
- [Round 4 — block-bootstrap augmentation](#round-4)
- [Round 5 — wider observed context](#round-5)
- [Round 6–7 — rebuilding the basket, and the basket-quality trap](#round-67)
- [Round 8 — bootstrap augmentation across all five folds](#round-8)
- [Summary: what each round eliminated](#summary)
- [Conclusions — what this project established](#conclusions)

---

<a name="round-0"></a>
## Round 0 — architecture grid, and the benchmark that changed the project

**Testing:** all 9 combinations of encoder (linear / MLP / attention) × training
regime (frozen / finetune / end-to-end), 3 seeds each, on a 10-asset
sector-SPDR basket with the absolute log-return reward.

The first full grid was invalid and had to be rerun — `finetune` and `e2e`
produced byte-identical metrics for the linear and MLP encoders, which is the
kind of coincidence that never is one. Root cause and fix are in
[NOTES.md](NOTES.md#bug-1-sb3-silently-discarded-the-pretrained-encoder); it
had been corrupting every pretrained cell since the original smoke test.

**Post-fix grid** (test Sharpe, 3-seed mean, 10-asset basket):

| cell | Sharpe | turnover |
|---|---|---|
| **`attention_frozen`** | **1.150** | **0.154** |
| `mlp_frozen` | 0.991 | 0.283 |
| `linear_frozen` | 0.924 | 0.348 |
| `attention_finetune` | 0.861 | 0.424 |
| `attention_e2e` | 0.858 | 0.407 |
| `linear_finetune` / `linear_e2e` | 0.869 / 0.793 | 0.356 / 0.387 |
| `mlp_finetune` / `mlp_e2e` | 0.782 / 0.823 | 0.372 / 0.408 |

`frozen` beat `finetune` and `e2e` for all three encoder types — the opposite
of what an undetected weight-wiping bug would produce, which is good evidence
the fix was real. Attention won on Sharpe and used less than half the turnover
of any other cell, matching the project's stated prior (see
[DESIGN.md](DESIGN.md)) that attention should win if any neural encoder does.

**Then the pivotal check:** a naive daily-rebalanced equal-weight basket scored
Sharpe **1.28** over the same window — *beating* the winning cell's 1.15.
Everything since has been an attempt to close that gap.

Attention's low turnover was recorded here as its "one clear edge", and that
was only ever true *relative to the other RL cells*. Measured against the
benchmark it is a handicap: the best cell trades ~11×/year (0.044 daily,
≈0.55%/yr at 5 bps) while equal-weight barely trades at all — cost-matching the
benchmark costs it just 0.007 Sharpe. There is no cost-efficiency story here,
and later rounds should not have leaned on one.

Two follow-ups, both negative:

- **Basket expansion, 10 → 13 assets** (added TLT long treasuries, GLD gold,
  EFA international — sector SPDRs alone are mostly undifferentiated equity
  beta). The benchmark's Sharpe rose *more* than any cell's, to 1.395. Gap
  unchanged. First sighting of what round 6–7 named the basket-quality trap.
- **Cost fairness**, since the equal-weight baseline rebalances for free: a
  cost-matched variant (1.388) and a true buy-and-hold variant (1.360) both
  still beat every RL cell. Not a costing artifact.

**Reward shaping** was the one thing that did help here. The original reward,
`log(1 + return − cost)`, asks the policy to maximize its own return with no
reference to the benchmark; if no strong rotation signal exists, its natural
equilibrium is "diversify like the benchmark, but pay turnover getting there."
Adding `excess_return_net_cost` — return minus the same-day equal-weight basket
return, net of cost — gave:

| cell (13-asset basket, 3 seeds) | Sharpe | sd | turnover |
|---|---|---|---|
| `attention_frozen_excess` | **1.259** | 0.053 | 0.109 |
| `mlp_frozen_excess` | 1.168 | 0.140 | 0.267 |

Higher *and* tighter across seeds than the log-return era, against a 1.395
benchmark — the gap looked like it had narrowed from ~0.18 to ~0.14. On one
window. Round 1 tested whether that survived.

<a name="round-12"></a>
## Round 1–2 — reward shaping × architecture, across 5 regimes

**Testing:** both reward types × both promising encoders (attention, linear)
across 5 rolling walk-forward folds spanning distinct regimes, 3 seeds each —
60 models. This is the experiment that overturned round 0's conclusion.

| cell | 2021 | 2022 bear | 2023 | 2024 | 2025+ |
|---|---|---|---|---|---|
| **equal-weight benchmark** | **2.225** | **−0.436** | **1.140** | **1.404** | **1.409** |
| `attention_frozen` | 1.829 | −0.764 | 0.897 | 1.295 | 1.217 |
| `attention_frozen_excess` | 1.805 | −0.760 | _1.148_ | 1.209 | 1.231 |
| `linear_frozen` | 1.796 | −0.700 | 1.030 | 1.192 | 1.053 |
| `linear_frozen_excess` | 1.784 | −0.737 | 0.934 | 1.152 | 1.118 |

**1 win in 20 fold-level comparisons**, and it is noise: fold 3,
`attention_frozen_excess`, +0.008 on a seed std of 0.212.

- Reward shaping is a **wash** — helps in some folds, hurts in others. Round
  0's apparent narrowing was single-window luck.
- Linear ≈ attention on Sharpe, but linear has meaningfully higher seed
  variance and ~2× the turnover. Attention stays the default on behaviour, not
  on Sharpe.
- The 2022 bear fold is catastrophically bad for every cell (−0.70 to −0.76
  against a −0.436 benchmark). Losing *more* than equal-weight in a drawdown is
  a specific failure, not general underperformance — see the defensiveness
  finding in round 6–7.

<a name="round-3"></a>
## Round 3 — is PPO even converging?

Every cell so far trained 200k timesteps with entropy loss and policy std
almost flat: the policy visibly never sharpened. If it simply hadn't finished
learning, nothing above would mean anything. Isolated on the best cell
(`attention_frozen_excess`) × the two most informative folds (2022 bear, the
worst; 2025+, the most decision-relevant), 3 seeds:

| variant | 2022 bear | 2025+ |
|---|---|---|
| baseline | −0.760 | 1.231 |
| `_bighead` — `net_arch` `[32]` → `[64,64]` | −0.758 | 0.992 |
| `_longtrain` — 200k → 600k timesteps | −0.582 | 1.308 |
| `_lowlr` — lr 3e-4 → 1e-4 | **−0.564** | **1.351** |

- **Bigger head: ruled out.** No help on the bear fold, clearly worse on 2025+.
- **3× training: the decisive test.** Entropy and policy std *do* genuinely
  converge given the budget (entropy −19.9 → −9.1, std 1.0 → 0.49) — and Sharpe
  moves barely. **The policy is demonstrably capable of converging, and
  convergence doesn't close the gap, so optimization budget is not the
  bottleneck.** That result is what redirected rounds 4–5 toward data and
  information rather than training.
- **Lower learning rate: best lever in the ablation**, and worth reading
  carefully — entropy didn't move much, so this is smaller, more conservative
  gradient steps helping generalization, not "the policy converged further."
  `_lowlr` became the base config for round 4.

Added in this round: optional `ppo.net_arch` and `ppo.learning_rate` config
keys (backward-compatible), and a `--folds` filter on
`scripts/run_rolling_validation.py` so an ablation needn't retrain all 5 folds.

<a name="round-4"></a>
## Round 4 — block-bootstrap augmentation

**Hypothesis:** if PPO converges fine but finds no edge, the constraint is
probably too little distinct history — each fold trains on only 1–4 years of
daily data, a few hundred rows.

`features/bootstrap.py::generate_synthetic_panel()` resamples ~21-day blocks of
real historical returns, volume and macro *together* — the same block indices
across every series, so real cross-asset co-movement and fat tails survive.
Deliberately not parametric GBM, which assumes Gaussian returns and would
smooth over exactly the regime shifts being tested against. Training data is
doubled with synthetic history; **val and test stay 100% real**, same protocol
as every other run here.

| | 2022 bear | 2025+ |
|---|---|---|
| `_lowlr` baseline | −0.564 | 1.351 |
| `_lowlr` + bootstrap | **−0.423** | 1.319 |
| equal-weight benchmark | −0.436 | 1.409 |

**The first crossing in the whole search**: the worst fold went from −0.564 to
−0.423, past its −0.436 benchmark, while 2025+ held flat (no regression).

Honest reading of the caveats: the margin is 0.013 on a 3-seed spread of 0.173
(individual seeds −0.260 / −0.405 / −0.604), and it has only been run on 2 of 5
folds. That is a lead, not a result. Confirming it — all 5 folds, more seeds —
is the current work.

<a name="round-5"></a>
## Round 5 — wider observed context

Every round to this point used the same 70-column panel. This one widens what
the encoder *observes* without touching what the policy can *trade*: 6 more
FRED series (`BAA10Y` credit spread, `T10Y3M` curve, `ICSA` jobless claims,
`DTWEXBGS` dollar, `DCOILWTICO` oil, `USSLIND` leading index) and 6
observed-only ETFs (`HYG`/`LQD` credit, `IEF`/`SHY` curve shape, `EEM` emerging
markets, `UUP` dollar).

These fold into the existing macro block as `macro__ctx_*` columns via a new
optional `context_tickers` parameter on `build_panel()`, so every encoder picks
them up with **zero changes to `encoders/*.py` and no change to the action
space** — which keeps the comparison clean.

| | 2022 bear | 2025+ |
|---|---|---|
| baseline | −0.760 | 1.231 |
| `_widectx` | **−0.652** | **1.296** |

A real, modest lift on both folds — still losing to benchmark, but the first
gain that came from genuinely new information rather than more training. Credit
spreads and curve shape were the largest gap in the pipeline: nothing in it saw
credit risk at all before this.

<a name="round-67"></a>
## Round 6–7 — rebuilding the basket, and the basket-quality trap

**Which assets does the model actually like?** `scripts/asset_promise.py`
gives two independent, zero-retraining views: revealed preference (average
weight the trained models actually allocate per asset) and a model-free
cross-check (standalone buy-and-hold Sharpe per asset).

A **leakage bug was caught here before it shipped**: the first pass computed
per-asset Sharpe across all 5 folds' *test* windows, which would leak if used
to pick a basket then evaluated on those same folds. Redone on pre-2020 data
only — the one cutoff legitimately out-of-sample for every fold — several
rankings flip: `XLE` goes from 4th best to dead last (its rank was the 2022 oil
spike), `TLT` goes from worst to mid-pack (its collapse was the 2022+ regime,
invisible pre-2020). Any future screening must use the pre-2020 method; see
[NOTES.md](NOTES.md#3-screen-only-on-data-that-predates-every-test-window).

**The defensiveness finding.** Average cash weight in the 2022 bear fold is
0.027 — no higher than the best fold's 0.042, and *lower* than several good
folds. **The model never learned to get defensive.** This is a distinct
explanation for why it loses specifically in bad regimes, separate from "no
rotation signal," and it is still largely unaddressed.

**Widening the candidate universe.** `scripts/screen_etf_universe.py` screens
~67 liquid, non-leveraged, pre-2016-inception ETFs across style factors,
single countries, credit and duration tiers, commodities, currencies and
sub-sectors. Explicitly *not* "top 100 by volume" — that list is dominated by
leveraged, inverse and single-stock daily-target products with decay mechanics
(`SOXS`, `TZA`, `MSTU`, `TQQQ`), which are excluded categorically. `LQD`/`HYG`
(credit) and `USMV` (min-vol) rank near the top; `BIL` (T-bills) tops it, but
that is a near-zero-volatility denominator artifact, not an opportunity.

**Does "diversified" mean diversified?** `scripts/build_diversified_basket.py`
computes the actual pre-2020 pairwise correlation matrix, because a basket of
different sector names can still be one bet. It confirms
`XLK`/`XLI`/`XLF`/`SMH` are 0.61–0.86 correlated with each other, and even
`EFA` (international) is 0.68–0.75 correlated with US equity sectors. The real
find: **`LQD` (investment-grade credit) is ~uncorrelated with equities (0.05)
while `HYG` (high-yield) is not (0.66)** — junk bonds carry equity-like risk,
investment grade genuinely doesn't. But pure greedy min-correlation selection
picks odd-if-mathematically-diverse baskets (Brazil, Hong Kong, `ARKK`). The
tension between correlation-optimal and economically sensible is **unresolved**
— it wants a properly constrained selection, not another ad hoc pick.

**Three narrowed baskets, each against its own freshly computed equal-weight
benchmark** (never the 13-asset one — "equal weight" means something different
at 6 assets):

| basket | 2022 bear: model / bench (gap) | 2025+: model / bench (gap) |
|---|---|---|
| 13-asset (reference) | −0.760 / −0.436 (−0.324) | 1.231 / 1.409 (−0.178) |
| `basket_div` — XLK, XLI, XLF, XLU, EFA, GLD | −0.774 / −0.404 (−0.370) | **1.446** / 1.638 (−0.192) |
| `basket_sharpe` — XLK, XLU, XLY, XLI, XLF, XLV | −0.825 / −0.578 (−0.247) | 0.992 / 1.166 (−0.174) |
| `basket_universe` — XLK, XLI, XLF, SMH, LQD, HYG, BIL, GLD, EFA | −0.790 / −0.722 (**−0.068**) | 1.583 / 1.688 (**−0.105**) |

**The basket-quality trap.** `basket_div` produced the highest absolute Sharpe
anywhere in this project (1.446) — and its own benchmark rose further (1.638),
leaving the gap slightly *wider* than the 13-asset reference. This is the same
pattern round 0 saw when the basket went 10 → 13. Raising basket quality raises
the model and the naive benchmark at least in lockstep, so **picking better
assets cannot close the gap by itself.** The model needs to do something
equal-weight structurally cannot.

`basket_universe` — the screen's validated sectors plus credit and a T-bill
safe harbour — has the tightest gap of any variant tested on both folds. The
credit and safe-harbour sleeves pull their weight even though the basket still
doesn't clear the bar. Worth noting against the unresolved tension above:
`basket_universe`'s *own benchmark* in the bear fold (−0.722) is worse than
`basket_div`'s (−0.404) despite holding more genuine diversifiers, which is
most likely its highly-correlated `XLK`/`XLI`/`XLF`/`SMH` core dominating
benchmark behaviour.

**A second real bug surfaced here**, from adding `BIL`: a fold came back with
59 days of test data instead of ~360, because T-bills are stable enough that
the 14-day RSI window often has zero down-days → `NaN` → one poisoned column
silently wiped 81% of rows for *every* asset via the panel's global `dropna()`.
Fixed; details and the detection method are in
[NOTES.md](NOTES.md#bug-2-a-near-zero-volatility-asset-silently-truncated-the-panel).
Any pre-fix `basket_universe` number (the one with `n_days=59`) is invalid; the
table above is the post-fix rerun.

<a name="round-8"></a>
## Round 8 — bootstrap augmentation, 10 seeds across all five folds

Round 4 claimed the project's only benchmark crossing, from 2 hand-picked folds
and 3 seeds, with a margin (0.013) an order of magnitude smaller than its seed
spread (0.173). Round 8 ran the same cell with and without augmentation across
**all 5 folds at 10 seeds** — 100 runs, as a Slurm job array on Princeton's
Adroit cluster (see `scripts/adroit/`).

Both cells are `attention_frozen_excess_lowlr`; the benchmark is
`equal_weight_benchmark_round8.json`, recomputed on the same data vintage as the
models. `t` is Welch's, from `scripts/compare_cells.py`; |t| > 2 is the bar for
calling a difference real.

| fold (test year) | baseline (10 seeds) | + bootstrap (10 seeds) | difference | t | verdict |
|---|---|---|---|---|---|
| 1 (2021) | 2.057 ± 0.126 | 2.116 ± 0.217 | +0.059 | 0.74 | noise |
| 2 (2022 bear) | −0.656 ± 0.166 | **−0.510** ± 0.144 | **+0.146** | **2.10** | **bootstrap better** |
| 3 (2023) | 1.198 ± 0.109 | 1.146 ± 0.158 | −0.053 | −0.87 | noise |
| 4 (2024) | 1.410 ± 0.147 | 1.418 ± 0.161 | +0.008 | 0.12 | noise |
| 5 (2025+) | 1.250 ± 0.102 | 1.293 ± 0.140 | +0.043 | 0.79 | noise |

### Round 4's crossing does not survive

| fold | 3 seeds | 10 seeds | benchmark |
|---|---|---|---|
| 2 (2022 bear) | −0.423 (**beat** −0.436) | **−0.510** (loses by 0.074) | −0.436 |
| 4 (2024) | 1.525 at 3 seeds, 1.592 at 2 | **1.418** (+0.014) | 1.404 |

Seeds 0–2 on the bear fold returned −0.26, −0.41, −0.60; the remaining seven
returned −0.49, −0.54, −0.49, −0.75, −0.59, −0.62, −0.34. The first three
happened to include the two best runs of ten. Fold 4's apparent +0.188 margin
at two seeds decayed to +0.120 at three and +0.014 at ten. **Every "crossing"
this project found before round 8 was a small-sample artifact.**

### What augmentation actually does

Against the benchmark, per fold, using each cell's own standard error:

| fold | benchmark | baseline gap (t) | bootstrap gap (t) |
|---|---|---|---|
| 1 (2021) | 2.225 | −0.168 (**−4.22**) | −0.109 (−1.59) |
| 2 (2022 bear) | −0.436 | −0.220 (**−4.18**) | −0.074 (−1.61) |
| 3 (2023) | 1.140 | +0.058 (1.70) | +0.005 (0.10) |
| 4 (2024) | 1.404 | +0.005 (0.12) | +0.014 (0.27) |
| 5 (2025+) | 1.344 | −0.094 (**−2.90**) | −0.050 (−1.14) |

This is the clearest statement of the result: **the un-augmented baseline is
significantly worse than equal-weight on 3 of 5 folds; the augmented cell is
not significantly worse on any, and not significantly better on any either.**
Block-bootstrap augmentation does not make the agent beat the benchmark — it
makes the agent statistically indistinguishable from it. After eight rounds,
"ties with buy-and-hold" is the ceiling reached so far.

The one fold-level effect that is real (bear fold, +0.146, t = 2.10) is also the
one where the benchmark itself loses money, i.e. augmentation helps most in a
sustained drawdown. That is consistent with what the method does: resampled
blocks give the policy more examples of falling markets than the 1–4 years of
real training history contain.

### The methodological lesson

Three seeds cannot resolve the effects this project is chasing. Fold-level seed
standard deviations run 0.10–0.25, so a 3-seed mean carries a standard error of
roughly 0.06–0.14 — larger than every margin ever claimed here. At 10 seeds the
standard error falls to 0.03–0.07, which is what made round 8 decisive in either
direction. **No future claim in this project should rest on fewer than 10
seeds**, and differences should be reported with `scripts/compare_cells.py`
rather than by comparing a mean to a spread by eye.

### Vintage note

Fold 5 is the only fold whose test window runs to the end of available data, so
its length tracks the data vintage: 358 days on the cache round 4 used, 363 on
the current one, which moves the equal-weight benchmark by 0.066 — larger than
most margins here. All round-8 numbers are single-vintage; the six superseded
fold-5 runs are preserved under `runs_archive/vintage_358days/`. Folds 1–4 have
fixed `test_end` dates and are vintage-independent.

### Round 8 addendum — the policy stays near equal weight by choice

A structural hypothesis worth recording because it was tested and **refuted**.

The env's action space is `Box(low=-1, high=1)`, and SB3 clips actions to those
bounds before `step()`. Softmaxing 14 logits confined to [−1, 1] therefore cannot
reach an arbitrary simplex point — the reachable per-slot range is
**[1.03%, 36.24%]**, not [0%, 100%]. That looked like a candidate ceiling on
everything: it would cap de-risking at 36% cash and cap conviction at 36% in one
asset, which would explain the "never goes defensive" finding, the tight
benchmark tracking, and the general unresponsiveness to levers.

Measured against trained rollouts (`scripts/check_action_space.py`, 3 runs
across 2 folds, 251 days each), **the cap is not binding**:

| run | realized weight range | cash mean / max | most concentrated asset | days within 90% of cap |
|---|---|---|---|---|
| baseline, bear fold | 1.6% – 14.4% | 4.7% / 6.2% | 14.4% | 0.00% |
| bootstrap, bear fold | 1.6% – 12.2% | 2.0% / 2.7% | 12.2% | 0.00% |
| baseline, 2024 fold | 2.1% – 15.5% | 2.4% / 2.7% | 15.5% | 0.00% |

Every weight sits deep inside the feasible set — never above 16% where 36% is
allowed, cash at 2–6% where 36% is available, and not a single day in 753 comes
within 90% of the bound. Equal weight is 7.14%, so **the learned policy is a
mildly tilted equal-weight basket, by choice rather than by constraint.**

Two consequences:

1. **Widening the action bounds is not a lever.** The agent does not use the
   range it already has.
2. **"Never goes defensive" is a learning failure, not a plumbing one.** The
   capacity to hold 36% cash exists and goes unused, which points at the reward
   and exploration rather than the interface. Candidates: the excess-return
   reward is on the order of 1e-4 per day, `ent_coef=0` leaves nothing pushing
   exploration, and the advantage scale may be too small to move the Gaussian
   policy's mean off its initialization — consistent with round 3's observation
   that entropy and policy std barely budge without a 3× step budget.

This also reframes every earlier round: if the policy is structurally a
near-equal-weight basket, then tracking the benchmark closely is the *expected*
outcome, and "beating it" was always going to come down to small tilts.

<a name="summary"></a>
## Summary: what each round eliminated

| round | hypothesis tested | outcome |
|---|---|---|
| 0 | which encoder / training regime is best | `attention_frozen`; but loses to equal-weight |
| 0 | more assets close the gap | no — benchmark rises more |
| 0 | benchmark is unfairly cost-free | no — cost-matched and buy-and-hold both still win |
| 0 | reward should reference the benchmark | promising on one window… |
| 1–2 | …does that survive across regimes? | no — 1 win in 20, inside noise |
| 3 | PPO is under-trained | no — it converges given 3× budget, Sharpe doesn't follow |
| 3 | policy/value head too small | no — bigger is worse |
| 4 | too little training history | **the one crossing found (unconfirmed)** |
| 5 | too little observed information | real, modest lift on both folds |
| 6–7 | better asset selection | no — the basket-quality trap |
| 6–7 | does the model de-risk in drawdowns? | **no — and that's a distinct unexplored failure** |
| 8 | does round 4's crossing hold across all regimes? | direction yes (mean gap −0.089 → −0.016, beats benchmark on 3/5); magnitude still inside seed noise |

---

<a name="conclusions"></a>
## Conclusions — what this project established

**Status: concluded.** The question it set out to answer has been answered, and
the answer is stable across eight rounds, ~240 trained models and five
regime-diverse test windows.

### The result

A PPO allocator over 13 liquid ETFs, fed daily bars and macro series, **does not
beat an equal-weight buy-and-hold of the same basket.** The best configuration
found — attention encoder, frozen after unsupervised pretraining,
excess-return reward, lr 1e-4, block-bootstrap-augmented training data — reaches
a *statistical tie*: at 10 seeds it is not significantly worse than the
benchmark on any fold, and not significantly better on any either. Every apparent
crossing found before round 8 was a small-sample artifact.

### Four things learned about the problem

**1. The benchmark is the hard part, not the model.** An equal-weight basket is
not a weak baseline; it is the rational allocation under no predictive
information, it rebalances for almost nothing, and it captures the same
cross-sectional risk premia the model is reaching for. Beating it requires a
predictive edge. No amount of better optimization substitutes for one.

**2. When the baseline is built from your inputs, improving the inputs doesn't
help.** The basket-quality trap (round 6–7) is the sharpest finding here.
Concentrating into stronger assets lifted the model *and* its benchmark in
lockstep: `basket_div` posted the highest absolute Sharpe in the project (1.446)
and still lost to its own benchmark (1.638) by more than the original 13-asset
gap. This generalizes past finance — any relative-performance objective whose
baseline is a function of the same inputs is immune to input quality.

**3. A near-uniform policy is the correct response to no signal, not a training
failure.** Measured on trained rollouts: realized weights 1.6–15.5% against a
7.14% uniform, cash 2–6% where 36% was reachable, never within 90% of the
action-space bound on any of 753 days. The policy had the freedom to concentrate
and declined. Round 0 predicted this equilibrium in words — "diversify like the
benchmark, but pay turnover getting there" — and round 8 measured it. It also
explains why *every* lever moved Sharpe so little: they were all adjusting the
size of a small tilt.

**4. The one place a learned policy has structural room is where fixed weights
cannot adapt.** The only statistically real fold-level effect in the entire
project is bootstrap augmentation on the 2022 bear market (+0.146, t = 2.10) —
the one window where the benchmark loses money. Equal weight cannot de-risk by
construction; a policy can. That points at a different objective (drawdown
control, volatility targeting) rather than a different lever, and it is the one
direction the evidence actually supports.

### Five things learned about doing the research

These transfer further than the finance, and cost more to learn.

**1. Establish measurement precision before running experiments.** Fold-level
seed standard deviations here are 0.10–0.25 Sharpe. A 3-seed mean therefore
carries a standard error of 0.06–0.14 — **larger than every margin this project
ever claimed.** Eight rounds were run at a precision incapable of resolving the
effects being tested, and two "findings" duly evaporated at 10 seeds. The seed
variance should have been measured in round 1 and every experiment sized against
it.

**2. Single-window results mislead in a specific, predictable direction.** Round
0's reward shaping looked like a real improvement on one test window (1.259 vs a
1.395 benchmark, tighter across seeds). Across five regimes it won 1 of 20
comparisons, by 0.008. A result from one window is a hypothesis, not a finding.

**3. Bugs in research code produce plausible numbers, not errors.** Both bugs
found here were silent. SB3 re-applied orthogonal init over the pretrained
encoder, so three training regimes were secretly identical — detectable only
because two cells agreed to the last decimal. A T-bill's RSI went `NaN` on a
zero-loss window and the global `dropna()` deleted 81% of rows for every asset —
detectable only because one fold reported 59 test days instead of ~360. Neither
raised an exception; both yielded believable Sharpe ratios. **Validate
invariants, not just outputs**: do differently-configured runs actually differ,
does the panel contain NaN, do the row counts match across folds.

**4. Leakage arrives through side doors.** The obvious paths were guarded from
the start — train-only normalization, no future-return labels, forward returns
used solely as reward. It nearly entered anyway through *asset screening*:
ranking assets by Sharpe computed across the test windows, then selecting a
basket to evaluate on those same windows. Correcting it to pre-2020-only data
flipped `XLE` from 4th-best to last. Feature hygiene does not protect you if the
experiment's *design* peeks.

**5. Derived artifacts must match what they judge.** Two near-misses, both from
comparing against a baseline computed on different terms. Fold 5's test window
tracks the data vintage, so a six-day-newer cache moved its benchmark by 0.066 —
larger than most effects under study. And a reporting script silently scored
6-asset baskets against the 13-asset benchmark, manufacturing three crossings
that did not exist. Benchmarks need the same universe, the same window and the
same vintage as the models they score, and that needs to be enforced in code
rather than remembered.

### What would justify resuming

Not another optimizer lever. The remaining untested knobs (`ent_coef`, reward
scale, `turnover_penalty`) would make the policy tilt harder; with no signal to
tilt on, the likeliest outcome is unchanged mean return at higher variance.

Three things would genuinely change the odds:

- **A different objective.** Drawdown-constrained or volatility-targeted
  allocation, judged on Calmar or max-drawdown rather than Sharpe-vs-equal-weight.
  Competes where a policy has structural advantage instead of requiring it to
  out-predict the market. Reuses nearly all of this code.
- **Different information.** Point-in-time macro vintages, intraday bars,
  options-implied volatility, higher-frequency credit, cross-sectional
  fundamentals. Daily adjusted closes on the most liquid ETFs in existence, with
  features standard since the 1980s, is the most heavily mined dataset in
  finance; the null result is what that prior predicts.
- **A less efficient market.** Different asset class entirely, at the cost of
  more noise and more regime risk.

### What is worth keeping regardless

A pipeline that is honest by construction (no future-return supervision,
train-only normalization, walk-forward validation, per-basket benchmarks), a
harness that runs 100 experiments as a Slurm array, tooling that reports
t statistics instead of impressions, and a written record of what was tried and
why it didn't work. The negative result is legible, which is the part that makes
it useful.
