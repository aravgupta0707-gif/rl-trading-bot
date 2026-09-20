# Engineering notes

Conventions, traps, and bug post-mortems. Most of what's here was learned the
expensive way, and several items **silently corrupt results rather than
failing loudly** — worth reading before changing a basket, a feature, or a
screening step.

---

## Conventions

### 1. Every distinct ticker/macro set needs its own `cache_dir`

`fetch_etf_prices()` and `fetch_macro()` cache to **one parquet per
`cache_dir`, keyed only by directory** — not by which tickers or series were
requested. A cache hit returns the whole cached frame regardless of what the
config asked for. Reusing a `cache_dir` after changing `data.etfs` or
`data.macro` therefore silently trains on the old universe.

Every experiment with a distinct universe gets its own dir
(`cache_widectx`, `cache_basket_div`, …). Follow the pattern, or pass
`refresh=True`.

### 2. A narrowed basket needs its own equal-weight benchmark

"Equal weight" means something different at 6 assets than at 13, so a 6-asset
model compared against the 13-asset benchmark is comparing two different
things. Generate the basket's own:

```bash
python scripts/rolling_benchmark.py --config <that basket's config> \
    --out runs_rolling/equal_weight_benchmark_<basket>.json
```

This is not a nitpick — it's how the basket-quality trap was found
([RESULTS.md](RESULTS.md#round-67)). Narrowed baskets look like big wins
against the wrong benchmark.

### 3. Screen only on data that predates every test window

Any asset selection, Sharpe ranking, or correlation matrix used to pick a
basket must be computed on data ending before the earliest `train_end` of the
folds you will evaluate on — in practice, pre-2020.

Screening on aggregate test-window data leaks, and not subtly: correcting this
flipped `XLE` from 4th-best to last (its rank was entirely the 2022 oil spike)
and `TLT` from worst to mid-pack (its collapse was the 2022+ regime, invisible
pre-2020). A basket picked on the leaky ranking would have looked good for
reasons that were already in the test set.

### 4. Verify numbers from files on disk

Read results from `runs*/**/metrics.json`, not from console tails or
notification text. Interleaved background jobs truncate and interleave output.
The RSI bug below was caught precisely by cross-checking disk state against
what a summary appeared to say.

### 5. Long runs must be resumable, never blocking

Every runner skips any cell that already has a `metrics.json`, so relaunching
the same command resumes rather than retrains. Multi-hour rounds go through a
wrapper that loops until the expected file count exists — see
`scripts/run_bootstrap_confirmation.sh` — launched with `nohup … & disown`.
This project has a history of long background jobs being killed unpredictably;
assume it will happen and make it cheap.

### 6. `python`, not `python3`

On the machine this project was developed on, `python3` resolved to a different
interpreter without `pandas_datareader`. Cost real debugging time once.

---

<a name="fold-naming"></a>
## Fold definitions (and a naming quirk)

Expanding train window, 1-year validation, 1-year test. Defined in
`scripts/run_rolling_validation.py::FOLDS`:

| fold | train through | validation | test | regime |
|---|---|---|---|---|
| `fold1_test2020` | 2019-12-31 | 2020 | **2021** | post-COVID bull |
| `fold2_test2021` | 2020-12-31 | 2021 | **2022** | rate-hike bear |
| `fold3_test2022` | 2021-12-31 | 2022 | **2023** | recovery |
| `fold4_test2023` | 2022-12-31 | 2023 | **2024** | continued bull |
| `fold5_test2024` | 2023-12-31 | 2024 | **2025 → end of data** | most recent |

**The `test<year>` in each fold name is the validation year, not the test
year.** The actual test window is the year after the name suggests — so
`fold2_test2021`, the fold described everywhere as "the 2022 bear," tests on
2022. The names are kept as-is because ~145 committed run directories embed
them; the tables in [RESULTS.md](RESULTS.md) and [README.md](README.md) label
folds by their real test year.

Fold 5's test window runs to the end of available data, so **its length depends
on the data vintage** — the original runs had 358 test days, later caches give
364. Folds 1–4 have fixed `test_end` dates and are vintage-independent.

This is not a rounding concern. Five extra trading days move fold 5's
equal-weight benchmark from **1.409 to 1.344** — a 0.066 shift, larger than most
model-vs-benchmark margins this project measures (round 4's crossing was 0.013).
Two rules follow:

- **Recompute the benchmark on the same vintage as the models it judges**
  (`scripts/rolling_benchmark.py --out <vintage-specific file>`).
- **Watch out for the skip-existing interaction.** Runners skip any cell with a
  `metrics.json`, so adding seeds to an old fold 5 keeps the old seeds on the old
  window and puts new seeds on the new one — one fold, two test windows, no
  warning. Move the stale runs aside to force a clean re-run; see
  `runs_archive/vintage_358days/` for a worked example.

---

## Bug post-mortems

Both bugs below produced plausible-looking numbers rather than errors, which is
the only reason they're worth this much text.

<a name="bug-1-sb3-silently-discarded-the-pretrained-encoder"></a>
### Bug 1 — SB3 silently discarded the pretrained encoder

**Symptom.** In the first full 9-cell grid, `finetune` and `e2e` produced
*byte-identical* metrics for the linear and MLP encoders.

**Cause.** `stable-baselines3`'s `ActorCriticPolicy._build()` re-applies its
default orthogonal initialization to every `nn.Linear`/`nn.Conv2d` in the
*entire* features extractor — including the custom encoder — immediately after
`PPO(...)` construction. Every pretrained weight `prepare_encoder()` produced
was overwritten before training started, so all three regimes were secretly the
same untrained encoder. True since the original smoke test; every pretrained
result before the fix was invalid.

**Fix.** `training/train_ppo.py::train()` snapshots the encoder's
`state_dict()` before `PPO(...)` and restores it after.

**Why the fix is trustworthy.** Post-fix, `frozen` beat `finetune`/`e2e` for
all three encoder types — the opposite of what a still-broken pipeline would
produce, and the three regimes now visibly diverge at smoke scale.

**Lesson.** Two cells agreeing to the last decimal is a bug report, not a
coincidence. Assert that differently-configured runs actually differ.

<a name="bug-2-a-near-zero-volatility-asset-silently-truncated-the-panel"></a>
### Bug 2 — a near-zero-volatility asset silently truncated the panel

**Symptom.** After adding `BIL` (T-bills) to a basket, one fold's test result
came back with **59 days of data instead of ~360**. The Sharpe computed on
those 59 days looked entirely reasonable.

**Cause.** `BIL`'s price is stable enough that its 14-day RSI window often
contains **zero down-days** → `loss = 0` → `gain/0 = NaN`. `build_panel()` ends
with a global `dropna()`, so that one column wiped ~81% of recent rows for
**every asset in the basket**. None of the original 13 tickers ever triggered
it, because normal volatility always supplies some down-days.

**Fix.** `features/panel.py::_etf_features()` now applies the standard RSI
convention — no losses means RSI 100, not `NaN`:

```python
rsi = rsi.where(loss != 0, 100.0)
```

Verified: 99 → 512 rows recovered, no NaN remaining, and the rerun's `n_days`
matched every other fold.

**Lesson, and the check to run.** A single poisoned column silently truncates
the entire panel via `dropna()`. After touching feature code or adding any
low-volatility instrument (short-duration bonds, cash equivalents, currency
hedges), verify explicitly:

```python
assert not panel.isna().any().any()
# and spot-check per-fold row counts -- the tell here was 59 days, not an error
```

---

## Environment notes

Developed on Windows locally; also runs in a Linux container. Two things
differ in the container:

- **Yahoo Finance is blocked** by the sandbox's egress policy, so `cache/`
  cannot be re-fetched there. It can be reconstructed from the committed
  `cache_widectx/` parquet, which holds all 13 default tickers plus a macro
  superset — subset it to the 13 tickers and 5 default series, and folds 1–4
  reproduce the original row counts exactly.
- **`pyarrow`** is required for the parquet caches and is in
  `requirements.txt`; older checkouts predate it and fail on first cache read.
