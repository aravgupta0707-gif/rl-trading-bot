# Archived: fold5 runs on the 358-day data vintage

These six runs (`attention_frozen_excess_lowlr`, with and without bootstrap,
3 seeds each) are valid results — they are archived rather than deleted because
they were produced on an **older data vintage** than everything else in round 8,
and mixing vintages inside one comparison is not safe at the margins this
project is measuring.

## What differs

Fold 5 is the only fold whose test window runs to the end of available data
rather than a fixed `test_end`, so its length tracks the data vintage:

| | test days | equal-weight benchmark Sharpe |
|---|---|---|
| original cache (these runs) | 358 | 1.409 |
| current cache (`cache_widectx` vintage, data through 2026-07-09) | 363 | **1.344** |

Five extra trading days move the benchmark by **0.066** — larger than several of
the model-vs-benchmark margins under investigation (round 4's crossing was
0.013). Judging a 358-day model against a 363-day benchmark, or vice versa,
would therefore manufacture or erase a "crossing" on arithmetic alone.

Folds 1–4 have fixed `test_end` dates, are vintage-independent, and reproduce
identically across caches (252/251/249/251 days, benchmarks unchanged to three
decimals).

## Why they were moved rather than kept in place

The runners skip any cell that already has a `metrics.json`. Leaving these in
`runs_rolling/`/`runs_bootstrap/` meant every later sweep — including the
10-seed Slurm array — would silently *keep* the old-vintage fold 5 and only add
new seeds on the current one, producing a fold whose seeds came from two
different test windows. Moving them here forces a clean re-run.

## Status of other cells

Every other cell's fold 5 results (`attention_frozen_excess`, `_widectx`,
`basket_*`, …) are still on their original vintages and were not re-run — they
are not part of round 8's comparison. Cross-round fold 5 comparisons therefore
carry a vintage caveat of roughly this size; cross-round comparisons on folds
1–4 do not.
