# Running on a Slurm cluster (Princeton Adroit)

Why bother: a single run is ~5 minutes on 2 cores, so the bottleneck is never
one run — it's that the interesting designs are wide. Round 8 as specified here
is 2 cells × 5 folds × 5 seeds = **50 runs**, which is ~2 hours sequentially and
about **5 minutes** as a job array. That matters because round 4's result — the
only benchmark crossing in the project — has a margin of 0.013 against a seed
spread of 0.173. Only more seeds settle it, and seeds are exactly what a cluster
makes free.

**Ask for CPUs, not GPUs.** The policy head is one 32-unit layer over an 8-dim
latent, and PPO here is bottlenecked on Python-side environment stepping.
`torch.cuda.is_available()` being `False` costs this workload nothing.

## Steps

```bash
# 1. on your machine (must be on the campus VPN)
ssh <netid>@adroit.princeton.edu

# 2. clone into scratch -- /home is small and not meant for run artifacts
cd /scratch/network/$USER
git clone https://github.com/aravgupta0707-gif/rl-trading-bot.git
cd rl-trading-bot
git checkout claude/jolly-ramanujan-g9s56x

# 3. one-time setup: venv + deps + data cache. Login node, needs network.
bash scripts/adroit/setup_env.sh

# 4. submit
sbatch scripts/adroit/round8_array.slurm
squeue -u $USER

# 5. when it drains, rebuild the summaries and read the table
python scripts/aggregate_results.py --write-summaries
python scripts/rolling_benchmark.py --config configs/attention_frozen_excess_lowlr.yaml \
    --out runs_rolling/equal_weight_benchmark_round8.json
python scripts/aggregate_results.py --benchmark runs_rolling/equal_weight_benchmark_round8.json
```

## Things to check before trusting the first submission

- **Partition and account.** The job script sets neither, so it lands in the
  cluster default. Run `sinfo -s` and, if that queue is wrong for your
  allocation, add `#SBATCH --partition=<name>` (and `--account=` if your site
  requires one).
- **Array range must match the design.** `--array=0-49` corresponds to the
  `CELLS`/`FOLDS`/`SEEDS` arrays inside the script (2 × 5 × 5). Change the
  seed list and the range together, or the extra combinations simply never run.
- **Module names drift.** `setup_env.sh` tries `anaconda3/2024.6`, then
  `anaconda3`, then `python`, then the system interpreter, and prints which one
  it used. If all fail, `module avail anaconda3` shows what the cluster has.
- **No network on compute nodes is fine.** `setup_env.sh` rebuilds `cache/` from
  the committed `cache_widectx/` parquet, so training never touches Yahoo
  Finance. Folds 1–4 reproduce the original row counts exactly
  (252/251/249/251); fold 5 tracks the cache vintage (364 days here vs 358 in
  the earliest runs), which is why step 5 recomputes fold 5's benchmark on the
  same data before comparing.

## Getting results back

Run directories are small (~200 KB each: `model.zip`, `config.json`,
`metrics.json`) and this repo tracks them by convention:

```bash
git add runs_rolling runs_bootstrap && git commit -m "Round 8 results from Adroit"
git push origin claude/jolly-ramanujan-g9s56x
```

Pushing from the cluster needs a credential for the remote — an SSH deploy key
or a PAT in a credential helper. If you'd rather not set that up there,
`rsync -av <netid>@adroit.princeton.edu:/scratch/network/<netid>/rl-trading-bot/runs_bootstrap ./`
from your laptop works just as well.

## Concurrency caveat

Every array task runs one fold/seed, writes its own
`runs_*/<cell>_<fold>_seed<n>/` directory — no collisions there — but they all
rewrite the same `<cell>_*_summary.json`, and those writes race. The per-run
`metrics.json` files are the source of truth and are written once on
completion, so the losing writes cost nothing; `scripts/aggregate_results.py
--write-summaries` rebuilds every summary from them afterwards. Don't read a
summary file written during a live array.
