#!/bin/bash
# One-time environment setup for Princeton Adroit (or any Slurm cluster).
#
# Run this ONCE on the LOGIN node (it needs outbound network for pip; compute
# nodes may not have it). It creates a venv and populates the data cache from
# the committed parquet, so the training jobs themselves need no network at all.
#
# Usage:
#   ssh <netid>@adroit.princeton.edu
#   cd /scratch/network/$USER && git clone <repo-url> rl-trading-bot
#   cd rl-trading-bot && bash scripts/adroit/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKDIR="$(pwd)"
VENV="${VENV:-$WORKDIR/.venv}"

echo "=== repo: $WORKDIR"
echo "=== venv: $VENV"

# Princeton clusters expose Python through modules. Versions change, so try a
# few and fall back to the system interpreter; check `module avail anaconda3`
# if none of these resolve.
if command -v module &>/dev/null; then
  module load anaconda3/2024.6 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || module load python 2>/dev/null \
    || true
fi

PY="$(command -v python3 || command -v python)"
echo "=== interpreter: $PY ($($PY -V 2>&1))"

"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

# CPU-only torch is the right build here: the policy is a [32]-unit head over an
# 8-dim latent, and PPO is bottlenecked on Python-side env stepping, not matmul.
# A GPU buys essentially nothing; parallel CPU tasks are the whole win.
# If download.pytorch.org is reachable, the CPU wheel is ~5x smaller than PyPI's
# CUDA default -- otherwise plain PyPI works fine.
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
  || python -m pip install --quiet torch
python -m pip install --quiet -r requirements.txt

python -c "import pandas, numpy, torch, gymnasium, stable_baselines3, pyarrow; \
print('deps OK, torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# Data: rebuild the gitignored default cache/ from the committed parquet rather
# than hitting Yahoo Finance, which is often blocked on clusters.
python scripts/rebuild_default_cache.py

echo
echo "=== setup complete. Submit training with:"
echo "    sbatch scripts/adroit/round8_array.slurm"
