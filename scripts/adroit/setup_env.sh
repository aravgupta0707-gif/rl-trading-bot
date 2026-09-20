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
# Deliberately NOT 'set -u': this script sources third-party init scripts (the
# cluster's module setup, and the venv's own activate), which reference unset
# variables like PS1 in a non-interactive shell. Under 'set -u' that aborts with
# "PS1: unbound variable" before anything is installed.
set -eo pipefail
cd "$(dirname "$0")/../.."

WORKDIR="$(pwd)"
VENV="${VENV:-$WORKDIR/.venv}"

echo "=== repo: $WORKDIR"
echo "=== venv: $VENV"

# `module` is a shell FUNCTION set up by the cluster's login profile, and
# `bash setup_env.sh` runs non-interactively, so it is usually not defined here
# even though it works fine when typed at a prompt. Source its init explicitly
# before trying to use it.
if ! command -v module &>/dev/null; then
  for init in /usr/licensed/Modules/init/bash \
              /usr/share/Modules/init/bash \
              /usr/share/lmod/lmod/init/bash \
              /etc/profile.d/modules.sh; do
    if [ -r "$init" ]; then
      echo "=== sourcing module init: $init"
      # shellcheck disable=SC1090
      source "$init" && break
    fi
  done
fi

# Versions drift; try newest-known first and fall back. `module avail anaconda3`
# lists what this cluster actually has.
if command -v module &>/dev/null; then
  for mod in anaconda3/2024.6 anaconda3/2025.6 anaconda3 python; do
    if module load "$mod" 2>/dev/null; then
      echo "=== loaded module: $mod"
      break
    fi
  done
else
  echo "=== no module command available; using the system interpreter"
fi

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "!!! no python found. Run 'module avail anaconda3', load one by hand, and re-run." >&2
  exit 1
fi
echo "=== interpreter: $PY ($($PY -V 2>&1))"

# Some system pythons ship without ensurepip, which makes `python -m venv` fail.
# Fall back to virtualenv, then to a venv seeded by get-pip, before giving up.
if ! "$PY" -m venv "$VENV" 2>/dev/null; then
  echo "=== 'python -m venv' failed; trying virtualenv"
  if "$PY" -m virtualenv "$VENV" 2>/dev/null; then
    echo "=== created with virtualenv"
  elif "$PY" -m venv --without-pip "$VENV" 2>/dev/null; then
    echo "=== created without pip; bootstrapping pip"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python
  else
    echo "!!! could not create a virtualenv with $PY." >&2
    echo "!!! Load an anaconda3 module by hand and re-run, e.g.:" >&2
    echo "!!!   module load anaconda3/2024.6 && bash scripts/adroit/setup_env.sh" >&2
    exit 1
  fi
fi

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

# Slurm opens the --output file before the job script runs, so this directory
# has to exist at SUBMIT time -- the mkdir inside round8_array.slurm is too late
# to save a job whose output path is missing.
mkdir -p slurm_logs

echo
echo "=== setup complete. Submit training with:"
echo "    sbatch scripts/adroit/round8_array.slurm"
