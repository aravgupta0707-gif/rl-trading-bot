"""Compare a trained policy's REALIZED weight range against the analytic bound
implied by the action space.

The env's action space is Box(low=-1, high=1) and SB3 clips actions to those
bounds before step(), so softmaxing n logits confined to [-1, 1] cannot produce
an arbitrary simplex point: for 14 slots the reachable per-slot range is
[1.03%, 36.24%] rather than [0%, 100%]. The open question this answers is
whether that cap is actually BINDING -- i.e. whether the policy presses against
it -- or whether the policy operates well inside its feasible set.

Answer (see RESULTS.md, round 8 addendum): not binding. Trained policies keep
every weight between ~1.6% and ~16%, hold 2-6% cash where 36% is reachable, and
never come within 90% of the cap on any day. The portfolio is a mildly tilted
equal-weight basket by choice, not by constraint.

Usage:
    python scripts/check_action_space.py
"""
import sys, json, glob
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from stable_baselines3 import PPO
from configs import load_config
from data import fetch_etf_prices, fetch_macro
from envs import align_features_and_returns, compute_forward_returns, make_env
from features import apply_normalizer, build_panel, fit_normalizer, rolling_walk_forward_split
from run_rolling_validation import FOLDS

cfg = load_config('configs/attention_frozen_excess_lowlr.yaml')
d, f = cfg['data'], cfg['features']
prices = fetch_etf_prices(d['etfs'], d['start_date'], d['end_date'], d['cache_dir'])
macro  = fetch_macro(d['macro'], d['start_date'], d['end_date'], d['cache_dir'])
panel  = build_panel(prices, macro, d['etfs'], f)
fwd    = compute_forward_returns(prices, d['etfs'])
n = len(d['etfs']) + 1

def sm(x):
    x = np.asarray(x, float); e = np.exp(x - x.max()); return e/e.sum()
hi = sm([1]+[-1]*(n-1))[0]; lo = sm([-1]+[1]*(n-1))[0]
print(f"analytic bound for {n} slots, logits clipped to [-1,1]: [{lo:.4f}, {hi:.4f}]")
print(f"equal weight = {1/n:.4f}\n")

runs = [
  ('runs_rolling/attention_frozen_excess_lowlr_fold2_test2021_seed0', 'fold2_test2021', 'baseline  bear fold'),
  ('runs_bootstrap/attention_frozen_excess_lowlr_bootstrap_fold2_test2021_seed0', 'fold2_test2021', 'bootstrap bear fold'),
  ('runs_rolling/attention_frozen_excess_lowlr_fold4_test2023_seed0', 'fold4_test2023', 'baseline  2024 fold'),
]
for run_dir, fold_name, label in runs:
    fold = next(fo for fo in FOLDS if fo['name']==fold_name)
    tr, va, te = rolling_walk_forward_split(panel, fold['train_end'], fold['val_end'], fold['test_end'])
    mean, std = fit_normalizer(tr)
    te_n = apply_normalizer(te, mean, std)
    model = PPO.load(f'{run_dir}/model.zip', device='cpu')
    feats, fr = align_features_and_returns(te_n, fwd)
    env = make_env(feats, fr, cfg['env'])
    obs, _ = env.reset(); W=[]; done=False
    while not done:
        a,_ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        W.append(info['weights']); done = term or trunc
    W = np.array(W)
    print(f"{label}  ({len(W)} days)")
    print(f"   realized weight range over all slots: [{W.min():.4f}, {W.max():.4f}]")
    print(f"   cash slot: mean={W[:,-1].mean():.4f}  max={W[:,-1].max():.4f}  min={W[:,-1].min():.4f}")
    print(f"   most concentrated single asset on any day: {W[:,:-1].max():.4f}")
    print(f"   fraction of days at >90% of the analytic cap in some slot: {(W.max(axis=1) > 0.9*hi).mean():.2%}")
