"""Config loading with deep-merge over configs/default.yaml.

Experiment configs only need to specify what differs from the default
(e.g. just `encoder.type` and `encoder.regime`); everything else is inherited.
"""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict:
    with open(DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if path is not None:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    return config
