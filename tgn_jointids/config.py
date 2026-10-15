"""YAML configuration loading with attribute-style access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Recursive dict with attribute access (cfg.model.hidden_dim)."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else copy.deepcopy(v))
                for k, v in self.items()}


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = Config(data)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg
