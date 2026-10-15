"""Shared CLI plumbing for the scripts (adds the repo root to sys.path)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tgn_jointids.config import load_config  # noqa: E402
from tgn_jointids.utils import ensure_dir, env_path  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config entries, e.g. continual.replay.mode=class_specific")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


def _parse_value(v: str):
    import yaml
    return yaml.safe_load(v)


def load(args):
    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: _parse_value(v) for k, v in overrides.items()}
    cfg = load_config(args.config, overrides)
    out = ensure_dir(env_path(cfg.paths.output_dir))
    return cfg, out
