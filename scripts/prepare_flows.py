"""Derive a reduced flow file from an official NetFlow-v3 CSV ("common20_rare100" rule).

    python scripts/prepare_flows.py --csv <official.csv> --out <derived.csv> \
        [--frac 0.2] [--min-flows 200000] [--seed 42]

Rule (the one found in NF-ToN-IoT-v3_common20_rare100.csv, reconstructed by
comparison with the official NF-ToN-IoT-v3 file, results/nf_cicids2018_v3/
downsampling_rule.json): every class with at least --min-flows flows keeps a uniform
random --frac of its flows (per flow, without replacement); smaller classes are kept whole.
Rows are copied unchanged, so the derived file is an exact subset of the official one and
spans the same time. No oversampling. Writes <out> and <out>.manifest.json (input / output
SHA-256, per-class counts before and after, the rule and the seed).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def select_rows(labels: pd.Series, frac: float, min_flows: int, seed: int) -> np.ndarray:
    """Boolean mask of kept rows. Classes are processed in sorted order with one RNG,
    so the selection depends only on the labels, frac, min_flows and seed."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(labels), bool)
    codes = labels.to_numpy()
    for c in sorted(labels.unique()):
        idx = np.flatnonzero(codes == c)
        if len(idx) >= min_flows:
            idx = rng.choice(idx, int(round(frac * len(idx))), replace=False)
        keep[idx] = True
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--min-flows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    df = pd.read_csv(args.csv)
    labels = df["Attack"].astype(str).str.strip().str.lower()
    keep = select_rows(labels, args.frac, args.min_flows, args.seed)
    before, after = labels.value_counts(), labels[keep].value_counts()
    df.loc[keep].to_csv(out, index=False)          # original row order and values
    manifest = {"input": args.csv, "input_sha256": sha256(args.csv), "output": str(out),
                "output_sha256": sha256(out),
                "rule": f"classes with >= {args.min_flows} flows: uniform random {args.frac:.0%} "
                        f"of flows without replacement; other classes kept whole",
                "frac": args.frac, "min_flows": args.min_flows, "seed": args.seed,
                "rows_before": int(len(df)), "rows_after": int(keep.sum()),
                "per_class": {c: {"before": int(before[c]), "after": int(after.get(c, 0)),
                                  "downsampled": bool(before[c] >= args.min_flows)}
                              for c in before.index}}
    with open(str(out) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
