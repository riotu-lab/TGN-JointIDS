"""Five-seed robustness study (Tables 8 and 11): train -> warm-up -> EWC -> stream,
with a fixed split and an independent initialisation per seed.

    python scripts/run_seeds.py --config configs/nf_ton_iot_v3.yaml
    python scripts/run_seeds.py --config configs/nf_cicids2018_v3.yaml --seeds 13 42
"""

import gc

import pandas as pd
import torch

from _common import base_parser, load
from stream_eval import stream_and_report
from tgn_jointids.pipeline import load_graph_bundle
from tgn_jointids.utils import get_device
from train import train_one_seed


def flatten(seed, m):
    row = {"seed": seed, "n_windows": m["n_windows"]}
    row.update({f"bin_{k}": v for k, v in m["binary"].items()})
    for key in ("attack_type_cascade", "attack_type_raw"):
        row.update({f"{key}_{k}": v for k, v in m[key].items() if k != "recall"})
    row.update({f"recall_{k}": v for k, v in m["tracked_recall"].items()})
    row.update({f"latency_{k}": v for k, v in m["latency"].items()})
    return row


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    cfg, out = load(args)
    device = get_device(args.device)
    seeds = args.seeds or list(cfg.training.seeds)
    bundle = load_graph_bundle(cfg)

    rows = []
    for seed in seeds:
        print("=" * 80 + f"\nSEED {seed}\n" + "=" * 80)
        model, memory, trainer, _, _ = train_one_seed(cfg, seed, device, out, bundle,
                                                      verbose=not args.quiet)
        _, metrics = stream_and_report(cfg, model, memory, trainer, bundle, device, out, seed)
        rows.append(flatten(seed, metrics))
        pd.DataFrame(rows).to_csv(out / "seeds_per_run.csv", index=False)
        del model, memory, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    num = df.drop(columns=["seed"]).select_dtypes("number")
    summary = pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1)})
    summary.to_csv(out / "seeds_summary.csv")
    print("\nMean ± std over seeds", seeds)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
