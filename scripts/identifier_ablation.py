"""Persistent-identifier ablation (Section 5.9, Figure 7).

Same checkpoint, same seed, three node-ID policies used as memory keys:
  stable_ip        original IPv4-derived integers
  permuted_stable  random one-to-one remapping, persistent across windows
  nonpersistent    fresh IDs for every node occurrence (no memory retrieval)

    python scripts/identifier_ablation.py --config configs/nf_ton_iot_v3.yaml \
        --checkpoint results/nf_ton_iot_v3/checkpoints/seed_2026.pt
"""

import numpy as np
import pandas as pd
import torch

from _common import base_parser, load
from tgn_jointids.pipeline import (load_graph_bundle, make_label_space, make_memory, make_model,
                                   make_trainer, prepare_for_streaming, summarize_stream)
from tgn_jointids.streaming import run_stream
from tgn_jointids.utils import ensure_dir, get_device, save_json, set_seed

VARIANTS = ("stable_ip", "permuted_stable", "nonpersistent")


def remap(bundle, variant, seed):
    splits = {k: [g.clone() for g in bundle[k]] for k in ("train", "val", "stream")}
    if variant == "stable_ip":
        return splits
    if variant == "permuted_stable":
        ids = sorted({int(n) for k in splits for g in splits[k] for n in g.node_ids.tolist()})
        perm = np.random.default_rng(seed).permutation(len(ids)) + 1_000_000
        mapping = dict(zip(ids, map(int, perm)))
        for k in splits:
            for g in splits[k]:
                g.node_ids = torch.tensor([mapping[int(n)] for n in g.node_ids.tolist()],
                                          dtype=torch.long)
        return splits
    if variant == "nonpersistent":
        counter = 10_000_000
        for k in ("train", "val", "stream"):
            for g in splits[k]:
                n = int(g.num_nodes)
                g.node_ids = torch.arange(counter, counter + n, dtype=torch.long)
                counter += n
        return splits
    raise ValueError(variant)


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    cfg, out = load(args)
    device = get_device(args.device)
    labels = make_label_space(cfg)
    bundle = load_graph_bundle(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    d = ensure_dir(out / "identifier_ablation")

    rows = []
    for variant in VARIANTS:
        print(f"--- {variant} ---")
        splits = remap(bundle, variant, args.seed)
        set_seed(args.seed)  # identical replay draws across variants
        model = make_model(cfg, bundle["node_dim"], bundle["edge_dim"], labels.num_classes).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        memory = make_memory(cfg)
        trainer = make_trainer(cfg, model, memory, splits["train"], labels, device)
        prepare_for_streaming(cfg, trainer, splits["train"], device)
        df, _ = run_stream(model, memory, trainer, splits["stream"], device,
                           update_interval=cfg.continual.update_interval,
                           conf_threshold=cfg.continual.alert_threshold, log_every=0)
        df.to_csv(d / f"stream_results_{variant}.csv", index=False)
        m = summarize_stream(cfg, df, labels, memory)
        save_json(m, d / f"metrics_{variant}.json")
        rows.append({"variant": variant, "bin_accuracy": m["binary"]["accuracy"],
                     "benign_fpr": m["binary"]["benign_fpr"],
                     "attack_f1_weighted": m["attack_type_cascade"]["f1_weighted"],
                     **{f"recall_{k}": v for k, v in m["tracked_recall"].items()},
                     "memory_keys": m["memory"]["keys"], "memory_bytes": m["memory"]["bytes"]})
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(d / "identifier_ablation.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
