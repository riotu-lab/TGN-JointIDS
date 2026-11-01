"""Per-window causal inference latency distribution (mean, median, P95, P99).

    python scripts/measure_latency.py --config configs/nf_ton_iot_v3.yaml \
        --runs 13=results/nf_ton_iot_v3/tgn_jointids/seed_13 ... --out results/nf_ton_iot_v3/latency_idle.json

The graph bundle comes from --config; each checkpoint's model is rebuilt from the config saved
in the checkpoint. For each checkpoint the earliest N stream (test) windows
are processed in time order, one window per forward pass, exactly as in causal
evaluation: lane reset at block start, memory read, encode, predict (softmax),
memory write. Each window is timed individually with CUDA synchronisation, after a
warm-up pass. Run on an otherwise idle GPU.
"""

import argparse
import json
import time

import numpy as np
import torch

from _common import ROOT  # noqa: F401
from tgn_jointids.config import Config, load_config
from tgn_jointids.lanes import UnitLaneScheduler
from tgn_jointids.pipeline import (load_graph_bundle, make_label_space, make_model,
                                   make_tbptt_trainer)
from tgn_jointids.utils import get_device


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_windows(tr, graphs):
    tr.model.eval()
    memory = tr.new_memory()
    sched = UnitLaneScheduler(graphs, 1, tr.unit_ms, tr.t0, shuffle=False)
    out = []
    for step in sched.steps(0):
        sync()
        t0 = time.perf_counter()
        data, node_lanes, (bin_l, multi_l, _, node_emb, nids) = tr._forward(graphs, step, memory)
        torch.softmax(multi_l, 1), torch.softmax(bin_l, 1)
        tr._write(memory, node_lanes, nids, node_emb, keep_grad=False)
        sync()
        out.append(1000 * (time.perf_counter() - t0))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dataset config (graph bundle)")
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir, e.g. 13=results/.../seed_13")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = get_device(args.device)
    runs = dict(r.split("=", 1) for r in args.runs)
    bundle = load_graph_bundle(load_config(args.config))
    order = np.argsort([int(g.window_id.item()) for g in bundle["stream"]], kind="stable")[:args.windows]
    subset = [bundle["stream"][i] for i in order]
    res = {}
    for name, run in runs.items():
        ck = torch.load(f"{run}/checkpoint.pt", map_location=device, weights_only=False)
        cfg = Config(ck["config"])
        labels = make_label_space(cfg)
        model = make_model(cfg, bundle["node_dim"], bundle["edge_dim"], labels.num_classes).to(device)
        model.load_state_dict(ck["model_state_dict"])
        tr = make_tbptt_trainer(cfg, model, bundle["train"], labels, device, t0=bundle["t0"])
        time_windows(tr, subset[:300])                                   # warm-up
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ms = time_windows(tr, subset)
        res[name] = {"windows": int(len(ms)), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
                     "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
                     "max_ms": float(ms.max()),
                     "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)}
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
        del model, tr
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
