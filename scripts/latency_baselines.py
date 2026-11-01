"""Idle-machine latency of the baselines, one window per call (as for TGN-JointIDS).

    python scripts/latency_baselines.py --dataset NF-CICIDS2018-v3 --model egraphsage

One (dataset, model) per process, so memory is released between measurements; results are
merged into the JSON file.

Run after all baseline trainings have finished (the latency stored in their run_info.json was
measured while other jobs shared the GPU/CPU). Neural baselines: the selected checkpoint.
Tree baselines: the models are not stored, so the selected configuration is refitted with the
same seed (deterministic for both libraries on the same data) and timed. Model inference only;
building the window graph / feature vector is not included, as for TGN-JointIDS.
Writes results/baselines/latency_idle.json: {dataset: {model: {seed: stats}}}.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_baseline_neural as N  # noqa: E402
import train_baseline_tree as T  # noqa: E402
from _common import load  # noqa: E402

B = ROOT / "results/baselines"
SEEDS = [13, 42, 123, 777, 2026]
DS = {"NF-ToN-IoT-v3": ("nf_ton_iot_v3", "configs/nf_ton_iot_v3.yaml",
                        "data/processed/baselines/nf_ton_iot_v3_winfeat.npz"),
      "NF-CICIDS2018-v3": ("nf_cicids2018_v3", "configs/nf_cicids2018_v3.yaml",
                           "data/processed/baselines/nf_cicids2018_v3_winfeat.npz")}


def stats(lat):
    return {"windows": int(len(lat)), "mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
            "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DS))
    ap.add_argument("--model", required=True, choices=["xgboost", "random_forest", "window_gru", "egraphsage"])
    a = ap.parse_args()
    path = B / "latency_idle.json"
    all_res = json.load(open(path)) if path.exists() else {}
    dev = torch.device("cuda")
    for ds, (dd, cfg_path, feats) in [(a.dataset, DS[a.dataset])]:
        res = {ds: {}}
        d = np.load(feats)
        C = len(d["classes"])
        for m in [x for x in ("xgboost", "random_forest") if x == a.model]:
            res[ds][m] = {}
            for s in SEEDS:
                hp = json.load(open(B / dd / m / f"seed_{s}/run_info.json"))["selected"]
                clf = T.make(m, hp, s, C)
                if m == "xgboost":
                    clf.fit(d["train_X"], d["train_y_multi"], sample_weight=T.compute_sample_weight("balanced", d["train_y_multi"]),
                            eval_set=[(d["val_X"], d["val_y_multi"])], verbose=False)
                else:
                    clf.fit(d["train_X"], d["train_y_multi"])
                X = d["stream_X"][:2000]
                lat = []
                for i in range(len(X)):
                    t0 = time.perf_counter()
                    clf.predict_proba(X[i:i + 1])
                    lat.append((time.perf_counter() - t0) * 1000)
                res[ds][m][str(s)] = stats(np.array(lat[20:]))
                print(ds, m, s, round(res[ds][m][str(s)]["mean_ms"], 3), flush=True)
        if a.model == "window_gru":
            pairs = [("window_gru", N.SeqTask(feats, dev))]
        elif a.model == "egraphsage":
            args = N.base_parser("").parse_args(["--config", cfg_path, "--set", f"paths.output_dir=results/baselines/{dd}"])
            cfg, _ = load(args)
            pairs = [("egraphsage", N.GraphTask(cfg, dev))]
        else:
            pairs = []
        for m, task in pairs:
            res[ds][m] = {}
            for s in SEEDS:
                model = task.model().to(dev)
                model.load_state_dict(torch.load(B / dd / m / f"seed_{s}/checkpoint.pt"))
                res[ds][m][str(s)] = stats(task.latency(model))
                print(ds, m, s, round(res[ds][m][str(s)]["mean_ms"], 3), flush=True)
        all_res.setdefault(ds, {}).update(res[ds])
    json.dump(all_res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
