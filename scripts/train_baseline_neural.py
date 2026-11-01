"""Neural baselines, all seeds in one process (the bundle is loaded once).

    python scripts/train_baseline_neural.py --config configs/nf_cicids2018_v3.yaml \
        --model egraphsage --seeds 13 42 123 777 2026 --out results/baselines/<dataset>/egraphsage
    python scripts/train_baseline_neural.py --config ... --model window_gru \
        --feats data/processed/baselines/<dataset>_winfeat.npz --seeds ... --out ...

egraphsage  E-GraphSAGE (Lo et al., NOMS 2022): edge-feature GraphSAGE layers
            h_N(v) = mean_{u->v} W_m [h_u || e_uv],  h_v = ReLU(W_a [h_v || h_N(v)]), 2 layers,
            128-d, both edge directions; edge embedding [h_u || h_v]. Initial node features are
            the same 35-d host features as TGN-JointIDS (the original uses a constant vector).
            Window readout = [mean of node emb || mean and max of edge emb] -> MLP -> classes.
            Static: no memory across windows.
window_gru  Sequence model without a graph: per-window feature vectors (baseline_features.py,
            standardised on training windows) -> MLP -> unidirectional GRU (128) over the
            windows of each 60 s block in time order, reset at every block (the same reset unit
            as the TGN-JointIDS memory) -> classes. Causal: the output at a window sees only
            that window and the earlier windows of its block.

Both use the TGN-JointIDS training budget: 7,000 updates, AdamW (lr 1e-3, wd 1e-4), cosine LR
to 1e-5, grad clip 1.0, validation every 100 updates, checkpoint = best validation macro-F1.
Loss: class-weighted cross-entropy (inverse class frequency on training windows, normalised to
mean 1, capped at 30x like the ENS cap). Binary decision: P(attack) = 1 - P(benign) >= 0.5.
Writes <out>/seed_<s>/: history.json, checkpoint.pt, predictions_val.npz,
predictions_stream.npz (window_id order, same keys as TGN-JointIDS), run_info.json.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from _common import base_parser, load
from tgn_jointids.utils import count_parameters, set_seed

UPDATES, EVAL_EVERY, LR, MIN_LR, WD, CLIP, BATCH = 7000, 100, 1e-3, 1e-5, 1e-4, 1.0, 32


# ---------------------------------------------------------------------------- E-GraphSAGE
class ESAGELayer(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr="mean")
        self.msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        h_n = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(self.upd(torch.cat([x, h_n], -1)))

    def message(self, x_j, edge_attr):
        return self.msg(torch.cat([x_j, edge_attr], -1))


class EGraphSAGE(nn.Module):
    def __init__(self, node_dim, edge_dim, C, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        dims = [node_dim] + [hidden] * layers
        self.layers = nn.ModuleList(ESAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, C))

    def forward(self, data):
        x, ei, ea, b = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
        ei2 = torch.cat([ei, ei.flip(0)], 1)                 # both directions
        ea2 = torch.cat([ea, ea], 0)
        for layer in self.layers:
            x = self.drop(layer(x, ei2, ea2))
        B = int(data.num_graphs)
        z = torch.cat([x[ei[0]], x[ei[1]]], -1)              # edge embedding [h_u || h_v]
        eb = b[ei[0]]
        g = torch.cat([scatter(x, b, 0, dim_size=B, reduce="mean"),
                       scatter(z, eb, 0, dim_size=B, reduce="mean"),
                       scatter(z, eb, 0, dim_size=B, reduce="max")], -1)
        return self.head(g)


# ---------------------------------------------------------------------------- Window-GRU
class WindowGRU(nn.Module):
    def __init__(self, in_dim, C, hidden=128, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, C))

    def forward(self, x):                                    # x [B, T, D] -> logits [B, T, C]
        h, _ = self.gru(self.inp(x))
        return self.head(h)


def blocks_of(X, wid, t0, block_ms=60000):
    """Split one partition (sorted by window_id) into per-block sequences: list of index arrays."""
    blk = (wid - t0) // block_ms
    cuts = np.flatnonzero(np.diff(blk)) + 1
    return np.split(np.arange(len(wid)), cuts)


def pad(X, idx_list, device):
    T = max(len(i) for i in idx_list)
    out = np.zeros((len(idx_list), T, X.shape[1]), np.float32)
    mask = np.zeros((len(idx_list), T), bool)
    for k, i in enumerate(idx_list):
        out[k, :len(i)] = X[i]
        mask[k, :len(i)] = True
    return torch.from_numpy(out).to(device), torch.from_numpy(mask).to(device)


# ---------------------------------------------------------------------------- shared
def class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(float)
    w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1), 0.0)
    w = w / w[cnt > 0].mean()
    w = np.minimum(w, 30 * w[cnt > 0].min())
    return torch.tensor(w / w[cnt > 0].mean(), dtype=torch.float32)


def lr_at(u):
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * u / UPDATES))


def to_predictions(prob, y_multi, y_bin, wid):
    pa = 1.0 - prob[:, 0]
    return {"prob_multi": prob.astype(np.float32), "prob_attack": pa.astype(np.float32), "y_multi": y_multi,
            "y_bin": y_bin, "pred_multi": prob.argmax(1), "pred_bin": (pa >= 0.5).astype(int), "window_id": wid}


class GraphTask:
    def __init__(self, cfg, device):
        from tgn_jointids.pipeline import load_graph_bundle
        b = load_graph_bundle(cfg)
        self.device, self.C = device, len(cfg.data.classes)
        self.node_dim, self.edge_dim = b["node_dim"], b["edge_dim"]
        self.parts = {}
        for p in ("train", "val", "stream"):
            gs = b[p]
            wid = np.array([int(g.window_id) for g in gs])
            order = np.argsort(wid, kind="stable")
            gs = [gs[i] for i in order]
            for g in gs:      # drop fields the baseline does not use (and must not see)
                for k in ("node_ids", "y_node", "attack_dist", "attack_ratio", "edge_type"):
                    if k in g:
                        del g[k]
            self.parts[p] = {"graphs": gs, "window_id": wid[order],
                             "y_multi": np.array([int(g.y_multi) for g in gs]),
                             "y_bin": np.array([int(g.y_bin) for g in gs])}
        self.y_train = self.parts["train"]["y_multi"]

    def model(self):
        return EGraphSAGE(self.node_dim, self.edge_dim, self.C)

    def batches(self, rng):
        while True:
            loader = DataLoader(self.parts["train"]["graphs"], batch_size=BATCH, shuffle=True,
                                generator=torch.Generator().manual_seed(int(rng.integers(1 << 31))))
            for data in loader:
                data = data.to(self.device)
                yield data, data.y_multi.view(-1)

    def forward(self, model, batch):
        return model(batch)

    @torch.no_grad()
    def predict(self, model, part):
        model.eval()
        out = []
        for data in DataLoader(self.parts[part]["graphs"], batch_size=512):
            out.append(F.softmax(model(data.to(self.device)), -1).cpu())
        model.train()
        return torch.cat(out).numpy()

    def latency(self, model, n=2000):
        model.eval()
        lat = []
        with torch.no_grad():
            for g in self.parts["stream"]["graphs"][:n]:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                d = next(iter(DataLoader([g], batch_size=1))).to(self.device)
                model(d)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)
        return np.array(lat[20:])


class SeqTask:
    def __init__(self, feats, device):
        d = np.load(feats)
        self.device, self.C, t0 = device, len(d["classes"]), int(d["t0"])
        mu, sd = d["train_X"].mean(0), d["train_X"].std(0) + 1e-6
        self.parts = {}
        for p in ("train", "val", "stream"):
            X = ((d[f"{p}_X"] - mu) / sd).astype(np.float32)
            self.parts[p] = {"X": X, "window_id": d[f"{p}_window_id"], "y_multi": d[f"{p}_y_multi"],
                             "y_bin": d[f"{p}_y_bin"], "blocks": blocks_of(X, d[f"{p}_window_id"], t0)}
        self.y_train = self.parts["train"]["y_multi"]
        self.in_dim = self.parts["train"]["X"].shape[1]

    def model(self):
        return WindowGRU(self.in_dim, self.C)

    def batches(self, rng):
        P = self.parts["train"]
        y = torch.from_numpy(P["y_multi"]).to(self.device)
        while True:
            perm = rng.permutation(len(P["blocks"]))
            for s in range(0, len(perm) - BATCH + 1, BATCH):     # 32 blocks per update
                idx = [P["blocks"][k] for k in perm[s:s + BATCH]]
                x, m = pad(P["X"], idx, self.device)
                flat = torch.from_numpy(np.concatenate(idx)).to(self.device)
                yield (x, m), y[flat]

    def forward(self, model, batch):
        x, m = batch
        return model(x)[m]                                   # row-major mask = concatenated block order

    @torch.no_grad()
    def predict(self, model, part):
        model.eval()
        P = self.parts[part]
        prob = np.zeros((len(P["X"]), self.C), np.float32)
        for s in range(0, len(P["blocks"]), 256):
            idx = P["blocks"][s:s + 256]
            x, m = pad(P["X"], idx, self.device)
            prob[np.concatenate(idx)] = F.softmax(model(x)[m], -1).cpu().numpy()
        model.train()
        return prob

    def latency(self, model, n=2000):
        """One window per call, carrying the GRU state (streaming)."""
        model.eval()
        X = torch.from_numpy(self.parts["stream"]["X"][:n]).to(self.device)
        lat, h = [], None
        with torch.no_grad():
            for i in range(len(X)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, h = model.gru(model.inp(X[i].view(1, 1, -1)), h)
                model.head(out)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)
        return np.array(lat[20:])


def run_seed(task, seed, out, device):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = task.model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    w = class_weights(task.y_train, task.C).to(device)
    labels = list(range(task.C))
    yv = task.parts["val"]["y_multi"]
    run = Path(out) / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    hist = {"update": [], "val_macro_f1": [], "loss": []}
    best, best_u, losses = -1.0, 0, []
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    model.train()
    for u, (batch, y) in zip(range(1, UPDATES + 1), task.batches(rng)):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(u - 1)
        loss = F.cross_entropy(task.forward(model, batch), y, weight=w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        losses.append(loss.item())
        if u % EVAL_EVERY == 0:
            f = f1_score(yv, task.predict(model, "val").argmax(1), labels=labels, average="macro", zero_division=0)
            hist["update"].append(u)
            hist["val_macro_f1"].append(float(f))
            hist["loss"].append(float(np.mean(losses)))
            losses = []
            if f > best:
                best, best_u = f, u
                torch.save(model.state_dict(), run / "checkpoint.pt")
            if u % 1000 == 0:
                print(f"seed {seed} update {u} loss {hist['loss'][-1]:.4f} val macro-F1 {f:.4f} "
                      f"(best {best:.4f} @ {best_u})", flush=True)
    t_train = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**20
    model.load_state_dict(torch.load(run / "checkpoint.pt"))
    for part in ("val", "stream"):
        P = task.parts[part]
        pr = to_predictions(task.predict(model, part), P["y_multi"], P["y_bin"], P["window_id"])
        np.savez_compressed(run / f"predictions_{part}.npz", **pr, loss=np.array(np.nan))
    lat = task.latency(model)
    json.dump(hist, open(run / "history.json", "w"))
    info = {"seed": seed, "parameters": count_parameters(model), "train_seconds_total": t_train,
            "peak_gpu_mb_training": peak, "best_update": best_u, "best_val_macro_f1": best,
            "class_weights": w.cpu().tolist(),
            "latency": {"windows": int(len(lat)), "mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
                        "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99))}}
    json.dump(info, open(run / "run_info.json", "w"), indent=1)
    print(f"seed {seed} done: best val macro-F1 {best:.4f} @ {best_u}, {t_train / 60:.1f} min", flush=True)


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, choices=["egraphsage", "window_gru"])
    p.add_argument("--feats")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke-updates", type=int, default=0, help="Smoke test: train this many updates only.")
    args = p.parse_args()
    if args.smoke_updates:
        global UPDATES, EVAL_EVERY
        UPDATES, EVAL_EVERY = args.smoke_updates, max(args.smoke_updates // 2, 1)
    cfg, _ = load(args)
    device = torch.device(args.device)
    task = GraphTask(cfg, device) if args.model == "egraphsage" else SeqTask(args.feats, device)
    for s in args.seeds:
        if (Path(args.out) / f"seed_{s}" / "predictions_stream.npz").exists():
            print(f"seed {s} exists, skipped", flush=True)
            continue
        run_seed(task, s, args.out, device)


if __name__ == "__main__":
    main()
