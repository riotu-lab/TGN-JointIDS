"""Train TGN-JointIDS (or one of its ablations) for one seed with chronological lanes and
truncated BPTT, then write causal predictions (memory reset at every 60 s block, frozen
parameters) for the validation and test partitions.

    python scripts/train.py --config configs/nf_ton_iot_v3.yaml --seed 13

    # ablations: same config, one override each, separate output folder
    python scripts/train.py --config configs/nf_ton_iot_v3.yaml --seed 13 \
        --set memory.arm=graph_only paths.output_dir=results/nf_ton_iot_v3/ablations/graph_only

    # smoke test: a few hundred steps, records runtime, GPU memory, GRU gradients
    python scripts/train.py --config ... --seed 13 --smoke-steps 240 --no-eval-stream

Writes to <output_dir>/seed_<seed>/: history.json, checkpoint.pt, predictions_val.npz,
predictions_stream.npz (stream = test partition), run_info.json (smoke.json with
--smoke-steps).
"""

import json
import time

import numpy as np
import torch

from _common import base_parser, load
from tgn_jointids.pipeline import (load_graph_bundle, make_label_space, make_model,
                                   make_tbptt_trainer)
from tgn_jointids.utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def save_predictions(path, pred: dict) -> None:
    np.savez_compressed(path, **{k: v for k, v in pred.items() if isinstance(v, np.ndarray)},
                        loss=np.array(pred["loss"]))


def main():
    p = base_parser(__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Train for this many lane-steps (one epoch) and report cost only.")
    p.add_argument("--no-eval-stream", action="store_true")
    p.add_argument("--measure-latency", type=int, default=0,
                   help="Per-window causal inference latency on the first N stream windows "
                        "(one lane, i.e. one window per forward pass).")
    args = p.parse_args()
    cfg, out = load(args)
    if cfg.memory.get("mode") != "tbptt":
        raise ValueError("This script requires memory.mode: tbptt")
    if cfg.training.get("schedule_seed") is None:
        cfg.training.schedule_seed = args.seed
    device = get_device(args.device)
    arm = cfg.memory.arm
    run_dir = ensure_dir(out / f"seed_{args.seed}")

    bundle = load_graph_bundle(cfg)
    labels = make_label_space(cfg)
    set_seed(args.seed)
    model = make_model(cfg, bundle["node_dim"], bundle["edge_dim"], labels.num_classes).to(device)
    trainer = make_tbptt_trainer(cfg, model, bundle["train"], labels, device, t0=bundle["t0"])
    import hashlib
    split_sha = hashlib.sha256(json.dumps({k: list(map(int, v)) for k, v in bundle["split"].items()},
                                          sort_keys=True).encode()).hexdigest()
    info = {"arm": arm, "seed": args.seed, "graphs": cfg.paths.graphs, "split_sha256": split_sha,
            "feature_scaler": bundle.get("scaler", {}).get("mode"),
            "parameters": count_parameters(model), "config": cfg.to_dict()}
    print(f"arm={arm} seed={args.seed} params={info['parameters']:,} "
          f"scaler={info['feature_scaler']} graphs={cfg.paths.graphs}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.smoke_steps:
        t0 = time.perf_counter()
        tr = trainer.train_epoch(bundle["train"], epoch=0, max_steps=args.smoke_steps)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        va = trainer.causal_predict(bundle["val"])
        t_val = time.perf_counter() - t0
        smoke = {"arm": arm, "lane_steps": tr["steps"], "optimizer_steps": tr["opt_steps"],
                 "train_seconds": t_train, "seconds_per_lane_step": t_train / max(tr["steps"], 1),
                 "full_epoch_lane_steps": trainer_steps(trainer, bundle["train"]),
                 "val_causal_seconds": t_val, "val_graphs": len(bundle["val"]),
                 "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2**20
                                 if torch.cuda.is_available() else None),
                 "gru_grad_norms": tr["gru_grad_norms"], "total_grad_norm": tr["total_grad_norm"],
                 "train_loss": tr["loss"],
                 "val_acc_after_smoke": float(np.mean(va["pred_multi"] == va["y_multi"]))}
        smoke["est_epoch_seconds"] = (smoke["seconds_per_lane_step"] * smoke["full_epoch_lane_steps"]
                                      + t_val)
        save_json(smoke, run_dir / "smoke.json")
        print({k: v for k, v in smoke.items() if k != "gru_grad_norms"})
        print("GRU grad norms per optimizer step:", [f"{g:.2e}" for g in tr["gru_grad_norms"]])
        return

    t_fit = time.perf_counter()
    max_updates = int(cfg.training.get("max_updates", 0) or 0)
    if max_updates:
        history = trainer.fit_fixed_updates(
            bundle["train"], bundle["val"], max_updates=max_updates,
            eval_every=int(cfg.training.get("eval_every_updates", 100)),
            lr_schedule=cfg.training.get("lr_schedule", "cosine"))
        info.update({"budget": "fixed_updates", "total_updates": trainer.total_updates,
                     "best_update": trainer.best_epoch})
    else:
        history = trainer.fit(bundle["train"], bundle["val"], epochs=int(cfg.training.epochs))
    info["train_seconds_total"] = time.perf_counter() - t_fit
    if torch.cuda.is_available():
        info["peak_gpu_mb_training"] = torch.cuda.max_memory_allocated() / 2**20
    save_json(history, run_dir / "history.json")
    torch.save({"model_state_dict": model.state_dict(), "best_epoch": trainer.best_epoch,
                "best_score": trainer.best_score, "arm": arm, "seed": args.seed,
                "config": cfg.to_dict(), "feature_scaler": bundle.get("scaler")},
               run_dir / "checkpoint.pt")
    save_predictions(run_dir / "predictions_val.npz", trainer.causal_predict(bundle["val"]))
    if not args.no_eval_stream:
        save_predictions(run_dir / "predictions_stream.npz",
                         trainer.causal_predict(bundle["stream"]))
    if args.measure_latency:
        info["latency"] = measure_latency(trainer, bundle["stream"], args.measure_latency)
    info.update({"best_epoch": trainer.best_epoch, "best_val_macro_f1": trainer.best_score})
    save_json(info, run_dir / "run_info.json")
    print(f"done -> {run_dir}")


def measure_latency(trainer, stream, n: int) -> dict:
    """Causal per-window latency: earliest n stream windows in time order, one
    lane (one window per forward pass, memory read + encode + predict + write)."""
    order = np.argsort([int(g.window_id.item()) for g in stream], kind="stable")[:n]
    subset = [stream[i] for i in order]
    trainer.causal_predict(subset[: min(200, n)], num_lanes=1)            # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.causal_predict(subset, num_lanes=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return {"windows": len(subset), "mean_ms_per_window": 1000 * sec / len(subset),
            "note": "includes host-side batching and memory dictionary operations"}


def trainer_steps(trainer, graphs) -> int:
    from tgn_jointids.lanes import UnitLaneScheduler
    return UnitLaneScheduler(graphs, trainer.num_lanes, trainer.unit_ms, trainer.t0,
                             seed=trainer.schedule_seed).n_steps(0)


if __name__ == "__main__":
    main()
