"""Step 3 — memory warm-up, EWC reference and prequential streaming evaluation.

    python scripts/stream_eval.py --config configs/nf_ton_iot_v3.yaml \
        --checkpoint results/nf_ton_iot_v3/checkpoints/seed_2026.pt

Writes to <output_dir>/seed_<seed>/: stream_results.csv, metrics.json,
attack_type_per_class.csv, attack_type_confusion.csv, classification_report.txt,
alerts.json.
"""

from pathlib import Path

import pandas as pd
import torch

from _common import base_parser, load
from tgn_jointids.metrics import attack_only_metrics, full_classification_report
from tgn_jointids.pipeline import (load_graph_bundle, make_label_space, make_memory, make_model,
                                   make_trainer, prepare_for_streaming, summarize_stream)
from tgn_jointids.streaming import run_stream
from tgn_jointids.utils import ensure_dir, get_device, save_json, set_seed


def stream_and_report(cfg, model, memory, trainer, bundle, device, out_dir: Path, seed: int):
    labels = make_label_space(cfg)
    print("Phase 2 — memory warm-up + EWC reference")
    prepare_for_streaming(cfg, trainer, bundle["train"], device)
    print(f"  memory keys after warm-up: {len(memory):,}")
    warm_keys = len(memory)

    print(f"Phase 3 — streaming {len(bundle['stream']):,} windows")
    df, alerts = run_stream(model, memory, trainer, bundle["stream"], device,
                            update_interval=cfg.continual.update_interval,
                            conf_threshold=cfg.continual.alert_threshold,
                            id2attack=labels.id2attack)

    d = ensure_dir(out_dir / f"seed_{seed}")
    df["type_true"] = df["multi_true"].map(labels.id2attack)
    df["type_pred"] = df["multi_pred"].map(labels.id2attack)
    df.to_csv(d / "stream_results.csv", index=False)

    metrics = summarize_stream(cfg, df, labels, memory)
    metrics["memory"]["keys_after_warmup"] = warm_keys
    metrics["n_alerts"] = len(alerts)
    save_json(metrics, d / "metrics.json")
    save_json({"alerts": alerts}, d / "alerts.json")

    _, per_class, cm = attack_only_metrics(df, labels, cascade=True)
    per_class.to_csv(d / "attack_type_per_class.csv", index=False)
    names = [labels.id2attack[i] for i in labels.attack_ids]
    pd.DataFrame(cm, index=names, columns=names + ["benign"]).to_csv(d / "attack_type_confusion.csv")
    (d / "classification_report.txt").write_text(full_classification_report(df, labels))

    b, a = metrics["binary"], metrics["attack_type_cascade"]
    print(f"Binary acc {b['accuracy']:.4f} | F1(attack) {b['f1_attack']:.4f} | "
          f"FNR {b['fnr']:.4f} | benign FPR {b['benign_fpr']:.4f}")
    print(f"Attack-type weighted F1 {a['f1_weighted']:.4f} | macro F1 {a['f1_macro']:.4f}")
    print(f"Latency mean {metrics['latency']['mean_ms']:.2f} ms | P95 {metrics['latency']['p95_ms']:.2f} ms")
    print(f"Results -> {d}")
    return df, metrics


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    cfg, out = load(args)
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    seed = int(ckpt.get("seed", 0))
    set_seed(seed)

    bundle = load_graph_bundle(cfg)
    labels = make_label_space(cfg)
    model = make_model(cfg, bundle["node_dim"], bundle["edge_dim"], labels.num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    memory = make_memory(cfg)
    trainer = make_trainer(cfg, model, memory, bundle["train"], labels, device)
    stream_and_report(cfg, model, memory, trainer, bundle, device, out, seed)


if __name__ == "__main__":
    main()
