"""Step 2 — offline training of TGN-JointIDS for one seed.

    python scripts/train.py --config configs/nf_ton_iot_v3.yaml --seed 2026
"""

import torch

from _common import base_parser, load
from tgn_jointids.pipeline import (load_graph_bundle, make_label_space, make_loaders, make_memory,
                                   make_model, make_trainer)
from tgn_jointids.utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def train_one_seed(cfg, seed, device, out_dir, bundle=None, verbose=True):
    bundle = bundle or load_graph_bundle(cfg)
    labels = make_label_space(cfg)
    set_seed(seed)
    model = make_model(cfg, bundle["node_dim"], bundle["edge_dim"], labels.num_classes).to(device)
    memory = make_memory(cfg)
    trainer = make_trainer(cfg, model, memory, bundle["train"], labels, device)
    if verbose:
        print(f"Seed {seed} | parameters: {count_parameters(model):,}")
    train_loader, val_loader = make_loaders(cfg, bundle["train"], bundle["val"])
    history = trainer.train(train_loader, val_loader, epochs=cfg.training.epochs, verbose=verbose)

    ckpt = ensure_dir(out_dir / "checkpoints") / f"seed_{seed}.pt"
    torch.save({"seed": seed, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(), "history": history,
                "best_val_acc": trainer.best_acc, "epochs_executed": len(history["train_loss"]),
                "node_dim": bundle["node_dim"], "edge_dim": bundle["edge_dim"],
                "config": cfg.to_dict()}, ckpt)
    save_json(history, out_dir / f"seed_{seed}" / "history.json")
    print(f"Best val attack-type acc {trainer.best_acc:.4f} | checkpoint -> {ckpt}")
    return model, memory, trainer, bundle, ckpt


def main():
    p = base_parser(__doc__)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    cfg, out = load(args)
    train_one_seed(cfg, args.seed, get_device(args.device), out)


if __name__ == "__main__":
    main()
