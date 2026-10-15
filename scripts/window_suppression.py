"""Window-level composition / majority-vote suppression analysis (Section 5.1, Eq. 2).

For every non-empty window (before priority-aware labelling) and every attack
class c present in it, reports:
  mixed %       share of c-windows containing more than one label
  suppressed %  S_c = |{w : n_c(w) < max_{j≠c} n_j(w)}| / |{w : c ∈ w}|
  mean share %  mean of n_c(w) / |w| over c-windows

    python scripts/window_suppression.py --config configs/nf_ton_iot_v3.yaml
    python scripts/window_suppression.py --config ... --attack-only   # j ranges over attacks only
"""

from collections import defaultdict

import pandas as pd

from _common import base_parser, load
from tgn_jointids.data import balance_classes, iter_time_windows, load_flows
from tgn_jointids.utils import env_path


def main():
    p = base_parser(__doc__)
    p.add_argument("--attack-only", action="store_true",
                   help="Compare n_c only against other ATTACK classes (exclude benign).")
    p.add_argument("--no-balance", action="store_true", help="Analyse the raw flow file.")
    args = p.parse_args()
    cfg, out = load(args)
    benign = cfg.data.benign_label

    df = load_flows(env_path(cfg.paths.flows_csv))
    if cfg.data.balance_target and not args.no_balance:
        df = balance_classes(df, cfg.data.balance_target, seed=cfg.data.balance_seed)

    n_win = mixed_win = 0
    present, mixed, suppressed, share = (defaultdict(int), defaultdict(int),
                                         defaultdict(int), defaultdict(float))
    for _, snap in iter_time_windows(df, window_ms=cfg.graph.window_ms, stride_ms=cfg.graph.stride_ms):
        counts = snap["Attack"].value_counts()
        n_win += 1
        is_mixed = len(counts) > 1
        mixed_win += is_mixed
        for c, n_c in counts.items():
            if c == benign:
                continue
            others = counts.drop(c)
            if args.attack_only:
                others = others.drop(benign, errors="ignore")
            present[c] += 1
            mixed[c] += is_mixed
            suppressed[c] += int(len(others) > 0 and n_c < others.max())
            share[c] += n_c / len(snap)

    rows = [{"class": c, "windows": present[c], "mixed_pct": 100 * mixed[c] / present[c],
             "suppressed_pct": 100 * suppressed[c] / present[c],
             "mean_share_pct": 100 * share[c] / present[c]} for c in present]
    table = pd.DataFrame(rows).sort_values("suppressed_pct", ascending=False)
    table.to_csv(out / "window_suppression.csv", index=False)
    print(f"Non-empty windows: {n_win:,} | mixed: {100 * mixed_win / max(n_win, 1):.2f}%")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
