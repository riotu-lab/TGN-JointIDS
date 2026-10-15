# TGN-JointIDS

**Long-Tail-Aware Streaming Intrusion Detection in IoT Networks via Temporal Graph Memory and Multi-Task Learning**

Official PyTorch / PyTorch Geometric implementation.

TGN-JointIDS detects intrusions on a single-pass, chronologically ordered stream of IoT NetFlow
records. Every 5 s window becomes a communication graph (nodes = IPv4 hosts, edges = flows); a
persistent IP-keyed TGN memory carries each host's behavioural state across windows; a shared
edge-aware GAT + SAGPool encoder feeds three heads (binary alert, attack type, per-host
participation); and a long-tail-aware objective plus EWC-regularised replay keeps rare classes
alive throughout continual adaptation.

```
flows ─► sliding-window graphs ─► memory read (Eq. 6) ─► [x‖m̃] MLP ─► 3×GAT ─► SAGPool ─┬─► binary head
          W=5 s, S=2.5 s           IPv4 integer key                                     ├─► attack-type head
                                                                                        └─► node head
          prequential: predict ─► memory write-back (GRU, Eq. 7) ─► label revealed ─► replay ─► update every K=32 (EWC)
```

## Key results (NF-ToN-IoT-v3, 29,453 streaming windows)

| Metric | Single run (seed 2026) | 5 seeds (mean ± std) |
|---|---|---|
| Binary accuracy | 97.46 % | 97.59 ± 0.26 % |
| False-negative rate | 1.61 % | 1.57 ± 0.25 % |
| Attack-type weighted F1 | 0.9743 | 0.9610 ± 0.0231 |
| MITM / Ransomware recall | 0.9216 / 0.8672 | 0.9443 / 0.9303 |
| Mean / P95 latency (Quadro RTX 8000) | 1.95 / 2.25 ms | — |

External validation on NF-CICIDS2018-v3 (5 seeds): binary accuracy 91.74 ± 0.49 %, multiclass
weighted F1 85.46 ± 3.06 %.

## Repository structure

```
TGN-JointIDS/
├── configs/
│   ├── nf_ton_iot_v3.yaml          # all hyper-parameters of Table 3
│   └── nf_cicids2018_v3.yaml
├── tgn_jointids/
│   ├── data/
│   │   ├── preprocessing.py        # loading, chronological sort, class re-sampling
│   │   ├── features.py             # label space, IPv4 → integer key (Eq. 5)
│   │   ├── graph_builder.py        # windowing, node/edge features, priority labels (Eqs. 1–4)
│   │   └── split.py                # stratified 60/15/25 split
│   ├── models/
│   │   ├── memory.py               # persistent TGN node memory (Eqs. 6–7)
│   │   └── tgn_jointids.py         # GAT + SAGPool encoder, three heads (Eqs. 8–13)
│   ├── losses.py                   # ENS weights, focal loss, joint objective (Eqs. 14–16)
│   ├── continual.py                # EWC (Eq. 17), replay buffers
│   ├── trainer.py                  # offline training, warm-up, incremental updates
│   ├── streaming.py                # prequential test-then-train loop (Algorithm 1)
│   ├── metrics.py                  # binary, attack-only, OvR, threshold, latency metrics
│   └── pipeline.py                 # component factories used by the scripts
├── scripts/
│   ├── build_graphs.py             # flows → graphs + fixed split
│   ├── train.py                    # offline training (one seed)
│   ├── stream_eval.py              # warm-up + EWC + streaming evaluation
│   ├── run_seeds.py                # five-seed study
│   ├── identifier_ablation.py      # stable vs permuted vs non-persistent IDs
│   ├── threshold_analysis.py       # PR curve, threshold sweep
│   ├── window_suppression.py       # window composition / suppression analysis
│   └── make_figures.py             # confusion matrices, rolling accuracy, confidence
├── notebooks/                      # original research notebooks (outputs cleared)
├── data/README.md                  # where to get the datasets
└── results/                        # created at run time
```

## Installation

```bash
git clone https://github.com/<user>/TGN-JointIDS.git
cd TGN-JointIDS
conda env create -f environment.yml && conda activate tgn-jointids
# or: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10–3.12, PyTorch 2.5.1 (CUDA 12.1) and PyTorch Geometric 2.x.

## Quick start

Place the flow CSV in `data/raw/` (see [`data/README.md`](data/README.md)), then:

```bash
CFG=configs/nf_ton_iot_v3.yaml

python scripts/build_graphs.py --config $CFG                       # graphs + fixed split
python scripts/train.py        --config $CFG --seed 2026           # offline training
python scripts/stream_eval.py  --config $CFG \
       --checkpoint results/nf_ton_iot_v3/checkpoints/seed_2026.pt  # streaming evaluation
python scripts/make_figures.py --config $CFG \
       --stream-results results/nf_ton_iot_v3/seed_2026/stream_results.csv
```

Any config entry can be overridden from the command line:

```bash
python scripts/run_seeds.py --config $CFG --set continual.replay.mode=class_specific split.sort_by=window_id
```

## Reproducing the paper

| Paper item | Command |
|---|---|
| Table 4 – window suppression | `scripts/window_suppression.py` |
| Tables 5–7, Figs. 2, 3, 5, 6 – single run | `scripts/train.py --seed 2026` → `scripts/stream_eval.py` → `scripts/make_figures.py` |
| Table 8 / Table 11 – five seeds | `scripts/run_seeds.py` (ToN / CICIDS config) |
| Table 9, Fig. 4 – operating point | `scripts/threshold_analysis.py` |
| Table 10 – latency | `latency` block of `metrics.json` |
| Fig. 7 – identifier persistence | `scripts/identifier_ablation.py` |

The modular code is a refactor of the notebooks in `notebooks/`. It was checked against them:
the model initialises identically under the same seed (state-dict keys and values match, so
notebook checkpoints load directly), and the graph builder produces bit-identical graphs.

Outputs per run (`results/<experiment>/seed_<s>/`): `stream_results.csv` (one row per window,
including `attack_prob`), `metrics.json`, `attack_type_per_class.csv`,
`attack_type_confusion.csv`, `classification_report.txt`, `alerts.json`.

Attack-type metrics are reported in two forms: `attack_type_cascade` (benign whenever the binary
head says benign — the rule of Section 4.4) and `attack_type_raw` (arg max of the attack-type head).

## Reproducibility notes

The defaults in `configs/` reproduce the code path that generated the reported numbers. Where that
code differs from an idealised description, the behaviour is exposed as a config switch:

| Setting | Default (as run) | Alternative |
|---|---|---|
| `continual.replay.mode` | `fifo` – FIFO buffer, capacity 5000 | `class_specific` – per-class reservoirs, k_min = 10, overflow 2000 |
| `split.sort_by` | `null` – subsets keep the stratified-split order | `window_id` – each subset re-sorted chronologically |
| ToN feature `NUM_PK TS_128_TO_256_BYTES` | kept verbatim (column absent → zero-filled) | `NUM_PKTS_128_TO_256_BYTES` |
| `loss.ens_source` | ToN: `flow_counts`, CICIDS: `train_graphs` | — |

The NF-CICIDS2018-v3 single-run validation report (Table 12) used lr = 5e-4 and dropout = 0.20
(`--set training.lr=5e-4 model.dropout=0.2`); the five-seed study uses the config defaults.

## Citation

```bibtex
@article{tgnjointids2026,
  title   = {Long-Tail-Aware Streaming Intrusion Detection in Internet of Things Networks
             via Temporal Graph Memory and Multi-Task Learning},
  author  = {Ben Atitallah, Safa and others},
  journal = {TODO},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The datasets are subject to their own licences.
