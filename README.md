# TGN-JointIDS

**TGN-JointIDS: Temporal Host Memory for Real-Time Graph-Based Intrusion Detection under Class Imbalance**

Official PyTorch / PyTorch Geometric implementation, with the configurations, scripts and result
tables used in the paper.

TGN-JointIDS classifies network traffic window by window. Every 5 s window of NetFlow records
(2.5 s stride) becomes a host graph: nodes are IPv4 hosts (35-d features), edges are flows (5-d
features). A GRU memory keyed by host IP carries each host's recent behaviour across the windows of
a 60 s block; an edge-aware graph-attention encoder with SAGPool readout feeds three heads (attack
vs benign, attack type, per-host participation); an ENS-weighted focal loss handles the long tail of
rare attack classes. Every window is predicted before its own memory update (causal inference).

```
flows ─► 5 s window graphs ─► memory read (per IP) ─► [x ‖ m] MLP ─► 3 × edge-aware GAT ─► SAGPool ─┬─► binary head
                                     ▲                                                              ├─► attack-type head
                                     └──── GRU memory write (reset every 60 s block) ◄──────────────┴─► node head
```

## Main results

Test partition, mean of five seeds (13, 42, 123, 777, 2026); 60 s block-stratified split
(whole 60 s blocks go to train / validation / test, 60/15/25, every class in every partition).

| Metric | NF-ToN-IoT-v3 (10 classes) | NF-CICIDS2018-v3 (15 classes) |
| --- | --- | --- |
| Macro-F1 | 0.923 ± 0.004 | 0.887 ± 0.002 |
| Accuracy | 0.966 | 0.957 |
| Binary F1 | 0.983 | 0.940 |
| Benign false-positive rate | 0.071 | 0.010 |
| Latency per window (RTX 4090) | 2.1 ms | 2.6 ms |
| Parameters | 302,224 | 302,709 |

Ablations (macro-F1): removing the temporal memory (Graph-only) costs 0.022 (ToN) and 0.028
(CICIDS), significant on both datasets (paired time-cluster bootstrap). External baselines on the
same windows and splits:

| Model | ToN macro-F1 | CICIDS macro-F1 |
| --- | --- | --- |
| Random Forest | 0.888 | 0.756 |
| XGBoost | 0.908 | 0.820 |
| Window-GRU (temporal, no graph) | 0.905 | 0.834 |
| E-GraphSAGE (static graph) | 0.903 | 0.840 |
| **TGN-JointIDS** | **0.923** | **0.887** |

Test blocks share attack sessions with training blocks, so these numbers measure recognition of
known attack campaigns. A later-day test on NF-CICIDS2018-v3 (train 14–28 Feb, test 1–2 Mar) is in
`results/nf_cicids2018_v3/later_day/`. All result tables, statistics and figures are indexed in
[results/README.md](results/README.md).

## Repository structure

```
configs/
  nf_ton_iot_v3.yaml                 NF-ToN-IoT-v3, block-stratified split
  nf_cicids2018_v3.yaml              NF-CICIDS2018-v3, block-stratified split
  nf_cicids2018_v3_later_day.yaml    NF-CICIDS2018-v3, later-day (forward-in-time) test
tgn_jointids/
  data/features.py                   label space, IPv4 -> integer key
  data/graph_builder.py              windowing, node / edge features, window labels
  data/normalization.py              signed log1p + z-score, fitted on training windows only
  data/preprocessing.py              flow loading
  models/tgn_jointids.py             encoder (GAT + SAGPool) and the three heads
  models/lane_memory.py              per-lane IP-keyed GRU memory
  lanes.py                           chronological lanes of 60 s blocks
  tbptt.py                           training with truncated BPTT, causal evaluation
  trainer.py                         optimiser, joint loss, checkpoint bookkeeping
  losses.py                          ENS class weights, focal loss, joint objective
  pipeline.py                        factories used by the scripts
  stats.py                           time-cluster bootstrap
scripts/                             data preparation, training, baselines, analysis (below)
tests/                               unit tests (python -m pytest tests -q)
results/                             result tables, statistics and figures (see results/README.md)
data/                                datasets (not distributed, see data/README.md)
```

## Installation

```bash
conda env create -f environment.yml && conda activate tgn-jointids
pip install -e .
```

or, with an existing PyTorch install, `pip install -r requirements.txt && pip install -e .`.
The paper's runs used Python 3.11, PyTorch 2.3.0 + CUDA 12.1, PyTorch Geometric 2.6.1, one
RTX 4090 (`results/nf_ton_iot_v3/environment.json`).

## Reproducing the paper

All commands run from the repository root. `<s>` is a seed in {13, 42, 123, 777, 2026}.

**1. Data** (see [data/README.md](data/README.md)). Classes with at least 200,000 flows keep a
random 20% of their flows; smaller classes are kept whole:

```bash
python scripts/prepare_flows.py --csv <official NF-CICIDS2018-v3.csv> \
    --out data/raw/NF-CICIDS2018-v3_common20_rare100.csv --frac 0.2 --min-flows 200000 --seed 42
python scripts/build_graphs.py        --config configs/nf_ton_iot_v3.yaml
python scripts/build_graphs_cicids.py --config configs/nf_cicids2018_v3.yaml
python scripts/build_graphs_cicids.py --config configs/nf_cicids2018_v3_later_day.yaml
```

**2. TGN-JointIDS and its ablations** (7,000 updates each; replace the dataset name for CICIDS):

```bash
D=nf_ton_iot_v3
python scripts/train.py --config configs/$D.yaml --seed <s>        # -> results/$D/tgn_jointids/seed_<s>
python scripts/train.py --config configs/$D.yaml --seed <s> --set memory.arm=graph_only     paths.output_dir=results/$D/ablations/graph_only
python scripts/train.py --config configs/$D.yaml --seed <s> --set memory.arm=frozen_gru     paths.output_dir=results/$D/ablations/frozen_gru
python scripts/train.py --config configs/$D.yaml --seed <s> --set loss.multi_weighting=none paths.output_dir=results/$D/ablations/w_o_ens
python scripts/train.py --config configs/$D.yaml --seed <s> --set loss.focal_gamma=0.0      paths.output_dir=results/$D/ablations/w_o_focal
python scripts/train.py --config configs/nf_cicids2018_v3_later_day.yaml --seed <s>
python scripts/measure_latency.py --config configs/$D.yaml --out results/$D/latency_idle.json \
    --runs 13=results/$D/tgn_jointids/seed_13 42=results/$D/tgn_jointids/seed_42 ...
```

**3. External baselines**:

```bash
python scripts/baseline_features.py --config configs/$D.yaml          # -> data/processed/baselines/$D_winfeat.npz
python scripts/train_baseline_tree.py --feats data/processed/baselines/${D}_winfeat.npz \
    --model xgboost --seed <s> --out results/baselines/$D/xgboost   # also --model random_forest
python scripts/train_baseline_neural.py --config configs/$D.yaml --set paths.output_dir=results/baselines/$D \
    --model egraphsage --seeds 13 42 123 777 2026 --out results/baselines/$D/egraphsage
python scripts/train_baseline_neural.py --config configs/$D.yaml --set paths.output_dir=results/baselines/$D \
    --model window_gru --feats data/processed/baselines/${D}_winfeat.npz --seeds 13 42 123 777 2026 \
    --out results/baselines/$D/window_gru
python scripts/latency_baselines.py --dataset NF-ToN-IoT-v3 --model xgboost   # one process per model
```

**4. Tables, statistics and figures** (in this order):

```bash
python scripts/analyze_ton.py
python scripts/analyze_ton_ablations.py
python scripts/analyze_cicids.py --ablations
python scripts/analyze_cicids_later_day.py --config configs/nf_cicids2018_v3_later_day.yaml
python scripts/analyze_ablation_statistics.py
python scripts/plot_ablation_statistics.py
python scripts/plot_design_justification.py
python scripts/analyze_focal.py
python scripts/analyze_rare_attacks.py
python scripts/analyze_baselines.py
python scripts/collect_paper_figures.py
```

`scripts/audit_netflow_v3.py` is the dataset audit of the official NF-CICIDS2018-v3 file
(timeline, label coverage, protocol feasibility; `results/nf_cicids2018_v3/dataset_audit/`).

## Tests

```bash
python -m pytest tests -q
```

The tests cover the lane memory and TBPTT (gradient flow, causality, resets, reproducibility), the
fixed update budget, the ablation switches, the losses and ENS weights, train-only scaling, block
assignment, and the NF-CICIDS2018-v3 graph building and leakage checks.

## Citation

```bibtex
@article{tgnjointids2026,
  title   = {Long-Tail-Aware Streaming Intrusion Detection in Internet of Things Networks
             via Temporal Graph Memory and Multi-Task Learning},
  author  = {Driss, Maha and
             Ben Atitallah, Safa and
             Saidam, Wahaj and
             Alnasi, Lina and
             Boulila, Wadii},
  journal = {XX},
  year    = {2026}
}
```

See also [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
