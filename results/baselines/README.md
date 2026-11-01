# External baselines: TGN-JointIDS vs XGBoost, Random Forest, E-GraphSAGE, Window-GRU

Five seeds (13, 42, 123, 777, 2026) on both datasets. Every baseline uses exactly the windows, the
60 s block-stratified split, the labels and the train-only log1p z-score scaling of TGN-JointIDS, and
is selected on validation only. Test predictions are aligned window by window with TGN-JointIDS
(checked), so all comparisons are paired.

| Baseline | Input | Script | Selection / training |
| --- | --- | --- | --- |
| Random Forest | per-window vector (163-d: mean/std/min/max of the 35-d host and 5-d flow features, log counts) | `scripts/train_baseline_tree.py` | 500 trees, balanced class weights; grid (depth, leaf size) chosen on val macro-F1 |
| XGBoost | same vector | same | balanced sample weights; grid (depth, min child weight) on val macro-F1, early stopping on val log-loss |
| Window-GRU | same vector, sequence of the windows of each 60 s block (reset per block, causal) | `scripts/train_baseline_neural.py` | 7,000 updates, cosine LR, best val macro-F1 every 100 updates |
| E-GraphSAGE (Lo et al., 2022) | the same window graphs (host features + flow edge features), no memory | same | same budget as Window-GRU / TGN-JointIDS |

Neural baselines use class-weighted cross-entropy (inverse frequency, capped at 30x). For all
baselines the binary decision is P(attack) = 1 - P(benign) >= 0.5 (TGN-JointIDS has a binary head).
Features: `scripts/baseline_features.py` -> `data/processed/baselines/*_winfeat.npz`.

## Results (test, mean ± SD of five seeds; `report/table_main_*.csv`)

| NF-ToN-IoT-v3 | Macro-F1 | Binary F1 | FPR | FNR | Rare F1 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.888 ± 0.001 | 0.978 | 0.158 | 0.015 | 0.657 | 33.3 |
| XGBoost | 0.908 ± 0.001 | 0.980 | 0.134 | 0.016 | 0.742 | 10.1 |
| Window-GRU | 0.905 ± 0.002 | 0.978 | 0.169 | 0.013 | 0.744 | 0.07 |
| E-GraphSAGE | 0.903 ± 0.002 | 0.979 | 0.156 | 0.014 | 0.720 | 0.76 |
| Graph-only (ours, no memory) | 0.901 ± 0.007 | 0.980 | 0.068 | 0.026 | 0.710 | – |
| **TGN-JointIDS** | **0.923 ± 0.004** | **0.983** | 0.071 | 0.022 | **0.806** | 2.08 |

| NF-CICIDS2018-v3 | Macro-F1 | Binary F1 | FPR | FNR | Rare F1 | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.756 ± 0.002 | 0.734 | 0.172 | 0.178 | 0.711 | 33.1 |
| XGBoost | 0.820 ± 0.004 | 0.845 | 0.057 | 0.168 | 0.778 | 53.6 |
| Window-GRU | 0.834 ± 0.002 | 0.795 | 0.172 | 0.065 | 0.787 | 0.07 |
| E-GraphSAGE | 0.840 ± 0.003 | 0.852 | 0.100 | 0.078 | 0.750 | 0.74 |
| Graph-only (ours, no memory) | 0.859 ± 0.007 | 0.928 | 0.011 | 0.111 | 0.771 | – |
| **TGN-JointIDS** | **0.887 ± 0.002** | **0.940** | **0.010** | 0.092 | **0.801** | 2.63 |

Paired statistics (`report/all_metrics_paired.csv`, 10- and 60-min cluster bootstrap):
- TGN-JointIDS has the best macro-F1 on both datasets, 5/5 seeds against every baseline.
  Significant on NF-CICIDS2018-v3 against RF, XGBoost and Window-GRU (E-GraphSAGE +0.047, 5/5
  seeds, 60-min CI includes 0) and on NF-ToN-IoT-v3 against RF and E-GraphSAGE (XGBoost +0.015 and
  Window-GRU +0.017: 5/5 seeds, CI includes 0).
- Binary F1 and false alarms: significantly better than every baseline on both datasets
  (FPR 0.07 vs 0.13-0.17 on ToN; 0.010 vs 0.06-0.17 on CICIDS).
- Rare-class recall: significantly better than every baseline on CICIDS, and than RF, XGBoost and
  Window-GRU on ToN.
- Where baselines are better (`report/where_tgn_is_worse.csv`): binary recall / FNR on
  NF-ToN-IoT-v3 (RF, Window-GRU, E-GraphSAGE; about 0.6-0.8 points) - they flag more windows at
  the cost of 2x more false alarms; ToN Scanning recall (Window-GRU) and CICIDS infiltration
  recall (Window-GRU).

Caveats: on NF-ToN-IoT-v3 the threshold-free binary average precision is the same for all models
(0.998-0.999), so the binary FPR/FNR differences there are differences of operating point, not of
separability; on NF-CICIDS2018-v3 TGN-JointIDS also has the highest AP (0.987 vs 0.85-0.95).
Tree latency is model inference on CPU via the library API for one row (no batching) and
is implementation-dependent; all latencies exclude building the window graph / vector.

## Contents

- `nf_ton_iot_v3/`, `nf_cicids2018_v3/`: `<model>/seed_<s>/` with predictions (val, stream = test),
  `run_info.json` (selected hyper-parameters, timings) and, for the neural baselines, `history.json`
  and `checkpoint.pt`.
- `report/`: main tables (CSV + LaTeX), per-class F1, paired statistics against TGN-JointIDS,
  `fig_baselines_main`, `fig_baselines_per_class` (`scripts/analyze_baselines.py`).
- `latency_idle.json`: idle-machine latency, one window per call (`scripts/latency_baselines.py`,
  one process per dataset and model).
- `logs/`: training logs. The NF-CICIDS2018-v3 Random Forest seed 777 crashed once at start-up
  (segfault) and was rerun with identical settings.
