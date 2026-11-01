# NF-ToN-IoT-v3

- **Config:** `configs/nf_ton_iot_v3.yaml`; graph bundle `data/processed/nf_ton_iot_v3.pt`.
- **Data:** 5,674,750 flows (classes with ≥ 200k flows keep 20%, the others are kept whole), 10
  classes; 76,361 / 23,793 / 30,908 window graphs in train / validation / test (60 s
  block-stratified split, 60/15/25 of flows, every class in every partition).
- **Training:** 7,000 updates, cosine LR 1e-3 → 1e-5, validation every 100 updates, checkpoint =
  best validation macro-F1; binary threshold fixed at 0.5. Five seeds.

| Folder / file | Content |
| --- | --- |
| `tgn_jointids/seed_<s>/` | TGN-JointIDS runs |
| `ablations/{graph_only, frozen_gru, w_o_ens, w_o_focal}/seed_<s>/` | ablation runs (see `../README.md`) |
| `report/` | `scripts/analyze_ton.py`: overall (`table1_overall`), per class (`table2_per_class_stream`), critical classes (`table3_critical`), compute (`table4_compute`), time-cluster CIs (`statistics.json`), figures |
| `ablations/report/` | `scripts/analyze_ton_ablations.py`: dataset (`table1_dataset`), TGN-JointIDS overall and per class (`table2`–`table4`), memory and ENS ablations (`table5`, `table6`), efficiency (`table7`), one-vs-rest rates (`table8`), `tables.md`, figures 1–9 |
| `latency_idle.json`, `ablations/latency_idle.json` | per-window causal latency on an idle GPU (`scripts/measure_latency.py`) |
| `environment.json` | software and hardware of the runs |
| `paper_figures/` | figures selected for the paper, with captions (`README.md`) |
| `logs/` | training logs |

Headline (test, mean ± SD of five seeds): macro-F1 0.923 ± 0.004, accuracy 0.966, binary F1
0.983, benign FPR 0.071, 2.1 ms per window.
