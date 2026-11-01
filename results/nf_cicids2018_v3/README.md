# NF-CICIDS2018-v3

Same architecture, training protocol and hyper-parameters as NF-ToN-IoT-v3; only the class list
(15 classes), the mixed-window priority order (SQL injection > XSS > web brute force, rarest
first) and the data paths differ.

- **Data:** the official release (20,115,529 flows) reduced with the NF-ToN-IoT-v3 rule
  (`scripts/prepare_flows.py`; classes with ≥ 200k flows keep 20%: benign, DDoS-HOIC, FTP-BF,
  DDoS-LOIC-HTTP, bot) → 4,571,570 flows. `downsampling_rule.json` documents how the rule was
  reconstructed from the NF-ToN-IoT-v3 file.
- **Graph build:** `scripts/build_graphs_cicids.py` (strict label and feature checks; leakage checks:
  every flow and every 60 s block in exactly one partition).
- **Block-stratified split** (`configs/nf_cicids2018_v3.yaml`): whole 60 s blocks to train /
  validation / test, 60/15/25 of flows; 81,267 / 20,428 / 33,400 window graphs; all 15 classes in
  every partition. Test blocks share attack sessions (and, for FTP, SSH and the four DoS classes,
  the single attacker IP) with training blocks.

| Folder / file | Content |
| --- | --- |
| `tgn_jointids/seed_<s>/` | TGN-JointIDS runs |
| `ablations/{graph_only, frozen_gru, w_o_ens, w_o_focal}/seed_<s>/` | ablation runs (see `../README.md`); `ablations/latency_idle_*.json` |
| `report/` | `scripts/analyze_cicids.py --ablations`: `REPORT.md` (concise report); tables 1–9 (CSV + LaTeX): overall with the source-IP identity baseline, per class (raw and binary-gated), minority classes, compute, dataset, graph sizes, ablations, paired bootstrap, threshold sensitivity; figures 1–7; `statistics.json` |
| `dataset/` | flows and window graphs per class and partition |
| `dataset_audit/` | audit of the official file: timeline, episodes, label coverage, protocol feasibility (`scripts/audit_netflow_v3.py`) |
| `later_day/` | forward-in-time test, see `later_day/README.md` |
| `latency_idle.json` | per-window causal latency on an idle GPU |
| `paper_figures/` | figures selected for the paper, with captions (`README.md`) |
| `logs/` | training logs |

Headline (test, mean ± SD of five seeds): macro-F1 0.887 ± 0.002, accuracy 0.957, binary F1
0.940, benign FPR 0.010, 2.6 ms per window. Source-IP identity baseline: macro-F1 0.44.
