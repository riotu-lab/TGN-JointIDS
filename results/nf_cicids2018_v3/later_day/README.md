# NF-CICIDS2018-v3: later-day test

- **Config:** `configs/nf_cicids2018_v3_later_day.yaml`; bundle
  `data/processed/nf_cicids2018_v3_later_day.pt` (same flows as the block-stratified bundle).
- **Split:** train and validation = flows before 2018-03-01 04:00 UTC (14–28 February); the cut
  lies in an overnight gap with no traffic, and no flow crosses it. Validation = held-out whole
  60 s blocks of that period, used only for checkpoint selection. Test = 1–2 March.
- **Evaluation** (`scripts/analyze_cicids_later_day.py`): benign windows of 1–2 March and the
  1 March infiltration windows (infiltration is a training class, seen on an earlier day).
  Test windows labelled bot (2 March), not a training class, are excluded.
- **Training:** TGN-JointIDS with the same protocol as the main runs; five seeds.

| Folder / file | Content |
| --- | --- |
| `tgn_jointids/seed_<s>/` | runs |
| `report/table_later_day_test.csv` | binary and attack-type metrics, mean ± SD and per seed, with an IP-only baseline (score = highest training attack share of the window's hosts) |
| `report/table_compute.csv`, `report/statistics.json` | compute and full statistics |
| `report/fig_confusion_binary_test`, `fig_pr_curve_test`, `fig_validation_curve` | figures |
| `dataset/` | split statistics |
| `latency_idle.json` | per-window latency |

Result (17,061 test windows: 8,542 benign, 8,519 infiltration): binary precision 0.939, recall
0.529, benign FPR 0.034, average precision 0.879 and AUROC 0.854 (IP-only baseline: 0.619 and
0.671); infiltration precision 0.973 and recall 0.499 on the attack-type head. The model ranks
later-day traffic better than the IP rule with few false alarms, but detects about half of the
infiltration windows.
