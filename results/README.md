# Results

Every number in the paper comes from the folders below. All runs: seeds 13, 42, 123, 777, 2026;
one RTX 4090; 7,000 updates; checkpoint selected on validation macro-F1 only. In file names,
`stream` means the **test** partition and `val` the validation partition.

| Model name (paper) | Folder name | What differs from TGN-JointIDS |
| --- | --- | --- |
| **TGN-JointIDS** | `tgn_jointids/` | – |
| Graph-only | `ablations/graph_only/` | no temporal memory (`memory.arm=graph_only`) |
| Frozen GRU | `ablations/frozen_gru/` | memory kept, GRU updater fixed at initialisation (`memory.arm=frozen_gru`) |
| w/o ENS | `ablations/w_o_ens/` | focal loss without ENS class weights (`loss.multi_weighting=none`) |
| w/o focal | `ablations/w_o_focal/` | ENS-weighted cross-entropy (`loss.focal_gamma=0.0`) |

In the analysis code the same models are keyed `A` (TGN-JointIDS), `B` (w/o ENS), `C` (w/o focal),
`graph_only` and `frozen_gru`.

## Folders

| Folder | Content | Script(s) |
| --- | --- | --- |
| `nf_ton_iot_v3/` | NF-ToN-IoT-v3: TGN-JointIDS runs, `report/` (main tables and figures), `ablations/` (four ablations and `ablations/report/`), `latency_idle.json`, `environment.json`, `paper_figures/` | `analyze_ton.py`, `analyze_ton_ablations.py` |
| `nf_cicids2018_v3/` | NF-CICIDS2018-v3: TGN-JointIDS runs, `report/` (main tables, ablation table, figures), `ablations/`, `dataset/` (split statistics), `dataset_audit/` (audit of the official file), `later_day/` (forward-in-time test), `paper_figures/` | `analyze_cicids.py --ablations`, `analyze_cicids_later_day.py` |
| `ablation_statistics/` | Paired comparison of TGN-JointIDS with each ablation on every metric, both datasets (Δ, 95% CIs, verdicts); `focal/`: what the focal term changes | `analyze_ablation_statistics.py`, `plot_ablation_statistics.py`, `plot_design_justification.py`, `analyze_focal.py` |
| `rare_attack_analysis/` | Rare classes (< 1% of training windows): TGN-JointIDS vs Graph-only and w/o ENS | `analyze_rare_attacks.py` |
| `baselines/` | Random Forest, XGBoost, Window-GRU and E-GraphSAGE on the same windows and splits; paired statistics against TGN-JointIDS | `baseline_features.py`, `train_baseline_*.py`, `latency_baselines.py`, `analyze_baselines.py` |

Each run folder `seed_<s>/` holds `run_info.json` (configuration, split hash, timing),
`history.json` (validation curve) and, when distributed, `checkpoint.pt` and the frozen causal
predictions `predictions_val.npz` / `predictions_stream.npz` from which every table is computed.
`run_info.json` files record the data paths and output folders in use when the runs were made
(before the repository was reorganised); the split hash identifies the data.

## Statistics

Paired cluster bootstrap over 10- and 60-minute time clusters (the same resampled clusters for
both models and all five seeds). "Significant" = both 95% CIs exclude 0; "consistent" = the same
sign on 5/5 seeds but a CI includes 0. Macro averages use only the classes present in the ground
truth.

## Protocol notes

- Both datasets use a 60 s block-stratified split (whole blocks to train / validation / test,
  60/15/25, every class in every partition). Test blocks share attack sessions with training
  blocks; on NF-CICIDS2018-v3, FTP, SSH and the four DoS classes each come from a single attacker
  IP, and a source-IP identity baseline with no traffic features reaches macro-F1 0.44
  (`nf_cicids2018_v3/report/table1_overall.csv`).
- `nf_cicids2018_v3/later_day/` trains on 14–28 February and tests on 1–2 March (benign and
  infiltration windows; infiltration is a training class).
