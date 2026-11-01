# Ablation statistics: TGN-JointIDS vs each ablation, every metric, both datasets

Script: `scripts/analyze_ablation_statistics.py`. It reads only the existing five-seed test predictions.

## Method

- **Comparisons:**
  - TGN-JointIDS vs **Graph-only**: the same model without the temporal memory;
  - TGN-JointIDS vs **w/o ENS**: focal loss without ENS class weights;
  - TGN-JointIDS vs **w/o focal**: ENS-weighted cross-entropy (`loss.focal_gamma=0`).
- **Metrics:**

  | Group | Metrics |
  | --- | --- |
  | Multiclass | Accuracy, macro-F1, weighted F1 |
  | Binary | Accuracy, precision, recall, F1, FPR, FNR, average precision |
  | Rare classes (< 1% of training windows) | F1, recall, precision, detection |
  | Per class | F1 and recall of every attack class |

- **Δ:** TGN-JointIDS minus the ablation, five paired seeds. 95% CIs come from a 1,000-resample
  cluster bootstrap with 10- and 60-min time clusters. Macro averages use only classes present
  in the ground truth.
- **Verdicts:**

  | Verdict | Rule |
  | --- | --- |
  | Significant gain | Both CIs exclude 0 in the favourable direction (for FPR and FNR, lower is better) |
  | Consistent gain | Favourable on 5/5 seeds, but a CI includes 0 |
  | Significant loss | Both CIs exclude 0 in the unfavourable direction |

## Files

| File | Content |
| --- | --- |
| `all_metrics_paired.csv` | Every metric × comparison × dataset |
| `where_tgn_improves.csv` / `.tex` | Significant and consistent gains only |
| `focal/` | Focal-specific analysis (hard windows, calibration, convergence, stability) and its verdict |
| `where_tgn_is_worse.csv` | Significant losses (trade-offs of ENS and of the focal term, see below) |

## Charts (PNG + PDF; `python scripts/plot_ablation_statistics.py`)

| File | Content |
| --- | --- |
| `fig_ablation_summary` | Key metric values (macro-F1, binary F1, FNR, FPR, rare F1, rare recall) of Graph-only, w/o ENS and TGN-JointIDS on both datasets; one zoomed panel per metric. * = TGN-JointIDS significantly better than that ablation, † = significantly worse |
| `fig_ablation_overall` | Forest plot of Δ (TGN-JointIDS − ablation) for every multiclass and binary metric, with 95% CIs (thick: 60-min clusters, thin: 10-min clusters), coloured by verdict; 2 datasets × 2 comparisons |
| `fig_ablation_rare_classes` | The same layout for the rare-class metrics and for every per-class metric that shows a gain or a loss |

| `fig_justify_memory`, `fig_justify_ens`, `fig_justify_focal` | One figure per design choice (paper-ready): ablation vs TGN-JointIDS on the metrics that justify it, mean ± SD of five seeds with seed dots, Δ and 95% CI, * / † significance (`scripts/plot_design_justification.py`) |

FPR and FNR are plotted sign-flipped in the forest plots: positive = fewer false alarms or fewer
missed attacks.

## Significant gains of TGN-JointIDS

### vs Graph-only (the temporal memory)

| Metric | NF-ToN-IoT-v3 | NF-CICIDS2018-v3 |
| --- | --- | --- |
| Multiclass macro-F1 | 0.9225 vs 0.9009 (+0.022) | 0.8866 vs 0.8591 (+0.028) |
| Multiclass accuracy | +0.005 | +0.008 |
| Weighted F1 | +0.004 | +0.008 |
| Binary accuracy | +0.0035 | +0.006 |
| Binary F1 | +0.002 | +0.012 |
| Binary recall | +0.005 | +0.019 |
| Binary FNR | 0.0263 → 0.0216 | 0.1108 → 0.0918 |
| Rare-class F1 | 0.710 → 0.806 (+0.097) | +0.030 (60-min CI only) |
| Rare-class precision | +0.130 | +0.030 (60-min CI only) |
| Rare detection (binary head) | 0.725 → 0.872 (+0.148) | — |
| Per class | Ransomware F1 +0.063, DDoS F1 +0.002, DoS F1 +0.004 | Web-BF F1 +0.22 (recall +0.25), XSS F1 +0.13 (recall +0.12), SQL-Inj F1 +0.05 |

Also consistent on 5/5 seeds, with a CI that includes 0:
- NF-ToN-IoT-v3: ransomware recall, injection recall, XSS, DoS recall;
- NF-CICIDS2018-v3: average precision, infiltration F1, Slowloris F1, rare recall and rare
  precision with 10-min clusters.

### vs w/o ENS (class-weighting)

| Metric | NF-ToN-IoT-v3 | NF-CICIDS2018-v3 |
| --- | --- | --- |
| Binary FPR | 0.0955 → 0.0711 | 0.0136 → 0.0103 |
| Binary precision | +0.004 | +0.008 |
| Rare recall | +0.036 (5/5 seeds; significant with 60-min clusters only) | +0.036 (significant) |
| Per class | Ransomware recall +0.047, scanning recall +0.054, injection recall +0.006 | XSS F1 +0.23 (recall +0.26), Web-BF recall +0.079, infiltration recall +0.008 |

Also significant on NF-CICIDS2018-v3: binary accuracy +0.0015.

## Significant losses (be transparent in the paper)

All losses come from ENS, not from the memory:
- **NF-ToN-IoT-v3:** without ENS the model has higher multiclass macro-F1 (0.9309 vs 0.9225),
  higher rare F1 (0.841 vs 0.806) and higher rare precision. ENS trades precision for rare
  recall.
- **NF-CICIDS2018-v3:** ENS lowers SQL-Inj F1 by −0.11.

## Recommended wording

- The temporal memory significantly improves overall and rare-attack detection on both datasets:
  macro-F1 +0.022 / +0.028; on NF-ToN-IoT-v3, rare-class F1 +0.10 and rare-attack detection
  +0.15; on NF-CICIDS2018-v3, web-attack F1 +0.13 to +0.22.
- ENS class weighting is a recall-oriented trade-off. It lowers the benign false-positive rate on
  both datasets and raises recall of the rarest attacks (ransomware, scanning, XSS), but reduces
  rare-class precision and, on NF-ToN-IoT-v3, macro-F1.
