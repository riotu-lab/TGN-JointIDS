# NF-CICIDS2018-v3: figures for the paper

Each figure is provided as PDF (vector, for LaTeX) and PNG (300 dpi). Model names follow the paper:

| Name in figures | Model |
| --- | --- |
| **TGN-JointIDS** | The selected model |
| Graph-only | The same model without the temporal memory |
| Frozen GRU | Memory with the GRU updater fixed at initialisation |
| w/o ENS | Focal loss without ENS class weights |
| w/o focal | ENS-weighted cross-entropy (focal γ = 0) |

- **Evaluation:** 60 s block-stratified split, every class in every partition; five seeds
  (13, 42, 123, 777, 2026); 7,000 updates; checkpoint = best validation macro-F1; binary
  threshold 0.5 fixed a priori.
- **Disclosure for every caption:** test blocks share attack sessions with training blocks,
  so the split does not measure forward-in-time generalisation.
- **Source:** copies of `../report/` (`scripts/analyze_cicids.py --ablations`) and of the
  cross-dataset folders, made by `scripts/collect_paper_figures.py`.

## Main figures (recommended for the paper; they justify the design choices)

| # | File | Suggested caption | Justifies |
| --- | --- | --- | --- |
| 1 | `fig_cicids_justify_memory` | Effect of the temporal memory on both datasets: TGN-JointIDS vs the same model without memory (Graph-only). Mean ± SD of five seeds (dots = seeds); Δ and 95% CI from a 60-min cluster bootstrap; * significant. The memory raises macro-F1 (+0.022 / +0.028), rare-class F1 (+0.097 on NF-ToN-IoT-v3) and accuracy, and lowers missed attacks (FNR −0.005 / −0.019) | Temporal (GRU) memory |
| 2 | `fig_cicids_justify_ens` | Effect of ENS class weighting: TGN-JointIDS vs focal loss without ENS weights. ENS lowers false alarms (FPR −0.024 / −0.003) and raises binary precision and rare-class recall (+0.036 on both datasets), at the cost of macro-F1 on NF-ToN-IoT-v3 (−0.008, †) | ENS weights (a recall-oriented trade-off) |
| 3 | `fig_cicids_justify_focal` | Effect of the focal term: TGN-JointIDS vs ENS-weighted cross-entropy. Dataset-dependent and mostly small: on NF-CICIDS2018-v3 fewer false alarms (FPR −0.0015, *), macro-F1 +0.006 (5/5 seeds, n.s.) and rare recall +0.016 (5/5 seeds, n.s.); on NF-ToN-IoT-v3 no gain and more false alarms (FPR +0.0048, †). Details: `../ablation_statistics/focal/` | Focal loss (dataset-dependent; report honestly) |
| 4 | `fig_cicids_rare_attacks_ablation` | Rare attacks (< 1% of training windows) on both datasets: rare-class F1, recall, precision and detection per model, F1 per rare class, and paired differences | Memory and ENS on rare attacks |
| 5 | `fig_cicids_ablation_summary` | Key metrics of TGN-JointIDS vs each ablation on both datasets; * / † = significantly better / worse | Summary of all ablations |

Figures 1–5 show both datasets, so they are identical in the two paper_figures folders; include
each once. The dataset-specific figures follow below. Scripts:
`scripts/plot_design_justification.py`, `scripts/analyze_rare_attacks.py`,
`scripts/plot_ablation_statistics.py`. Full statistics: `results/ablation_statistics/`.

## All figures

| File | Suggested caption |
| --- | --- |
| `fig_cicids_dataset_graphs` | NF-CICIDS2018-v3: flows per class in the official release and in the file used (heavy classes reduced to 20% of flows, others complete, the same rule as NF-ToN-IoT-v3); window graphs per class and partition; graph size (edges per 5 s window) |
| `fig_cicids_confusion_15class` | 15-class confusion matrix of TGN-JointIDS on the test partition (row-normalised, mean over five seeds). The remaining errors concentrate in the interleaved web-attack sessions (Web-BF / XSS / SQL injection) and in LOIC-HTTP → LOIC-UDP |
| `fig_cicids_confusion_binary` | Binary confusion matrix of TGN-JointIDS (mean counts over five seeds; FPR 0.0103 ± 0.0015, FNR 0.0918 ± 0.0060) |
| `fig_cicids_per_class_f1_recall` | Per-class F1 and recall of TGN-JointIDS (mean ± SD, dots = seeds) with a source-IP identity baseline that uses no traffic features (macro-F1 0.44). Single-attacker classes (SSH, DoS) are partly identifiable from the host alone |
| `fig_cicids_pr_threshold` | Binary-head precision–recall curves per seed (AP 0.987 ± 0.001) and threshold sensitivity (descriptive only; 0.5 used) |
| `fig_cicids_convergence` | Validation macro-F1 and loss over the 7,000 updates for five seeds; ★ = selected checkpoint (update 6,200 ± 580) |
| `fig_cicids_rare_attacks_ablation` | Same figure as `fig_ton_rare_attacks_ablation` (both datasets). On NF-CICIDS2018-v3, memory raises rare-attack recall and precision (+0.03 each, CIs exclude 0), and ENS raises rare recall (+0.036, CI excludes 0) |
| `fig_cicids_ablation_summary` | TGN-JointIDS vs Graph-only, w/o ENS and w/o focal on the key metrics of both datasets (macro-F1, binary F1, FNR, FPR, rare-class F1 and recall); * / † = TGN-JointIDS significantly better / worse (paired cluster bootstrap). Source: `../../ablation_statistics/` (forest plots with all metrics there) |
| `fig_cicids_ablation_metrics` | Ablations (five seeds; Graph-only, Frozen GRU, w/o ENS): (a) test macro-F1; (b) F1 on the difficult classes; (c) paired differences with 95% CIs (60-min cluster bootstrap) |

Key numbers (test, five seeds):

| Model | Macro-F1 |
| --- | --- |
| TGN-JointIDS | 0.8866 ± 0.0022 (accuracy 0.9572, weighted F1 0.9575, binary F1 0.9396) |
| Graph-only | 0.8591 ± 0.0072 |
| Frozen GRU | 0.8822 ± 0.0052 |
| w/o ENS | 0.8842 ± 0.0070 |
| w/o focal | 0.8806 |

Paired differences in macro-F1:

| Comparison | Δ | 95% CI, 60-min clusters | 95% CI, 10-min clusters | Seeds > 0 |
| --- | --- | --- | --- | --- |
| TGN-JointIDS − Graph-only | +0.028 | [0.017, 0.036] | [0.000, 0.038] | 5/5 |
| Frozen GRU − Graph-only | +0.023 | [0.014, 0.032] | [−0.004, 0.032] | 5/5 |
| TGN-JointIDS − Frozen GRU | +0.004 | includes 0 | includes 0 | 4/5 |
| TGN-JointIDS − w/o ENS | +0.002 | includes 0 | includes 0 | 3/5 |

ENS weights:
- lower the binary FPR: Δ −0.0033, CI [−0.0050, −0.0019];
- raise XSS F1: Δ +0.23, CI [0.08, 0.30].

Latency: mean 2.63 ± 0.06 ms per 5 s window (median 2.56, P95 3.03, P99 3.51 ms; idle RTX 4090).

## Comparison with external baselines

- `fig_cicids_baselines`: macro-F1, binary F1, FPR, FNR and rare-class F1 of Random Forest, XGBoost,
  Window-GRU, E-GraphSAGE, Graph-only and TGN-JointIDS on both datasets (five seeds).
  Suggested caption: "TGN-JointIDS vs external baselines on the same windows and splits (test, mean ±
  SD of five seeds; dots = seeds). All baselines are selected on validation only."
- `fig_cicids_baselines_per_class`: per-class F1 heatmap of the same models.
Source and statistics: `results/baselines/`.
