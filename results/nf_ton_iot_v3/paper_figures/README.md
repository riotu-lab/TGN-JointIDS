# NF-ToN-IoT-v3: figures for the paper

Each figure is provided as PDF (vector, for LaTeX) and PNG (300 dpi). Model names follow the paper:

| Name in figures | Model |
| --- | --- |
| **TGN-JointIDS** | The selected model |
| Graph-only | The same model without the temporal memory |
| Frozen GRU | Memory with the GRU updater fixed at initialisation |
| w/o ENS | Focal loss without ENS class weights |
| w/o focal | ENS-weighted cross-entropy (focal γ = 0) |

- **Data:** five seeds (13, 42, 123, 777, 2026); 60 s block-stratified split; causal test
  evaluation with memory reset at every block.
- **Training:** 7,000 updates, checkpoint = best validation macro-F1.
- **Regenerate:** `python scripts/analyze_ton.py`, `python scripts/analyze_ton_ablations.py`,
  then `python scripts/collect_paper_figures.py` (builds the dataset figure and copies the others).

## Main figures (recommended for the paper; they justify the design choices)

| # | File | Suggested caption | Justifies |
| --- | --- | --- | --- |
| 1 | `fig_ton_justify_memory` | Effect of the temporal memory on both datasets: TGN-JointIDS vs the same model without memory (Graph-only). Mean ± SD of five seeds (dots = seeds); Δ and 95% CI from a 60-min cluster bootstrap; * significant. The memory raises macro-F1 (+0.022 / +0.028), rare-class F1 (+0.097 on NF-ToN-IoT-v3) and accuracy, and lowers missed attacks (FNR −0.005 / −0.019) | Temporal (GRU) memory |
| 2 | `fig_ton_justify_ens` | Effect of ENS class weighting: TGN-JointIDS vs focal loss without ENS weights. ENS lowers false alarms (FPR −0.024 / −0.003) and raises binary precision and rare-class recall (+0.036 on both datasets), at the cost of macro-F1 on NF-ToN-IoT-v3 (−0.008, †) | ENS weights (a recall-oriented trade-off) |
| 3 | `fig_ton_justify_focal` | Effect of the focal term: TGN-JointIDS vs ENS-weighted cross-entropy. Dataset-dependent and mostly small: on NF-CICIDS2018-v3 fewer false alarms (FPR −0.0015, *), macro-F1 +0.006 (5/5 seeds, n.s.) and rare recall +0.016 (5/5 seeds, n.s.); on NF-ToN-IoT-v3 no gain and more false alarms (FPR +0.0048, †). Details: `../ablation_statistics/focal/` | Focal loss (dataset-dependent; report honestly) |
| 4 | `fig_ton_rare_attacks_ablation` | Rare attacks (< 1% of training windows) on both datasets: rare-class F1, recall, precision and detection per model, F1 per rare class, and paired differences | Memory and ENS on rare attacks |
| 5 | `fig_ton_ablation_summary` | Key metrics of TGN-JointIDS vs each ablation on both datasets; * / † = significantly better / worse | Summary of all ablations |

Figures 1–5 show both datasets, so they are identical in the two paper_figures folders; include
each once. The dataset-specific figures follow below. Scripts:
`scripts/plot_design_justification.py`, `scripts/analyze_rare_attacks.py`,
`scripts/plot_ablation_statistics.py`. Full statistics: `results/ablation_statistics/`.

## All figures

| File | Suggested caption | Source |
| --- | --- | --- |
| `fig_ton_dataset_graphs` | NF-ToN-IoT-v3: flows per class (official release vs. the 20%/100% reduced file used), window graphs per class and partition, and graph size (edges per 5 s window) | `scripts/collect_paper_figures.py` |
| `fig_ton_confusion_10class` | 10-class confusion matrix of TGN-JointIDS on the test partition (row-normalised, mean over five seeds) | `../ablations/report/fig2` |
| `fig_ton_confusion_binary` | Binary confusion matrix of TGN-JointIDS (row-normalised; mean count per seed) | `../ablations/report/fig1` |
| `fig_ton_seed_results_minority` | Macro-F1 per seed (validation and test) and per-seed recall / precision of the minority classes (MITM, ransomware, scanning, injection) | `../report/fig1` |
| `fig_ton_memory_ablation` | Memory ablation: test macro-F1 per seed (paired lines) for Graph-only, Frozen GRU and TGN-JointIDS, and critical-class recall | `../ablations/report/fig3` |
| `fig_ton_minority_ablation` | Minority-class recall and F1 for the four models (mean ± SD, five seeds) | `../ablations/report/fig5` |
| `fig_ton_ens_tradeoff` | Effect of ENS weighting on the precision–recall trade-off of the critical classes (arrows: w/o ENS → TGN-JointIDS, same seed) | `../ablations/report/fig4` |
| `fig_ton_pr_threshold` | Binary-head precision–recall curve and threshold sensitivity (0.5 fixed a priori; other thresholds descriptive only) | `../ablations/report/fig6` |
| `fig_ton_rare_attacks_ablation` | Rare attacks (< 1% of training windows) on NF-ToN-IoT-v3 (row a) and NF-CICIDS2018-v3 (row b): rare-class F1, recall, precision and binary detection per model, F1 per rare class, and paired differences with 95% CIs. Memory raises rare F1 by +0.10 and rare detection by +0.15 on NF-ToN-IoT-v3; ENS raises rare recall by +0.036 on both datasets | `../../rare_attack_analysis/` |
| `fig_ton_ablation_summary` | TGN-JointIDS vs Graph-only, w/o ENS and w/o focal on the key metrics of both datasets (macro-F1, binary F1, FNR, FPR, rare-class F1 and recall); * / † = TGN-JointIDS significantly better / worse (paired cluster bootstrap). Source: `../../ablation_statistics/` (forest plots with all metrics there) |
| `fig_ton_convergence` | Validation convergence of the four models under the matched 7,000-update protocol | `../ablations/report/fig9` |

Key numbers for the captions and text, test partition, five seeds (report tables of the two
source folders):
- TGN-JointIDS macro-F1 0.923 ± 0.004;
- Graph-only 0.901;
- Frozen GRU 0.921;
- w/o ENS 0.931; w/o focal 0.926;
- TGN-JointIDS − Graph-only = +0.022, 95% CI excludes 0, positive on 5/5 seeds.

## Comparison with external baselines

- `fig_ton_baselines`: macro-F1, binary F1, FPR, FNR and rare-class F1 of Random Forest, XGBoost,
  Window-GRU, E-GraphSAGE, Graph-only and TGN-JointIDS on both datasets (five seeds).
  Suggested caption: "TGN-JointIDS vs external baselines on the same windows and splits (test, mean ±
  SD of five seeds; dots = seeds). All baselines are selected on validation only."
- `fig_ton_baselines_per_class`: per-class F1 heatmap of the same models.
Source and statistics: `results/baselines/`.
