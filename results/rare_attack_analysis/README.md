# Rare-attack analysis of the TGN-JointIDS ablations (both datasets)

Models compared: **TGN-JointIDS**, **Graph-only** (the same model without the temporal memory)
and **w/o ENS** (focal loss without ENS class weights). The frozen-GRU variant is left out of
this analysis; it is reported in the ablation reports of each dataset.

Script: `scripts/analyze_rare_attacks.py`. It reads only the existing five-seed predictions;
no training was added.

## Setup

Macro averages and their bootstrap use only the classes present in the ground truth (the same
class set for both models); fixed 2026-10-22.

- **Rare classes:** fixed BEFORE looking at results as attack classes with less than 1% of the
  training windows.

  | Dataset | Rare classes (share of training windows) |
  | --- | --- |
  | NF-ToN-IoT-v3 | MITM (0.75%), ransomware (0.63%) |
  | NF-CICIDS2018-v3 (block-stratified split) | GoldenEye (0.19%), Hulk (0.23%), XSS (0.28%), LOIC-UDP (0.36%), SQL injection (0.65%), Slowloris (0.78%) |

- **Metrics** (test partition, five seeds, raw attack-type head):
  - rare F1, recall and precision, each averaged over the rare classes;
  - rare detection: the share of rare-class windows that the binary head flags as attacks.
- **Statistics:** paired difference TGN-JointIDS minus each ablation, with 95% CIs from a
  cluster bootstrap over 10- and 60-min time clusters (2,000 resamples).

## Outputs

| File | Content |
| --- | --- |
| `table_rare_summary` | Rare-class F1, recall, precision and detection per model and dataset |
| `table_rare_per_class` | F1 and recall per rare class and model |
| `table_rare_paired` | Paired Δ, CI, whether the CI excludes 0, and seeds > 0 (10- and 60-min clusters) |
| `fig_rare_attacks_ablation` | Row a: NF-ToN-IoT-v3, row b: NF-CICIDS2018-v3. (1) rare-class means, (2) per rare class, (3) paired differences. Copied to both paper_figures folders |

Tables are in CSV and LaTeX; the figure in PNG and PDF. `statistics.json` holds all the numbers.

## Results (mean over five seeds)

| | NF-ToN-IoT-v3 | NF-CICIDS2018-v3 |
| --- | --- | --- |
| Rare F1: TGN-JointIDS / Graph-only | 0.806 / 0.710 | 0.801 / 0.771 |
| Rare recall: TGN-JointIDS / Graph-only | 0.890 / 0.875 | 0.868 / 0.842 |
| Rare detection: TGN-JointIDS / Graph-only | 0.872 / 0.725 | 0.987 / 0.986 |

**Memory (TGN-JointIDS vs Graph-only)**

- **NF-ToN-IoT-v3:**
  - rare F1 +0.097, CI [0.018, 0.170], 5/5 seeds;
  - rare precision +0.130, CI excludes 0;
  - rare detection by the binary head +0.148, CI [0.044, 0.293];
  - MITM F1 rises from 0.62 to 0.75.
- **NF-CICIDS2018-v3** (60-min clusters, 5/5 seeds for each):

  | Metric | Δ | 95% CI |
  | --- | --- | --- |
  | Rare F1 | +0.030 | [0.013, 0.064] |
  | Rare recall | +0.026 | [0.008, 0.067] |
  | Rare precision | +0.030 | [0.009, 0.064] |

  With 10-min clusters these CIs include 0, because the six rare classes occupy few clusters.
  XSS F1 rises from 0.37 to 0.49.

**ENS class weights**

- Recall-oriented, on both datasets: rare recall +0.036 on 5/5 seeds.
  - ToN: CI [0.002, 0.077] (60-min clusters).
  - CICIDS: CI [0.024, 0.074] (60-min) and [0.020, 0.057] (10-min).
- Precision drops: −0.092 on ToN (CI excludes 0) and −0.029 on CICIDS.
- Rare F1 is therefore lower with ENS on ToN (−0.035, CI excludes 0) and not significantly
  different on CICIDS (+0.014).

## What the paper can claim

- The temporal memory is the component that improves rare-attack detection.
  - On NF-ToN-IoT-v3, rare-class F1 rises by +0.10 and the binary detection of rare attacks by
    +0.15 (both CIs exclude 0).
  - On NF-CICIDS2018-v3, rare-attack F1, recall and precision each rise by about +0.03 (60-min CIs
    exclude 0, 5/5 seeds).
- ENS weighting makes the detector recall-oriented on rare attacks: about 3.6 percentage points
  fewer missed rare-attack windows on both datasets, at the cost of precision. It should be
  presented as a deliberate trade-off (missed attacks are costlier than false alarms), not as an
  F1 gain. On NF-ToN-IoT-v3 it lowers rare F1.
- NF-CICIDS2018-v3 caveat: the split is block-stratified (shared sessions), and several rare
  CICIDS classes (GoldenEye, Hulk, Slowloris) come from one attacker IP each.
