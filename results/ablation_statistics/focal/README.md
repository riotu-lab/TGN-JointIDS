# What the focal term changes: TGN-JointIDS vs w/o focal (both datasets)

w/o focal = TGN-JointIDS with `loss.focal_gamma=0` (ENS-weighted cross-entropy); every other
setting is unchanged (five seeds, same split, 7,000 updates).

Runs:
- `nf_cicids2018_v3/ablations/w_o_focal/`;
- `nf_ton_iot_v3/ablations/w_o_focal/`.

Scripts:
- `scripts/analyze_focal.py`: this folder;
- `scripts/analyze_ablation_statistics.py`: standard metrics, in `../all_metrics_paired.csv`;
- `scripts/plot_design_justification.py`: `../fig_justify_focal`.

"sig." means both the 10- and 60-min cluster-bootstrap CIs exclude 0.

## Verdict: the effect of the focal term depends on the dataset and is mostly small

### NF-CICIDS2018-v3: modest benefits

| Effect | Δ | Evidence |
| --- | --- | --- |
| Macro-F1 | +0.006 | 5/5 seeds, CI includes 0 |
| Rare-class recall | +0.016 | 5/5 seeds, CI includes 0 |
| XSS recall (the hardest rare class) | 0.28 → 0.47 (+0.185) | sig. |
| XSS F1 | +0.14 | |
| False alarms (FPR) | −0.0015 | sig. |
| Binary precision | +0.0035 | sig. |
| Mean F1 of the confusable web / XSS / SQL-injection group | +0.038 | 5/5 seeds |
| Validation macro-F1 at 1,000 / 2,000 updates | +0.040 / +0.025 | early training is faster |
| Seed SD of test macro-F1 | 0.0022 vs 0.0060 | more stable across seeds |

Costs on NF-CICIDS2018-v3:
- more missed attacks (FNR +0.0065, sig.) and lower binary F1 (−0.002, sig.);
- lower SQL-injection recall (−0.095) and infiltration F1 (−0.005), both sig.;
- lower detection on stealthy windows, where attack flows are under 10% of the window (−0.012, sig.);
- worse calibration (attack-type ECE +0.019, sig.).

### NF-ToN-IoT-v3: no benefit

| Effect | Δ | Evidence |
| --- | --- | --- |
| Macro-F1 | −0.003 | 1/5 seeds better; n.s. with 10-min clusters, a loss with 60-min |
| Rare-class F1 | −0.011 | n.s. |
| Rare precision | −0.025 | sig. loss |
| False alarms (FPR) | +0.0048 | sig. loss |
| Scanning recall | −0.015 | sig. loss |
| Binary recall | +0.001 | sig. gain |
| Detection of stealthy attack windows | +0.013 | sig. gain |
| Attack-type ECE | −0.007 | better calibration, 5/5 seeds |

Convergence on NF-ToN-IoT-v3 is slightly slower with focal loss.

## What the paper can say

- Do **not** claim that the focal term improves detection in general.
- An honest statement: on NF-CICIDS2018-v3, the focal term speeds up early training, makes runs
  more stable across seeds, lowers false alarms and strongly improves the hardest, most confusable
  rare class (XSS); on NF-ToN-IoT-v3 it gives no gain. Most of the long-tail benefit comes from the
  ENS weights and the temporal memory.
- Optional design change: ENS-weighted cross-entropy is a simpler alternative with equal or better
  results on NF-ToN-IoT-v3. Changing the final model would mean re-labelling the selected
  architecture, so it is the authors' decision.

## Files

| File | Content |
| --- | --- |
| `focal_specific_metrics.csv` / `.tex` | Hard windows (stealthy, low-identity, confusable), calibration (ECE, Brier), convergence, stability |
| `fig_focal_convergence.png` / `.pdf` | Validation macro-F1 over training, mean and range over five seeds |

The NF-CICIDS2018-v3 low-identity subset has only 64 windows and is not interpretable.
