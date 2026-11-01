# TGN-JointIDS on NF-CICIDS2018-v3: block-stratified evaluation (technical report)

**In short:** TGN-JointIDS, retrained unchanged on NF-CICIDS2018-v3, reaches test macro-F1
0.8866 ± 0.0022 over five seeds. The memory adds +0.028 macro-F1 over the graph-only model.
The split shares attack sessions between training and test, so these figures do not show
generalisation forward in time.

## Setup

- **Data:** the official NF-CICIDS2018-v3 release (20,115,529 flows, checksum-verified) is reduced
  with the NF-ToN-IoT-v3 rule. Classes with at least 200k flows (benign, DDoS-HOIC, FTP-BF,
  DDoS-LOIC-HTTP, bot) keep a random 20% of their flows; the others are kept whole. The result
  has 4,571,570 flows, with no oversampling.
- **Graphs:** 5 s windows with a 2.5 s stride inside 60 s blocks.
  - Strict label and feature checks.
  - Mixed-attack windows take the rarest attack class (SQL injection > XSS > web).
- **Split:** whole 60 s blocks to train / val / test = 60/15/25 of flows (81,267 / 20,428 / 33,400
  window graphs), with all 15 classes in every partition.
- **Leakage checks:** every flow is in one partition; no identical flow record appears in two
  partitions; windows stay inside their blocks.
- **Model and training:** identical to NF-ToN-IoT-v3 except the 15-way attack-type head.
  - 7,000 updates, cosine LR, TBPTT K = 24, 32 lanes.
  - Checkpoint = best validation macro-F1; binary threshold 0.5.
  - Causal frozen-model evaluation with memory reset per block.
  - The test set was never used for any choice.
- **Seeds:** 13, 42, 123, 777, 2026. All runs (TGN-JointIDS and its ablations × 5 seeds) share one split
  (SHA-256 `321d1458...`) and completed all 7,000 updates. The seed-13 pilot was rerun once after
  a native segfault.

## Results: TGN-JointIDS (test, mean ± SD over five seeds)

| Metric | Value |
| --- | --- |
| Binary accuracy | 0.9659 ± 0.0014 |
| Binary precision / recall | 0.9732 ± 0.0038 / 0.9082 ± 0.0060 |
| Binary F1 | 0.9396 ± 0.0026 |
| Binary FPR / FNR | 0.0103 ± 0.0015 / 0.0918 ± 0.0060 |
| Binary average precision | 0.9870 ± 0.0010 |
| Multiclass accuracy (raw / gated) | 0.9572 ± 0.0007 / 0.9587 ± 0.0013 |
| Macro-F1 (raw / gated) | 0.8866 ± 0.0022 / 0.8906 ± 0.0020 |
| Weighted F1 (raw / gated) | 0.9575 ± 0.0007 / 0.9579 ± 0.0014 |
| Latency per window (mean / P95) | 2.63 / 3.03 ms |
| Training per run | 237 min (four runs in parallel); 1.15 GB GPU |

"Raw" is the attack-type head's prediction; "gated" overrides it with benign whenever the binary
head says benign.

**Per class (F1):**
- **≥ 0.98:** SSH, Slowloris, HOIC, bot, Hulk, FTP, SlowHTTP, GoldenEye.
- **0.88–0.98:** benign 0.976, LOIC-HTTP 0.907, infiltration 0.882.
- **Weak:**

  | Class | F1 | Cause |
  | --- | --- | --- |
  | LOIC-UDP | 0.802 | 52 LOIC-HTTP windows per seed predicted as LOIC-UDP |
  | Web-BF | 0.752 | Confused with XSS and SQL injection |
  | SQL injection | 0.538 | About 177 false positives per seed from XSS and web windows |
  | XSS | 0.490 | Half of its windows predicted as SQL injection |

  The web, XSS and SQL-injection classes are interleaved in time, and their labels are noisy
  (audit).

## Ablations (five seeds; 95% CI from a 60-min cluster bootstrap)

| Model | Macro-F1 | Binary FPR | XSS F1 |
| --- | --- | --- | --- |
| TGN-JointIDS | 0.8866 ± 0.0022 | 0.0103 | 0.490 |
| Frozen GRU | 0.8822 ± 0.0052 | 0.0124 | 0.428 |
| w/o ENS | 0.8842 ± 0.0070 | 0.0136 | 0.260 |
| Graph-only (no memory) | 0.8591 ± 0.0072 | 0.0113 | 0.365 |

- **Memory helps:** TGN-JointIDS − Graph-only = +0.028, CI [0.017, 0.036], 5/5 seeds. Frozen
  GRU − Graph-only = +0.023, 5/5 seeds. With 10-min clusters the CIs are wider: [0.000, 0.038]
  and [−0.004, 0.032]. The gain is concentrated in Web-BF and XSS. This replicates
  NF-ToN-IoT-v3 (+0.022).
- **Training the GRU adds little:** TGN-JointIDS − frozen GRU = +0.004, and the CI includes 0.
  The trainable GRU lowers the binary FPR (Δ −0.0021, CI excludes 0) and is more stable across
  seeds.
- **ENS weights:** macro-F1 Δ +0.002, CI includes 0. They lower the FPR (Δ −0.0033, CI excludes 0)
  and raise XSS F1 (+0.23, CI [0.08, 0.30]), at the cost of SQL injection and LOIC-UDP.

## Limitations and required disclosures

1. **Class-covered, not temporal.** Test blocks come from the same attack sessions as training
   blocks.
2. **Identity shortcut.** FTP, SSH and the four DoS classes each come from a single attacker IP.
   A source-IP identity baseline with no traffic features reaches macro-F1 0.44, and F1 0.66–0.80
   on SSH, GoldenEye, Slowloris, Hulk and SlowHTTP. The near-perfect scores on these classes are
   therefore partly host identification.
3. **Later days.** Trained on 14–28 Feb and tested on 1–2 Mar (benign and infiltration windows,
   five seeds; `../later_day/`), the model keeps few false alarms (FPR 0.034) and ranks windows
   better than an IP-only rule (AUROC 0.85 vs 0.67), but detects about half of the infiltration
   windows (recall 0.53).
4. **Label noise.** Web, XSS and SQL-injection flows use implausible ports; 55% of infiltration
   flows are DNS queries; 23 February carries no attack labels.
5. **Reduced file.** Heavy classes are subsampled per flow, which thins their graphs. The same
   rule was used for NF-ToN-IoT-v3.

The tables (CSV + LaTeX) and figures (PNG + PDF) are in this folder; the paper selection is in
`../paper_figures/`.
