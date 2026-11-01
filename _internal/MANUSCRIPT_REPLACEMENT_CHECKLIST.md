# Manuscript replacement checklist (audit of Tgn_JointIDS_Paper (10).pdf, 2026-09-30)

All numbers in the manuscript come from the legacy pipeline (window-level random split with overlapping
windows, resampling, unscaled features, non-standard focal loss, untrained GRU, online EWC/replay).
They are not comparable with the corrected results. Full reasoning: technical report, Section 9.

## Claims to remove or rewrite
1. Persistent per-IP memory "across graph snapshots" / "survives streaming and batching" -> short-term within-block context (reset every 60 s, gamma = 1).
2. GRU memory "detached, preventing gradients" -> GRU trained by TBPTT (K = 24); frozen GRU performs as well.
3. "Forgetting-free adaptation" (Contribution 4, Sec. 3.6, Algorithm 1, keyword "Continual Learning") -> remove or report as negative.
4. "Chronological evaluation protocol ... non-overlapping temporal segments" (Sec. 4.3) -> block-stratified, not forward-in-time.
5. "Single pass with no future look-ahead" (Sec. 7) -> qualify.
6. Unseen-attack generalisation -> not evaluated.
7. Protocol type "incorporated into edge-aware message passing" (Sec. 3.1) -> false; edge_type unused.
8. Suppression-aware labelling (Contribution 3, Eq. 3, Table 4, Sec. 5.1) -> never changes a label on the corrected data; Eq. 3 argmax must be over attack classes.
9. ENS on "pre-balancing frequencies" improves long-tail performance -> ENS on training-graph counts; lowers macro-F1 (-0.008), raises ransomware/scanning recall, lowers binary FPR.
10. Class-specific replay "guarantees" rare exemplars -> not used; quota ineffective under uniform sampling.
11. Memory-SAGPool "positive feedback loop" -> remove (untested).
12. Scanning "precision 0.9552 ... not spurious alerts" -> precision 0.706; benign->scanning is the main error.
13. MITM 0.18->0.92 / ransomware 0.00->0.65 from memory+replay -> replace with controlled Tables 5-6.
14. "~2,500x faster than real time" -> ~1,200x (2.1 ms per window, one window every 2.5 s).

## Tables and figures
- Abstract/Conclusion numbers -> five-seed results (report Table 2).
- Table 1 (gaps) -> revise Memory/Continual columns.
- Table 2 (class distribution) -> keep; add window counts (report Table 1).
- Table 3 (hyperparameters) -> final protocol (32 lanes, K = 24, 7,000 updates, cosine 1e-3->1e-5, val every 100, reset per 60 s block, gamma = 1; remove ES, batch 32, plateau LR, decay 0.98, EWC, replay, alert tau).
- Sec. 4.3 -> block-stratified protocol, causal frozen evaluation, identity controls.
- Sec. 4.5 -> RTX 4090, PyTorch 2.3.0 + CUDA 12.1, PyG 2.6.1.
- Table 4 / Sec. 5.1 -> replace (suppression_corrected_dataset.json) or remove.
- Table 5 -> report Table 2 binary-head rows (30,908 stream windows, 4,836 benign).
- Table 6 -> report Table 3 (raw 10-class); recompute attack-only cascade if that view is kept.
- Table 7 -> report Table 8.   Table 8 -> report Table 2.
- Table 9 -> five-seed threshold table (report Section 7).   Table 10 -> report Table 7.
- Table 11 / Fig. 7 (identifier persistence) -> remove.
- Tables 11-12 (NF-CICIDS2018-v3) -> rerun under corrected protocol or remove.
- Tables 13-17, Figs. 8-9 (DynGNN-2H/TGN-Proto) -> report Tables 5 (memory) and 6 (ENS).
- Table 18 (SOTA) -> remove legacy numbers; compare only under the corrected protocol.
- Fig. 2 -> fig1_binary_confusion; Fig. 3 -> fig2_multiclass_confusion; Fig. 4 -> fig6_binary_pr_threshold;
  Fig. 5 -> fig8_rolling_stream (non-contiguous-block caveat); Fig. 6 -> fig7_confidence.
- New: fig3_memory_ablation, fig4_ens_tradeoff, fig5_minority_comparison, fig9_convergence.
- Equations: fix Eq. 3; memory Eqs. 5-7 (gamma = 1, reset, TBPTT); ENS (training-graph counts); focal (p* = true-class probability); remove/move Eq. 16 (EWC).
- Add reproducibility statement and limitations (report Section 8).
