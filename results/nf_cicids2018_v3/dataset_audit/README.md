# NF-CICIDS2018-v3 dataset audit

Audit of the official file before any graph building or training (2026-10-21).
- Official release: https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22
- Local copy: data/raw/nf_cicids2018_v3_official/ (BagIt package; NF-CICIDS2018-v3.csv,
  20,115,529 rows, SHA-1 verified against the manifest). See official_dataset_provenance.json.

Contents
- `official/`: audit of the official file (summary, episodes, timeline, window composition,
  forward-in-time counts, protocol feasibility, label_coverage_checks.json, timeline figure).

Command used for official/ (read-only):

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/audit_netflow_v3.py \
        --csv data/raw/<official NF-CICIDS2018-v3>.csv --config configs/nf_cicids2018_v3.yaml \
        --out results/nf_cicids2018_v3/dataset_audit/official
