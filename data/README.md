# Data

Datasets are **not** redistributed here. Download them from the University of
Queensland NetFlow repository and place the CSV files under `data/raw/`.

| Dataset | Source | Flows | Classes |
|---|---|---|---|
| NF-ToN-IoT-v3 | Luay et al., 2025 (arXiv:2503.04404) | 5,674,750 considered | benign + 9 attacks |
| NF-CICIDS2018-v3 | https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22 | 20,115,529 | benign + 14 attacks |

Expected file names (edit `paths.flows_csv` in the config to change them):

```
data/raw/NF-ToN-IoT-v3_exp2_common20_rare100.csv
data/raw/NF-CICIDS2018-v3_resampled_by_rarity.csv
```

Required columns: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`,
`PROTOCOL`, `FLOW_START_MILLISECONDS`, `Attack`, plus the node/edge features listed
in the YAML config.

`scripts/build_graphs.py` then (i) re-samples every class to
`data.balance_target` flows, (ii) builds 5 s / 2.5 s sliding-window graphs and
(iii) stores the graphs together with the fixed train/val/stream split in
`data/processed/`.

> **Note.** Both file names above refer to *derived* flow files (a class-dependent
> sub-sample of the official release). The script that produces them from the
> official CSVs still has to be added to this repository (`scripts/prepare_flows.py`).
