# Data

The datasets are **not** redistributed here. Download the official NetFlow-v3 releases from the
University of Queensland NIDS dataset repository and derive the flow files used in the paper.

| Dataset | Official flows | Classes | Flows used |
| --- | --- | --- | --- |
| NF-ToN-IoT-v3 (Luay et al., 2025, arXiv:2503.04404) | 27,520,260 | benign + 9 attacks | 5,674,750 |
| NF-CICIDS2018-v3 (https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22) | 20,115,529 | benign + 14 attacks | 4,571,570 |

**Reduction rule (both datasets).** Every class with at least 200,000 flows keeps a uniform random
20% of its flows; smaller classes are kept whole. Rows are copied unchanged (no oversampling), so
the derived file is a time-preserving subset of the official one:

```bash
python scripts/prepare_flows.py --csv <official.csv> --out data/raw/<derived.csv> \
    --frac 0.2 --min-flows 200000 --seed 42
```

The NF-CICIDS2018-v3 file of the paper was produced by this command (its manifest records the input
and output SHA-256). The NF-ToN-IoT-v3 file of the paper predates the script; the script implements
the same rule, reconstructed from that file
(`results/nf_cicids2018_v3/downsampling_rule.json`), so a regenerated file follows the same rule
but is not row-identical.

Expected layout (paths are set in `configs/*.yaml`, `paths.flows_csv` and `paths.graphs`):

```
data/raw/NF-ToN-IoT-v3_common20_rare100.csv
data/raw/NF-CICIDS2018-v3_common20_rare100.csv            (+ .manifest.json)
data/processed/nf_ton_iot_v3.pt                           scripts/build_graphs.py
data/processed/nf_cicids2018_v3.pt                        scripts/build_graphs_cicids.py
data/processed/nf_cicids2018_v3_later_day.pt              scripts/build_graphs_cicids.py
data/processed/baselines/<dataset>_winfeat.npz            scripts/baseline_features.py
```

Required columns: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`,
`FLOW_START_MILLISECONDS`, `Attack`, plus the node and edge features listed in the configs.

Each graph bundle stores the window graphs, the fixed train / validation / test ("stream") split
and the block assignment; its `.manifest.json` records the split hash, per-class window counts per
partition and the leakage checks (no flow and no 60 s block in two partitions).
