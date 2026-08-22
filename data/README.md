Runtime data are intentionally not versioned.

Expected local layout:

```text
data/raw/nottem/
    dataset.csv
    metadata.json
    documentation.txt

data/processed/nottem/
    development.csv
    final-holdout.csv
```

Acquire the source series with:

```bash
python -m scripts.download_data \
  rdataset nottem \
  --package datasets \
  --destination data/raw/nottem
```

`raw` is the local materialization of the R dataset `datasets::nottem`,
acquired programmatically through `statsmodels.datasets.get_rdataset` (see
`scripts.download_data.acquire_rdataset`). It contains the original two-column
`time`/`value` transport table plus source metadata and documentation, with no
UCI or other external repository involved.

`processed` is produced by Notebook 02, which reconstructs a monthly
`PeriodIndex` and splits the series into two disjoint temporal scopes:

- `development.csv` — `1920-01` through `1938-12` (228 monthly observations),
  used for exploration, preparation, and model selection; and
- `final-holdout.csv` — `1939-01` through `1939-12` (12 monthly observations),
  sealed until Notebook 04 opens it exactly once for final evaluation.

These runtime files remain ignored by Git. The notebooks authenticate their
lineage and content fingerprints through persisted manifests and handoffs
(`artifacts/exploration/nottem/`, `artifacts/preparation/nottem/`,
`artifacts/model-selection/nottem/`, `artifacts/models/nottem/`), each of
which records the SHA-256 of the data it depends on.
