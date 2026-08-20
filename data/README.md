Runtime data are intentionally not versioned.

Expected local layout:

```text
data/raw/concrete-compressive-strength/
data/processed/concrete-compressive-strength/
```

Acquire the UCI source snapshot with:

```bash
python -m scripts.download_data \
  uci 165 \
  --destination data/raw/concrete-compressive-strength
```

`raw` is the local materialization of the UCI source. `processed` is produced
by Notebook 02 and includes the prepared projection and local split artifacts.
These runtime files remain ignored by Git. The notebooks authenticate their
lineage and content fingerprints through persisted manifests and handoffs.
