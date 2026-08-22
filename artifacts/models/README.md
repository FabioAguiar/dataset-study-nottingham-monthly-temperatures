Model runtime artifacts are generated locally and ignored by Git.

Expected `nottem` finalization outputs after running Notebook 04:

```text
artifacts/models/nottem/final-pipeline.joblib
artifacts/models/nottem/final-model-manifest.json
artifacts/models/nottem/final-test-evidence.json
artifacts/models/nottem/inference-bundle.json
artifacts/models/nottem/final-model-handoff.json
```

These files are local runtime outputs, not source code, and remain ignored by
Git. Notebook 04 materializes them; Notebook 05 consumes them read-only and
creates no new scientific artifacts. `final-pipeline.joblib` holds a frozen,
forecasting-specific artifact (not a scikit-learn `Pipeline`).

Schemas:

- `final-pipeline.joblib` — `forecasting-frozen-model.v1`
- `final-model-manifest.json` — `forecasting-final-model-manifest.v1`
- `final-test-evidence.json` — `forecasting-final-test-evidence.v1`
- `inference-bundle.json` — `forecasting-inference-bundle.v1`
- `final-model-handoff.json` — `forecasting-final-model-handoff.v1`

The serialized joblib model must be loaded only after its SHA-256 has been
validated against the bundle and handoff contract; treat it as a trusted local
artifact only (`joblib.load` is unsafe for untrusted bytes). Recreate the
outputs by running the notebooks in order after acquiring the `nottem` source.
