Model runtime artifacts are generated locally and ignored by Git.

Expected Concrete finalization outputs after running Notebook 04:

```text
artifacts/models/concrete-compressive-strength/final-pipeline.joblib
artifacts/models/concrete-compressive-strength/final-model-manifest.json
artifacts/models/concrete-compressive-strength/final-test-evidence.json
artifacts/models/concrete-compressive-strength/inference-bundle.json
artifacts/models/concrete-compressive-strength/final-model-handoff.json
```

These files are local runtime outputs, not source code, and remain ignored by
Git. Notebook 04 materializes them; Notebook 05 consumes them read-only. The
serialized joblib model must be loaded only after its SHA-256 has been validated
according to the bundle and handoff contract. Recreate the outputs by running
the notebooks in order after acquiring the UCI data.
