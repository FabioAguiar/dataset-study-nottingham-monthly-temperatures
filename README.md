# Nottingham Monthly Temperatures — Reproducible Univariate Forecasting Study

## Overview

This repository is a standalone, educational, reproducible study of
**univariate time-series forecasting** on the `datasets::nottem` series
(Nottingham Castle monthly average air temperature, 1920–1939). The workflow
runs as five ordered notebooks (01 → 05) that each start from persisted,
content-authenticated handoffs rather than live in-memory variables from the
previous notebook: exploration → preparation → model selection (temporally
ordered backtesting) → final model freeze and one-time holdout evaluation →
a read-only inference demo.

The study is organized as a strictly temporal problem: no random, shuffled, or
K-fold split is used anywhere, evaluation windows never overlap, and the final
12 months are opened for scoring exactly once, after the model is frozen. The
study is a self-contained scientific reference; it is not integrated with any
external platform or deployment surface at this stage.

| Item | Value |
|---|---|
| Problem type | `time_series_forecasting` |
| Forecasting mode | `univariate` |
| Scientific reference | `datasets::nottem` |
| Target | `temperature` |
| Target unit | degrees Fahrenheit |
| Frequency | monthly (`M`) |
| Forecast horizon | 12 months |
| Selected model | `seasonal_trend_ols` (`DeterministicSeasonalTrendOLS`) |
| Operational modeling ready | `false` |

## Visual summary

### Historical evolution

![Monthly temperature evolution and annual-level trend signal](docs/images/nottem_temperature_evolution.png)

### Seasonal structure

![Calendar-month temperature profile across years](docs/images/nottem_calendar_month_profile.png)

### Model selection

![Backtesting MAE by eligible forecasting specification](docs/images/nottem_model_selection_mae.png)

### Final 1939 forecast

![Final 1939 forecast versus observed temperature](docs/images/nottem_final_forecast.png)

## Exploratory diagnostics

![Distribution of monthly average temperature](docs/images/nottem_target_distribution.png)

![STL decomposition of the monthly temperature series](docs/images/nottem_stl_decomposition.png)

![Autocorrelation diagnostics](docs/images/nottem_autocorrelation.png)

The `docs/images/` directory contains stable, versioned documentation assets.
Repository-relative image paths are kept stable so downstream documentation
consumers can resolve them from a published repository revision.

## Dataset

| Item | Value |
|---|---|
| Source | R datasets package (Rdatasets), via `statsmodels.datasets.get_rdataset` |
| Scientific reference | `datasets::nottem` |
| Series | Nottingham Castle monthly air temperature |
| Semantics | Monthly average air temperature at Nottingham Castle |
| Unit | degrees Fahrenheit |
| Frequency | monthly |
| Coverage | `1920-01` → `1939-12` |
| Observations | 240 |
| Source exogenous predictors | none |

The Python acquisition layer materializes the source as a two-column
transport table (`time`, `value`); Notebooks 01/02 reconstruct a monthly
`PeriodIndex` from it and rename the value column to `temperature`. This
transport representation does not redefine the target semantics or unit.

## Acquisition

```bash
python -m scripts.download_data \
  rdataset nottem \
  --package datasets \
  --destination data/raw/nottem
```

This calls `scripts.download_data.acquire_rdataset(...)`, which wraps
`statsmodels.datasets.get_rdataset`. Notebooks 01 and 02 use the same helper
directly rather than re-implementing acquisition. The materialized raw layout
is:

```text
data/raw/nottem/dataset.csv
data/raw/nottem/metadata.json
data/raw/nottem/documentation.txt
```

No UCI Machine Learning Repository dataset is involved in this study; the
`ucimlrepo` dependency and the `uci` acquisition mode remain in this
repository only as reusable, generic infrastructure shared with other
dataset-study repositories.

## Problem contract

| Field | Value |
|---|---|
| `problem_type` | `time_series_forecasting` |
| `forecasting_mode` | `univariate` |
| `target` | `temperature` |
| target unit | degrees Fahrenheit |
| `frequency` | `M` |
| forecast horizon | 12 months |

## Temporal boundary

| Scope | Range | Observations |
|---|---|---:|
| Development | `1920-01` → `1938-12` | 228 |
| Final holdout | `1939-01` → `1939-12` | 12 |

Random, shuffled, or `KFold`-style splitting is not meaningful for this study:
temporal order carries information (trend and seasonality), and shuffling
would leak future context into training folds. All evaluation in this study
therefore uses temporally ordered, non-overlapping windows.

**Caveat on exploration visibility.** Notebook 01 explores the complete
1920–1939 series retrospectively, before the final holdout is sealed for
model selection and final evaluation. From Notebook 02 onward, the 1939
holdout is a **prospectively sealed model-selection/final-evaluation
holdout** — it is never used to choose or tune anything before its single
final-evaluation opening — but it is **not** an external, exploration-blind
test set in the strongest sense, because Notebook 01's descriptive/statistical
exploration already had visibility into it. This is a known, documented
limitation of the study design, not an evaluation-contract violation.

## Backtesting contract

| Field | Value |
|---|---|
| Mode | expanding window |
| Initial training | 120 months |
| Forecast horizon | 12 months |
| Origin step | 12 months |
| Folds | 9 |
| Validation years | 1930 → 1938 |
| Forecasts per complete specification | 108 |

Each fold trains on an expanding window starting at `1920-01`, forecasts the
next 12 months, and steps the origin forward by 12 months for the next fold.
Validation windows are calendar years and never overlap.

## Metrics

- **Primary:** MAE
- **Secondary:** RMSE, seasonal MASE(12)
- **Diagnostic:** horizon-wise absolute error / MAE

`MAPE` and `sMAPE` are deliberately **not** used: Fahrenheit has an arbitrary
zero point, so a percentage-of-value metric is not scale-meaningful for this
target (a temperature near 0 °F would produce huge or undefined percentage
errors that say nothing about forecast quality).

## Baselines

| Baseline | Definition |
|---|---|
| `seasonal_naive_12` | `y_hat[T+h] = y[T+h-12]` |
| `naive_last_value` | `y_hat[T+h] = y[T]` |

`seasonal_naive_12` is the primary baseline for comparison; `naive_last_value`
is secondary.

## Selected model

The frozen selection contract chose **`seasonal_trend_ols`**
(family `DeterministicSeasonalTrendOLS`) from a catalog that also included
smoothing (Holt-Winters), SARIMA, and AutoReg candidates, plus the two
baselines. Its design:

- intercept;
- linear time trend;
- 11 calendar-month dummies (February–December), January as reference; and
- constructed with `statsmodels.regression.linear_model.OLS`, fit from
  scratch on each training fold (`fit_from_scratch_on_each_training_fold_only`).

The frozen final artifact (`final-pipeline.joblib`) is a forecasting-specific
serialized artifact, **not** a scikit-learn `Pipeline`.

`seasonal_trend_ols` won the frozen practical-tie selection rule (tolerance
0.05 °F on pooled MAE) among five statistically indistinguishable finalists
(`seasonal_trend_ols`, `holt_winters_additive_no_trend`, `sarima_100_011_12`,
`holt_winters_additive_damped_trend`, `holt_winters_additive_trend`), using
the tie-break order: pooled seasonal MASE, pooled RMSE, fold-to-fold MAE
standard deviation, long-horizon (h7–h12) MAE, model complexity rank,
candidate id.

## Model-selection results (backtesting)

| Metric | Backtesting |
|---|---:|
| MAE (°F) | 1.839438 |
| RMSE (°F) | 2.320801 |
| Seasonal MASE(12) | 0.656960 |

`seasonal_trend_ols` improved MAE over `seasonal_naive_12` by 0.798525 °F
(≈30.3%) in the replayed, frozen selection contract.

## Final evaluation

| Field | Value |
|---|---|
| Final forecast origin | `1938-12` |
| Final evaluation period | `1939-01` → `1939-12` |
| Forecasts produced | exactly 12 |

| Metric | Final (1939 holdout) |
|---|---:|
| MAE (°F) | 1.526584 |
| RMSE (°F) | 1.859967 |
| Seasonal MASE(12) | 0.555495 |

The final metrics are numerically better than the backtesting reference
metrics on this single 1939 holdout window; this is reported as observed
evidence, not claimed as proof of a generally superior model, since it is a
single 12-month evaluation window. The contract is strict: the model was
fit exactly once on the full development scope (`1920-01` → `1938-12`),
frozen before the holdout was opened, evaluated on the holdout exactly once,
and never retuned, re-selected, or refit afterward.

## Workflow 01 → 05

| Notebook | Role |
|---|---|
| 01 — Time Series Understanding and Exploration | Retrospective, non-mutating exploration of the complete series; publishes the exploration handoff. |
| 02 — Forecasting Data Preparation | Reconstructs the canonical series, splits development vs. sealed final holdout, materializes the backtesting schedule; publishes the preparation handoff. |
| 03 — Model Selection and Evaluation | Runs the frozen expanding-window backtest over the candidate catalog and baselines on development data only; freezes the selection handoff. |
| 04 — Final Forecasting Model and Bundle | Reconstructs the selected specification, fits once on full development, opens the sealed holdout exactly once, materializes the model bundle and final handoff. |
| 05 — Forecasting Inference Demo | Read-only consumer: authenticates the handoff/bundle, verifies the model SHA-256, loads the frozen model, and demonstrates deterministic inference without refitting. |

## Artifact flow

| Notebook | Artifact(s) |
|---|---|
| 01 | `artifacts/exploration/nottem/exploration-handoff.json` |
| 02 | `artifacts/preparation/nottem/preparation-handoff.json` |
| 03 | `artifacts/model-selection/nottem/*` (manifest, candidate results, validation evidence, selection analysis, handoff) |
| 04 | `artifacts/models/nottem/*` (final pipeline, manifest, test evidence, inference bundle, handoff) |
| 05 | none — read-only consumer of the Notebook 04 outputs |

All runtime JSON/CSV/joblib outputs under `data/` and `artifacts/` are
ignored by Git (see `.gitignore`) and must be materialized locally by running
the notebooks in order.

## Inference

Notebook 05 and `scripts/forecasting_inference.py` define the read-only
inference contract for the current frozen model:

**Input** — a `pandas.DataFrame` with exactly the columns:

- `period`
- `temperature`

The history must be monthly, unique, strictly increasing, and contiguous;
`temperature` must be numeric-coercible, finite, and free of missing values.
There are no exogenous predictors. The history must extend at least through
the frozen training end (`1938-12`); `refit_on_input` is always `false` — the
supplied history establishes the forecast chronology/origin only. For the
current `seasonal_trend_ols` model specifically, its coefficients are frozen
at training time and historical values passed at inference time never update
them.

**Output** — 12 future monthly rows with columns:

- `period`
- `forecast` (degrees Fahrenheit)

This is a narrow, forecasting-specific contract, not a general tabular
Mapping/Series/batch-prediction API.

## Installation / environment

Python `>= 3.10` is required.

```bash
python -m pip install -e ".[notebook,test]"
```

Key runtime dependencies (see `pyproject.toml` for the full, unmodified list):
`pandas`, `statsmodels`, `scikit-learn`, `joblib`; notebook/test extras add
`jupyterlab`, `ipykernel`, `matplotlib`, and `pytest`. The `ucimlrepo`
dependency is retained as generic, reusable acquisition infrastructure shared
across dataset-study repositories, even though this study does not use it.

## Reproducibility

```bash
# 1. acquire the source series
python -m scripts.download_data rdataset nottem --package datasets --destination data/raw/nottem

# 2. run the notebooks in order
jupyter lab   # execute 01, 02, 03, 04, then 05

# 3. run the tests
python -m pytest -q
```

During normal execution, owner Notebooks 01, 03, and 04 publish their
corresponding documentation figures to `docs/images/` through the shared
`scripts.export_figures.export_figure` helper. Future regeneration therefore
uses the owner notebook source plus the generic export helper; there is no
dataset-specific exporter.

Git does not contain any runtime data or artifacts — `data/raw/`,
`data/processed/`, and the generated files under `artifacts/` are excluded by
`.gitignore` and must be regenerated locally by running the workflow above.

## Repository structure

```text
notebooks/       official nottem notebooks 01-05
scripts/         reusable acquisition, preparation, selection, finalization, and inference code
tests/           scientific contracts, compatibility, corruption, and hygiene tests
docs/images/      versioned documentation figures
data/            local raw/processed runtime data plus data documentation
artifacts/       local handoffs, evidence, manifests, and model outputs
README.md
pyproject.toml
```

## Boundaries and limitations

- This is a standalone educational/scientific study, not a production system.
- There is no integration with any external orchestration or deployment
  platform at this repository stage; the study is intended as a scientific
  reference ahead of any future integration.
- There is no deployment, serving API, or operational production surface.
- The final model is frozen: no online learning, no refit-on-input, no
  post-holdout retuning.
- `operational_modeling_ready = false` is recorded in the final handoff and
  manifest, and must not be reinterpreted as a readiness claim.
- Notebook 01's exploratory analysis had visibility into the full series,
  including the 1939 evaluation window (see **Temporal boundary** above);
  this is a documented scope limitation of the study design.
