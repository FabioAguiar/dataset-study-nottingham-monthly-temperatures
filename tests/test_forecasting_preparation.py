"""Forecasting-specific preparation and safe handoff tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.forecasting_preparation import (
    ForecastingPreparationError,
    build_forecasting_preparation_handoff,
    load_and_validate_forecasting_preparation_handoff,
    materialize_temporal_scopes,
    reconstruct_backtesting_schedule,
    reconstruct_monthly_series,
    seasonal_mase_scale,
    validate_forecasting_source,
    validate_target_contract,
    write_forecasting_preparation_artifacts,
)
from scripts.prepare_data import ArtifactConflictError, fingerprint_file


def _raw() -> pd.DataFrame:
    n = 240
    return pd.DataFrame({
        "time": 1920.0 + np.arange(n) / 12.0,
        "value": 49.0 + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + np.arange(n) / 100,
    })


def _handoff(raw_sha: str = "x" * 64) -> dict:
    series = reconstruct_monthly_series(_raw())
    development = series.iloc[:228]
    schedule = []
    for i in range(9):
        training = development.iloc[:120 + 12 * i]
        validation = development.iloc[120 + 12 * i:132 + 12 * i]
        schedule.append({
            "fold": i + 1, "train_start": str(training.index[0]),
            "train_end_forecast_origin": str(training.index[-1]),
            "training_observations": len(training), "complete_training_cycles": len(training) // 12,
            "validation_start": str(validation.index[0]), "validation_end": str(validation.index[-1]),
            "validation_observations": 12,
            "seasonal_mase_scale_from_training": seasonal_mase_scale(training),
        })
    return {
        "schema_version": "exploration-handoff.v1", "artifact_type": "exploration_handoff", "dataset_slug": "nottem",
        "source": {"repository": "Rdatasets", "source_reference": "datasets::nottem", "dataset_name": "nottem", "package": "datasets", "provider": "statsmodels.datasets.get_rdataset", "path": "data/raw/nottem/dataset.csv", "sha256": raw_sha, "row_count": 240, "column_count": 2, "column_order": ["time", "value"]},
        "prediction_contract": {"problem_type": "time_series_forecasting", "forecasting_mode": "univariate", "target_column": "temperature", "target_classes": [], "target_semantics": "Monthly average air temperature at Nottingham Castle", "target_unit": "degrees Fahrenheit", "frequency": "M", "index_type": "PeriodIndex", "source_exogenous_predictors": 0, "forecast_horizon": 12},
        "temporal_contract": {"source_start": "1920-01", "source_end": "1939-12", "source_observations": 240, "development_start": "1920-01", "development_end": "1938-12", "development_observations": 228, "final_forecast_origin": "1938-12", "final_holdout_start": "1939-01", "final_holdout_end": "1939-12", "final_holdout_observations": 12},
        "backtesting_contract": {"mode": "expanding_window", "initial_training_months": 120, "forecast_horizon": 12, "origin_step_months": 12, "fold_count": 9, "validation_forecast_count": 108, "validation_targets_overlap": False, "schedule": schedule},
        "evaluation_contract": {"primary_metric": "mae", "secondary_metrics": ["rmse", "seasonal_mase_12"], "seasonal_mase_period": 12, "horizon_wise_diagnostic": "mae_h1_to_h12", "primary_baseline": "seasonal_naive_12", "secondary_baseline": "naive_last_value", "percentage_error_metrics": "excluded"},
        "readiness": {"notebook_01_complete": True, "deterministic_preparation_ready": True, "temporal_backtesting_ready": True, "split_execution_ready": False, "model_selection_ready": False},
    }


def _metadata() -> dict:
    return {"source_archive": "Rdatasets", "source_reference": "datasets::nottem", "dataset": "nottem", "package": "datasets", "provider": "statsmodels.datasets.get_rdataset"}


def _bundle(root: Path):
    raw = _raw(); raw_path = root / "data/raw/nottem/dataset.csv"; raw_path.parent.mkdir(parents=True); raw.to_csv(raw_path, index=False)
    handoff = _handoff(fingerprint_file(raw_path))
    exploration = root / "artifacts/exploration/nottem/exploration-handoff.json"; exploration.parent.mkdir(parents=True)
    exploration.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")
    source_validation = validate_forecasting_source(raw, _metadata(), source_file=raw_path, handoff=handoff, project_root=root)
    series = reconstruct_monthly_series(raw); validate_target_contract(series, handoff)
    dev, hold = materialize_temporal_scopes(series, handoff)
    schedule = reconstruct_backtesting_schedule(series.iloc[:228], handoff)
    payload = build_forecasting_preparation_handoff(handoff=handoff, exploration_handoff_path="artifacts/exploration/nottem/exploration-handoff.json", exploration_sha256=fingerprint_file(exploration), source_validation=source_validation, development=dev, final_holdout=hold, schedule=schedule)
    return handoff, series, dev, hold, payload


def test_reconstructs_fractional_year_to_monthly_period_index() -> None:
    series = reconstruct_monthly_series(_raw())
    assert isinstance(series.index, pd.PeriodIndex) and series.index.freqstr == "M"
    assert (str(series.index[0]), str(series.index[-1]), series.name) == ("1920-01", "1939-12", "temperature")


def test_rejects_non_integer_month_coordinate() -> None:
    raw = _raw(); raw.loc[1, "time"] += 0.001
    with pytest.raises(ForecastingPreparationError, match="integer month"):
        reconstruct_monthly_series(raw)


@pytest.mark.parametrize("mutation,message", [
    (lambda x: pd.concat([x.iloc[[1, 0]], x.iloc[2:]], ignore_index=True), "order"),
    (lambda x: x.assign(time=lambda f: f["time"].mask(f.index == 1, f.loc[0, "time"])), "Duplicate"),
    (lambda x: x.drop(index=1).reset_index(drop=True), "gap"),
])
def test_rejects_invalid_temporal_identity(mutation, message: str) -> None:
    with pytest.raises(ForecastingPreparationError, match=message):
        reconstruct_monthly_series(mutation(_raw()))


def test_source_identity_rejects_sha_and_reference_mismatch(tmp_path: Path) -> None:
    raw = _raw(); path = tmp_path / "data/raw/nottem/dataset.csv"; path.parent.mkdir(parents=True); raw.to_csv(path, index=False)
    handoff = _handoff("0" * 64)
    with pytest.raises(ForecastingPreparationError, match="raw_sha256"):
        validate_forecasting_source(raw, _metadata(), source_file=path, handoff=handoff, project_root=tmp_path)
    handoff["source"]["sha256"] = fingerprint_file(path); metadata = _metadata(); metadata["source_reference"] = "other::nottem"
    with pytest.raises(ForecastingPreparationError, match="source_reference"):
        validate_forecasting_source(raw, metadata, source_file=path, handoff=handoff, project_root=tmp_path)


def test_target_contract_and_temporal_scopes() -> None:
    handoff = _handoff(); series = reconstruct_monthly_series(_raw())
    contract = validate_target_contract(series, handoff); dev, hold = materialize_temporal_scopes(series, handoff)
    assert contract["problem_type"] == "time_series_forecasting" and contract["forecasting_mode"] == "univariate"
    assert (len(dev), len(hold)) == (228, 12)
    assert pd.Period(dev.iloc[-1]["period"], freq="M") + 1 == pd.Period(hold.iloc[0]["period"], freq="M")
    assert set(dev.period).isdisjoint(hold.period)


def test_reconstructs_exact_frozen_backtest() -> None:
    handoff = _handoff(); series = reconstruct_monthly_series(_raw()); schedule = reconstruct_backtesting_schedule(series.iloc[:228], handoff)
    assert len(schedule) == 9 and sum(x["validation_observations"] for x in schedule) == 108
    assert schedule[0]["training_observations"] == 120 and schedule[-1]["training_observations"] == 216
    assert schedule[0]["train_end_forecast_origin"] == "1929-12" and schedule[-1]["train_end_forecast_origin"] == "1937-12"
    validations = [(x["validation_start"], x["validation_end"]) for x in schedule]
    assert all(end < "1939-01" for _, end in validations) and len({p for pair in validations for p in pair}) == 18


def test_mase_is_positive_training_only_and_validation_invariant() -> None:
    handoff = _handoff(); series = reconstruct_monthly_series(_raw()); training = series.iloc[:120]
    scale = seasonal_mase_scale(training); changed = series.copy(); changed.iloc[120:132] += 10000
    assert np.isfinite(scale) and scale > 0 and seasonal_mase_scale(changed.iloc[:120]) == scale
    assert seasonal_mase_scale(changed.iloc[:120]) == handoff["backtesting_contract"]["schedule"][0]["seasonal_mase_scale_from_training"]


def test_holdout_perturbation_cannot_change_folds_or_scales() -> None:
    handoff = _handoff(); series = reconstruct_monthly_series(_raw())
    before = reconstruct_backtesting_schedule(series.iloc[:228], handoff); series.iloc[228:] += 1e6
    assert reconstruct_backtesting_schedule(series.iloc[:228], handoff) == before


def test_contract_has_no_random_split_or_global_transform(tmp_path: Path) -> None:
    _, _, _, _, payload = _bundle(tmp_path); text = json.dumps(payload)
    assert "random" not in text.lower() and "shuffle" not in text.lower()
    assert payload["readiness"]["split_execution_ready"] is False
    assert all(payload["preprocessing_contract"][key] == "none" for key in ("global_scaling", "global_differencing", "global_power_transformation", "global_decomposition", "global_detrending"))


def test_writer_is_idempotent_and_rejects_conflict(tmp_path: Path) -> None:
    _, _, dev, hold, payload = _bundle(tmp_path)
    first = write_forecasting_preparation_artifacts(project_root=tmp_path, development=dev, final_holdout=hold, payload=payload)
    second = write_forecasting_preparation_artifacts(project_root=tmp_path, development=dev, final_holdout=hold, payload=payload)
    assert {status for _, status in first.statuses} == {"created"}
    assert {status for _, status in second.statuses} == {"reused_equivalent"}
    changed = dev.copy(); changed.loc[0, "temperature"] += 1
    with pytest.raises(ArtifactConflictError):
        write_forecasting_preparation_artifacts(project_root=tmp_path, development=changed, final_holdout=hold, payload=payload)


def _written(root: Path):
    _, _, dev, hold, payload = _bundle(root)
    write_forecasting_preparation_artifacts(project_root=root, development=dev, final_holdout=hold, payload=payload)
    return root / "artifacts/preparation/nottem/preparation-handoff.json"


def test_safe_loader_fresh_reload_and_no_holdout_values(tmp_path: Path) -> None:
    path = _written(tmp_path)
    loaded = load_and_validate_forecasting_preparation_handoff(path.relative_to(tmp_path), project_root=tmp_path)
    assert len(loaded.development) == 228 and loaded.readiness["model_selection_ready"] is True
    assert not hasattr(loaded, "final_holdout")
    assert "temperature" not in loaded.sealed_holdout_integrity


@pytest.mark.parametrize("kind", ["upstream", "development", "holdout"])
def test_loader_validates_all_file_hashes(tmp_path: Path, kind: str) -> None:
    path = _written(tmp_path)
    targets = {"upstream": tmp_path / "artifacts/exploration/nottem/exploration-handoff.json", "development": tmp_path / "data/processed/nottem/development.csv", "holdout": tmp_path / "data/processed/nottem/final-holdout.csv"}
    targets[kind].write_bytes(targets[kind].read_bytes() + b"tamper")
    with pytest.raises(ForecastingPreparationError, match="SHA"):
        load_and_validate_forecasting_preparation_handoff(path.relative_to(tmp_path), project_root=tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("final_holdout_evaluated", True, "unevaluated"),
    ("split_execution_ready", True, "split execution"),
    ("temporal_backtesting_ready", False, "readiness"),
    ("model_selection_ready", False, "readiness"),
])
def test_loader_rejects_unsafe_readiness(tmp_path: Path, field: str, value, message: str) -> None:
    path = _written(tmp_path); payload = json.loads(path.read_text()); payload["readiness"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForecastingPreparationError, match=message):
        load_and_validate_forecasting_preparation_handoff(path.relative_to(tmp_path), project_root=tmp_path)


def test_loader_rejects_tampered_artifact_type(tmp_path: Path) -> None:
    path = _written(tmp_path); payload = json.loads(path.read_text()); payload["artifact_type"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForecastingPreparationError, match="artifact type"):
        load_and_validate_forecasting_preparation_handoff(path.relative_to(tmp_path), project_root=tmp_path)


def test_notebook_02_structure_and_clean_state() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/02_data_preparation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8")); code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert all(c.get("execution_count") is None and c.get("outputs") == [] for c in notebook["cells"] if c["cell_type"] == "code")
    for prohibited in ("train_test_split", ".sample(", "shuffle=True", ".fit(", "ContinuousRegressionSplitPolicy", "ClassificationSplitPolicy", "split_continuous_regression_dataset", "split_classification_dataset"):
        assert prohibited not in code
    assert "load_and_validate_forecasting_exploration_handoff" in code
    assert "write_forecasting_preparation_artifacts" in code
    assert "load_and_validate_forecasting_preparation_handoff" in code
