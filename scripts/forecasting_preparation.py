"""Fail-closed preparation contracts for univariate monthly forecasting.

This module deliberately has no model fitting, scoring, random partitioning, or
API that opens the sealed final holdout for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.prepare_data import (
    ArtifactWriteResult,
    fingerprint_dataframe_csv,
    fingerprint_file,
    write_preparation_artifacts,
)
from scripts.target_contract import define_univariate_forecasting_target_contract


SCHEMA_VERSION = "forecasting-preparation-handoff.v1"
ARTIFACT_TYPE = "forecasting_preparation_handoff"


class ForecastingPreparationError(RuntimeError):
    """Raised when temporal preparation cannot be authenticated safely."""


@dataclass(frozen=True, slots=True)
class ModelSelectionForecastingPreparation:
    """Model-selection-safe view: development is loaded; holdout is not."""

    development: pd.DataFrame
    prediction_contract: Mapping[str, Any]
    backtesting_contract: Mapping[str, Any]
    evaluation_contract: Mapping[str, Any]
    preprocessing_contract: Mapping[str, Any]
    readiness: Mapping[str, Any]
    sealed_holdout_integrity: Mapping[str, Any]


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ForecastingPreparationError(message)


def _project_path(root: Path, relative: str | Path) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ForecastingPreparationError("Artifact path escapes project root.") from exc
    return candidate


def validate_exploration_readiness(handoff: Mapping[str, Any]) -> None:
    """Enforce every Notebook-01 gate consumed by preparation."""
    _fail(handoff.get("schema_version") == "exploration-handoff.v1", "Unexpected exploration schema.")
    _fail(handoff.get("artifact_type") == "exploration_handoff", "Unexpected exploration artifact type.")
    _fail(handoff.get("dataset_slug") == "nottem", "Unexpected dataset slug.")
    pc = handoff.get("prediction_contract", {})
    expected = {
        "problem_type": "time_series_forecasting",
        "forecasting_mode": "univariate",
        "target_column": "temperature",
        "target_classes": [],
        "source_exogenous_predictors": 0,
        "frequency": "M",
        "forecast_horizon": 12,
    }
    for key, value in expected.items():
        _fail(pc.get(key) == value, f"Unexpected prediction contract field: {key}.")
    ready = handoff.get("readiness", {})
    for key in ("notebook_01_complete", "deterministic_preparation_ready", "temporal_backtesting_ready"):
        _fail(ready.get(key) is True, f"Exploration readiness gate is not true: {key}.")
    _fail(ready.get("split_execution_ready") is False, "Snapshot split execution must remain false.")
    _fail(ready.get("model_selection_ready") is False, "Upstream model selection readiness must remain false.")


def validate_forecasting_source(
    raw: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    source_file: str | Path,
    handoff: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Authenticate Rdatasets identity, raw bytes, shape, and semantics."""
    validate_exploration_readiness(handoff)
    root = Path(project_root).resolve()
    source_path = Path(source_file).resolve()
    source = handoff["source"]
    checks = {
        "repository": (metadata.get("source_archive"), source.get("repository")),
        "source_reference": (metadata.get("source_reference"), source.get("source_reference")),
        "dataset_name": (metadata.get("dataset"), source.get("dataset_name")),
        "package": (metadata.get("package"), source.get("package")),
        "provider": (metadata.get("provider"), source.get("provider")),
        "raw_logical_path": (source_path.relative_to(root).as_posix(), source.get("path")),
        "raw_sha256": (fingerprint_file(source_path), source.get("sha256")),
        "row_count": (len(raw), source.get("row_count")),
        "column_count": (len(raw.columns), source.get("column_count")),
        "column_order": (list(raw.columns), source.get("column_order")),
    }
    for field, (observed, expected) in checks.items():
        _fail(observed == expected, f"Source {field} mismatch: observed={observed!r}, expected={expected!r}.")
    _fail(list(raw.columns) == ["time", "value"], "Raw columns must be exactly time,value.")
    pc = handoff["prediction_contract"]
    _fail(pc["problem_type"] == "time_series_forecasting", "Source problem type mismatch.")
    _fail(pc["forecasting_mode"] == "univariate", "Source forecasting mode mismatch.")
    _fail(pc["target_semantics"] == "Monthly average air temperature at Nottingham Castle", "Target semantics mismatch.")
    _fail(pc["target_unit"] == "degrees Fahrenheit", "Target unit mismatch.")
    _fail(pc["frequency"] == "M" and pc["source_exogenous_predictors"] == 0, "Frequency/exogenous contract mismatch.")
    return {key: observed for key, (observed, _) in checks.items()} | {"source_identity_validated": True}


def reconstruct_monthly_series(
    raw: pd.DataFrame,
    *,
    target_name: str = "temperature",
    tolerance: float = 1e-8,
) -> pd.Series:
    """Reconstruct and strictly validate a monthly PeriodIndex."""
    _fail(list(raw.columns) == ["time", "value"], "Raw columns must be exactly time,value.")
    _fail(len(raw) > 0, "Raw series is empty.")
    times = pd.to_numeric(raw["time"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(raw["value"], errors="coerce").to_numpy(dtype=float)
    _fail(np.isfinite(times).all(), "Time coordinates must be finite numeric values.")
    _fail(np.isfinite(values).all(), "Target values must be finite numeric values.")
    years = np.floor(times).astype(int)
    month_positions = (times - years) * 12.0
    rounded = np.rint(month_positions)
    _fail(np.all(np.abs(month_positions - rounded) <= tolerance), "Fractional-year coordinate is not an integer month.")
    months_zero = rounded.astype(int)
    _fail(np.all((months_zero >= 0) & (months_zero <= 11)), "Fractional-year month is outside 0..11.")
    labels = [f"{year:04d}-{month:02d}" for year, month in zip(years, months_zero + 1)]
    index = pd.PeriodIndex(labels, freq="M", name="period")
    _fail(index.is_monotonic_increasing, "Temporal order must be strictly increasing.")
    _fail(index.is_unique, "Duplicate monthly period detected.")
    expected = pd.period_range(index[0], index[-1], freq="M", name="period")
    _fail(index.equals(expected), "Monthly coverage contains a gap or unexpected period.")
    return pd.Series(values, index=index, name=target_name)


def validate_target_contract(series: pd.Series, handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the target contract and compare it with Notebook 01."""
    pc = handoff["prediction_contract"]
    contract = define_univariate_forecasting_target_contract(
        series,
        target=pc["target_column"],
        problem_type=pc["problem_type"],
        forecasting_mode=pc["forecasting_mode"],
        target_semantics=pc["target_semantics"],
        target_unit=pc["target_unit"],
        expected_frequency=pc["frequency"],
        source_exogenous_predictors=pc["source_exogenous_predictors"],
    )
    observed = {
        "problem_type": contract.problem_type,
        "forecasting_mode": contract.forecasting_mode,
        "target_column": contract.target,
        "target_semantics": contract.target_semantics,
        "target_unit": contract.target_unit,
        "frequency": contract.frequency,
        "index_type": contract.index_type,
        "source_exogenous_predictors": contract.source_exogenous_predictors,
        "forecast_horizon": pc["forecast_horizon"],
    }
    for key, value in observed.items():
        _fail(pc.get(key) == value, f"Reconstructed target contract mismatch: {key}.")
    _fail(pc.get("target_classes") == [], "Forecast target classes must be empty.")
    return dict(pc)


def materialize_temporal_scopes(series: pd.Series, handoff: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create explicit period,target CSV projections without mutation."""
    temporal = handoff["temporal_contract"]
    _fail(str(series.index[0]) == temporal["source_start"] and str(series.index[-1]) == temporal["source_end"], "Source coverage mismatch.")
    _fail(len(series) == temporal["source_observations"], "Source observation count mismatch.")
    development = series.loc[temporal["development_start"]:temporal["development_end"]]
    holdout = series.loc[temporal["final_holdout_start"]:temporal["final_holdout_end"]]
    _fail(len(development) == temporal["development_observations"] == 228, "Development boundary mismatch.")
    _fail(len(holdout) == temporal["final_holdout_observations"] == 12, "Final holdout boundary mismatch.")
    _fail(development.index[-1] + 1 == holdout.index[0], "Development/holdout boundary is not adjacent.")
    _fail(len(development.index.intersection(holdout.index)) == 0, "Development and holdout overlap.")
    def frame(part: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"period": part.index.astype(str), "temperature": part.to_numpy()})
    return frame(development), frame(holdout)


def seasonal_mase_scale(training: pd.Series, period: int = 12) -> float:
    """Compute a scale from training history only."""
    _fail(period == 12, "This frozen contract requires seasonal period 12.")
    scale = float(training.diff(period).dropna().abs().mean())
    _fail(math.isfinite(scale) and scale > 0.0, "Seasonal MASE denominator must be finite and positive.")
    return scale


def reconstruct_backtesting_schedule(development: pd.Series, handoff: Mapping[str, Any], *, tolerance: float = 1e-12) -> list[dict[str, Any]]:
    """Rebuild expanding folds and authenticate upstream fold-local scales."""
    bc = handoff["backtesting_contract"]
    _fail(bc["mode"] == "expanding_window", "Backtesting mode must be expanding_window.")
    initial, horizon, step, count = (bc["initial_training_months"], bc["forecast_horizon"], bc["origin_step_months"], bc["fold_count"])
    _fail((initial, horizon, step, count) == (120, 12, 12, 9), "Frozen backtesting parameters diverged.")
    schedule: list[dict[str, Any]] = []
    validation_periods: list[pd.Period] = []
    upstream = bc.get("schedule", [])
    _fail(len(upstream) == count, "Upstream fold count mismatch.")
    for offset in range(count):
        train_n = initial + offset * step
        training = development.iloc[:train_n]
        validation = development.iloc[train_n:train_n + horizon]
        _fail(len(validation) == horizon, "Validation window is incomplete.")
        origin = training.index[-1]
        _fail(validation.index[0] == origin + 1, "Validation does not start after forecast origin.")
        _fail(validation.index[-1] < pd.Period("1939-01", freq="M"), "A fold touches the final holdout.")
        scale = seasonal_mase_scale(training)
        row = {
            "fold": offset + 1,
            "train_start": str(training.index[0]),
            "train_end_forecast_origin": str(origin),
            "training_observations": len(training),
            "complete_training_cycles": len(training) // 12,
            "validation_start": str(validation.index[0]),
            "validation_end": str(validation.index[-1]),
            "validation_observations": len(validation),
            "seasonal_mase_scale_from_training": scale,
        }
        expected = upstream[offset]
        for key in row.keys() - {"seasonal_mase_scale_from_training"}:
            _fail(row[key] == expected.get(key), f"Reconstructed fold {offset + 1} mismatch: {key}.")
        _fail(math.isclose(scale, float(expected.get("seasonal_mase_scale_from_training")), rel_tol=0.0, abs_tol=tolerance), f"Fold {offset + 1} MASE scale mismatch.")
        schedule.append(row)
        validation_periods.extend(validation.index)
    _fail(len(validation_periods) == bc["validation_forecast_count"] == 108, "Validation forecast count mismatch.")
    _fail(len(set(validation_periods)) == len(validation_periods), "Validation windows overlap.")
    _fail(bc["validation_targets_overlap"] is False, "Upstream overlap flag must be false.")
    return schedule


def build_forecasting_preparation_handoff(
    *, handoff: Mapping[str, Any], exploration_handoff_path: str | Path,
    exploration_sha256: str, source_validation: Mapping[str, Any],
    development: pd.DataFrame, final_holdout: pd.DataFrame,
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic forecasting-specific preparation manifest."""
    source, pc, bc, ec = handoff["source"], handoff["prediction_contract"], handoff["backtesting_contract"], handoff["evaluation_contract"]
    preprocessing = {
        "original_target_scale_preserved": True, "imputation": "none", "interpolation": "none",
        "outlier_mutation": "none", "global_scaling": "none", "global_differencing": "none",
        "global_power_transformation": "none", "global_decomposition": "none", "global_detrending": "none",
        "exogenous_predictors_added": 0,
    }
    consumer = {
        "model_selection_must_not_resplit": True, "model_selection_must_use_frozen_backtest": True,
        "model_selection_must_not_open_final_holdout": True,
        "model_selection_must_use_fold_local_learned_operations": True,
        "final_holdout_sealed": True, "final_holdout_evaluated": False,
    }
    readiness = {
        "notebook_01_handoff_validated": True, "source_independently_revalidated": True,
        "canonical_series_reconstructed": True, "prepared_projection_materialized": True,
        "development_scope_materialized": True, "final_holdout_sealed": True,
        "final_holdout_evaluated": False, "backtesting_schedule_materialized": True,
        "fold_local_mase_scaling_validated": True, "temporal_leakage_controls_validated": True,
        "preparation_handoff_reloadable": True, "split_execution_ready": False,
        "temporal_backtesting_ready": True, "model_selection_ready": True,
        "model_selected": False, "final_model_trained": False,
    }
    return {
        "schema_version": SCHEMA_VERSION, "artifact_type": ARTIFACT_TYPE, "dataset_slug": "nottem",
        "exploration_handoff": {"path": Path(exploration_handoff_path).as_posix(), "sha256": exploration_sha256,
            "schema_version": handoff["schema_version"], "dataset_slug": handoff["dataset_slug"], "source_sha256": source["sha256"]},
        "source_validation": dict(source_validation),
        "prediction_contract": dict(pc),
        "prepared_data": {
            "development": {"path": "data/processed/nottem/development.csv", "sha256": fingerprint_dataframe_csv(development), "row_count": 228, "start": "1920-01", "end": "1938-12"},
            "sealed_final_holdout": {"path": "data/processed/nottem/final-holdout.csv", "sha256": fingerprint_dataframe_csv(final_holdout), "row_count": 12, "start": "1939-01", "end": "1939-12", "sealed": True, "evaluated": False, "exposed_to_model_selection": False},
        },
        "backtesting_contract": {key: bc[key] for key in ("mode", "initial_training_months", "forecast_horizon", "origin_step_months", "fold_count", "validation_forecast_count", "validation_targets_overlap")} | {"schedule": schedule},
        "evaluation_contract": dict(ec), "preprocessing_contract": preprocessing,
        "consumer_contract": consumer, "readiness": readiness,
    }


def write_forecasting_preparation_artifacts(*, project_root: str | Path, development: pd.DataFrame, final_holdout: pd.DataFrame, payload: Mapping[str, Any]) -> ArtifactWriteResult:
    """Atomically create or reuse the two projections and handoff."""
    return write_preparation_artifacts(
        project_root=project_root,
        csv_artifacts={"data/processed/nottem/development.csv": development, "data/processed/nottem/final-holdout.csv": final_holdout},
        json_artifacts={"artifacts/preparation/nottem/preparation-handoff.json": payload},
        overwrite=False,
    )


def load_and_validate_forecasting_preparation_handoff(
    preparation_handoff_path: str | Path = "artifacts/preparation/nottem/preparation-handoff.json",
    *, project_root: str | Path = ".", expected_dataset_slug: str = "nottem",
) -> ModelSelectionForecastingPreparation:
    """Authenticate all lineage and integrity without opening holdout values."""
    root = Path(project_root).resolve()
    path = _project_path(root, preparation_handoff_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ForecastingPreparationError("Preparation handoff is unreadable.") from exc
    _fail(payload.get("schema_version") == SCHEMA_VERSION, "Unexpected preparation schema.")
    _fail(payload.get("artifact_type") == ARTIFACT_TYPE, "Unexpected preparation artifact type.")
    _fail(payload.get("dataset_slug") == expected_dataset_slug, "Unexpected preparation dataset slug.")
    upstream = payload.get("exploration_handoff", {})
    _fail(upstream.get("dataset_slug") == expected_dataset_slug and upstream.get("schema_version") == "exploration-handoff.v1", "Upstream exploration identity mismatch.")
    _fail(fingerprint_file(_project_path(root, upstream["path"])) == upstream.get("sha256"), "Upstream exploration SHA mismatch.")
    _fail(payload.get("source_validation", {}).get("raw_sha256") == upstream.get("source_sha256"), "Upstream source SHA mismatch.")
    ready = payload.get("readiness", {})
    for key in ("temporal_backtesting_ready", "model_selection_ready", "final_holdout_sealed", "preparation_handoff_reloadable"):
        _fail(ready.get(key) is True, f"Required readiness gate is false: {key}.")
    _fail(ready.get("split_execution_ready") is False, "Snapshot split execution must remain false.")
    _fail(ready.get("final_holdout_evaluated") is False, "Final holdout must remain unevaluated.")
    consumer = payload.get("consumer_contract", {})
    _fail(consumer.get("final_holdout_evaluated") is False and consumer.get("model_selection_must_not_open_final_holdout") is True, "Unsafe final-holdout consumer contract.")
    prepared = payload.get("prepared_data", {})
    dev_ref, hold_ref = prepared.get("development", {}), prepared.get("sealed_final_holdout", {})
    dev_path, hold_path = _project_path(root, dev_ref["path"]), _project_path(root, hold_ref["path"])
    _fail(fingerprint_file(dev_path) == dev_ref.get("sha256"), "Development SHA mismatch.")
    _fail(fingerprint_file(hold_path) == hold_ref.get("sha256"), "Sealed holdout SHA mismatch.")
    _fail(hold_ref.get("sealed") is True and hold_ref.get("evaluated") is False and hold_ref.get("exposed_to_model_selection") is False, "Unsafe sealed holdout reference.")
    development = pd.read_csv(dev_path)
    _fail(list(development.columns) == ["period", "temperature"] and len(development) == dev_ref.get("row_count") == 228, "Development artifact structure mismatch.")
    periods = pd.PeriodIndex(development["period"], freq="M")
    _fail(str(periods[0]) == dev_ref.get("start") and str(periods[-1]) == dev_ref.get("end"), "Development coverage mismatch.")
    bc = payload.get("backtesting_contract", {})
    _fail(bc.get("fold_count") == 9 and bc.get("validation_forecast_count") == 108 and bc.get("validation_targets_overlap") is False, "Backtesting summary mismatch.")
    _fail(all(row["validation_end"] < "1939-01" for row in bc.get("schedule", [])), "A fold reaches the sealed holdout.")
    return ModelSelectionForecastingPreparation(
        development=development,
        prediction_contract=dict(payload["prediction_contract"]),
        backtesting_contract=dict(bc), evaluation_contract=dict(payload["evaluation_contract"]),
        preprocessing_contract=dict(payload["preprocessing_contract"]), readiness=dict(ready),
        sealed_holdout_integrity={key: hold_ref[key] for key in ("path", "sha256", "row_count", "start", "end", "sealed", "evaluated", "exposed_to_model_selection")},
    )
