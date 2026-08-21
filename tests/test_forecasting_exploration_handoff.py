"""Tests for the univariate forecasting Notebook-01 exploration handoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.forecasting_exploration_handoff import (
    ForecastingExplorationHandoffError,
    build_univariate_forecasting_exploration_handoff,
    load_and_validate_forecasting_exploration_handoff,
)
from scripts.target_contract import define_univariate_forecasting_target_contract


SLUG = "nottem"
SOURCE_REFERENCE = "datasets::nottem"


def _bundle(tmp_path: Path):
    index = pd.period_range("1920-01", "1939-12", freq="M", name="period")
    values = 49.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(len(index)) / 12.0)
    series = pd.Series(values, index=index, name="temperature")
    development = series.loc[:"1938-12"]

    raw = pd.DataFrame(
        {
            "time": 1920.0 + np.arange(len(index)) / 12.0,
            "value": values,
        }
    )
    source = tmp_path / "data/raw/nottem/dataset.csv"
    source.parent.mkdir(parents=True)
    raw.to_csv(source, index=False)

    target_contract = define_univariate_forecasting_target_contract(
        series,
        target="temperature",
        problem_type="time_series_forecasting",
        forecasting_mode="univariate",
        target_semantics="Monthly average air temperature at Nottingham Castle",
        target_unit="degrees Fahrenheit",
        expected_frequency="M",
        source_exogenous_predictors=0,
    )

    rows = []
    for fold, year in enumerate(range(1930, 1939), start=1):
        train_end = pd.Period(f"{year - 1}-12", freq="M")
        training = development.loc[:train_end]
        scale = float(training.diff(12).dropna().abs().mean())
        rows.append(
            {
                "fold": fold,
                "train_start": "1920-01",
                "train_end_forecast_origin": str(train_end),
                "training_observations": len(training),
                "complete_training_cycles": len(training) // 12,
                "validation_start": f"{year}-01",
                "validation_end": f"{year}-12",
                "validation_observations": 12,
                "seasonal_mase_scale_from_training": scale,
            }
        )
    schedule = pd.DataFrame(rows)

    prep = pd.DataFrame(
        [{"Decision": "Backtest mode", "Frozen value": "expanding_window"}]
    )
    boundary = pd.DataFrame(
        [{"Boundary item": "Final holdout", "Frozen value": "1939-01 -> 1939-12"}]
    )
    leakage = pd.DataFrame(
        [{"Risk": "Future target leakage", "Policy": "prohibited"}]
    )
    insights = pd.DataFrame(
        [{"Insight": "Seasonal structure", "Frozen evidence / decision": "strong"}]
    )

    return {
        "dataset_slug": SLUG,
        "source_repository": "Rdatasets",
        "source_reference": SOURCE_REFERENCE,
        "source_dataset_name": "nottem",
        "source_package": "datasets",
        "source_provider": "statsmodels.datasets.get_rdataset",
        "source_file": source,
        "project_root": tmp_path,
        "source_dataframe": raw,
        "target_contract": target_contract,
        "canonical_series": series,
        "forecast_horizon": 12,
        "development_history": development,
        "final_forecast_origin": pd.Period("1938-12", freq="M"),
        "final_holdout_start": pd.Period("1939-01", freq="M"),
        "final_holdout_end": pd.Period("1939-12", freq="M"),
        "primary_baseline_id": "seasonal_naive_12",
        "secondary_baseline_id": "naive_last_value",
        "backtesting_schedule": schedule,
        "preparation_backtesting_contract": prep,
        "evaluation_boundary_contract": boundary,
        "leakage_risk_register": leakage,
        "key_exploratory_insights": insights,
        "primary_metric": "mae",
        "secondary_metrics": ("rmse", "seasonal_mase_12"),
        "mase_seasonal_period": 12,
    }


def test_builds_ready_forecasting_handoff(tmp_path: Path) -> None:
    report = build_univariate_forecasting_exploration_handoff(**_bundle(tmp_path))

    assert report.is_structurally_valid
    assert report.is_handoff_ready
    assert report.payload["prediction_contract"]["problem_type"] == "time_series_forecasting"
    assert report.payload["prediction_contract"]["forecasting_mode"] == "univariate"
    assert report.payload["feature_contract"]["feature_columns"] == []
    assert report.payload["temporal_contract"]["development_end"] == "1938-12"
    assert report.payload["temporal_contract"]["final_holdout_start"] == "1939-01"
    assert report.payload["backtesting_contract"]["fold_count"] == 9
    assert report.payload["backtesting_contract"]["validation_forecast_count"] == 108
    assert report.payload["readiness"]["split_execution_ready"] is False
    assert report.payload["readiness"]["temporal_backtesting_ready"] is True
    assert report.payload["readiness"]["model_selection_ready"] is False


def test_write_and_reload_forecasting_handoff(tmp_path: Path) -> None:
    report = build_univariate_forecasting_exploration_handoff(**_bundle(tmp_path))
    destination = tmp_path / "artifacts/exploration/nottem/exploration-handoff.json"

    persisted = report.write(destination)
    payload = load_and_validate_forecasting_exploration_handoff(
        destination,
        expected_dataset_slug=SLUG,
        expected_source_reference=SOURCE_REFERENCE,
    )

    assert persisted.path == destination.resolve()
    assert len(persisted.sha256) == 64
    assert payload["source"]["path"] == "data/raw/nottem/dataset.csv"
    assert payload["evaluation_contract"]["primary_metric"] == "mae"
    assert payload["evaluation_contract"]["primary_baseline"] == "seasonal_naive_12"


def test_rejects_backtest_that_reaches_final_holdout(tmp_path: Path) -> None:
    params = _bundle(tmp_path)
    schedule = params["backtesting_schedule"].copy()
    schedule.loc[schedule.index[-1], "validation_end"] = "1939-12"
    params["backtesting_schedule"] = schedule

    report = build_univariate_forecasting_exploration_handoff(**params)

    assert not report.is_handoff_ready
    with pytest.raises(ForecastingExplorationHandoffError, match="not ready"):
        report.raise_if_invalid()


def test_loader_rejects_snapshot_split_authorization(tmp_path: Path) -> None:
    report = build_univariate_forecasting_exploration_handoff(**_bundle(tmp_path))
    destination = tmp_path / "exploration-handoff.json"
    report.write(destination)

    import json

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["readiness"]["split_execution_ready"] = True
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ForecastingExplorationHandoffError,
        match="must not authorize snapshot split",
    ):
        load_and_validate_forecasting_exploration_handoff(destination)
