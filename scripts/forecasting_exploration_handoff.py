"""Notebook-01 exploration handoff for univariate time-series forecasting.

The module serializes the validated forecasting exploration contract without
executing preparation, backtesting, model fitting, or final-holdout evaluation.
The resulting JSON is designed to be reloaded from a fresh kernel by Notebook 02.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


HANDOFF_SCHEMA_VERSION = "exploration-handoff.v1"
HANDOFF_ARTIFACT_TYPE = "exploration_handoff"

_ISSUE_COLUMNS = ["Scope", "Issue", "Details"]
_SUMMARY_COLUMNS = ["Gate", "Ready", "Interpretation"]
_NEXT_STEP_COLUMNS = [
    "Notebook",
    "Sequence",
    "Action",
    "Status",
    "Acceptance criterion",
]
_EXPECTED_OUTPUT_COLUMNS = ["Notebook", "Expected output", "Purpose"]
_ARTIFACT_COLUMNS = ["Artifact", "Value"]


class ForecastingExplorationHandoffError(RuntimeError):
    """Raised when a forecasting exploration handoff is invalid."""


def _text(value: object) -> str:
    return str(value).strip()


def _unique_text_tuple(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if not item:
            continue
        if item in seen:
            raise ForecastingExplorationHandoffError(
                f"Duplicate contract value is not allowed: {item!r}."
            )
        seen.add(item)
        result.append(item)
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise ForecastingExplorationHandoffError(
            "Expected a pandas DataFrame report."
        )
    return _json_safe(frame.to_dict(orient="records"))


def _next_steps() -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 1,
            "Action": "Reacquire and independently revalidate the R source",
            "Status": "Ready",
            "Acceptance criterion": (
                "Source reference, raw-file SHA-256, row count, column order, "
                "canonical monthly coverage, and target semantics match the "
                "persisted exploration handoff."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 2,
            "Action": "Reconstruct the canonical monthly target and seal 1939",
            "Status": "Ready",
            "Acceptance criterion": (
                "The canonical target remains unchanged on the Fahrenheit scale; "
                "model-development history ends at 1938-12 and 1939-01 through "
                "1939-12 is unavailable to all learned preparation decisions."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 3,
            "Action": "Materialize the frozen expanding-window backtesting schedule",
            "Status": "Ready",
            "Acceptance criterion": (
                "Exactly nine non-overlapping 12-month validation folds are "
                "materialized from origins 1929-12 through 1937-12."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 4,
            "Action": "Validate fold-local metric scaling and temporal causality",
            "Status": "Ready",
            "Acceptance criterion": (
                "Seasonal MASE denominators use training history only and no "
                "validation/final-holdout target influences fitted state, features, "
                "transformations, or baseline inputs."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 5,
            "Action": "Persist and reload the forecasting preparation handoff",
            "Status": "Ready",
            "Acceptance criterion": (
                "Notebook 03 can reconstruct temporal boundaries, baselines, metrics, "
                "and folds from a fresh kernel without Notebook 01 live state."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 1,
            "Action": "Score frozen baselines and candidate forecasting families",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "All candidates use the same nine folds, 12-month horizon, MAE "
                "primary metric, secondary RMSE/seasonal-MASE diagnostics, and "
                "fold-local learned operations."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=_NEXT_STEP_COLUMNS)


def _expected_outputs(dataset_slug: str) -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": (
                f"artifacts/preparation/{dataset_slug}/preparation-handoff.json"
            ),
            "Purpose": (
                "Freeze the canonical development series, sealed holdout boundary, "
                "backtesting schedule, baselines, metrics, and leakage controls."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Expected output": (
                f"artifacts/model-selection/{dataset_slug}/model-selection-handoff.json"
            ),
            "Purpose": (
                "Freeze the selected forecasting specification before one-time "
                "final-holdout evaluation."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=_EXPECTED_OUTPUT_COLUMNS)


@dataclass(frozen=True, slots=True)
class PersistedForecastingExplorationHandoff:
    """Metadata for one materialized forecasting exploration handoff."""

    path: Path
    sha256: str
    size_bytes: int

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Artifact": "Path", "Value": self.path.as_posix()},
                {"Artifact": "SHA-256", "Value": self.sha256},
                {"Artifact": "Bytes", "Value": self.size_bytes},
            ],
            columns=_ARTIFACT_COLUMNS,
        )


@dataclass(frozen=True, slots=True)
class ForecastingExplorationHandoffReport:
    """Validated, serializable Notebook-01 forecasting transition contract."""

    payload: dict[str, Any]
    issues: pd.DataFrame
    next_steps: pd.DataFrame
    expected_outputs: pd.DataFrame

    @property
    def is_structurally_valid(self) -> bool:
        return self.issues.empty

    @property
    def is_handoff_ready(self) -> bool:
        readiness = self.payload.get("readiness", {})
        return bool(
            self.is_structurally_valid
            and readiness.get("notebook_01_complete") is True
            and readiness.get("deterministic_preparation_ready") is True
            and readiness.get("temporal_backtesting_ready") is True
        )

    def summary_frame(self) -> pd.DataFrame:
        readiness = self.payload["readiness"]
        rows = [
            {
                "Gate": "Notebook 01 analysis complete",
                "Ready": readiness["notebook_01_complete"],
                "Interpretation": (
                    "Source, temporal, forecasting, quality, leakage, and evaluation "
                    "contracts are complete."
                ),
            },
            {
                "Gate": "Handoff artifact structurally valid",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "Serialized source, temporal boundaries, metrics, baselines, and "
                    "continuation contracts are coherent."
                ),
            },
            {
                "Gate": "Deterministic preparation may begin",
                "Ready": readiness["deterministic_preparation_ready"],
                "Interpretation": (
                    "Notebook 02 may independently reconstruct the validated monthly "
                    "series and preserve its original scale."
                ),
            },
            {
                "Gate": "Temporal backtesting may be materialized",
                "Ready": readiness["temporal_backtesting_ready"],
                "Interpretation": (
                    "The nine-fold expanding-window geometry is frozen and 1939 "
                    "remains outside model development."
                ),
            },
            {
                "Gate": "Model selection may begin",
                "Ready": readiness["model_selection_ready"],
                "Interpretation": (
                    "Expected to remain false until Notebook 02 publishes a validated "
                    "preparation handoff."
                ),
            },
            {
                "Gate": "Notebook 01 handoff ready",
                "Ready": self.is_handoff_ready,
                "Interpretation": (
                    "Notebook 02 can continue from a fresh kernel using the persisted "
                    "forecasting contract."
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def open_reviews_frame(self) -> pd.DataFrame:
        return pd.DataFrame(deepcopy(self.payload["open_reviews"]))

    def next_steps_frame(self) -> pd.DataFrame:
        return self.next_steps.copy(deep=True)

    def expected_outputs_frame(self) -> pd.DataFrame:
        return self.expected_outputs.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        return self.issues.copy(deep=True)

    def raise_if_invalid(self) -> None:
        if self.is_handoff_ready:
            return
        if not self.issues.empty:
            details = "; ".join(
                f"{row['Scope']}: {row['Issue']} ({row['Details']})"
                for _, row in self.issues.iterrows()
            )
        else:
            details = "handoff readiness gates are not satisfied"
        raise ForecastingExplorationHandoffError(
            f"Notebook-01 forecasting handoff is not ready: {details}."
        )

    def write(
        self,
        destination: str | Path,
    ) -> PersistedForecastingExplorationHandoff:
        """Atomically persist the validated JSON handoff."""
        self.raise_if_invalid()
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            _json_safe(self.payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        return PersistedForecastingExplorationHandoff(
            path=path,
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )


def build_univariate_forecasting_exploration_handoff(
    *,
    dataset_slug: str,
    source_repository: str,
    source_reference: str,
    source_dataset_name: str,
    source_package: str,
    source_provider: str,
    source_file: str | Path,
    project_root: str | Path | None,
    source_dataframe: pd.DataFrame,
    target_contract: object,
    canonical_series: pd.Series,
    forecast_horizon: int,
    development_history: pd.Series,
    final_forecast_origin: object,
    final_holdout_start: object,
    final_holdout_end: object,
    primary_baseline_id: str,
    secondary_baseline_id: str,
    backtesting_schedule: pd.DataFrame,
    preparation_backtesting_contract: pd.DataFrame,
    evaluation_boundary_contract: pd.DataFrame,
    leakage_risk_register: pd.DataFrame,
    key_exploratory_insights: pd.DataFrame,
    primary_metric: str,
    secondary_metrics: Sequence[object],
    mase_seasonal_period: int,
) -> ForecastingExplorationHandoffReport:
    """Build the final Notebook-01 handoff without mutating analytical data."""
    issues: list[dict[str, str]] = []

    slug = _text(dataset_slug)
    repository = _text(source_repository)
    reference = _text(source_reference)
    dataset_name = _text(source_dataset_name)
    package = _text(source_package)
    provider = _text(source_provider)
    source = Path(source_file).expanduser().resolve()
    project = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )

    problem_type = _text(getattr(target_contract, "problem_type", ""))
    forecasting_mode = _text(getattr(target_contract, "forecasting_mode", ""))
    target = _text(getattr(target_contract, "target", ""))
    target_semantics = _text(getattr(target_contract, "target_semantics", ""))
    target_unit = _text(getattr(target_contract, "target_unit", ""))
    frequency = _text(getattr(target_contract, "frequency", ""))
    prediction_output = _text(getattr(target_contract, "prediction_output", ""))
    source_exogenous_predictors = int(
        getattr(target_contract, "source_exogenous_predictors", -1)
    )

    if not all((slug, repository, reference, dataset_name, package, provider)):
        issues.append({
            "Scope": "Source",
            "Issue": "Incomplete source identity",
            "Details": "Dataset/source identity fields must all be non-empty",
        })
    if not source.is_file():
        issues.append({
            "Scope": "Source",
            "Issue": "Source file unavailable",
            "Details": source.as_posix(),
        })
    if not isinstance(source_dataframe, pd.DataFrame) or source_dataframe.empty:
        issues.append({
            "Scope": "Source",
            "Issue": "Invalid source dataframe",
            "Details": "A non-empty pandas DataFrame is required",
        })
    if problem_type != "time_series_forecasting":
        issues.append({
            "Scope": "Target",
            "Issue": "Unexpected problem type",
            "Details": repr(problem_type),
        })
    if forecasting_mode != "univariate":
        issues.append({
            "Scope": "Target",
            "Issue": "Unexpected forecasting mode",
            "Details": repr(forecasting_mode),
        })
    if source_exogenous_predictors != 0:
        issues.append({
            "Scope": "Target",
            "Issue": "Unexpected source exogenous predictors",
            "Details": repr(source_exogenous_predictors),
        })

    if not isinstance(canonical_series, pd.Series) or canonical_series.empty:
        issues.append({
            "Scope": "Temporal series",
            "Issue": "Invalid canonical series",
            "Details": "A non-empty pandas Series is required",
        })
    elif not isinstance(canonical_series.index, pd.PeriodIndex):
        issues.append({
            "Scope": "Temporal series",
            "Issue": "Canonical index must be a PeriodIndex",
            "Details": type(canonical_series.index).__name__,
        })
    elif canonical_series.index.freqstr != frequency:
        issues.append({
            "Scope": "Temporal series",
            "Issue": "Canonical frequency mismatch",
            "Details": f"{canonical_series.index.freqstr!r} != {frequency!r}",
        })

    if not isinstance(development_history, pd.Series) or development_history.empty:
        issues.append({
            "Scope": "Evaluation boundary",
            "Issue": "Invalid development history",
            "Details": "A non-empty pandas Series is required",
        })

    try:
        horizon = int(forecast_horizon)
        mase_period = int(mase_seasonal_period)
    except (TypeError, ValueError):
        horizon = 0
        mase_period = 0
    if horizon <= 0:
        issues.append({
            "Scope": "Forecast contract",
            "Issue": "Invalid forecast horizon",
            "Details": repr(forecast_horizon),
        })
    if mase_period <= 0:
        issues.append({
            "Scope": "Metric contract",
            "Issue": "Invalid seasonal MASE period",
            "Details": repr(mase_seasonal_period),
        })

    try:
        origin = pd.Period(final_forecast_origin, freq=frequency)
        holdout_start = pd.Period(final_holdout_start, freq=frequency)
        holdout_end = pd.Period(final_holdout_end, freq=frequency)
    except (TypeError, ValueError) as exc:
        issues.append({
            "Scope": "Evaluation boundary",
            "Issue": "Invalid final temporal boundary",
            "Details": str(exc),
        })
        origin = holdout_start = holdout_end = None

    if origin is not None and holdout_start is not None and holdout_start != origin + 1:
        issues.append({
            "Scope": "Evaluation boundary",
            "Issue": "Final holdout is not adjacent to forecast origin",
            "Details": f"origin={origin}, holdout_start={holdout_start}",
        })
    if (
        holdout_start is not None
        and holdout_end is not None
        and horizon > 0
        and holdout_end != holdout_start + horizon - 1
    ):
        issues.append({
            "Scope": "Evaluation boundary",
            "Issue": "Final holdout length does not match forecast horizon",
            "Details": f"start={holdout_start}, end={holdout_end}, horizon={horizon}",
        })
    if (
        isinstance(development_history, pd.Series)
        and not development_history.empty
        and origin is not None
        and development_history.index[-1] != origin
    ):
        issues.append({
            "Scope": "Evaluation boundary",
            "Issue": "Development history does not end at final forecast origin",
            "Details": f"{development_history.index[-1]} != {origin}",
        })

    required_schedule_columns = {
        "fold",
        "train_start",
        "train_end_forecast_origin",
        "training_observations",
        "complete_training_cycles",
        "validation_start",
        "validation_end",
        "validation_observations",
        "seasonal_mase_scale_from_training",
    }
    if not isinstance(backtesting_schedule, pd.DataFrame):
        issues.append({
            "Scope": "Backtesting",
            "Issue": "Invalid backtesting schedule",
            "Details": "Expected a pandas DataFrame",
        })
    else:
        missing_schedule = sorted(
            required_schedule_columns - set(backtesting_schedule.columns)
        )
        if missing_schedule:
            issues.append({
                "Scope": "Backtesting",
                "Issue": "Backtesting schedule is incomplete",
                "Details": repr(missing_schedule),
            })
        elif not backtesting_schedule.empty:
            if not bool((backtesting_schedule["validation_observations"] == horizon).all()):
                issues.append({
                    "Scope": "Backtesting",
                    "Issue": "Backtest validation horizon mismatch",
                    "Details": repr(
                        tuple(backtesting_schedule["validation_observations"].tolist())
                    ),
                })
            if holdout_start is not None:
                last_validation = pd.Period(
                    backtesting_schedule.iloc[-1]["validation_end"],
                    freq=frequency,
                )
                if last_validation >= holdout_start:
                    issues.append({
                        "Scope": "Backtesting",
                        "Issue": "Backtesting reaches sealed final holdout",
                        "Details": f"last validation={last_validation}",
                    })

    report_frames = (
        ("Preparation/backtesting contract", preparation_backtesting_contract),
        ("Evaluation boundary contract", evaluation_boundary_contract),
        ("Leakage risk register", leakage_risk_register),
        ("Key exploratory insights", key_exploratory_insights),
    )
    for name, frame in report_frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            issues.append({
                "Scope": "Upstream contract",
                "Issue": f"{name} is unavailable",
                "Details": "Expected a non-empty pandas DataFrame",
            })

    baseline_primary = _text(primary_baseline_id)
    baseline_secondary = _text(secondary_baseline_id)
    metric_primary = _text(primary_metric)
    metric_secondary = _unique_text_tuple(secondary_metrics)

    source_columns = (
        tuple(str(value) for value in source_dataframe.columns)
        if isinstance(source_dataframe, pd.DataFrame)
        else ()
    )
    source_sha256 = _sha256_file(source) if source.is_file() else None
    fold_count = (
        int(len(backtesting_schedule))
        if isinstance(backtesting_schedule, pd.DataFrame)
        else 0
    )
    forecast_count = (
        int(backtesting_schedule["validation_observations"].sum())
        if isinstance(backtesting_schedule, pd.DataFrame)
        and "validation_observations" in backtesting_schedule
        else 0
    )

    next_steps = _next_steps()
    expected_outputs = _expected_outputs(slug)

    open_reviews = [
        {
            "review_id": "REV-001",
            "theme": "Final-holdout exploration exposure",
            "blocking": False,
            "summary": (
                "The 1939 observations were inspected during retrospective Notebook 01 "
                "exploration before the final boundary was frozen."
            ),
            "continuation": (
                "Treat 1939 as prospectively sealed from Notebook 02 onward; no new "
                "learned or selection decision may use those values."
            ),
        },
        {
            "review_id": "REV-002",
            "theme": "Model-specific differencing and transformations",
            "blocking": False,
            "summary": (
                "No global differencing, power transform, detrending, or decomposition "
                "policy is frozen."
            ),
            "continuation": (
                "Evaluate candidate-specific transformations only inside each training "
                "window of the frozen backtesting design."
            ),
        },
        {
            "review_id": "REV-003",
            "theme": "Candidate model and multi-step strategy",
            "blocking": False,
            "summary": (
                "Forecasting model families and their 12-step implementation strategies "
                "remain intentionally open."
            ),
            "continuation": (
                "Notebook 03 must compare candidates with seasonal_naive_12 without "
                "consulting the 1939 holdout."
            ),
        },
    ]

    payload: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "artifact_type": HANDOFF_ARTIFACT_TYPE,
        "dataset_slug": slug,
        "source": {
            "repository": repository,
            "source_reference": reference,
            "dataset_name": dataset_name,
            "package": package,
            "provider": provider,
            "path": _relative_posix(source, project),
            "sha256": source_sha256,
            "row_count": (
                int(len(source_dataframe))
                if isinstance(source_dataframe, pd.DataFrame)
                else 0
            ),
            "column_count": len(source_columns),
            "column_order": list(source_columns),
        },
        "prediction_contract": {
            "problem_type": problem_type,
            "forecasting_mode": forecasting_mode,
            "target_column": target,
            "target_classes": [],
            "target_semantics": target_semantics,
            "target_unit": target_unit,
            "frequency": frequency,
            "index_type": getattr(target_contract, "index_type", None),
            "source_exogenous_predictors": source_exogenous_predictors,
            "forecast_horizon": horizon,
            "prediction_output": prediction_output,
        },
        "feature_contract": {
            "identifier_columns": [],
            "feature_columns": [],
            "numerical_features": [],
            "categorical_features": [],
            "baseline_feature_count": 0,
            "source_exogenous_predictors": source_exogenous_predictors,
            "temporal_coordinate_role": "index_only_not_predictor",
        },
        "temporal_contract": {
            "source_start": (
                str(canonical_series.index[0])
                if isinstance(canonical_series, pd.Series)
                and not canonical_series.empty
                else None
            ),
            "source_end": (
                str(canonical_series.index[-1])
                if isinstance(canonical_series, pd.Series)
                and not canonical_series.empty
                else None
            ),
            "source_observations": (
                int(len(canonical_series))
                if isinstance(canonical_series, pd.Series)
                else 0
            ),
            "development_start": (
                str(development_history.index[0])
                if isinstance(development_history, pd.Series)
                and not development_history.empty
                else None
            ),
            "development_end": (
                str(development_history.index[-1])
                if isinstance(development_history, pd.Series)
                and not development_history.empty
                else None
            ),
            "development_observations": (
                int(len(development_history))
                if isinstance(development_history, pd.Series)
                else 0
            ),
            "final_forecast_origin": str(origin) if origin is not None else None,
            "final_holdout_start": (
                str(holdout_start) if holdout_start is not None else None
            ),
            "final_holdout_end": (
                str(holdout_end) if holdout_end is not None else None
            ),
            "final_holdout_observations": horizon,
            "final_holdout_exploration_blind": False,
            "prospectively_sealed_from_notebook_02": True,
        },
        "preparation_contract": {
            "canonical_target_preserved": True,
            "original_target_scale_preserved": True,
            "global_imputation": "none",
            "global_outlier_mutation": "none",
            "global_differencing": "none",
            "global_power_transformation": "none",
            "global_decomposition_or_detrending": "none",
            "decisions": _frame_records(preparation_backtesting_contract),
        },
        "backtesting_contract": {
            "mode": "expanding_window",
            "initial_training_months": int(
                backtesting_schedule.iloc[0]["training_observations"]
            ) if fold_count else None,
            "forecast_horizon": horizon,
            "origin_step_months": horizon,
            "fold_count": fold_count,
            "validation_forecast_count": forecast_count,
            "validation_targets_overlap": False,
            "schedule": _frame_records(backtesting_schedule),
        },
        "evaluation_contract": {
            "primary_metric": metric_primary,
            "secondary_metrics": list(metric_secondary),
            "seasonal_mase_period": mase_period,
            "horizon_wise_diagnostic": "mae_h1_to_h12",
            "percentage_error_metrics": "excluded",
            "primary_baseline": baseline_primary,
            "secondary_baseline": baseline_secondary,
            "point_forecasts_required": horizon,
            "forecast_intervals_required": False,
        },
        "evaluation_boundary": {
            "contract": _frame_records(evaluation_boundary_contract),
            "risk_register": _frame_records(leakage_risk_register),
        },
        "exploratory_synthesis": {
            "key_insights": _frame_records(key_exploratory_insights),
        },
        "open_reviews": open_reviews,
        "continuation": {
            "next_steps": _frame_records(next_steps),
            "expected_outputs": _frame_records(expected_outputs),
            "fresh_kernel_required": True,
            "notebook_02_must_revalidate_source": True,
            "notebook_03_must_consume_frozen_preparation_handoff": True,
        },
        "readiness": {
            "notebook_01_complete": len(issues) == 0,
            "deterministic_preparation_ready": len(issues) == 0,
            "split_execution_ready": False,
            "temporal_backtesting_ready": len(issues) == 0,
            "model_selection_ready": False,
            "model_selection_waits_for_notebook_02": True,
        },
    }

    return ForecastingExplorationHandoffReport(
        payload=_json_safe(payload),
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
        next_steps=next_steps,
        expected_outputs=expected_outputs,
    )


def load_and_validate_forecasting_exploration_handoff(
    path: str | Path,
    *,
    expected_dataset_slug: str | None = None,
    expected_source_reference: str | None = None,
) -> dict[str, Any]:
    """Load and validate the persisted forecasting exploration handoff."""
    handoff_path = Path(path).expanduser().resolve()
    if not handoff_path.is_file():
        raise ForecastingExplorationHandoffError(
            f"Exploration handoff artifact is missing: {handoff_path.as_posix()}."
        )

    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ForecastingExplorationHandoffError(
            f"Exploration handoff JSON is invalid: {exc}."
        ) from exc

    required = {
        "schema_version",
        "artifact_type",
        "dataset_slug",
        "source",
        "prediction_contract",
        "feature_contract",
        "temporal_contract",
        "preparation_contract",
        "backtesting_contract",
        "evaluation_contract",
        "evaluation_boundary",
        "continuation",
        "readiness",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ForecastingExplorationHandoffError(
            f"Exploration handoff is missing required fields: {missing!r}."
        )
    if payload["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ForecastingExplorationHandoffError(
            "Unexpected exploration handoff schema version: "
            f"{payload['schema_version']!r}."
        )
    if payload["artifact_type"] != HANDOFF_ARTIFACT_TYPE:
        raise ForecastingExplorationHandoffError(
            "Unexpected exploration handoff artifact type: "
            f"{payload['artifact_type']!r}."
        )
    if expected_dataset_slug is not None and payload["dataset_slug"] != expected_dataset_slug:
        raise ForecastingExplorationHandoffError(
            "Exploration handoff dataset slug mismatch: "
            f"{payload['dataset_slug']!r} != {expected_dataset_slug!r}."
        )
    source = payload["source"]
    if (
        expected_source_reference is not None
        and source.get("source_reference") != expected_source_reference
    ):
        raise ForecastingExplorationHandoffError(
            "Exploration handoff source reference mismatch."
        )

    prediction = payload["prediction_contract"]
    if prediction.get("problem_type") != "time_series_forecasting":
        raise ForecastingExplorationHandoffError(
            "Persisted handoff is not a time-series forecasting contract."
        )
    if prediction.get("forecasting_mode") != "univariate":
        raise ForecastingExplorationHandoffError(
            "Persisted handoff is not a univariate forecasting contract."
        )
    if prediction.get("target_classes") not in ([], None):
        raise ForecastingExplorationHandoffError(
            "Forecasting handoff must not declare target classes."
        )
    if int(prediction.get("source_exogenous_predictors", -1)) != 0:
        raise ForecastingExplorationHandoffError(
            "Univariate source contract must declare zero exogenous predictors."
        )

    temporal = payload["temporal_contract"]
    horizon = int(prediction.get("forecast_horizon", 0))
    if horizon <= 0 or int(temporal.get("final_holdout_observations", 0)) != horizon:
        raise ForecastingExplorationHandoffError(
            "Forecast horizon and final-holdout length are inconsistent."
        )
    if temporal.get("prospectively_sealed_from_notebook_02") is not True:
        raise ForecastingExplorationHandoffError(
            "Final holdout is not prospectively sealed from Notebook 02."
        )

    backtesting = payload["backtesting_contract"]
    if backtesting.get("mode") != "expanding_window":
        raise ForecastingExplorationHandoffError(
            "Unexpected forecasting backtesting mode."
        )
    if int(backtesting.get("forecast_horizon", 0)) != horizon:
        raise ForecastingExplorationHandoffError(
            "Backtesting horizon does not match the forecast contract."
        )
    if backtesting.get("validation_targets_overlap") is not False:
        raise ForecastingExplorationHandoffError(
            "Frozen forecasting validation targets must be non-overlapping."
        )

    readiness = payload["readiness"]
    if readiness.get("notebook_01_complete") is not True:
        raise ForecastingExplorationHandoffError(
            "Exploration handoff does not declare Notebook 01 complete."
        )
    if readiness.get("deterministic_preparation_ready") is not True:
        raise ForecastingExplorationHandoffError(
            "Exploration handoff does not authorize deterministic preparation."
        )
    if readiness.get("temporal_backtesting_ready") is not True:
        raise ForecastingExplorationHandoffError(
            "Exploration handoff does not authorize temporal backtesting."
        )
    if readiness.get("split_execution_ready") is not False:
        raise ForecastingExplorationHandoffError(
            "Forecasting handoff must not authorize snapshot split execution."
        )
    if readiness.get("model_selection_ready") is not False:
        raise ForecastingExplorationHandoffError(
            "Notebook 01 must not authorize model selection before Notebook 02."
        )

    return deepcopy(payload)
