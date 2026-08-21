"""Fail-closed finalization for the frozen Nottingham forecasting model.

The historical filename ``final-pipeline.joblib`` contains a frozen forecasting
model artifact, not a :class:`sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.regression.linear_model import OLS

from scripts.forecasting_model_selection import (
    frozen_specification_catalog,
    load_and_validate_forecasting_model_selection_handoff,
)
from scripts.forecasting_preparation import seasonal_mase_scale


MODEL_SCHEMA = "forecasting-frozen-model.v1"
MANIFEST_SCHEMA = "forecasting-final-model-manifest.v1"
EVIDENCE_SCHEMA = "forecasting-final-test-evidence.v1"
BUNDLE_SCHEMA = "forecasting-inference-bundle.v1"
HANDOFF_SCHEMA = "forecasting-final-model-handoff.v1"
FINAL_FILENAMES = (
    "final-pipeline.joblib", "final-model-manifest.json",
    "final-test-evidence.json", "inference-bundle.json",
    "final-model-handoff.json",
)
DESIGN_COLUMNS = ("intercept", "linear_time_trend") + tuple(
    f"month_{month:02d}" for month in range(2, 13)
)

# Frozen semantic contracts for the Notebook 04 -> Notebook 05 boundary. These
# constants are the single source of truth for every static (winner-independent)
# field asserted by the auxiliary loaders below; cross-artifact (winner-dependent)
# fields are instead derived from the authenticated upstream contract/model/bundle
# by the main handoff loader, never hardcoded here.
_FORECASTING_IDENTITY = {
    "dataset_slug": "nottem", "problem_type": "time_series_forecasting",
    "forecasting_mode": "univariate",
}
_MODEL_SELECTION_HANDOFF_SCHEMA = "forecasting-model-selection-handoff.v1"
_TRAINING_SCOPE_LABEL = "full development 1920-01 -> 1938-12"

NOTEBOOK_05_INSTRUCTIONS = (
    "validate final-model handoff", "validate inference bundle", "verify model SHA before joblib load",
    "load frozen model without refit", "do not reopen model selection", "do not execute final training",
    "do not use final holdout as demo input", "use a valid monthly historical series",
    "generate 12 future forecasts", "demonstrate deterministic repeatability",
    "keep the model frozen", "declare operational_modeling_ready=false",
)

_MANIFEST_TRAINING_STATIC = {
    "scope": _TRAINING_SCOPE_LABEL, "start": "1920-01", "end": "1938-12",
    "observations": 228, "final_fit_count": 1,
}
_MANIFEST_TARGET_STATIC = {
    "column": "temperature", "semantics": "Monthly average air temperature at Nottingham Castle",
    "unit": "degrees Fahrenheit", "frequency": "M",
}
_MANIFEST_MODEL_STATE_STATIC = {
    "model_artifact_format": "joblib", "model_artifact_kind": "frozen_forecasting_model",
    "fitted": True, "frozen": True,
}
_MANIFEST_FORECAST_STATIC = {"horizon": 12}
_MANIFEST_HOLDOUT_STATIC = {
    "model_frozen_before_holdout_open": True, "holdout_open_count": 1,
    "start": "1939-01", "end": "1939-12", "row_count": 12,
}
_MANIFEST_FINAL_EVALUATION_STATIC = {
    "completed": True, "evaluation_count": 1, "used_for_adjustment": False,
    "used_for_retuning": False, "used_for_model_selection": False,
}
_MANIFEST_READINESS_STATIC = {
    "final_model_trained": True, "model_artifact_materialized": True,
    "final_holdout_evaluated": True, "inference_bundle_materialized": True,
    "final_model_handoff_ready": True, "inference_demo_ready": True,
    "operational_modeling_ready": False,
}

_EVIDENCE_PERIOD_STATIC = {"start": "1939-01", "end": "1939-12"}
_EVIDENCE_METRIC_FORMULAS = {
    "mae": "mean(abs(y_pred-y_true))", "rmse": "sqrt(mean((y_pred-y_true)^2))",
    "seasonal_mase_12": "mean(abs_error/full_development_seasonal_mase_denominator)",
}

_BUNDLE_TARGET_STATIC = {
    "column": "temperature", "semantics": "Monthly average air temperature at Nottingham Castle",
    "unit": "degrees Fahrenheit", "dtype": "numeric finite",
}
_BUNDLE_TIME_STATIC = {
    "expected_frequency": "M", "seasonal_period": 12, "forecast_horizon": 12,
    "forecast_origin": "last period of supplied history",
}
_BUNDLE_READINESS_STATIC = {
    "inference_demo_ready": True, "final_model_trained": True,
    "model_frozen": True, "operational_modeling_ready": False,
}
_BUNDLE_MODEL_ARTIFACT_STATIC = {
    "model_artifact_format": "joblib", "model_artifact_kind": "frozen_forecasting_model",
}
_BUNDLE_SECURITY_STATIC = {
    "verify_joblib_sha_before_load": True, "trusted_local_artifact_only": True,
}

_HANDOFF_TRAINING_STATIC = {"scope": _TRAINING_SCOPE_LABEL, "rows": 228, "fit_count": 1}
_HANDOFF_FINAL_EVALUATION_STATIC = {
    "origin": "1938-12", "period": {"start": "1939-01", "end": "1939-12"},
    "horizon": 12, "evaluation_count": 1,
}
_HANDOFF_BOUNDARY_STATIC = {
    "model_frozen_before_holdout_open": True, "holdout_evaluated": True,
    "holdout_evaluation_count": 1, "holdout_used_for_adjustment": False,
    "no_retuning_after_final_evaluation": True, "no_model_change_after_final_evaluation": True,
}


class ForecastingFinalizationError(RuntimeError):
    pass


class ForecastingFinalizationContractError(ForecastingFinalizationError):
    pass


class ForecastingHoldoutAccessError(ForecastingFinalizationError):
    pass


class ForecastingDuplicateFinalEvaluationError(ForecastingFinalizationError):
    pass


class ForecastingArtifactConflictError(ForecastingFinalizationError):
    pass


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ForecastingArtifactConflictError(f"Referenced artifact is unreadable: {path}.") from exc
    return hashlib.sha256(data).hexdigest()


def semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "semantic_fingerprint"}
    return hashlib.sha256(_canonical(clean)).hexdigest()


def _with_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["semantic_fingerprint"] = semantic_fingerprint(result)
    return result


def _confined(root: Path, path: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ForecastingArtifactConflictError("Artifact path escapes project root.") from exc
    return candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ForecastingArtifactConflictError("Artifact path escapes project root.") from exc


def _same(a: Any, b: Any, label: str, tol: float = 1e-12) -> None:
    if isinstance(b, Mapping):
        if not isinstance(a, Mapping) or set(a) != set(b):
            raise ForecastingFinalizationContractError(f"{label} keys differ.")
        for key in b:
            _same(a[key], b[key], f"{label}.{key}", tol)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            raise ForecastingFinalizationContractError(f"{label} differs.")
        for i, item in enumerate(b):
            _same(a[i], item, f"{label}[{i}]", tol)
    elif isinstance(b, (float, np.floating)):
        if not isinstance(a, (int, float)) or not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
            raise ForecastingFinalizationContractError(f"{label} differs.")
    elif a != b:
        raise ForecastingFinalizationContractError(f"{label} differs: {a!r} != {b!r}.")


def _require_fields(container: Any, expected: Mapping[str, Any], label: str) -> None:
    """Assert that each of ``expected``'s fields is present with that exact value.

    Unlike ``_same``, this checks a named subset of ``container``'s fields rather
    than an exact key set, so static (winner-independent) fields can be asserted
    without also constraining the dynamic (winner/lineage-dependent) fields that
    live alongside them in the same JSON object.
    """
    if not isinstance(container, Mapping):
        raise ForecastingFinalizationContractError(f"{label} must be an object.")
    for key, value in expected.items():
        _same(container.get(key), value, f"{label}.{key}")


@dataclass(frozen=True, slots=True)
class FrozenForecastingFinalizationContract:
    dataset_slug: str
    selected_candidate_id: str
    selected_role: str
    selected_family: str
    selected_specification: dict[str, Any]
    development_path: str
    development_sha256: str
    holdout_path: str
    holdout_sha256: str
    reference_metrics: dict[str, float]
    model_selection_handoff_path: str
    model_selection_handoff_sha256: str
    model_selection_semantic_fingerprint: str
    preparation_reference: dict[str, Any]
    training_start: str = "1920-01"
    training_end: str = "1938-12"
    training_observations: int = 228
    holdout_start: str = "1939-01"
    holdout_end: str = "1939-12"
    holdout_observations: int = 12
    forecast_horizon: int = 12


def validate_forecasting_finalization_contract(
    handoff: Mapping[str, Any], *, handoff_path: str | Path,
    project_root: str | Path | None = None,
) -> FrozenForecastingFinalizationContract:
    readiness = handoff.get("readiness", {})
    required_true = ("final_model_training_ready", "final_holdout_sealed",
                     "selected_specification_frozen", "model_selection_handoff_reloadable")
    required_false = ("final_model_trained", "model_artifact_materialized",
                      "model_bundle_materialized", "final_holdout_evaluated",
                      "operational_modeling_ready")
    if any(readiness.get(k) is not True for k in required_true) or any(
        readiness.get(k) is not False for k in required_false
    ):
        raise ForecastingFinalizationContractError("Upstream readiness does not authorize finalization.")
    if (handoff.get("dataset_slug"), handoff.get("problem_type"), handoff.get("forecasting_mode")) != (
        "nottem", "time_series_forecasting", "univariate"
    ):
        raise ForecastingFinalizationContractError("Forecasting identity differs.")
    spec = handoff.get("selected_specification")
    catalog = {x["candidate_id"]: x for x in frozen_specification_catalog()}
    winner = handoff.get("selected_candidate_id")
    if winner not in catalog or spec != catalog[winner] or handoff.get("selection", {}).get("selected_candidate_id") != winner:
        raise ForecastingFinalizationContractError("Selected specification diverges from authenticated catalog/replay.")
    if winner != "seasonal_trend_ols":
        raise ForecastingFinalizationContractError(f"Unsupported frozen winner: {winner!r}.")
    if handoff.get("selected_family") != "DeterministicSeasonalTrendOLS" or handoff.get("selected_role") != "candidate":
        raise ForecastingFinalizationContractError("Selected model identity differs.")
    instructions = handoff.get("final_training_instructions", {})
    expected_instructions = {
        "notebook": "notebooks/04_final_model_and_bundle.ipynb",
        "training_scope": "full development 1920-01 -> 1938-12",
        "reconstruct_exact_selected_specification": True,
        "freeze_before_final_holdout_open": True,
        "final_forecast_horizon": 12,
        "final_evaluation_period": "1939-01 -> 1939-12",
        "open_holdout_only_after_final_spec_fit": True,
        "evaluate_holdout_once": True,
        "do_not_retune": True,
        "do_not_change_candidate": True,
        "do_not_change_hyperparameters": True,
        "do_not_change_transform_policy": True,
        "target_scale": "original degrees Fahrenheit",
    }
    for key, value in expected_instructions.items():
        if instructions.get(key) != value:
            raise ForecastingFinalizationContractError(f"Final-training instruction {key!r} differs.")
    development = handoff.get("development", {})
    holdout = handoff.get("sealed_final_holdout_metadata_only", {})
    if (development.get("start"), development.get("end"), development.get("row_count")) != ("1920-01", "1938-12", 228):
        raise ForecastingFinalizationContractError("Development scope differs.")
    if (holdout.get("start"), holdout.get("end"), holdout.get("row_count"), holdout.get("sealed"), holdout.get("evaluated")) != ("1939-01", "1939-12", 12, True, False):
        raise ForecastingFinalizationContractError("Sealed holdout metadata differs.")
    metrics = handoff.get("evaluation", {}).get("selected_pooled_metrics", {})
    if set(metrics) != {"mae", "rmse", "seasonal_mase_12"}:
        raise ForecastingFinalizationContractError("Selected reference metrics differ.")
    handoff_file = Path(handoff_path)
    if project_root is not None and not handoff_file.is_absolute():
        handoff_file = _confined(Path(project_root), handoff_file)
    return FrozenForecastingFinalizationContract(
        dataset_slug="nottem", selected_candidate_id=winner,
        selected_role=handoff["selected_role"], selected_family=handoff["selected_family"],
        selected_specification=json.loads(json.dumps(spec)),
        development_path=development["path"], development_sha256=development["sha256"],
        holdout_path=holdout["path"], holdout_sha256=holdout["sha256"],
        reference_metrics={k: float(v) for k, v in metrics.items()},
        model_selection_handoff_path=str(handoff_path),
        model_selection_handoff_sha256=sha256_file(handoff_file),
        model_selection_semantic_fingerprint=handoff["semantic_fingerprint"],
        preparation_reference=dict(handoff["upstream_preparation_handoff"]),
    )


@dataclass(slots=True)
class ForecastingFinalizationGuard:
    final_fit_count: int = 0
    model_frozen: bool = False
    holdout_open_count: int = 0
    final_forecast_call_count: int = 0
    final_evaluation_count: int = 0
    post_holdout_fit_count: int = 0
    audit_log: list[str] = field(default_factory=list)

    def register_fit(self) -> None:
        if self.holdout_open_count:
            self.post_holdout_fit_count += 1
            raise ForecastingFinalizationContractError("Fit is forbidden after holdout access.")
        if self.final_fit_count:
            raise ForecastingFinalizationContractError("Final model may be fitted exactly once.")
        self.final_fit_count = 1
        self.audit_log.append("fit #1 completed")

    def mark_frozen(self) -> None:
        if self.final_fit_count != 1 or self.holdout_open_count:
            raise ForecastingFinalizationContractError("Freeze boundary prerequisites failed.")
        self.model_frozen = True
        self.audit_log.append("model frozen")

    def authorize_holdout_open(self) -> None:
        if not self.model_frozen or self.final_fit_count != 1:
            raise ForecastingHoldoutAccessError("Holdout target access is forbidden before trusted model freeze.")
        if self.holdout_open_count:
            raise ForecastingDuplicateFinalEvaluationError("Final holdout may be opened once.")
        self.holdout_open_count = 1
        self.audit_log.append("holdout opened")

    def register_forecast(self) -> None:
        if not self.model_frozen:
            raise ForecastingFinalizationContractError("Final forecast requires frozen model.")
        if self.final_forecast_call_count:
            raise ForecastingDuplicateFinalEvaluationError("Final forecast may be generated once.")
        self.final_forecast_call_count = 1
        self.audit_log.append("final forecast generated")

    def register_evaluation(self) -> None:
        if self.holdout_open_count != 1 or self.final_forecast_call_count != 1:
            raise ForecastingFinalizationContractError("Evaluation prerequisites failed.")
        if self.final_evaluation_count:
            raise ForecastingDuplicateFinalEvaluationError("Final evaluation may occur once.")
        self.final_evaluation_count = 1
        self.audit_log.append("final evaluation #1 completed")

    def reject_mutation(self, kind: str) -> None:
        raise ForecastingFinalizationContractError(f"Frozen {kind} cannot be changed.")


def _design(index: pd.PeriodIndex, start_ordinal: int) -> np.ndarray:
    t = np.asarray(index.asi8 - start_ordinal, dtype=float)
    return np.column_stack([np.ones(len(index)), t] + [
        (index.month == month).astype(float) for month in range(2, 13)
    ])


@dataclass(frozen=True, slots=True)
class FrozenForecastingModel:
    schema_version: str
    model_kind: str
    dataset_slug: str
    candidate_id: str
    family: str
    selected_specification: dict[str, Any]
    training_start: str
    training_end: str
    training_observations: int
    training_sha256: str
    training_start_ordinal: int
    design_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    coefficient_count: int
    target_column: str
    target_unit: str
    frequency: str
    forecast_horizon: int
    fitted: bool
    frozen: bool
    final_fit_count: int
    model_state_semantic_fingerprint: str

    def state_descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "model_kind": self.model_kind,
            "dataset_slug": self.dataset_slug, "candidate_id": self.candidate_id,
            "family": self.family, "selected_specification": self.selected_specification,
            "training_scope": {"start": self.training_start, "end": self.training_end},
            "training_observations": self.training_observations,
            "training_sha256": self.training_sha256,
            "training_start_ordinal": self.training_start_ordinal,
            "design_columns": list(self.design_columns),
            "coefficients": list(self.coefficients), "target_column": self.target_column,
            "target_unit": self.target_unit, "frequency": self.frequency,
            "forecast_horizon": self.forecast_horizon,
        }

    def forecast_periods(self, periods: pd.PeriodIndex) -> pd.Series:
        periods = pd.PeriodIndex(periods, freq="M", name="period")
        if len(periods) != self.forecast_horizon or not periods.equals(
            pd.period_range(periods[0], periods=self.forecast_horizon, freq="M", name="period")
        ):
            raise ForecastingFinalizationContractError("Forecast index must contain 12 contiguous monthly periods.")
        values = _design(periods, self.training_start_ordinal) @ np.asarray(self.coefficients)
        if not np.isfinite(values).all():
            raise ForecastingFinalizationContractError("Forecast contains non-finite values.")
        return pd.Series(values, index=periods, name="forecast")

    def predict_from_history(self, history: pd.DataFrame) -> pd.DataFrame:
        checked = validate_inference_history(history, training_end=self.training_end)
        future = pd.period_range(checked.index[-1] + 1, periods=12, freq="M", name="period")
        pred = self.forecast_periods(future)
        return pd.DataFrame({"period": future.astype(str), "forecast": pred.to_numpy(float)})


def validate_full_development(path: str | Path, *, expected_sha256: str) -> pd.Series:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ForecastingFinalizationContractError("Development byte SHA-256 mismatch.")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["period", "temperature"] or len(frame) != 228:
        raise ForecastingFinalizationContractError("Development columns/row count differ.")
    try:
        index = pd.PeriodIndex(frame["period"].astype(str), freq="M", name="period")
    except Exception as exc:
        raise ForecastingFinalizationContractError("Development periods are invalid.") from exc
    expected = pd.period_range("1920-01", "1938-12", freq="M", name="period")
    values = pd.to_numeric(frame["temperature"], errors="coerce").to_numpy(float)
    if not index.equals(expected) or index.has_duplicates or not np.isfinite(values).all():
        raise ForecastingFinalizationContractError("Development chronology/values differ.")
    return pd.Series(values, index=index, name="temperature")


def reconstruct_and_fit_selected_forecasting_model(
    contract: FrozenForecastingFinalizationContract, development: pd.Series,
    guard: ForecastingFinalizationGuard,
) -> FrozenForecastingModel:
    if contract.selected_candidate_id != "seasonal_trend_ols":
        raise ForecastingFinalizationContractError("Only the authenticated OLS winner is supported.")
    guard.register_fit()
    start = int(development.index[0].ordinal)
    coefficients = tuple(float(x) for x in OLS(
        development.to_numpy(float), _design(development.index, start)
    ).fit().params)
    descriptor = {
        "schema_version": MODEL_SCHEMA, "model_kind": "deterministic_seasonal_trend_ols",
        "dataset_slug": contract.dataset_slug, "candidate_id": contract.selected_candidate_id,
        "family": contract.selected_family, "selected_specification": contract.selected_specification,
        "training_scope": {"start": contract.training_start, "end": contract.training_end},
        "training_observations": len(development), "training_sha256": contract.development_sha256,
        "training_start_ordinal": start, "design_columns": list(DESIGN_COLUMNS),
        "coefficients": list(coefficients), "target_column": "temperature",
        "target_unit": "degrees Fahrenheit", "frequency": "M", "forecast_horizon": 12,
    }
    fingerprint = hashlib.sha256(_canonical(descriptor)).hexdigest()
    return FrozenForecastingModel(
        MODEL_SCHEMA, "deterministic_seasonal_trend_ols", contract.dataset_slug,
        contract.selected_candidate_id, contract.selected_family,
        contract.selected_specification, contract.training_start, contract.training_end,
        len(development), contract.development_sha256, start, DESIGN_COLUMNS,
        coefficients, len(coefficients), "temperature", "degrees Fahrenheit", "M", 12,
        True, True, guard.final_fit_count, fingerprint,
    )


def freeze_forecasting_model(model: FrozenForecastingModel) -> FrozenForecastingModel:
    validate_frozen_forecasting_model(model)
    return model


def validate_frozen_forecasting_model(model: Any) -> FrozenForecastingModel:
    if type(model) is not FrozenForecastingModel:
        raise ForecastingFinalizationContractError("Unexpected joblib object type.")
    if (model.schema_version, model.model_kind, model.dataset_slug, model.candidate_id,
        model.family, model.coefficient_count, len(model.coefficients), model.design_columns,
        model.fitted, model.frozen, model.final_fit_count) != (
        MODEL_SCHEMA, "deterministic_seasonal_trend_ols", "nottem", "seasonal_trend_ols",
        "DeterministicSeasonalTrendOLS", 13, 13, DESIGN_COLUMNS, True, True, 1
    ):
        raise ForecastingFinalizationContractError("Frozen model internal contract differs.")
    expected = hashlib.sha256(_canonical(model.state_descriptor())).hexdigest()
    if expected != model.model_state_semantic_fingerprint or not np.isfinite(model.coefficients).all():
        raise ForecastingFinalizationContractError("Frozen model semantic state differs.")
    return model


def serialize_frozen_forecasting_model_to_staging(model: FrozenForecastingModel, path: str | Path) -> str:
    validate_frozen_forecasting_model(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return sha256_file(path)


def validate_serialized_forecasting_model(
    path: str | Path, *, expected_byte_sha256: str,
    expected_state_fingerprint: str,
) -> FrozenForecastingModel:
    if sha256_file(path) != expected_byte_sha256:
        raise ForecastingArtifactConflictError("Model byte SHA-256 mismatch; refusing joblib.load.")
    model = validate_frozen_forecasting_model(joblib.load(path))
    if model.model_state_semantic_fingerprint != expected_state_fingerprint:
        raise ForecastingArtifactConflictError("Model state semantic fingerprint mismatch.")
    return model


def open_final_holdout_once(
    path: str | Path, *, expected_sha256: str,
    guard: ForecastingFinalizationGuard,
) -> pd.Series:
    guard.authorize_holdout_open()
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ForecastingFinalizationContractError("Final holdout SHA mismatch before parse.")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["period", "temperature"] or len(frame) != 12:
        raise ForecastingFinalizationContractError("Final holdout columns/rows differ.")
    try:
        index = pd.PeriodIndex(frame["period"].astype(str), freq="M", name="period")
    except Exception as exc:
        raise ForecastingFinalizationContractError("Final holdout periods are invalid.") from exc
    expected = pd.period_range("1939-01", "1939-12", freq="M", name="period")
    values = pd.to_numeric(frame["temperature"], errors="coerce").to_numpy(float)
    if not index.equals(expected) or index.has_duplicates or not np.isfinite(values).all():
        raise ForecastingFinalizationContractError("Final holdout chronology/values differ.")
    return pd.Series(values, index=index, name="temperature")


def generate_final_forecast_once(
    model: FrozenForecastingModel, *, origin: str,
    guard: ForecastingFinalizationGuard,
) -> pd.Series:
    if origin != model.training_end:
        raise ForecastingFinalizationContractError("Final forecast origin differs from training end.")
    guard.register_forecast()
    future = pd.period_range(pd.Period(origin, freq="M") + 1, periods=12, freq="M", name="period")
    return model.forecast_periods(future)


def evaluate_final_forecast_once(
    predictions: pd.Series, holdout: pd.Series, *, development: pd.Series,
    guard: ForecastingFinalizationGuard,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    if not predictions.index.equals(holdout.index):
        raise ForecastingFinalizationContractError("Forecast and holdout periods differ.")
    guard.register_evaluation()
    denominator = float(seasonal_mase_scale(development, period=12))
    rows = []
    for horizon, period in enumerate(predictions.index, 1):
        true, pred = float(holdout.loc[period]), float(predictions.loc[period])
        error = pred - true
        rows.append({"horizon": horizon, "forecast_period": str(period), "y_true": true,
                     "y_pred": pred, "error": error, "abs_error": abs(error),
                     "squared_error": error * error, "seasonal_mase_scale": denominator,
                     "scaled_abs_error": abs(error) / denominator})
    metrics = _metrics_from_rows(rows)
    return rows, metrics, denominator


def _metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {"mae": float(np.mean([r["abs_error"] for r in rows])),
            "rmse": float(np.sqrt(np.mean([r["squared_error"] for r in rows]))),
            "seasonal_mase_12": float(np.mean([r["scaled_abs_error"] for r in rows]))}


def build_forecasting_final_test_evidence(
    contract: FrozenForecastingFinalizationContract, model: FrozenForecastingModel,
    rows: list[dict[str, Any]], metrics: Mapping[str, float], denominator: float,
    guard: ForecastingFinalizationGuard,
) -> dict[str, Any]:
    deltas = {key: float(metrics[key] - contract.reference_metrics[key]) for key in metrics}
    return _with_fingerprint({
        "schema_version": EVIDENCE_SCHEMA, "artifact_type": "final_test_evidence",
        "dataset_slug": "nottem", "problem_type": "time_series_forecasting",
        "forecasting_mode": "univariate", "final_forecast_origin": "1938-12",
        "final_evaluation_period": {"start": "1939-01", "end": "1939-12"},
        "forecast_horizon": 12, "holdout_byte_sha256": contract.holdout_sha256,
        "model_state_semantic_fingerprint": model.model_state_semantic_fingerprint,
        "final_seasonal_mase_denominator": denominator,
        "metric_formulas": {"mae": "mean(abs(y_pred-y_true))",
                            "rmse": "sqrt(mean((y_pred-y_true)^2))",
                            "seasonal_mase_12": "mean(abs_error/full_development_seasonal_mase_denominator)"},
        "forecasts": rows, "metrics": dict(metrics),
        "model_selection_reference_metrics": contract.reference_metrics,
        "generalization_deltas": deltas, "final_fit_count": guard.final_fit_count,
        "final_forecast_call_count": guard.final_forecast_call_count,
        "final_evaluation_count": guard.final_evaluation_count,
        "model_frozen_before_holdout_open": True,
        "holdout_used_for_adjustment": False, "holdout_used_for_retuning": False,
        "holdout_used_for_model_selection": False, "candidate_changed_after_holdout": False,
        "specification_changed_after_holdout": False,
    })


def _input_contract() -> dict[str, Any]:
    return {
        "kind": "monthly_univariate_history", "representation": "period + temperature series/frame",
        "required_columns": ["period", "temperature"],
        "period_requirements": {"monthly": True, "monotonic_increasing": True,
                                "unique": True, "contiguous": True},
        "temperature_requirements": {"dtype": "numeric finite", "no_missing": True},
        "exogenous_predictors": "none", "refit_on_input": False,
        "history_required": True,
        "history_role": "establish forecast origin and validate monthly chronology",
        "historical_values_used_to_refit": False,
        "historical_values_used_to_update_coefficients": False,
        "forecast_origin": "last supplied historical period",
        "minimum_history_observations": 1, "supplied_history_end_minimum": "1938-12",
        "post_training_history_limitation": "values establish origin/context only; frozen coefficients are not updated",
    }


def _output_contract() -> dict[str, Any]:
    return {"row_count": 12, "columns": ["period", "forecast"],
            "periods": "origin+1 through origin+12", "forecast_dtype": "numeric finite",
            "unit": "degrees Fahrenheit"}


def build_forecasting_inference_bundle(
    contract: FrozenForecastingFinalizationContract, model: FrozenForecastingModel,
    *, model_path: str, model_sha256: str,
) -> dict[str, Any]:
    return _with_fingerprint({
        "schema_version": BUNDLE_SCHEMA, "artifact_type": "inference_bundle",
        "dataset_slug": "nottem", "problem_type": "time_series_forecasting", "forecasting_mode": "univariate",
        "model": {"selected_candidate_id": contract.selected_candidate_id,
                  "selected_family": contract.selected_family,
                  "selected_specification": contract.selected_specification,
                  "model_artifact_path": model_path, "model_artifact_sha256": model_sha256,
                  "model_state_semantic_fingerprint": model.model_state_semantic_fingerprint,
                  "model_artifact_format": "joblib", "model_artifact_kind": "frozen_forecasting_model"},
        "target": {"column": "temperature", "semantics": "Monthly average air temperature at Nottingham Castle",
                   "unit": "degrees Fahrenheit", "dtype": "numeric finite"},
        "time": {"expected_frequency": "M", "seasonal_period": 12, "forecast_horizon": 12,
                 "forecast_origin": "last period of supplied history"},
        "input_contract": _input_contract(), "output_contract": _output_contract(),
        "security": {"verify_joblib_sha_before_load": True, "trusted_local_artifact_only": True,
                     "warning": "joblib is unsafe for untrusted bytes"},
        "readiness": {"inference_demo_ready": True, "final_model_trained": True,
                      "model_frozen": True, "operational_modeling_ready": False},
    })


def build_forecasting_final_model_manifest(
    contract: FrozenForecastingFinalizationContract, model: FrozenForecastingModel,
    guard: ForecastingFinalizationGuard, *, model_path: str, model_sha256: str,
) -> dict[str, Any]:
    return _with_fingerprint({
        "schema_version": MANIFEST_SCHEMA, "artifact_type": "final_model_manifest",
        "dataset_slug": "nottem", "problem_type": "time_series_forecasting", "forecasting_mode": "univariate",
        "upstream": {"model_selection_handoff": {"path": contract.model_selection_handoff_path,
                    "schema_version": "forecasting-model-selection-handoff.v1",
                    "sha256": contract.model_selection_handoff_sha256,
                    "semantic_fingerprint": contract.model_selection_semantic_fingerprint},
                    "preparation_lineage": contract.preparation_reference},
        "selected_model": {"selected_candidate_id": contract.selected_candidate_id,
                           "selected_role": contract.selected_role, "selected_family": contract.selected_family,
                           "selected_specification": contract.selected_specification},
        "training": {"scope": _TRAINING_SCOPE_LABEL, "development_path": contract.development_path,
                     "development_sha256": contract.development_sha256, "start": "1920-01", "end": "1938-12",
                     "observations": 228, "final_fit_count": guard.final_fit_count},
        "target": {"column": "temperature", "semantics": "Monthly average air temperature at Nottingham Castle",
                   "unit": "degrees Fahrenheit", "frequency": "M"},
        "model_state": {"model_artifact_path": model_path, "model_artifact_format": "joblib",
                        "model_artifact_kind": "frozen_forecasting_model", "model_artifact_byte_sha256": model_sha256,
                        "model_state_semantic_fingerprint": model.model_state_semantic_fingerprint,
                        "fitted": True, "frozen": True},
        "forecast": {"horizon": 12, "multi_step_strategy": contract.selected_specification["multi_step_strategy"]},
        "holdout_boundary": {"model_frozen_before_holdout_open": True, "holdout_open_count": 1,
                             "path": contract.holdout_path, "sha256": contract.holdout_sha256,
                             "start": "1939-01", "end": "1939-12", "row_count": 12},
        "final_evaluation": {"completed": True, "evaluation_count": 1, "used_for_adjustment": False,
                             "used_for_retuning": False, "used_for_model_selection": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "statsmodels": statsmodels.__version__, "joblib": joblib.__version__},
        "readiness": {"final_model_trained": True, "model_artifact_materialized": True,
                      "final_holdout_evaluated": True, "inference_bundle_materialized": True,
                      "final_model_handoff_ready": True, "inference_demo_ready": True,
                      "operational_modeling_ready": False},
    })


def _reference(path: str, sha: str, fingerprint: str) -> dict[str, str]:
    return {"path": path, "byte_sha256": sha, "semantic_fingerprint": fingerprint}


def build_forecasting_final_model_handoff(
    contract: FrozenForecastingFinalizationContract, model: FrozenForecastingModel,
    evidence: Mapping[str, Any], bundle: Mapping[str, Any], references: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return _with_fingerprint({
        "schema_version": HANDOFF_SCHEMA, "artifact_type": "final_model_handoff",
        "dataset_slug": "nottem", "problem_type": "time_series_forecasting", "forecasting_mode": "univariate",
        "upstream": {"model_selection_handoff": {"path": contract.model_selection_handoff_path,
                    "sha256": contract.model_selection_handoff_sha256,
                    "semantic_fingerprint": contract.model_selection_semantic_fingerprint}},
        "final_references": dict(references),
        "selected_model": {"candidate_id": contract.selected_candidate_id, "role": contract.selected_role,
                           "family": contract.selected_family, "selected_specification": contract.selected_specification},
        "training": {"scope": _TRAINING_SCOPE_LABEL, "rows": 228, "fit_count": 1},
        "final_evaluation": {"origin": "1938-12", "period": {"start": "1939-01", "end": "1939-12"},
                             "horizon": 12, "evaluation_count": 1, "metrics": evidence["metrics"]},
        "inference": {"input_contract": bundle["input_contract"], "output_contract": bundle["output_contract"],
                      "model_artifact_reference": references["final-pipeline.joblib"]},
        "boundary": {"model_frozen_before_holdout_open": True, "holdout_evaluated": True,
                     "holdout_evaluation_count": 1, "holdout_used_for_adjustment": False,
                     "no_retuning_after_final_evaluation": True, "no_model_change_after_final_evaluation": True},
        "readiness": {"model_selection_handoff_validated": True, "selected_specification_reconstructed": True,
                      "final_training_scope_validated": True, "final_training_completed": True, "final_fit_count": 1,
                      "final_model_trained": True, "model_frozen": True, "model_artifact_materialized": True,
                      "final_holdout_opened_after_freeze": True, "final_holdout_evaluated": True,
                      "final_holdout_evaluation_count": 1, "final_forecast_call_count": 1,
                      "final_test_evidence_materialized": True, "inference_bundle_materialized": True,
                      "final_model_handoff_reloadable": True, "inference_demo_ready": True,
                      "operational_modeling_ready": False},
        "notebook_05_instructions": list(NOTEBOOK_05_INSTRUCTIONS),
    })


def _validate_json(path: Path, schema: str, artifact_type: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastingArtifactConflictError(f"Invalid JSON artifact: {path}.") from exc
    if payload.get("schema_version") != schema or payload.get("artifact_type") != artifact_type:
        raise ForecastingArtifactConflictError(f"Unexpected schema/type in {path.name}.")
    try:
        expected_fingerprint = semantic_fingerprint(payload)
    except (TypeError, ValueError) as exc:
        raise ForecastingArtifactConflictError(f"Unfingerprintable artifact: {path.name}.") from exc
    if payload.get("semantic_fingerprint") != expected_fingerprint:
        raise ForecastingArtifactConflictError(f"Semantic fingerprint mismatch in {path.name}.")
    return payload


def _validate_final_model_manifest_intrinsic(manifest: Mapping[str, Any]) -> None:
    """Authenticate every manifest field that can be checked without external authority."""
    _require_fields(manifest, _FORECASTING_IDENTITY, "manifest")
    _require_fields(manifest.get("training"), _MANIFEST_TRAINING_STATIC, "manifest.training")
    _require_fields(manifest.get("target"), _MANIFEST_TARGET_STATIC, "manifest.target")
    _require_fields(manifest.get("model_state"), _MANIFEST_MODEL_STATE_STATIC, "manifest.model_state")
    _require_fields(manifest.get("forecast"), _MANIFEST_FORECAST_STATIC, "manifest.forecast")
    _require_fields(manifest.get("holdout_boundary"), _MANIFEST_HOLDOUT_STATIC, "manifest.holdout_boundary")
    _require_fields(manifest.get("final_evaluation"), _MANIFEST_FINAL_EVALUATION_STATIC, "manifest.final_evaluation")
    _require_fields(manifest.get("readiness"), _MANIFEST_READINESS_STATIC, "manifest.readiness")
    upstream = manifest.get("upstream", {})
    if not isinstance(upstream, Mapping) or upstream.get("model_selection_handoff", {}).get("schema_version") != _MODEL_SELECTION_HANDOFF_SCHEMA:
        raise ForecastingFinalizationContractError("manifest.upstream.model_selection_handoff.schema_version differs.")


def load_and_validate_forecasting_final_model_manifest(*, project_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    manifest = _validate_json(_confined(Path(project_root), manifest_path), MANIFEST_SCHEMA, "final_model_manifest")
    _validate_final_model_manifest_intrinsic(manifest)
    return manifest


def load_and_validate_forecasting_final_test_evidence(*, project_root: str | Path, evidence_path: str | Path) -> dict[str, Any]:
    evidence = _validate_json(_confined(Path(project_root), evidence_path), EVIDENCE_SCHEMA, "final_test_evidence")
    _require_fields(evidence, _FORECASTING_IDENTITY, "evidence")
    _require_fields(evidence.get("final_evaluation_period"), _EVIDENCE_PERIOD_STATIC, "evidence.final_evaluation_period")
    _require_fields(evidence.get("metric_formulas"), _EVIDENCE_METRIC_FORMULAS, "evidence.metric_formulas")
    rows = evidence.get("forecasts", [])
    if len(rows) != 12 or [r.get("horizon") for r in rows] != list(range(1, 13)):
        raise ForecastingFinalizationContractError("Final evidence horizons differ.")
    periods = [r.get("forecast_period") for r in rows]
    if periods != [str(x) for x in pd.period_range("1939-01", periods=12, freq="M")]:
        raise ForecastingFinalizationContractError("Final evidence periods differ.")
    denominator = float(evidence.get("final_seasonal_mase_denominator", float("nan")))
    if not (math.isfinite(denominator) and denominator > 0.0):
        raise ForecastingFinalizationContractError("Final evidence seasonal MASE denominator must be finite and positive.")
    for row in rows:
        y_true, y_pred = float(row["y_true"]), float(row["y_pred"])
        if not (math.isfinite(y_true) and math.isfinite(y_pred)):
            raise ForecastingFinalizationContractError("Final evidence y_true/y_pred must be finite.")
        error = y_pred - y_true
        expected = {"error": error, "abs_error": abs(error), "squared_error": error * error,
                    "seasonal_mase_scale": denominator, "scaled_abs_error": abs(error) / denominator}
        for key, value in expected.items():
            if not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ForecastingFinalizationContractError(f"Final evidence {key} differs.")
    _same(evidence["metrics"], _metrics_from_rows(rows), "final metrics")
    expected_deltas = {k: evidence["metrics"][k] - evidence["model_selection_reference_metrics"][k]
                       for k in evidence["metrics"]}
    _same(evidence["generalization_deltas"], expected_deltas, "generalization deltas")
    required = {"final_fit_count": 1, "final_forecast_call_count": 1, "final_evaluation_count": 1,
                "model_frozen_before_holdout_open": True, "holdout_used_for_adjustment": False,
                "holdout_used_for_retuning": False, "holdout_used_for_model_selection": False,
                "candidate_changed_after_holdout": False, "specification_changed_after_holdout": False,
                "final_forecast_origin": "1938-12", "forecast_horizon": 12}
    for key, value in required.items():
        if evidence.get(key) != value:
            raise ForecastingFinalizationContractError(f"Final evidence flag {key} differs.")
    return evidence


def load_and_validate_forecasting_inference_bundle(*, project_root: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    """Authenticate everything about the bundle that does not require the upstream winner.

    The specific selected candidate/family/specification is winner-dependent and is
    therefore NOT authenticated here: a standalone bundle load has no upstream
    authority to check it against. The main handoff loader performs that
    cross-artifact reconciliation against the authenticated Notebook 03 winner.
    """
    bundle = _validate_json(_confined(Path(project_root), bundle_path), BUNDLE_SCHEMA, "inference_bundle")
    _require_fields(bundle, _FORECASTING_IDENTITY, "bundle")
    _require_fields(bundle.get("target"), _BUNDLE_TARGET_STATIC, "bundle.target")
    _require_fields(bundle.get("time"), _BUNDLE_TIME_STATIC, "bundle.time")
    if bundle.get("input_contract") != _input_contract() or bundle.get("output_contract") != _output_contract():
        raise ForecastingFinalizationContractError("Bundle inference contract differs.")
    _require_fields(bundle.get("readiness"), _BUNDLE_READINESS_STATIC, "bundle.readiness")
    _require_fields(bundle.get("security"), _BUNDLE_SECURITY_STATIC, "bundle.security")
    model = bundle.get("model", {})
    _require_fields(model, _BUNDLE_MODEL_ARTIFACT_STATIC, "bundle.model")
    if not isinstance(model.get("selected_candidate_id"), str) or not model.get("selected_candidate_id"):
        raise ForecastingFinalizationContractError("Bundle selected_candidate_id must be a non-empty string.")
    if not isinstance(model.get("selected_family"), str) or not model.get("selected_family"):
        raise ForecastingFinalizationContractError("Bundle selected_family must be a non-empty string.")
    if not isinstance(model.get("selected_specification"), Mapping):
        raise ForecastingFinalizationContractError("Bundle selected_specification must be an object.")
    if not isinstance(model.get("model_artifact_path"), str) or not model.get("model_artifact_path"):
        raise ForecastingFinalizationContractError("Bundle model_artifact_path must be a non-empty string.")
    if not isinstance(model.get("model_artifact_sha256"), str) or len(model.get("model_artifact_sha256", "")) != 64:
        raise ForecastingFinalizationContractError("Bundle model_artifact_sha256 must be a 64-character hex string.")
    if not isinstance(model.get("model_state_semantic_fingerprint"), str) or len(model.get("model_state_semantic_fingerprint", "")) != 64:
        raise ForecastingFinalizationContractError("Bundle model_state_semantic_fingerprint must be a 64-character hex string.")
    return bundle


def load_trusted_forecasting_model_from_bundle(*, project_root: str | Path, bundle_path: str | Path) -> FrozenForecastingModel:
    root = Path(project_root).resolve()
    bundle = load_and_validate_forecasting_inference_bundle(project_root=root, bundle_path=bundle_path)
    info = bundle["model"]
    path = _confined(root, info["model_artifact_path"])
    model = validate_serialized_forecasting_model(path, expected_byte_sha256=info["model_artifact_sha256"],
                                                  expected_state_fingerprint=info["model_state_semantic_fingerprint"])
    if model.selected_specification != info["selected_specification"]:
        raise ForecastingFinalizationContractError("Bundle/model specification differs.")
    return model


def validate_inference_history(history: pd.DataFrame, *, training_end: str = "1938-12") -> pd.Series:
    if not isinstance(history, pd.DataFrame) or list(history.columns) != ["period", "temperature"] or len(history) < 1:
        raise ForecastingFinalizationContractError("History must contain period and temperature columns.")
    try: index = pd.PeriodIndex(history["period"].astype(str), freq="M", name="period")
    except Exception as exc: raise ForecastingFinalizationContractError("History periods are invalid monthly values.") from exc
    values = pd.to_numeric(history["temperature"], errors="coerce").to_numpy(float)
    expected = pd.period_range(index[0], periods=len(index), freq="M", name="period")
    if not index.equals(expected) or index.has_duplicates or not index.is_monotonic_increasing or not np.isfinite(values).all():
        raise ForecastingFinalizationContractError("History chronology/values are invalid.")
    if index[-1] < pd.Period(training_end, freq="M"):
        raise ForecastingFinalizationContractError("History ends before frozen training end.")
    return pd.Series(values, index=index, name="temperature")


def _artifact_state(output: Path) -> str:
    present = [((output / name).is_file()) for name in FINAL_FILENAMES]
    return "complete" if all(present) else "absent" if not any(present) else "partial"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical(payload))


def _authenticate_final_model_manifest_cross_artifact(
    manifest: Mapping[str, Any], *, contract: FrozenForecastingFinalizationContract,
    model: FrozenForecastingModel, refs: Mapping[str, Mapping[str, str]],
    upstream_reference: Mapping[str, Any],
) -> None:
    """Reconcile manifest fields that require the authenticated upstream chain.

    Every expected value here is derived from the already-authenticated ``contract``
    (Notebook 03 winner + lineage), the trusted ``model``, or the handoff's own
    authenticated sibling references -- never a second independent literal.
    """
    manifest_upstream = manifest.get("upstream", {}).get("model_selection_handoff", {})
    if (manifest_upstream.get("path"), manifest_upstream.get("sha256"), manifest_upstream.get("semantic_fingerprint")) != (
        upstream_reference.get("path"), upstream_reference.get("sha256"), upstream_reference.get("semantic_fingerprint")
    ):
        raise ForecastingFinalizationContractError("manifest.upstream.model_selection_handoff differs from the authenticated handoff reference.")
    selected = manifest.get("selected_model", {})
    if (selected.get("selected_candidate_id"), selected.get("selected_role"), selected.get("selected_family"),
        selected.get("selected_specification")) != (
        contract.selected_candidate_id, contract.selected_role, contract.selected_family, contract.selected_specification
    ):
        raise ForecastingFinalizationContractError("manifest.selected_model differs from the authenticated winner.")
    training = manifest.get("training", {})
    if (training.get("development_path"), training.get("development_sha256")) != (
        contract.development_path, contract.development_sha256
    ):
        raise ForecastingFinalizationContractError("manifest.training development lineage differs from the authenticated contract.")
    model_state = manifest.get("model_state", {})
    model_ref = refs.get("final-pipeline.joblib", {})
    if (model_state.get("model_artifact_path"), model_state.get("model_artifact_byte_sha256"),
        model_state.get("model_state_semantic_fingerprint")) != (
        model_ref.get("path"), model_ref.get("byte_sha256"), model.model_state_semantic_fingerprint
    ):
        raise ForecastingFinalizationContractError("manifest.model_state differs from the trusted frozen model.")
    forecast = manifest.get("forecast", {})
    if forecast.get("multi_step_strategy") != contract.selected_specification.get("multi_step_strategy"):
        raise ForecastingFinalizationContractError("manifest.forecast.multi_step_strategy differs from the authenticated specification.")
    holdout = manifest.get("holdout_boundary", {})
    if (holdout.get("path"), holdout.get("sha256")) != (contract.holdout_path, contract.holdout_sha256):
        raise ForecastingFinalizationContractError("manifest.holdout_boundary lineage differs from the authenticated contract.")


def _authenticate_inference_bundle_cross_artifact(
    bundle: Mapping[str, Any], *, contract: FrozenForecastingFinalizationContract,
    model: FrozenForecastingModel, refs: Mapping[str, Mapping[str, str]],
) -> None:
    """Reconcile the bundle's winner-dependent fields against the authenticated chain."""
    info = bundle.get("model", {})
    if (info.get("selected_candidate_id"), info.get("selected_family"), info.get("selected_specification")) != (
        contract.selected_candidate_id, contract.selected_family, contract.selected_specification
    ):
        raise ForecastingFinalizationContractError("bundle.model selected identity differs from the authenticated winner.")
    model_ref = refs.get("final-pipeline.joblib", {})
    if (info.get("model_artifact_path"), info.get("model_artifact_sha256"), info.get("model_state_semantic_fingerprint")) != (
        model_ref.get("path"), model_ref.get("byte_sha256"), model.model_state_semantic_fingerprint
    ):
        raise ForecastingFinalizationContractError("bundle.model artifact reference differs from the trusted frozen model.")


def _authenticate_final_test_evidence_cross_artifact(
    evidence: Mapping[str, Any], *, contract: FrozenForecastingFinalizationContract,
    model: FrozenForecastingModel,
) -> None:
    """Reconcile evidence fields that require the authenticated holdout/model lineage."""
    if evidence.get("holdout_byte_sha256") != contract.holdout_sha256:
        raise ForecastingFinalizationContractError("evidence.holdout_byte_sha256 differs from the authenticated sealed holdout.")
    if evidence.get("model_state_semantic_fingerprint") != model.model_state_semantic_fingerprint:
        raise ForecastingFinalizationContractError("evidence.model_state_semantic_fingerprint differs from the trusted frozen model.")


def _authenticate_final_model_handoff_cross_artifact(
    handoff: Mapping[str, Any], *, contract: FrozenForecastingFinalizationContract,
    bundle: Mapping[str, Any], evidence: Mapping[str, Any], refs: Mapping[str, Mapping[str, str]],
) -> None:
    """Reconcile handoff blocks that were previously left unauthenticated or partially authenticated."""
    selected = handoff.get("selected_model", {})
    if (selected.get("candidate_id"), selected.get("role"), selected.get("family"), selected.get("selected_specification")) != (
        contract.selected_candidate_id, contract.selected_role, contract.selected_family, contract.selected_specification
    ):
        raise ForecastingFinalizationContractError("handoff.selected_model differs from the authenticated winner.")
    _require_fields(handoff.get("training"), _HANDOFF_TRAINING_STATIC, "handoff.training")
    final_evaluation = handoff.get("final_evaluation", {})
    _require_fields(final_evaluation, _HANDOFF_FINAL_EVALUATION_STATIC, "handoff.final_evaluation")
    _same(final_evaluation.get("metrics"), evidence.get("metrics"), "handoff.final_evaluation.metrics")
    inference = handoff.get("inference", {})
    if inference.get("input_contract") != bundle.get("input_contract") or inference.get("output_contract") != bundle.get("output_contract"):
        raise ForecastingFinalizationContractError("handoff.inference contract differs from the inference bundle.")
    if inference.get("model_artifact_reference") != refs.get("final-pipeline.joblib"):
        raise ForecastingFinalizationContractError("handoff.inference.model_artifact_reference differs from the final model reference.")
    _require_fields(handoff.get("boundary"), _HANDOFF_BOUNDARY_STATIC, "handoff.boundary")
    if list(handoff.get("notebook_05_instructions", [])) != list(NOTEBOOK_05_INSTRUCTIONS):
        raise ForecastingFinalizationContractError("handoff.notebook_05_instructions differs from the canonical Notebook 05 contract.")


def load_and_validate_forecasting_final_model_handoff(
    *, project_root: str | Path,
    handoff_path: str | Path = "artifacts/models/nottem/final-model-handoff.json",
) -> dict[str, Any]:
    root = Path(project_root).resolve(); path = _confined(root, handoff_path)
    handoff = _validate_json(path, HANDOFF_SCHEMA, "final_model_handoff")
    _require_fields(handoff, _FORECASTING_IDENTITY, "handoff")
    upstream = handoff.get("upstream", {}).get("model_selection_handoff", {})
    upstream_path = _confined(root, upstream.get("path", ""))
    if sha256_file(upstream_path) != upstream.get("sha256"):
        raise ForecastingArtifactConflictError("Upstream model-selection SHA differs.")
    selected = load_and_validate_forecasting_model_selection_handoff(
        project_root=root, handoff_path=upstream_path, expected_dataset_slug="nottem")
    contract = validate_forecasting_finalization_contract(selected, handoff_path=upstream_path)
    if upstream.get("semantic_fingerprint") != contract.model_selection_semantic_fingerprint:
        raise ForecastingArtifactConflictError("Handoff upstream semantic fingerprint differs.")
    refs = handoff.get("final_references", {})
    if set(refs) != set(FINAL_FILENAMES[:-1]):
        raise ForecastingArtifactConflictError("Final sibling reference set differs.")
    resolved = {}
    for name, ref in refs.items():
        sibling = _confined(root, ref.get("path", ""))
        if sibling.name != name or sha256_file(sibling) != ref.get("byte_sha256"):
            raise ForecastingArtifactConflictError(f"Final sibling {name} differs.")
        resolved[name] = sibling
    manifest = load_and_validate_forecasting_final_model_manifest(project_root=root, manifest_path=resolved["final-model-manifest.json"])
    evidence = load_and_validate_forecasting_final_test_evidence(project_root=root, evidence_path=resolved["final-test-evidence.json"])
    bundle = load_and_validate_forecasting_inference_bundle(project_root=root, bundle_path=resolved["inference-bundle.json"])
    for name, payload in (("final-model-manifest.json", manifest), ("final-test-evidence.json", evidence), ("inference-bundle.json", bundle)):
        if payload["semantic_fingerprint"] != refs[name]["semantic_fingerprint"]:
            raise ForecastingArtifactConflictError(f"Final sibling {name} fingerprint differs.")
    model = load_trusted_forecasting_model_from_bundle(project_root=root, bundle_path=resolved["inference-bundle.json"])
    if refs["final-pipeline.joblib"]["semantic_fingerprint"] != model.model_state_semantic_fingerprint:
        raise ForecastingArtifactConflictError("Handoff model state differs.")
    # Cross-artifact selected-model identity: authenticated Notebook 03 winner ==
    # FrozenForecastingModel == manifest.selected_model == bundle.model == handoff.selected_model.
    if (model.candidate_id, model.family, model.selected_specification) != (
        contract.selected_candidate_id, contract.selected_family, contract.selected_specification
    ):
        raise ForecastingFinalizationContractError("Frozen model selected identity differs from the authenticated winner.")
    # Cross-artifact training scope: catches a re-signed joblib whose internal
    # training metadata was widened (e.g. to include 1939) without a real refit,
    # since these bounds come from the authenticated contract, not the joblib itself.
    if (model.training_start, model.training_end, model.training_observations, model.training_sha256) != (
        contract.training_start, contract.training_end, contract.training_observations, contract.development_sha256
    ):
        raise ForecastingFinalizationContractError("Frozen model training scope differs from the authenticated contract.")
    _authenticate_final_model_manifest_cross_artifact(manifest, contract=contract, model=model, refs=refs, upstream_reference=upstream)
    _authenticate_inference_bundle_cross_artifact(bundle, contract=contract, model=model, refs=refs)
    _authenticate_final_test_evidence_cross_artifact(evidence, contract=contract, model=model)
    _same(evidence["model_selection_reference_metrics"], contract.reference_metrics, "reference metrics")
    _authenticate_final_model_handoff_cross_artifact(handoff, contract=contract, bundle=bundle, evidence=evidence, refs=refs)
    future = pd.period_range("1939-01", periods=12, freq="M", name="period")
    replay = model.forecast_periods(future).to_numpy(float)
    if not np.allclose(replay, [r["y_pred"] for r in evidence["forecasts"]], rtol=1e-12, atol=1e-12):
        raise ForecastingFinalizationContractError("Frozen-model replay differs from final evidence predictions.")
    expected_readiness = {"model_selection_handoff_validated": True, "selected_specification_reconstructed": True,
        "final_training_scope_validated": True, "final_training_completed": True, "final_fit_count": 1,
        "final_model_trained": True, "model_frozen": True, "model_artifact_materialized": True,
        "final_holdout_opened_after_freeze": True, "final_holdout_evaluated": True,
        "final_holdout_evaluation_count": 1, "final_forecast_call_count": 1,
        "final_test_evidence_materialized": True, "inference_bundle_materialized": True,
        "final_model_handoff_reloadable": True, "inference_demo_ready": True, "operational_modeling_ready": False}
    if handoff.get("readiness") != expected_readiness:
        raise ForecastingFinalizationContractError("Final handoff readiness differs.")
    return handoff


@dataclass(frozen=True, slots=True)
class ForecastingFinalizationResult:
    status: str
    output_directory: Path
    paths: tuple[str, ...]
    guard: Mapping[str, Any]
    handoff: Mapping[str, Any]


def write_forecasting_final_artifacts(
    *, project_root: Path, output: Path, staging: Path, model_staging: Path,
    contract: FrozenForecastingFinalizationContract, model: FrozenForecastingModel,
    guard: ForecastingFinalizationGuard, evidence: dict[str, Any], model_sha: str,
) -> dict[str, Any]:
    output_rel = _relative(project_root, output)
    paths = {name: f"{output_rel}/{name}" for name in FINAL_FILENAMES}
    bundle = build_forecasting_inference_bundle(contract, model, model_path=paths["final-pipeline.joblib"], model_sha256=model_sha)
    manifest = build_forecasting_final_model_manifest(contract, model, guard, model_path=paths["final-pipeline.joblib"], model_sha256=model_sha)
    for name, payload in (("final-test-evidence.json", evidence), ("inference-bundle.json", bundle), ("final-model-manifest.json", manifest)):
        _write_json(staging / name, payload)
    refs = {"final-pipeline.joblib": _reference(paths["final-pipeline.joblib"], model_sha, model.model_state_semantic_fingerprint)}
    for name, payload in (("final-model-manifest.json", manifest), ("final-test-evidence.json", evidence), ("inference-bundle.json", bundle)):
        refs[name] = _reference(paths[name], sha256_file(staging / name), payload["semantic_fingerprint"])
    handoff = build_forecasting_final_model_handoff(contract, model, evidence, bundle, refs)
    _write_json(staging / "final-model-handoff.json", handoff)
    if set(x.name for x in staging.iterdir()) != set(FINAL_FILENAMES):
        raise ForecastingArtifactConflictError("Staged artifact set is not exactly five files.")
    # Validate staged JSONs and trusted model before promotion.
    _validate_json(staging / "final-model-manifest.json", MANIFEST_SCHEMA, "final_model_manifest")
    load_and_validate_forecasting_final_test_evidence(project_root=staging.parent, evidence_path=staging / "final-test-evidence.json")
    _validate_json(staging / "inference-bundle.json", BUNDLE_SCHEMA, "inference_bundle")
    _validate_json(staging / "final-model-handoff.json", HANDOFF_SCHEMA, "final_model_handoff")
    return handoff


def run_forecasting_finalization(
    *, project_root: str | Path,
    handoff_path: str | Path = "artifacts/model-selection/nottem/model-selection-handoff.json",
    output_directory: str | Path = "artifacts/models/nottem",
) -> ForecastingFinalizationResult:
    root = Path(project_root).resolve(); upstream_path = _confined(root, handoff_path)
    if not upstream_path.is_file():
        raise ForecastingFinalizationContractError("Model-selection handoff is missing.")
    selected = load_and_validate_forecasting_model_selection_handoff(
        project_root=root, handoff_path=upstream_path, expected_dataset_slug="nottem")
    contract = validate_forecasting_finalization_contract(
        selected,
        handoff_path=_relative(root, upstream_path),
        project_root=root,
    )
    output = _confined(root, output_directory); state = _artifact_state(output)
    if state == "partial": raise ForecastingArtifactConflictError("Partial final artifact set; refusing recovery.")
    if state == "complete":
        final = load_and_validate_forecasting_final_model_handoff(project_root=root, handoff_path=output / "final-model-handoff.json")
        if final["upstream"]["model_selection_handoff"]["sha256"] != contract.model_selection_handoff_sha256:
            raise ForecastingArtifactConflictError("Complete set belongs to divergent upstream.")
        return ForecastingFinalizationResult("reused_equivalent", output,
            tuple(_relative(root, output / n) for n in FINAL_FILENAMES),
            {"final_fit_count": 0, "holdout_open_count": 0, "final_evaluation_count": 0}, final)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    guard = ForecastingFinalizationGuard()
    try:
        development = validate_full_development(_confined(root, contract.development_path), expected_sha256=contract.development_sha256)
        guard.audit_log.extend(["handoff authenticated", "development authenticated", "exact selected spec reconstructed"])
        model = freeze_forecasting_model(reconstruct_and_fit_selected_forecasting_model(contract, development, guard))
        model_path = staging / "final-pipeline.joblib"
        model_sha = serialize_frozen_forecasting_model_to_staging(model, model_path)
        guard.audit_log.append("model staged")
        validate_serialized_forecasting_model(model_path, expected_byte_sha256=model_sha,
                                              expected_state_fingerprint=model.model_state_semantic_fingerprint)
        guard.audit_log.append("model round-trip validated")
        guard.mark_frozen()
        holdout = open_final_holdout_once(_confined(root, contract.holdout_path), expected_sha256=contract.holdout_sha256, guard=guard)
        predictions = generate_final_forecast_once(model, origin="1938-12", guard=guard)
        rows, metrics, denominator = evaluate_final_forecast_once(predictions, holdout, development=development, guard=guard)
        evidence = build_forecasting_final_test_evidence(contract, model, rows, metrics, denominator, guard)
        handoff = write_forecasting_final_artifacts(project_root=root, output=output, staging=staging,
            model_staging=model_path, contract=contract, model=model, guard=guard, evidence=evidence, model_sha=model_sha)
        if output.exists(): raise ForecastingArtifactConflictError("Output appeared during staging.")
        os.replace(staging, output)
        guard.audit_log.append("final artifacts materialized")
        final = load_and_validate_forecasting_final_model_handoff(project_root=root, handoff_path=output / "final-model-handoff.json")
        guard.audit_log.append("final handoff reloaded")
        return ForecastingFinalizationResult("created", output,
            tuple(_relative(root, output / n) for n in FINAL_FILENAMES), asdict(guard), final)
    except Exception:
        if staging.exists(): shutil.rmtree(staging)
        raise
