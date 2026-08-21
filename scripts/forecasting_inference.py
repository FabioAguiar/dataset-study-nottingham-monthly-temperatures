"""Read-only Notebook 05 consumer for the frozen Nottingham forecasting model.

This module composes the existing authenticated, read-only primitives in
``scripts.forecasting_finalization`` (final-model handoff loader, inference
bundle loader, trusted frozen-model loader, and history validator) into a
small consumer-only API for the Notebook 05 inference demo.

It contains no training, finalization, scoring, model-selection, or artifact
writing logic. All SHA-before-deserialization protection and cross-artifact
authentication stay centralized in ``scripts.forecasting_finalization``; this
module never calls ``joblib.load`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.forecasting_finalization import (
    FrozenForecastingModel,
    load_and_validate_forecasting_final_model_handoff,
    load_and_validate_forecasting_inference_bundle,
    load_trusted_forecasting_model_from_bundle,
    validate_inference_history,
)


class ForecastingInferenceError(RuntimeError):
    """Raised when the Notebook 05 consumer boundary cannot be trusted."""


@dataclass(frozen=True, slots=True)
class AuthenticatedForecastingConsumer:
    """A read-only handle on the authenticated final-model handoff, bundle, and frozen model."""

    final_handoff: Mapping[str, Any]
    inference_bundle: Mapping[str, Any]
    model: FrozenForecastingModel

    def forecast(self, history: pd.DataFrame) -> pd.DataFrame:
        return forecast_from_history(self, history)


def load_authenticated_forecasting_consumer(
    *,
    project_root: str | Path,
    handoff_path: str | Path = "artifacts/models/nottem/final-model-handoff.json",
) -> AuthenticatedForecastingConsumer:
    """Load and cross-check the Notebook 04 -> Notebook 05 boundary.

    1. Authenticate the final-model handoff via the hardened finalizer loader.
    2. Derive the inference-bundle path from the authenticated handoff's own
       ``final_references`` (never a second, independently hardcoded path).
    3. Authenticate the bundle via the hardened bundle loader.
    4. Trusted-load the frozen model (SHA-before-deserialization is enforced
       inside ``load_trusted_forecasting_model_from_bundle``).
    5. Reconcile candidate/family/specification/state-fingerprint identity
       across handoff, bundle, and the loaded model.
    """
    root = Path(project_root).resolve()
    handoff = load_and_validate_forecasting_final_model_handoff(
        project_root=root, handoff_path=handoff_path,
    )
    try:
        bundle_reference = handoff["final_references"]["inference-bundle.json"]
        bundle_path = bundle_reference["path"]
    except (KeyError, TypeError) as exc:
        raise ForecastingInferenceError(
            "Authenticated handoff is missing its inference-bundle reference."
        ) from exc
    bundle = load_and_validate_forecasting_inference_bundle(
        project_root=root, bundle_path=bundle_path,
    )
    model = load_trusted_forecasting_model_from_bundle(
        project_root=root, bundle_path=bundle_path,
    )
    selected = handoff.get("selected_model", {})
    if (model.candidate_id, model.family, model.selected_specification) != (
        selected.get("candidate_id"), selected.get("family"), selected.get("selected_specification"),
    ):
        raise ForecastingInferenceError(
            "Trusted frozen-model identity differs from the authenticated final-model handoff."
        )
    bundle_model = bundle.get("model", {})
    if (bundle_model.get("selected_candidate_id"), bundle_model.get("selected_family"),
        bundle_model.get("selected_specification"), bundle_model.get("model_state_semantic_fingerprint")) != (
        model.candidate_id, model.family, model.selected_specification, model.model_state_semantic_fingerprint,
    ):
        raise ForecastingInferenceError(
            "Trusted frozen-model identity differs from the authenticated inference bundle."
        )
    return AuthenticatedForecastingConsumer(final_handoff=handoff, inference_bundle=bundle, model=model)


def normalize_forecasting_inference_history(
    consumer: AuthenticatedForecastingConsumer, history: pd.DataFrame,
) -> pd.Series:
    """Validate and normalize a supplied historical DataFrame into a monthly ``pd.Series``.

    Delegates entirely to ``validate_inference_history``, the single authority
    for the executable input contract (exact ``["period", "temperature"]``
    columns, monthly/monotonic/unique/contiguous periods, finite numeric-
    coercible temperatures, history ending at or after the frozen training end).
    """
    return validate_inference_history(history, training_end=consumer.model.training_end)


def forecast_from_history(
    consumer: AuthenticatedForecastingConsumer, history: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the deterministic future-horizon forecast for a supplied history.

    The forecast origin is the last validated historical period; the future
    horizon comes from the authenticated bundle, never a user-supplied value.
    The only forecasting call made is ``FrozenForecastingModel.forecast_periods``.
    """
    checked = normalize_forecasting_inference_history(consumer, history)
    horizon = consumer.inference_bundle["time"]["forecast_horizon"]
    if horizon != consumer.model.forecast_horizon:
        raise ForecastingInferenceError("Bundle forecast horizon differs from the frozen model.")
    origin = checked.index[-1]
    future = pd.period_range(origin + 1, periods=horizon, freq="M", name="period")
    predicted = consumer.model.forecast_periods(future)
    output = pd.DataFrame({
        "period": future.astype(str),
        "forecast": predicted.to_numpy(dtype=float),
    })
    return validate_forecasting_inference_output(output, consumer=consumer, forecast_origin=origin)


def validate_forecasting_inference_output(
    output: pd.DataFrame, *, consumer: AuthenticatedForecastingConsumer, forecast_origin: pd.Period,
) -> pd.DataFrame:
    """Fail-closed check of the output contract: exact columns, row count, periods, finiteness."""
    horizon = consumer.inference_bundle["time"]["forecast_horizon"]
    if not isinstance(output, pd.DataFrame) or list(output.columns) != ["period", "forecast"] or len(output) != horizon:
        raise ForecastingInferenceError("Forecast output must have exactly ['period', 'forecast'] and the bundle horizon row count.")
    expected_periods = [str(p) for p in pd.period_range(forecast_origin + 1, periods=horizon, freq="M")]
    if list(output["period"]) != expected_periods:
        raise ForecastingInferenceError("Forecast output periods differ from origin+1 through origin+horizon.")
    values = pd.to_numeric(output["forecast"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ForecastingInferenceError("Forecast output contains non-finite values.")
    return output
