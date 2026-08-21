"""Leakage-safe model selection for the frozen Nottingham forecast study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scripts.forecasting_preparation import (
    ForecastingPreparationError,
    load_and_validate_forecasting_preparation_handoff,
    seasonal_mase_scale,
)

PRACTICAL_MAE_TIE_TOLERANCE_F = 0.05
HANDOFF_SCHEMA = "forecasting-model-selection-handoff.v1"
REQUIRED_FILENAMES = (
    "model-selection-manifest.json", "candidate-results.json",
    "cross-validation-results.csv", "validation-evidence.json",
    "selection-analysis.json", "model-selection-handoff.json",
)
TIE_BREAK_ORDER = [
    "pooled_seasonal_mase", "pooled_rmse", "fold_mae_std",
    "long_horizon_mae_h7_h12", "complexity_rank", "candidate_id",
]


class ForecastingModelSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    status: str
    paths: tuple[str, ...]


def _spec(candidate_id: str, role: str, family: str, rank: int,
          constructor: str, params: Mapping[str, Any], **policies: Any) -> dict[str, Any]:
    base = {
        "candidate_id": candidate_id, "role": role, "family": family,
        "complexity_rank": rank, "constructor": constructor,
        "fixed_hyperparameters": dict(params),
        "learned_parameter_scope": "fit_from_scratch_on_each_training_fold_only" if role == "candidate" else "none",
        "preprocessing_policy": "original_target_scale_no_global_learned_preprocessing",
        "differencing_policy": "none", "trend_policy": "none",
        "seasonal_policy": "none", "seasonal_period": 12,
        "multi_step_strategy": "single_origin_12_step_without_validation_feedback",
        "exogenous_policy": "none_deterministic_calendar_is_not_source_exogenous",
    }
    base.update(policies)
    return base


_CATALOG = (
    _spec("seasonal_naive_12", "primary_baseline", "SeasonalNaive", 0, "seasonal_naive_forecast", {},
          seasonal_policy="known_history_lag_12", multi_step_strategy="direct_known_history_lookup_12_steps"),
    _spec("naive_last_value", "secondary_baseline", "NaiveLastValue", 1, "naive_last_value_forecast", {},
          multi_step_strategy="constant_last_training_value_12_steps"),
    _spec("seasonal_trend_ols", "candidate", "DeterministicSeasonalTrendOLS", 2, "statsmodels.regression.linear_model.OLS",
          {"intercept": True, "linear_time_trend": True, "calendar_month_dummies": 11, "reference_month": "January"},
          trend_policy="linear_time_trend", seasonal_policy="11_calendar_month_dummies_january_reference",
          multi_step_strategy="direct_known_calendar_design_12_steps"),
    _spec("holt_winters_additive_no_trend", "candidate", "ExponentialSmoothing", 3, "statsmodels.tsa.holtwinters.ExponentialSmoothing",
          {"trend": None, "damped_trend": False, "seasonal": "add", "seasonal_periods": 12, "initialization_method": "estimated", "use_boxcox": False},
          seasonal_policy="additive_estimated_period_12", multi_step_strategy="native_forecast_12"),
    _spec("holt_winters_additive_damped_trend", "candidate", "ExponentialSmoothing", 4, "statsmodels.tsa.holtwinters.ExponentialSmoothing",
          {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": 12, "initialization_method": "estimated", "use_boxcox": False},
          trend_policy="additive_damped", seasonal_policy="additive_estimated_period_12", multi_step_strategy="native_forecast_12"),
    _spec("holt_winters_additive_trend", "candidate", "ExponentialSmoothing", 5, "statsmodels.tsa.holtwinters.ExponentialSmoothing",
          {"trend": "add", "damped_trend": False, "seasonal": "add", "seasonal_periods": 12, "initialization_method": "estimated", "use_boxcox": False},
          trend_policy="additive_undamped", seasonal_policy="additive_estimated_period_12", multi_step_strategy="native_forecast_12"),
    _spec("autoreg_lag_1_12_ct", "candidate", "AutoReg", 6, "statsmodels.tsa.ar_model.AutoReg",
          {"lags": [1, 12], "trend": "ct", "seasonal": False, "old_names": False},
          trend_policy="constant_and_linear_time", multi_step_strategy="recursive_12_step"),
    _spec("autoreg_lag_1_2_12_ct", "candidate", "AutoReg", 7, "statsmodels.tsa.ar_model.AutoReg",
          {"lags": [1, 2, 12], "trend": "ct", "seasonal": False, "old_names": False},
          trend_policy="constant_and_linear_time", multi_step_strategy="recursive_12_step"),
    _spec("sarima_100_100_12", "candidate", "SARIMAX", 8, "statsmodels.tsa.statespace.sarimax.SARIMAX",
          {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 12], "trend": "ct", "enforce_stationarity": True, "enforce_invertibility": True},
          trend_policy="constant_and_linear_time", seasonal_policy="seasonal_ar_1_period_12", multi_step_strategy="native_state_space_12_step"),
    _spec("sarima_100_011_12", "candidate", "SARIMAX", 9, "statsmodels.tsa.statespace.sarimax.SARIMAX",
          {"order": [1, 0, 0], "seasonal_order": [0, 1, 1, 12], "trend": "n", "enforce_stationarity": True, "enforce_invertibility": True},
          differencing_policy="internal_fold_local_seasonal_difference_D1_period_12", seasonal_policy="seasonal_difference_and_ma_1_period_12", multi_step_strategy="native_state_space_12_step"),
)


def frozen_specification_catalog() -> list[dict[str, Any]]:
    return json.loads(json.dumps(_CATALOG))


def validate_catalog(catalog: Sequence[Mapping[str, Any]]) -> None:
    if len(catalog) != 10 or sum(s["role"].endswith("baseline") for s in catalog) != 2:
        raise ForecastingModelSelectionError("Frozen catalog must contain two baselines and eight candidates.")
    ids = [s["candidate_id"] for s in catalog]; ranks = [s["complexity_rank"] for s in catalog]
    if len(set(ids)) != 10 or len(set(ranks)) != 10 or list(ranks) != list(range(10)):
        raise ForecastingModelSelectionError("Catalog IDs/ranks diverged.")
    if json.dumps(list(catalog), sort_keys=True) != json.dumps(frozen_specification_catalog(), sort_keys=True):
        raise ForecastingModelSelectionError("Catalog differs from frozen specifications.")


def validate_development_series(development: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(development, pd.DataFrame):
        if list(development.columns) != ["period", "temperature"]:
            raise ForecastingModelSelectionError("Development columns mismatch.")
        idx = pd.PeriodIndex(development["period"].astype(str), freq="M", name="period")
        values = pd.to_numeric(development["temperature"], errors="coerce").to_numpy(float)
    else:
        idx = pd.PeriodIndex(development.index, freq="M", name="period")
        values = pd.to_numeric(development, errors="coerce").to_numpy(float)
    expected = pd.period_range("1920-01", "1938-12", freq="M", name="period")
    if not idx.equals(expected) or len(values) != 228 or not np.isfinite(values).all():
        raise ForecastingModelSelectionError("Development scope is not the authenticated 228-month series.")
    return pd.Series(values, index=idx, name="temperature")


def fold_slices(series: pd.Series, backtesting_contract: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], pd.Series, pd.PeriodIndex]]:
    required = (backtesting_contract.get("mode"), backtesting_contract.get("initial_training_months"),
                backtesting_contract.get("forecast_horizon"), backtesting_contract.get("origin_step_months"),
                backtesting_contract.get("fold_count"), backtesting_contract.get("validation_forecast_count"),
                backtesting_contract.get("validation_targets_overlap"))
    if required != ("expanding_window", 120, 12, 12, 9, 108, False):
        raise ForecastingModelSelectionError("Backtesting contract diverged.")
    out = []
    schedule = backtesting_contract.get("schedule", [])
    if len(schedule) != 9: raise ForecastingModelSelectionError("Schedule must have nine folds.")
    for i, row in enumerate(schedule):
        train = series.iloc[:120 + i * 12]
        future = pd.period_range(train.index[-1] + 1, periods=12, freq="M", name="period")
        scale = seasonal_mase_scale(train)
        if row.get("fold") != i + 1 or row.get("train_end_forecast_origin") != str(train.index[-1]) or row.get("validation_start") != str(future[0]) or row.get("validation_end") != str(future[-1]) or not math.isclose(scale, float(row.get("seasonal_mase_scale_from_training", -1)), rel_tol=0, abs_tol=1e-12):
            raise ForecastingModelSelectionError(f"Fold {i+1} authentication failed.")
        out.append((row, train, future))
    return out


def seasonal_naive_forecast(training: pd.Series, future: pd.PeriodIndex) -> pd.Series:
    if len(future) != 12: raise ForecastingModelSelectionError("Forecast horizon must be 12.")
    return pd.Series(training.iloc[-12:].to_numpy(float), index=future, name="forecast")


def naive_last_value_forecast(training: pd.Series, future: pd.PeriodIndex) -> pd.Series:
    return pd.Series(np.repeat(float(training.iloc[-1]), 12), index=future, name="forecast")


def _ols_design(index: pd.PeriodIndex, start_ordinal: int) -> np.ndarray:
    t = np.asarray(index.asi8 - start_ordinal, dtype=float)
    months = index.month
    return np.column_stack([np.ones(len(index)), t] + [(months == m).astype(float) for m in range(2, 13)])


def seasonal_trend_ols_forecast(training: pd.Series, future: pd.PeriodIndex) -> pd.Series:
    start = int(training.index[0].ordinal)
    fit = OLS(training.to_numpy(float), _ols_design(training.index, start)).fit()
    return pd.Series(fit.predict(_ols_design(future, start)), index=future, name="forecast")


def forecast_specification(spec: Mapping[str, Any], training: pd.Series, future: pd.PeriodIndex) -> tuple[pd.Series, bool | None]:
    cid = spec["candidate_id"]
    if cid == "seasonal_naive_12": return seasonal_naive_forecast(training, future), None
    if cid == "naive_last_value": return naive_last_value_forecast(training, future), None
    if cid == "seasonal_trend_ols": return seasonal_trend_ols_forecast(training, future), None
    p = dict(spec["fixed_hyperparameters"])
    if spec["family"] == "ExponentialSmoothing":
        use_boxcox = p.pop("use_boxcox")
        fitted = ExponentialSmoothing(training, **p, use_boxcox=use_boxcox).fit(optimized=True)
        pred = fitted.forecast(12); converged = None
    elif spec["family"] == "AutoReg":
        fitted = AutoReg(training, **p).fit()
        pred = fitted.predict(start=len(training), end=len(training) + 11, dynamic=False); converged = None
    elif spec["family"] == "SARIMAX":
        p["order"] = tuple(p["order"]); p["seasonal_order"] = tuple(p["seasonal_order"])
        fitted = SARIMAX(training, **p).fit(disp=False)
        converged = bool(fitted.mle_retvals.get("converged", False))
        pred = fitted.get_forecast(steps=12).predicted_mean
    else: raise ForecastingModelSelectionError(f"Unknown specification: {cid}")
    return pd.Series(np.asarray(pred, dtype=float), index=future, name="forecast"), converged


def backtest_specification(spec: Mapping[str, Any], series: pd.Series, backtesting_contract: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, audits = [], []
    for fold, training, future in fold_slices(series, backtesting_contract):
        caught: list[str] = []; failure = None; fit_status = forecast_status = "not_run"
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                pred, converged = forecast_specification(spec, training.copy(), future)
                caught = [f"{type(w.message).__name__}: {w.message}" for w in records]
            fit_status = "success"
            if converged is False: raise ForecastingModelSelectionError("Explicit optimizer non-convergence.")
            if len(pred) != 12 or not pred.index.equals(future) or not np.isfinite(pred.to_numpy()).all():
                raise ForecastingModelSelectionError("Forecast shape/index/finiteness invalid.")
            forecast_status = "success"
            # Targets are accessed only after the full 12-vector has been produced.
            truth = series.loc[future]
            scale = float(fold["seasonal_mase_scale_from_training"])
            for h, period in enumerate(future, 1):
                y_true, y_pred = float(truth.loc[period]), float(pred.loc[period]); error = y_pred-y_true
                rows.append({"candidate_id": spec["candidate_id"], "role": spec["role"], "family": spec["family"],
                    "fold": int(fold["fold"]), "forecast_origin": fold["train_end_forecast_origin"], "forecast_period": str(period), "horizon": h,
                    "y_true": y_true, "y_pred": y_pred, "error": error, "abs_error": abs(error), "squared_error": error**2,
                    "seasonal_mase_scale": scale, "scaled_abs_error": abs(error)/scale,
                    "fit_status": fit_status, "forecast_status": forecast_status, "warning_count": len(caught)})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if fit_status == "not_run": fit_status = "failed"
            if forecast_status == "not_run": forecast_status = "failed"
        audits.append({"fold": int(fold["fold"]), "fit_status": fit_status, "forecast_status": forecast_status,
                       "warnings": caught, "failure": failure})
    frame = pd.DataFrame(rows)
    complete = len(frame) == 108 and not any(a["failure"] for a in audits)
    if spec["role"].endswith("baseline") and not complete:
        raise ForecastingModelSelectionError(f"Baseline {spec['candidate_id']} failed; backtesting is blocked.")
    return frame, audits


def aggregate_metrics(spec: Mapping[str, Any], rows: pd.DataFrame, audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = len(rows) == 108 and not any(a["failure"] for a in audits)
    result = {"candidate_id": spec["candidate_id"], "family": spec["family"], "role": spec["role"],
              "complexity_rank": spec["complexity_rank"], "complete": complete, "eligible": complete,
              "folds_completed": int(rows["fold"].nunique()) if len(rows) else 0, "forecast_rows": len(rows),
              "warnings": [w for a in audits for w in a["warnings"]], "failures": [a["failure"] for a in audits if a["failure"]],
              "fold_summaries": [], "horizon_mae": {}}
    if not complete: return result
    for fold, g in rows.groupby("fold", sort=True):
        result["fold_summaries"].append({"fold": int(fold), "fold_mae": float(g.abs_error.mean()),
            "fold_rmse": float(np.sqrt(g.squared_error.mean())), "fold_seasonal_mase": float(g.scaled_abs_error.mean())})
    result.update({"pooled_mae": float(rows.abs_error.mean()), "pooled_rmse": float(np.sqrt(rows.squared_error.mean())),
                   "pooled_seasonal_mase": float(rows.scaled_abs_error.mean()),
                   "fold_mae_std": float(np.std([x["fold_mae"] for x in result["fold_summaries"]], ddof=0))})
    result["horizon_mae"] = {str(int(h)): float(g.abs_error.mean()) for h,g in rows.groupby("horizon", sort=True)}
    result["short_horizon_mae_h1_h6"] = float(rows[rows.horizon <= 6].abs_error.mean())
    result["long_horizon_mae_h7_h12"] = float(rows[rows.horizon >= 7].abs_error.mean())
    return result


def select_winner(results: Sequence[Mapping[str, Any]], tolerance: float = PRACTICAL_MAE_TIE_TOLERANCE_F) -> dict[str, Any]:
    eligible = [dict(r) for r in results if r.get("eligible") and all(math.isfinite(float(r[k])) for k in ("pooled_mae","pooled_rmse","pooled_seasonal_mase","fold_mae_std","long_horizon_mae_h7_h12"))]
    if not eligible: raise ForecastingModelSelectionError("No complete eligible specification.")
    best = min(float(r["pooled_mae"]) for r in eligible)
    finalists = [r for r in eligible if float(r["pooled_mae"]) <= best + tolerance]
    ranked_finalists = sorted(finalists, key=lambda r: (r["pooled_seasonal_mase"], r["pooled_rmse"], r["fold_mae_std"], r["long_horizon_mae_h7_h12"], r["complexity_rank"], r["candidate_id"]))
    winner = ranked_finalists[0]
    full_ranking = sorted(eligible, key=lambda r: (0 if r["candidate_id"] == winner["candidate_id"] else 1, r["pooled_mae"], r["candidate_id"]))
    return {"best_raw_mae": best, "practical_tie_tolerance_f": tolerance,
            "finalists": [r["candidate_id"] for r in ranked_finalists], "tie_break_order": list(TIE_BREAK_ORDER),
            "selected_candidate_id": winner["candidate_id"], "ranking": [r["candidate_id"] for r in full_ranking]}


def run_all_backtests(development: pd.DataFrame | pd.Series, backtesting_contract: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    series = validate_development_series(development); catalog = frozen_specification_catalog(); validate_catalog(catalog)
    frames=[]; summaries=[]; audit={}
    for spec in catalog:
        rows, folds = backtest_specification(spec, series, backtesting_contract)
        frames.append(rows); audit[spec["candidate_id"]] = folds; summaries.append(aggregate_metrics(spec, rows, folds))
    evidence = pd.concat(frames, ignore_index=True)
    baseline = next(r for r in summaries if r["candidate_id"] == "seasonal_naive_12")
    for r in summaries:
        if r["complete"]:
            r["delta_mae_vs_seasonal_naive"] = r["pooled_mae"] - baseline["pooled_mae"]
            r["relative_mae_improvement_pct"] = 100*(baseline["pooled_mae"]-r["pooled_mae"])/baseline["pooled_mae"]
    selection = select_winner(summaries)
    for r in summaries: r["deterministic_rank"] = selection["ranking"].index(r["candidate_id"])+1 if r["candidate_id"] in selection["ranking"] else None
    return evidence, summaries, audit


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_fingerprint(obj: Any) -> str:
    if isinstance(obj, Mapping): obj = {k:v for k,v in obj.items() if k not in {"artifact_sha256","semantic_fingerprint","runtime_versions"}}
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _confined(root: Path, rel: str | Path) -> Path:
    p=(root/rel).resolve()
    try: p.relative_to(root)
    except ValueError as exc: raise ForecastingModelSelectionError("Artifact path escapes project root.") from exc
    return p


def build_artifacts(*, project_root: str | Path, preparation_handoff_path: str,
                    preparation_payload: Mapping[str, Any], evidence: pd.DataFrame,
                    summaries: list[dict[str, Any]], audits: Mapping[str, Any]) -> dict[str, Any]:
    root=Path(project_root).resolve(); prep_path=_confined(root, preparation_handoff_path)
    catalog=frozen_specification_catalog(); selection=select_winner(summaries)
    selected=next(r for r in summaries if r["candidate_id"]==selection["selected_candidate_id"])
    selected_spec=next(s for s in catalog if s["candidate_id"]==selection["selected_candidate_id"])
    prep_ref={"path":preparation_handoff_path,"schema_version":preparation_payload["schema_version"],"sha256":sha256_file(prep_path)}
    dev=preparation_payload["prepared_data"]["development"]; hold=preparation_payload["prepared_data"]["sealed_final_holdout"]
    common={"dataset_slug":"nottem","problem_type":"time_series_forecasting","forecasting_mode":"univariate"}
    candidate={"schema_version":"forecasting-candidate-results.v1","artifact_type":"candidate_results",**common,"specifications":summaries,"selection":selection}
    validation={"schema_version":"forecasting-validation-evidence.v1","artifact_type":"validation_evidence",**common,
        "evaluation_protocol":"expanding_window_backtesting","fold_count":9,"forecasts_per_complete_specification":108,
        "metric_formulas":{"mae":"mean(abs(y_pred-y_true))","rmse":"sqrt(mean((y_pred-y_true)^2))","seasonal_mase_12":"mean(abs_error/fold_training_seasonal_scale)"},
        "seasonal_mase_scaling_policy":"fold_training_only_lag_12_mean_absolute_difference","fold_audits":audits,
        "pooled_summaries":summaries,"selected_candidate_id":selected["candidate_id"],"max_validation_period":str(evidence.forecast_period.max()),"final_holdout_evaluated":False}
    analysis={"schema_version":"forecasting-selection-analysis.v1","artifact_type":"selection_analysis",**common,
        "candidate_family_rationale":{"seasonal_strength":0.960,"trend_strength":0.241,"development_lag_1_acf":0.809,"development_lag_12_acf":0.886,"annual_mean_trend_f_per_year":0.057,"seasonal_difference_variance_ratio":0.165,"point_anomaly_candidates":0,"cusum_stability_p":0.438,
            "interpretation":"Strong annual seasonality and lag-12 dependence favor parsimonious classical seasonal specifications; modest trend and short persistence motivate the frozen OLS, smoothing, AutoReg, and SARIMA variants."},
        "catalog_frozen_before_evaluation":True,"selection":selection,"baseline_comparison":[{"candidate_id":r["candidate_id"],"delta_mae_vs_seasonal_naive":r.get("delta_mae_vs_seasonal_naive"),"relative_mae_improvement_pct":r.get("relative_mae_improvement_pct")} for r in summaries],
        "stability_comparison":{r["candidate_id"]:r.get("fold_mae_std") for r in summaries},"long_horizon_comparison":{r["candidate_id"]:r.get("long_horizon_mae_h7_h12") for r in summaries},
        "failed_or_ineligible":[r["candidate_id"] for r in summaries if not r["eligible"]],"warnings_summary":{r["candidate_id"]:len(r["warnings"]) for r in summaries},
        "scientific_interpretation":f"{selected['candidate_id']} is selected by the frozen practical-tie rule.","no_1939_values_influenced_selection":True}
    artifact_paths={name:f"artifacts/model-selection/nottem/{name}" for name in REQUIRED_FILENAMES}
    manifest={"schema_version":"forecasting-model-selection-manifest.v1","artifact_type":"model_selection_manifest",**common,
        "preparation_handoff":prep_ref,"development_integrity":dict(dev),"sealed_holdout_integrity_metadata_only":dict(hold),
        "prediction_contract":preparation_payload["prediction_contract"],"evaluation_contract":preparation_payload["evaluation_contract"],"backtesting_contract":preparation_payload["backtesting_contract"],
        "baselines":catalog[:2],"frozen_candidate_catalog":catalog,"selection_contract":{"primary_metric":"mae","practical_mae_tie_tolerance_f":PRACTICAL_MAE_TIE_TOLERANCE_F,"tie_break_order":TIE_BREAK_ORDER},
        "artifact_paths":artifact_paths,"runtime_versions":{"python":os.sys.version.split()[0],"pandas":pd.__version__},
        "final_holdout_sealed":True,"final_holdout_evaluated":False,"final_model_trained":False,"model_artifact_materialized":False}
    handoff={"schema_version":HANDOFF_SCHEMA,"artifact_type":"model_selection_handoff",**common,"upstream_preparation_handoff":prep_ref,
        "development":dict(dev),"sealed_final_holdout_metadata_only":dict(hold),"selected_candidate_id":selected["candidate_id"],"selected_role":selected["role"],"selected_family":selected["family"],"selected_specification":selected_spec,
        "backtest":{"mode":"expanding_window","initial_training_months":120,"forecast_horizon":12,"origin_step_months":12,"fold_count":9,"validation_forecast_count":108},
        "evaluation":{"primary_metric":"mae","secondary_metrics":["rmse","seasonal_mase_12"],"selected_pooled_metrics":{"mae":selected["pooled_mae"],"rmse":selected["pooled_rmse"],"seasonal_mase_12":selected["pooled_seasonal_mase"]},"selected_fold_mae_std":selected["fold_mae_std"],"selected_horizon_mae":selected["horizon_mae"],"seasonal_naive_metrics":next(r for r in summaries if r["candidate_id"]=="seasonal_naive_12"),"delta_mae_vs_primary_baseline":selected["delta_mae_vs_seasonal_naive"]},
        "selection":selection,"final_training_instructions":{"notebook":"notebooks/04_final_model_and_bundle.ipynb","training_scope":"full development 1920-01 -> 1938-12","reconstruct_exact_selected_specification":True,"fit_once_on_full_development_if_required":not selected["role"].endswith("baseline"),"freeze_before_final_holdout_open":True,"final_forecast_horizon":12,"final_evaluation_period":"1939-01 -> 1939-12","open_holdout_only_after_final_spec_fit":True,"evaluate_holdout_once":True,"do_not_retune":True,"do_not_change_candidate":True,"do_not_change_hyperparameters":True,"do_not_change_transform_policy":True,"target_scale":"original degrees Fahrenheit"},
        "readiness":{"preparation_handoff_validated":True,"development_only_model_selection":True,"temporal_backtesting_completed":True,"frozen_backtesting_contract_respected":True,"baselines_evaluated":True,"candidate_catalog_frozen_before_evaluation":True,"candidate_models_evaluated":True,"selected_specification_frozen":True,"metric_contract_frozen":True,"model_selection_handoff_reloadable":True,"final_holdout_sealed":True,"final_holdout_evaluated":False,"final_model_training_ready":True,"final_model_trained":False,"model_artifact_materialized":False,"model_bundle_materialized":False,"operational_modeling_ready":False},
        "model_artifact":None,"artifact_paths":artifact_paths}
    return {"model-selection-manifest.json":manifest,"candidate-results.json":candidate,"cross-validation-results.csv":evidence,
            "validation-evidence.json":validation,"selection-analysis.json":analysis,"model-selection-handoff.json":handoff}


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()


_EVIDENCE_NUMERIC_COLUMNS = (
    "fold", "horizon", "y_true", "y_pred", "error", "abs_error",
    "squared_error", "seasonal_mase_scale", "scaled_abs_error",
    "warning_count",
)


def _equivalent_evidence_csv(
    existing_path: Path,
    current: pd.DataFrame,
    *,
    absolute_tolerance: float = 1e-9,
    ineligible_candidate_ids: Sequence[str] = (),
) -> tuple[bool, str]:
    """Compare OOS evidence semantically, independent of CSV float rendering."""
    try:
        existing = pd.read_csv(existing_path)
    except Exception as exc:
        return False, f"persisted CSV is unreadable: {type(exc).__name__}: {exc}"
    if list(existing.columns) != list(current.columns) or len(existing) != len(current):
        return False, (
            f"shape/columns differ: persisted_shape={existing.shape}, current_shape={current.shape}, "
            f"persisted_columns={list(existing.columns)}, current_columns={list(current.columns)}"
        )
    keys = ["candidate_id", "fold", "horizon", "forecast_period"]
    if any(key not in current.columns for key in keys):
        return False, f"scientific row keys are missing; required={keys}"
    if existing.duplicated(keys).any() or current.duplicated(keys).any():
        return False, "duplicate scientific forecast-row key detected"
    existing = existing.sort_values(keys, kind="stable").reset_index(drop=True)
    current = current.sort_values(keys, kind="stable").reset_index(drop=True)
    ineligible = set(ineligible_candidate_ids)
    comparable_output_rows = ~current["candidate_id"].astype(str).isin(ineligible).to_numpy()
    model_output_columns = {"y_pred", "error", "abs_error", "squared_error", "scaled_abs_error"}
    text_columns = [c for c in current.columns if c not in _EVIDENCE_NUMERIC_COLUMNS]
    for column in text_columns:
        if existing[column].astype(str).tolist() != current[column].astype(str).tolist():
            mismatch = next(
                i for i, (left, right) in enumerate(
                    zip(existing[column].astype(str), current[column].astype(str))
                ) if left != right
            )
            return False, (
                f"text column {column!r} differs at aligned row {mismatch}: "
                f"persisted={existing.at[mismatch, column]!r}, current={current.at[mismatch, column]!r}"
            )
    for column in _EVIDENCE_NUMERIC_COLUMNS:
        if column not in current.columns:
            return False, f"required numeric column {column!r} is missing"
        left = pd.to_numeric(existing[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return False, f"numeric column {column!r} contains non-finite values"
        mask = comparable_output_rows if column in model_output_columns else np.ones(len(current), dtype=bool)
        if not np.allclose(left[mask], right[mask], rtol=0.0, atol=absolute_tolerance):
            difference = np.where(mask, np.abs(left-right), -1.0)
            location = int(np.argmax(difference))
            return False, (
                f"numeric column {column!r} differs; max_abs_difference={difference[location]:.17g} "
                f"at aligned key={tuple(current.loc[location, keys])}, "
                f"persisted={left[location]:.17g}, current={right[location]:.17g}, "
                f"tolerance={absolute_tolerance:.1e}"
            )
    return True, "equivalent after scientific-key alignment"


def write_forecasting_model_selection_artifacts(*, project_root: str | Path, artifacts: Mapping[str, Any]) -> ArtifactWriteResult:
    root=Path(project_root).resolve(); base=_confined(root,"artifacts/model-selection/nottem")
    if set(artifacts)!=set(REQUIRED_FILENAMES): raise ForecastingModelSelectionError("Exactly six artifacts are required.")
    # Add sibling byte and semantic fingerprints to the handoff after rendering stable siblings.
    rendered={}
    for name,obj in artifacts.items():
        if name=="cross-validation-results.csv": rendered[name]=_csv_bytes(obj)
        elif name!="model-selection-handoff.json": rendered[name]=_canonical(obj)
    handoff=json.loads(json.dumps(artifacts["model-selection-handoff.json"]))
    handoff["sibling_artifacts"]={n:{"path":f"artifacts/model-selection/nottem/{n}","sha256":hashlib.sha256(b).hexdigest(),
        "semantic_fingerprint":semantic_fingerprint(artifacts[n]) if n.endswith(".json") else hashlib.sha256(b).hexdigest()} for n,b in rendered.items()}
    handoff["semantic_fingerprint"]=semantic_fingerprint(handoff)
    rendered["model-selection-handoff.json"]=_canonical(handoff)
    existing=[base/n for n in REQUIRED_FILENAMES if (base/n).exists()]
    if existing:
        if len(existing)!=6:
            present = sorted(p.name for p in existing)
            missing = sorted(set(REQUIRED_FILENAMES) - set(present))
            raise ForecastingModelSelectionError(
                f"Existing model-selection artifact set is partial; present={present}, missing={missing}."
            )
        divergent: list[str] = []
        divergence_details: list[str] = []
        for name in REQUIRED_FILENAMES:
            if name == "model-selection-handoff.json":
                # The persisted handoff authenticates the persisted sibling bytes. If
                # every scientific sibling is equivalent, keep that internally
                # consistent handoff instead of rebuilding it around volatile runtime
                # metadata from the current kernel.
                continue
            path = base / name
            if name.endswith(".csv"):
                candidate_payload = artifacts.get("candidate-results.json", {})
                ineligible_ids = [
                    str(result.get("candidate_id"))
                    for result in candidate_payload.get("specifications", [])
                    if result.get("eligible") is False
                ]
                equivalent, detail = _equivalent_evidence_csv(
                    path,
                    artifacts[name],
                    ineligible_candidate_ids=ineligible_ids,
                )
                if not equivalent:
                    divergence_details.append(f"{name}: {detail}")
            else:
                try:
                    existing_payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    equivalent = False
                else:
                    equivalent = semantic_fingerprint(existing_payload) == semantic_fingerprint(artifacts[name])
            if not equivalent:
                divergent.append(name)
        if divergent:
            raise ForecastingModelSelectionError(
                "Existing model-selection artifacts are scientifically divergent: "
                + ", ".join(divergent)
                + ". Details: " + " | ".join(divergence_details)
            )
        # Authenticate the retained handoff and all of its exact persisted sibling
        # hashes before declaring semantic reuse.
        load_and_validate_forecasting_model_selection_handoff(
            project_root=root,
            handoff_path="artifacts/model-selection/nottem/model-selection-handoff.json",
            expected_dataset_slug="nottem",
        )
        return ArtifactWriteResult("reused_equivalent",tuple(str(base/n) for n in REQUIRED_FILENAMES))
    base.parent.mkdir(parents=True,exist_ok=True); stage=Path(tempfile.mkdtemp(prefix=".model-selection-",dir=base.parent))
    try:
        for n,b in rendered.items(): (stage/n).write_bytes(b)
        if set(p.name for p in stage.iterdir())!=set(REQUIRED_FILENAMES): raise ForecastingModelSelectionError("Staging validation failed.")
        os.replace(stage,base)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise
    return ArtifactWriteResult("created",tuple(str(base/n) for n in REQUIRED_FILENAMES))


def load_and_validate_forecasting_model_selection_handoff(handoff_path: str | Path="artifacts/model-selection/nottem/model-selection-handoff.json", *, project_root: str | Path=".", expected_dataset_slug: str="nottem") -> dict[str, Any]:
    root=Path(project_root).resolve(); path=_confined(root,handoff_path)
    try: handoff=json.loads(path.read_text())
    except Exception as exc: raise ForecastingModelSelectionError("Model-selection handoff is unreadable.") from exc
    if handoff.get("schema_version")!=HANDOFF_SCHEMA or handoff.get("artifact_type")!="model_selection_handoff" or handoff.get("dataset_slug")!=expected_dataset_slug or handoff.get("problem_type")!="time_series_forecasting" or handoff.get("forecasting_mode")!="univariate": raise ForecastingModelSelectionError("Handoff identity diverged.")
    if handoff.get("semantic_fingerprint")!=semantic_fingerprint(handoff): raise ForecastingModelSelectionError("Handoff semantic fingerprint mismatch.")
    siblings=handoff.get("sibling_artifacts",{})
    if set(siblings)!=set(REQUIRED_FILENAMES)-{"model-selection-handoff.json"}: raise ForecastingModelSelectionError("Sibling set mismatch.")
    loaded={}
    for name,ref in siblings.items():
        p=_confined(root,ref["path"])
        if sha256_file(p)!=ref["sha256"]: raise ForecastingModelSelectionError(f"Sibling hash mismatch: {name}")
        if name.endswith(".json"):
            loaded[name]=json.loads(p.read_text())
            if semantic_fingerprint(loaded[name])!=ref["semantic_fingerprint"]: raise ForecastingModelSelectionError(f"Sibling semantic fingerprint mismatch: {name}")
        else: loaded[name]=pd.read_csv(p)
    prep=handoff["upstream_preparation_handoff"]
    if sha256_file(_confined(root,prep["path"]))!=prep["sha256"]: raise ForecastingModelSelectionError("Preparation SHA mismatch.")
    try: preparation=load_and_validate_forecasting_preparation_handoff(project_root=root,preparation_handoff_path=prep["path"],expected_dataset_slug=expected_dataset_slug)
    except ForecastingPreparationError as exc: raise ForecastingModelSelectionError("Preparation handoff validation failed.") from exc
    manifest=loaded["model-selection-manifest.json"]; candidate=loaded["candidate-results.json"]; evidence=loaded["cross-validation-results.csv"]
    catalog=frozen_specification_catalog(); validate_catalog(manifest.get("frozen_candidate_catalog",[]))
    selection=select_winner(candidate.get("specifications",[]))
    cid=selection["selected_candidate_id"]; spec=next(s for s in catalog if s["candidate_id"]==cid); summary=next(r for r in candidate["specifications"] if r["candidate_id"]==cid)
    checks=[handoff.get("selected_candidate_id")==cid,handoff.get("selected_family")==spec["family"],handoff.get("selected_role")==spec["role"],handoff.get("selected_specification")==spec,handoff.get("selection")==selection,
            handoff.get("development")==manifest.get("development_integrity"),handoff.get("sealed_final_holdout_metadata_only")==manifest.get("sealed_holdout_integrity_metadata_only"),
            handoff.get("evaluation",{}).get("selected_pooled_metrics")=={"mae":summary["pooled_mae"],"rmse":summary["pooled_rmse"],"seasonal_mase_12":summary["pooled_seasonal_mase"]}]
    if not all(checks): raise ForecastingModelSelectionError("Cross-artifact selected specification disagreement.")
    winner=evidence[(evidence.candidate_id==cid)&(evidence.fit_status=="success")&(evidence.forecast_status=="success")]
    if len(winner)!=108 or winner.fold.nunique()!=9 or str(evidence.forecast_period.max())>"1938-12": raise ForecastingModelSelectionError("OOS evidence boundary/count mismatch.")
    ready=handoff.get("readiness",{}); hold=handoff.get("sealed_final_holdout_metadata_only",{}); instr=handoff.get("final_training_instructions",{})
    if not (hold.get("sealed") is True and hold.get("evaluated") is False and hold.get("exposed_to_model_selection") is False and ready.get("final_model_training_ready") is True and ready.get("final_model_trained") is False and instr.get("do_not_retune") is True and instr.get("open_holdout_only_after_final_spec_fit") is True): raise ForecastingModelSelectionError("Final boundary/readiness diverged.")
    if hasattr(preparation,"final_holdout"): raise ForecastingModelSelectionError("Preparation boundary exposed holdout values.")
    return handoff
