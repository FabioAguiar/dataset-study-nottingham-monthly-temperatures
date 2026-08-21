from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.forecasting_finalization as ff
import scripts.forecasting_inference as fi

# The synthetic five-artifact-set builder is test-only fixture logic already
# authored for the Notebook 04 -> Notebook 05 boundary hardening tests. It is
# reused here (not duplicated) to build fixtures in tmp_path; it is never
# invoked from Notebook 05 or from scripts/forecasting_inference.py itself.
_ff_tests = importlib.import_module("tests.test_forecasting_finalization")

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_HANDOFF = REPO_ROOT / "artifacts/models/nottem/final-model-handoff.json"


def _fx(tmp_path, monkeypatch):
    return _ff_tests.full_artifact_set(tmp_path, monkeypatch)


def _consumer(fx):
    return fi.load_authenticated_forecasting_consumer(
        project_root=fx["tmp_path"], handoff_path=fx["out"] / "final-model-handoff.json",
    )


# ======================================================================================
# Happy-path consumer loading and canonical/later-origin forecasts (synthetic fixture).
# ======================================================================================

def test_consumer_loads_from_synthetic_artifact_set(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    consumer = _consumer(fx)
    assert consumer.model.candidate_id == "seasonal_trend_ols"
    assert consumer.model.family == "DeterministicSeasonalTrendOLS"
    assert consumer.final_handoff["readiness"]["inference_demo_ready"] is True
    assert consumer.final_handoff["readiness"]["operational_modeling_ready"] is False


def test_canonical_origin_forecast_shape_and_periods(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    consumer = _consumer(fx)
    history = pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]})
    result = consumer.forecast(history)
    assert list(result.columns) == ["period", "forecast"]
    assert len(result) == 12
    assert list(result["period"]) == [str(p) for p in pd.period_range("1939-01", periods=12, freq="M")]
    assert np.isfinite(result["forecast"].to_numpy(dtype=float)).all()


def test_later_origin_forecast_periods_advance_without_refit(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    consumer = _consumer(fx)
    history = pd.DataFrame({"period": ["1940-10", "1940-11", "1940-12"], "temperature": [48.0, 49.0, 50.0]})
    result = fi.forecast_from_history(consumer, history)
    assert list(result["period"]) == [str(p) for p in pd.period_range("1941-01", periods=12, freq="M")]
    assert len(result) == 12


# ======================================================================================
# Valid input contract cases.
# ======================================================================================

def test_valid_one_row_history(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    history = pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]})
    result = consumer.forecast(history)
    assert len(result) == 12


def test_valid_multirow_contiguous_history(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    history = pd.DataFrame({"period": ["1940-10", "1940-11", "1940-12"], "temperature": [48.0, 49.0, 50.0]})
    result = consumer.forecast(history)
    assert len(result) == 12


def test_numeric_string_temperature_is_coerced(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    history = pd.DataFrame({"period": ["1938-12"], "temperature": ["50.0"]})
    result = consumer.forecast(history)
    assert len(result) == 12


# ======================================================================================
# Invalid input contract matrix -- all must fail closed.
# ======================================================================================

INVALID_HISTORY_CASES = {
    "none": lambda: None,
    "dict": lambda: {"period": ["1938-12"], "temperature": [50.0]},
    "series": lambda: pd.Series([50.0], index=pd.PeriodIndex(["1938-12"], freq="M")),
    "empty": lambda: pd.DataFrame({"period": [], "temperature": []}),
    "wrong_columns": lambda: pd.DataFrame({"temperature": [50.0]}),
    "wrong_order": lambda: pd.DataFrame({"temperature": [50.0], "period": ["1938-12"]})[["temperature", "period"]],
    "extra_columns": lambda: pd.DataFrame({"period": ["1938-12"], "temperature": [50.0], "extra": [1]}),
    "invalid_period": lambda: pd.DataFrame({"period": ["not-a-period"], "temperature": [50.0]}),
    "duplicate_period": lambda: pd.DataFrame({"period": ["1938-12", "1938-12"], "temperature": [50.0, 51.0]}),
    "reversed_periods": lambda: pd.DataFrame({"period": ["1938-12", "1938-11"], "temperature": [50.0, 51.0]}),
    "monthly_gap": lambda: pd.DataFrame({"period": ["1938-10", "1938-12"], "temperature": [50.0, 51.0]}),
    "nonnumeric_temperature": lambda: pd.DataFrame({"period": ["1938-12"], "temperature": ["warm"]}),
    "nan_temperature": lambda: pd.DataFrame({"period": ["1938-12"], "temperature": [np.nan]}),
    "posinf_temperature": lambda: pd.DataFrame({"period": ["1938-12"], "temperature": [np.inf]}),
    "neginf_temperature": lambda: pd.DataFrame({"period": ["1938-12"], "temperature": [-np.inf]}),
    "before_training_end": lambda: pd.DataFrame({"period": ["1938-11"], "temperature": [50.0]}),
}


@pytest.mark.parametrize("case", list(INVALID_HISTORY_CASES), ids=list(INVALID_HISTORY_CASES))
def test_invalid_history_matrix_is_rejected(tmp_path, monkeypatch, case):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    bad = INVALID_HISTORY_CASES[case]()
    with pytest.raises(Exception):
        consumer.forecast(bad)


# ======================================================================================
# No-mutation guarantees.
# ======================================================================================

def test_input_history_is_not_mutated(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    history = pd.DataFrame({"period": ["1940-10", "1940-11", "1940-12"], "temperature": [48.0, 49.0, 50.0]})
    before = history.copy(deep=True)
    consumer.forecast(history)
    consumer.forecast(history)
    pd.testing.assert_frame_equal(history, before)


# ======================================================================================
# Forecast-origin and history-value semantics.
# ======================================================================================

def test_history_value_invariance_for_seasonal_trend_ols(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    periods = ["1940-10", "1940-11", "1940-12"]
    low = pd.DataFrame({"period": periods, "temperature": [20.0, 20.0, 20.0]})
    high = pd.DataFrame({"period": periods, "temperature": [80.0, 80.0, 80.0]})
    result_low = consumer.forecast(low)
    result_high = consumer.forecast(high)
    pd.testing.assert_frame_equal(result_low, result_high)


def test_different_origin_changes_output_periods(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    a = consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    b = consumer.forecast(pd.DataFrame({"period": ["1940-12"], "temperature": [50.0]}))
    assert list(a["period"]) != list(b["period"])
    assert list(a["period"]) == [str(p) for p in pd.period_range("1939-01", periods=12, freq="M")]
    assert list(b["period"]) == [str(p) for p in pd.period_range("1941-01", periods=12, freq="M")]


# ======================================================================================
# Deterministic repeatability and model-state immutability.
# ======================================================================================

def test_deterministic_repeatability(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    history = pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]})
    first = consumer.forecast(history)
    second = consumer.forecast(history)
    assert list(first["period"]) == list(second["period"])
    np.testing.assert_allclose(
        first["forecast"].to_numpy(float), second["forecast"].to_numpy(float), rtol=1e-12, atol=1e-12,
    )


def test_model_state_immutable_across_repeated_and_different_origin_calls(tmp_path, monkeypatch):
    consumer = _consumer(_fx(tmp_path, monkeypatch))
    descriptor_before = consumer.model.state_descriptor()
    fingerprint_before = consumer.model.model_state_semantic_fingerprint
    coefficients_before = tuple(consumer.model.coefficients)

    consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    consumer.forecast(pd.DataFrame({"period": ["1940-12"], "temperature": [999.0]}))

    assert consumer.model.state_descriptor() == descriptor_before
    assert consumer.model.model_state_semantic_fingerprint == fingerprint_before
    assert tuple(consumer.model.coefficients) == coefficients_before


# ======================================================================================
# No-refit audit: fitting/finalization primitives must never be invoked by the consumer.
# ======================================================================================

def test_no_refit_or_finalization_calls_during_load_and_forecast(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)

    def _forbid_fit(*_args, **_kwargs):
        raise AssertionError("OLS.fit must not be called by the Notebook 05 consumer path")

    def _forbid_lstsq(*_args, **_kwargs):
        raise AssertionError("np.linalg.lstsq must not be called by the Notebook 05 consumer path")

    def _forbid_finalization(*_args, **_kwargs):
        raise AssertionError("run_forecasting_finalization must not be called by the Notebook 05 consumer path")

    monkeypatch.setattr(ff.OLS, "fit", _forbid_fit)
    monkeypatch.setattr(np.linalg, "lstsq", _forbid_lstsq)
    monkeypatch.setattr(ff, "run_forecasting_finalization", _forbid_finalization)

    consumer = _consumer(fx)
    result = consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    assert len(result) == 12


# ======================================================================================
# Final-holdout isolation audit.
# ======================================================================================

def test_final_holdout_csv_is_never_read(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    original_read_csv = pd.read_csv

    def guarded_read_csv(path, *args, **kwargs):
        if Path(path).name == "final-holdout.csv":
            raise AssertionError("final-holdout.csv must never be read by the Notebook 05 consumer path")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    consumer = _consumer(fx)
    result = consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    assert len(result) == 12


# ======================================================================================
# Entry-boundary / tamper / security gate tests.
# ======================================================================================

def test_missing_handoff_file_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        fi.load_authenticated_forecasting_consumer(
            project_root=fx["tmp_path"], handoff_path=fx["out"] / "does-not-exist.json",
        )


def test_corrupt_json_handoff_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    (fx["out"] / "final-model-handoff.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(Exception):
        _consumer(fx)


def test_inference_demo_ready_false_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["handoff"]["readiness"]["inference_demo_ready"] = False
    _ff_tests._finalize_tamper(fx, [])
    with pytest.raises(Exception):
        _consumer(fx)


def test_operational_modeling_ready_true_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["handoff"]["readiness"]["operational_modeling_ready"] = True
    _ff_tests._finalize_tamper(fx, [])
    with pytest.raises(Exception):
        _consumer(fx)


def test_missing_bundle_sibling_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    (fx["out"] / "inference-bundle.json").unlink()
    with pytest.raises(Exception):
        _consumer(fx)


def test_tampered_bundle_bytes_are_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    path = fx["out"] / "inference-bundle.json"
    path.write_bytes(path.read_bytes() + b"\n// tampered")
    with pytest.raises(Exception):
        _consumer(fx)


def test_missing_model_artifact_is_blocked(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    (fx["out"] / "final-pipeline.joblib").unlink()
    with pytest.raises(Exception):
        _consumer(fx)


def test_model_byte_sha_mismatch_is_rejected_before_joblib_load(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    model_path = fx["out"] / "final-pipeline.joblib"
    data = bytearray(model_path.read_bytes())
    data[0] ^= 0xFF
    model_path.write_bytes(bytes(data))

    called = []
    original_load = ff.joblib.load
    monkeypatch.setattr(ff.joblib, "load", lambda *a, **k: (called.append(1), original_load(*a, **k))[1])

    with pytest.raises(Exception):
        _consumer(fx)
    assert called == []


def test_model_state_fingerprint_mismatch_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["model"]["model_state_semantic_fingerprint"] = "0" * 64
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


def test_bundle_selected_model_mismatch_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["model"]["selected_candidate_id"] = "naive_last_value"
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


def test_bundle_horizon_mismatch_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["time"]["forecast_horizon"] = 6
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


def test_bundle_target_unit_mismatch_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["target"]["unit"] = "degrees Celsius"
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


def test_bundle_refit_on_input_true_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["input_contract"]["refit_on_input"] = True
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


def test_bundle_model_frozen_false_is_rejected(tmp_path, monkeypatch):
    fx = _fx(tmp_path, monkeypatch)
    fx["bundle"]["readiness"]["model_frozen"] = False
    _ff_tests._finalize_tamper(fx, ["bundle"])
    with pytest.raises(Exception):
        _consumer(fx)


# ======================================================================================
# Real runtime artifacts (skip if unavailable in this workspace).
# ======================================================================================

def _skip_if_no_real_artifacts():
    if not REAL_HANDOFF.is_file():
        pytest.skip("real runtime final-model artifacts are unavailable in this workspace")


def test_real_artifacts_consumer_loads_and_forecasts(monkeypatch):
    _skip_if_no_real_artifacts()

    fit_calls = []
    monkeypatch.setattr(ff.OLS, "fit", lambda self, *a, **k: fit_calls.append(1) or pytest.fail("consumer path triggered a fit"))
    original_read_csv = pd.read_csv

    def guarded_read_csv(path, *a, **k):
        if Path(path).name == "final-holdout.csv":
            pytest.fail("consumer path reopened the final holdout")
        return original_read_csv(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

    consumer = fi.load_authenticated_forecasting_consumer(project_root=REPO_ROOT)
    result = consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    assert list(result.columns) == ["period", "forecast"]
    assert len(result) == 12
    assert np.isfinite(result["forecast"].to_numpy(float)).all()
    assert fit_calls == []


def test_real_artifact_hashes_unchanged_after_consumer_use():
    _skip_if_no_real_artifacts()
    directory = REAL_HANDOFF.parent
    before = {p.name: ff.sha256_file(p) for p in directory.iterdir() if p.is_file()}

    consumer = fi.load_authenticated_forecasting_consumer(project_root=REPO_ROOT)
    consumer.forecast(pd.DataFrame({"period": ["1938-12"], "temperature": [50.0]}))
    consumer.forecast(pd.DataFrame({"period": ["1940-12"], "temperature": [50.0]}))

    after = {p.name: ff.sha256_file(p) for p in directory.iterdir() if p.is_file()}
    assert before == after


# ======================================================================================
# Fresh-process consumer validation (synthetic artifacts, independent interpreter).
# ======================================================================================

def test_fresh_process_consumer_validation():
    """Fresh-process validation must use the real authenticated artifact chain: the
    synthetic tmp_path fixture stubs the upstream model-selection loader via
    monkeypatch, which cannot follow into a spawned subprocess."""
    _skip_if_no_real_artifacts()
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import pandas as pd
from scripts.forecasting_inference import load_authenticated_forecasting_consumer, forecast_from_history

consumer = load_authenticated_forecasting_consumer(
    project_root={str(REPO_ROOT)!r},
    handoff_path="artifacts/models/nottem/final-model-handoff.json",
)
history = pd.DataFrame({{"period": ["1938-12"], "temperature": [50.0]}})
result = forecast_from_history(consumer, history)
assert len(result) == 12
print("FRESH_PROCESS_OK", len(result))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "FRESH_PROCESS_OK 12" in proc.stdout


# ======================================================================================
# Source and notebook boundary tests.
# ======================================================================================

_FORBIDDEN_IMPORTS = {"statsmodels", "sklearn"}
_FORBIDDEN_CALL_NAMES = {
    "run_forecasting_finalization",
    "reconstruct_and_fit_selected_forecasting_model",
    "open_final_holdout_once",
    "evaluate_final_forecast_once",
    "select_winner",
    "backtest_specification",
    "run_all_backtests",
}


def test_helper_module_has_no_forbidden_imports_or_calls():
    source = Path(fi.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not (imported_roots & _FORBIDDEN_IMPORTS), "helper must not import statsmodels/sklearn directly"
    assert "joblib" not in imported_roots, "helper must not import joblib directly (no direct joblib.load)"

    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called_names & _FORBIDDEN_CALL_NAMES)

    called_attrs = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    assert ("joblib", "load") not in called_attrs
    assert not any(attr == "fit" for _, attr in called_attrs)
    assert "lstsq" not in source
    assert "final-holdout" not in source


def test_notebook_05_has_no_consumer_side_training_or_selection_calls():
    notebook_path = REPO_ROOT / "notebooks/05_inference_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    forbidden = _FORBIDDEN_CALL_NAMES | {
        "np.linalg.lstsq", ".fit(", "OLS.fit", "final-holdout.csv", "final-holdout",
    }
    for token in forbidden:
        assert token not in code, f"forbidden token found in Notebook 05 code cells: {token!r}"
    assert "artifacts/models/nottem" in code or "final-model-handoff" in code


def test_notebook_05_never_writes_to_final_artifact_directory():
    notebook_path = REPO_ROOT / "notebooks/05_inference_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    forbidden_writes = ("write_bytes(", "write_text(", "to_json(", "to_csv(", "joblib.dump(", "open(")
    for token in forbidden_writes:
        assert token not in code, f"forbidden write-capable token found in Notebook 05 code cells: {token!r}"


def test_notebook_05_structure_has_required_sections():
    notebook_path = REPO_ROOT / "notebooks/05_inference_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    required_fragments = [
        "Inference Context", "Authenticated Final-Model Handoff", "Inference Bundle",
        "Trusted Frozen-Model Loading", "Input Contract", "Historical-Series Normalization",
        "Canonical 12-Month Forecast", "Later-Origin Forecast", "Forecast-Origin",
        "Deterministic Repeatability", "No-Refit", "Holdout and Model-Selection Isolation",
        "Output Contract", "Final Artifact Immutability", "Fresh-Process Consumer Validation",
        "Study Completion Readiness",
    ]
    for fragment in required_fragments:
        assert fragment in markdown, f"missing expected Notebook 05 section: {fragment!r}"
