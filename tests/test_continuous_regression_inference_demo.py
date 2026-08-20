from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

import scripts.smoke_predict as smoke


FEATURES = ["a", "b", "age"]


@pytest.fixture()
def continuous_fixture(tmp_path):
    frame = pd.DataFrame({"a": [1., 2., 3., 4., 5., 6.], "b": [6., 5., 4., 3., 2., 1.], "age": [1, 2, 3, 4, 5, 6]})
    preprocess = ColumnTransformer([("numerical", "passthrough", FEATURES)], remainder="drop", sparse_threshold=0.0)
    model = HistGradientBoostingRegressor(max_iter=5, random_state=42)
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)]).fit(frame, [2., 3., 4., 5., 6., 7.])
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir()
    joblib.dump(pipeline, model_path)
    digest = smoke._sha256_file(model_path)
    params = pipeline.named_steps["model"].get_params(deep=False)
    preprocessing = {
        "type": "Pipeline", "numerical_features": FEATURES, "categorical_features": [],
        "scale_numerical": False, "scaler": None, "fit_scope": "training_partition_or_training_fold_only",
    }
    descriptor = {
        "pipeline_class": "Pipeline", "step_names": ["preprocess", "model"],
        "preprocessor_class": "ColumnTransformer", "model_class": "HistGradientBoostingRegressor",
        "feature_order": FEATURES, "transformed_feature_names": list(preprocess.fit(frame).get_feature_names_out()),
        "preprocessing_contract": preprocessing,
        "selected_hyperparameters": {"model__max_iter": 5},
        "fixed_constructor_parameters": params,
        "fitted_state_indicators": {"n_features_in_": 3},
    }
    bundle = {
        "schema_version": "inference-bundle.v3", "artifact_type": "inference_bundle",
        "dataset_slug": "synthetic-regression", "problem_type": "continuous_regression",
        "selected_model_id": "synthetic_hgb", "selected_model_family": "HistGradientBoostingRegressor",
        "feature_order": FEATURES,
        "input_feature_dtypes": {
            "a": {"dtype": "float64", "role": "numerical_feature"},
            "b": {"dtype": "float64", "role": "numerical_feature"},
            "age": {"dtype": "int64", "role": "numerical_feature"},
        },
        "target_contract": {"column": "target", "semantics": "Continuous", "unit": "unit", "prediction_output": "continuous value"},
        "prediction_contract": {"type": "continuous_numeric", "scale": "original_target_scale", "unit": "unit"},
        "preprocessing_contract": preprocessing,
        "model_contract": {"family": "HistGradientBoostingRegressor", "selected_hyperparameters": {"model__max_iter": 5}, "fixed_constructor_parameters": params},
        "model_state_descriptor": descriptor, "model_state_fingerprint": "synthetic-state",
        "model_artifact_path": "models/model.joblib", "model_artifact_sha256": digest,
        "runtime_compatibility": smoke.current_runtime_versions(),
        "readiness": {"inference_demo_ready": True, "operational_modeling_ready": False, "operational_validity": "unconfirmed"},
    }
    handoff = {
        "schema_version": "final-model-handoff.v3", "artifact_type": "final_model_handoff",
        **{key: bundle[key] for key in ("dataset_slug", "problem_type", "selected_model_id", "selected_model_family", "feature_order", "target_contract", "prediction_contract", "preprocessing_contract")},
        "readiness": {**bundle["readiness"], "final_fit_count": 1, "test_partition_evaluation_count": 1, "test_prediction_call_count": 1, "test_partition_used_for_adjustment": False, "no_model_selection_decision_changed_after_test": True},
        "model_artifact_reference": {"path": bundle["model_artifact_path"], "sha256": digest, "state_fingerprint": "synthetic-state"},
        "bundle_reference": {"path": "models/bundle.json", "sha256": "bundle-hash"},
        "manifest_reference": {"path": "models/manifest.json", "sha256": "manifest-hash"},
        "sibling_references": {
            "inference-bundle.json": {"path": "models/bundle.json", "sha256": "bundle-hash"},
            "final-model-manifest.json": {"path": "models/manifest.json", "sha256": "manifest-hash"},
        },
    }
    manifest = {
        **{key: bundle[key] for key in ("dataset_slug", "problem_type", "selected_model_id", "selected_model_family", "target_contract", "preprocessing_contract")},
        "selected_hyperparameters": {"model__max_iter": 5},
        "model_artifact": {"path": bundle["model_artifact_path"], "byte_sha256": digest, "state_fingerprint": "synthetic-state", "state_descriptor": descriptor},
        "runtime_versions": smoke.current_runtime_versions(),
    }
    valid = pd.DataFrame([[2., 4., 7], [3., 3., 28]], columns=FEATURES, index=["x", "y"])
    return tmp_path, pipeline, bundle, handoff, manifest, valid


def test_valid_v3_readiness_alignment_and_pipeline(continuous_fixture):
    _, pipeline, bundle, handoff, manifest, _ = continuous_fixture
    smoke.validate_continuous_inference_readiness(handoff, bundle)
    smoke.validate_continuous_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    smoke.validate_continuous_loaded_pipeline_contract(pipeline, bundle=bundle, manifest=manifest)


def test_trusted_model_sha_is_checked_before_loader(continuous_fixture, monkeypatch):
    root, _, bundle, handoff, manifest, _ = continuous_fixture
    called = []
    assert smoke.validate_model_artifact_before_load(project_root=root, bundle=bundle, handoff=handoff, manifest=manifest).is_file()
    bad = deepcopy(bundle)
    bad["model_artifact_sha256"] = "0" * 64
    monkeypatch.setattr(joblib, "load", lambda *_: called.append(True))
    with pytest.raises(smoke.TrustedModelSourceError):
        smoke.validate_model_artifact_before_load(project_root=root, bundle=bad, handoff=handoff)
    assert called == []


@pytest.mark.parametrize("path", ["../model.joblib", "/tmp/model.joblib", r"C:\\model.joblib", r"\\\\server\\model.joblib"])
def test_unsafe_model_paths_fail(continuous_fixture, path):
    root, _, bundle, handoff, _, _ = continuous_fixture
    bad = deepcopy(bundle); bad["model_artifact_path"] = path
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_model_artifact_before_load(project_root=root, bundle=bad, handoff=handoff)


@pytest.mark.parametrize("mutation", ["missing", "extra", "order", "duplicate", "text", "nan", "posinf", "neginf", "fractional"])
def test_fail_closed_inputs(continuous_fixture, mutation):
    _, _, bundle, _, _, valid = continuous_fixture
    value = valid.copy(deep=True)
    if mutation == "missing": value = value.drop(columns=["a"])
    elif mutation == "extra": value["extra"] = 1
    elif mutation == "order": value = value[["b", "a", "age"]]
    elif mutation == "duplicate": value.columns = ["a", "a", "age"]
    elif mutation == "text": value["a"] = value["a"].astype(object); value.loc["x", "a"] = "bad"
    elif mutation == "nan": value.loc["x", "a"] = np.nan
    elif mutation == "posinf": value.loc["x", "a"] = np.inf
    elif mutation == "neginf": value.loc["x", "a"] = -np.inf
    else: value["age"] = value["age"].astype(float); value.loc["x", "age"] = 1.5
    with pytest.raises(smoke.InferenceInputError):
        smoke.normalize_continuous_inference_input(value, bundle=bundle)


def test_normalization_preserves_index_dtypes_and_input(continuous_fixture):
    _, _, bundle, _, _, valid = continuous_fixture
    before = valid.copy(deep=True)
    result = smoke.normalize_continuous_inference_input(valid, bundle=bundle)
    pd.testing.assert_frame_equal(valid, before)
    assert result.index.equals(valid.index)
    assert [str(result[c].dtype) for c in FEATURES] == ["float64", "float64", "int64"]


def test_prediction_shape_finite_unit_deterministic_and_immutable(continuous_fixture):
    _, pipeline, bundle, handoff, _, valid = continuous_fixture
    input_before, bundle_before, handoff_before = valid.copy(deep=True), deepcopy(bundle), deepcopy(handoff)
    first = smoke.predict_continuous_batch(pipeline, valid, bundle=bundle)
    second = smoke.predict_continuous_batch(pipeline, valid, bundle=bundle)
    assert first.shape == (2,) and np.isfinite(first).all() and np.array_equal(first, second)
    output = smoke.continuous_output_to_frame(first, bundle=bundle)
    assert output.columns.tolist() == ["example_id", "prediction", "unit"]
    assert output["example_id"].tolist() == ["x", "y"] and output["unit"].eq("unit").all()
    pd.testing.assert_frame_equal(valid, input_before)
    assert bundle == bundle_before and handoff == handoff_before


def test_exactly_one_predict_call_and_no_probability(continuous_fixture, monkeypatch):
    _, pipeline, bundle, _, _, valid = continuous_fixture
    calls = 0
    original = pipeline.predict
    def spy(value):
        nonlocal calls
        calls += 1
        return original(value)
    monkeypatch.setattr(pipeline, "predict", spy)
    smoke.predict_continuous_batch(pipeline, valid, bundle=bundle)
    assert calls == 1


@pytest.mark.parametrize(("field", "value"), [("type", "categorical"), ("scale", "other")])
def test_prediction_contract_mismatch_fails(continuous_fixture, field, value):
    _, _, bundle, _, _, valid = continuous_fixture
    bad = deepcopy(bundle); bad["prediction_contract"][field] = value
    with pytest.raises(smoke.InferenceContractError):
        smoke.normalize_continuous_inference_input(valid, bundle=bad)


def test_target_prediction_unit_mismatch_fails(continuous_fixture):
    _, _, bundle, _, _, valid = continuous_fixture
    bad = deepcopy(bundle); bad["target_contract"]["unit"] = "different"
    with pytest.raises(smoke.InferenceContractError):
        smoke.normalize_continuous_inference_input(valid, bundle=bad)


def test_helpers_and_notebook_preserve_consumer_boundary():
    source = inspect.getsource(smoke)
    assert ".fit(" not in source and ".fit_transform(" not in source and ".partial_fit(" not in source
    assert "read_csv" not in source
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/05_inference_demo.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    forbidden = (".fit(", ".fit_transform(", ".partial_fit(", "read_csv", "predict_proba", "predict_multiclass", "train.csv", "validation.csv", "test.csv", "accuracy_score", "confusion_matrix")
    assert not any(token in code for token in forbidden)


def test_synthetic_final_artifacts_are_not_modified(continuous_fixture):
    root, pipeline, bundle, handoff, manifest, valid = continuous_fixture
    directory = root / "models"
    before = {path.name: (smoke._sha256_file(path), path.stat().st_size) for path in directory.iterdir() if path.is_file()}
    smoke.validate_continuous_inference_readiness(handoff, bundle)
    smoke.validate_continuous_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    smoke.validate_model_artifact_before_load(project_root=root, bundle=bundle, handoff=handoff, manifest=manifest)
    smoke.predict_continuous_batch(pipeline, valid, bundle=bundle)
    after = {path.name: (smoke._sha256_file(path), path.stat().st_size) for path in directory.iterdir() if path.is_file()}
    assert before == after
