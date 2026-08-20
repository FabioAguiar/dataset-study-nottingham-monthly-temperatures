from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import scripts.smoke_predict as smoke


@pytest.fixture()
def fitted_contract() -> tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]:
    features = ["category", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = ["category", "SeniorCitizen"]
    numerical = ["tenure", "MonthlyCharges", "TotalCharges"]
    frame = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "SeniorCitizen": [0, 1, 0, 1, 1, 0, 1, 0],
            "tenure": [0, 1, 2, 3, 4, 5, 6, 7],
            "MonthlyCharges": [20.0, 80.0, 30.0, 90.0, 40.0, 75.0, 35.0, 70.0],
            "TotalCharges": [0.0, 80.0, 60.0, 270.0, 160.0, 375.0, 210.0, 490.0],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    preprocess = ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", numerical),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        l2_regularization=1.0,
        learning_rate=0.1,
        max_depth=3,
        max_iter=20,
        max_leaf_nodes=7,
        min_samples_leaf=1,
        random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)]).fit(frame, target)
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    runtime = smoke.current_runtime_versions()
    bundle: dict[str, Any] = {
        "dataset_slug": "synthetic-study",
        "model_id": "hist_gradient_boosting",
        "model_family": "HistGradientBoostingClassifier",
        "feature_columns": features,
        "required_input_columns": features,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "expected_input_dtypes": {
            "category": "string",
            "SeniorCitizen": "integer",
            "tenure": "integer",
            "MonthlyCharges": "numeric",
            "TotalCharges": "numeric",
        },
        "fitted_categorical_vocabularies": {
            column: [value.item() if isinstance(value, np.generic) else value for value in values]
            for column, values in zip(categorical, encoder.categories_, strict=True)
        },
        "missing_value_policy": {
            "preparation_rules": [
                {
                    "column": "TotalCharges",
                    "condition_column": "tenure",
                    "condition_value": 0,
                    "blank_replacement": 0.0,
                    "strip_strings": True,
                }
            ]
        },
        "prohibited_input_columns": ["customerID", "Churn"],
        "target_classes": ["No", "Yes"],
        "target_encoding": {"No": 0, "Yes": 1},
        "positive_class": "Yes",
        "positive_encoded_label": 1,
        "negative_class": "No",
        "target_column_metadata_only": "Churn",
        "educational_decision_threshold": 0.4,
        "model_artifact_path": "artifacts/models/synthetic/final-pipeline.joblib",
        "model_artifact_sha256": "placeholder",
        "model_state_fingerprint": "synthetic-state-fingerprint",
        "runtime_version_requirements": runtime,
        "preprocessing_embedded": True,
        "categorical_strategy": "one_hot",
        "unknown_category_policy": "ignore_and_report",
        "drop_category": None,
        "numerical_scaling": "none",
        "pipeline_class": "sklearn.pipeline.Pipeline",
        "selected_hyperparameters": {
            "model__l2_regularization": 1.0,
            "model__learning_rate": 0.1,
            "model__max_depth": 3,
            "model__max_iter": 20,
            "model__max_leaf_nodes": 7,
            "model__min_samples_leaf": 1,
        },
        "estimator_random_state": 42,
        "transformed_feature_names": list(
            pipeline.named_steps["preprocess"].get_feature_names_out()
        ),
        "readiness": {
            "educational_inference_demo_ready": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "operational_modeling_ready": False,
        },
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
        "output_contract": {"operational_prediction_available": False},
    }
    handoff: dict[str, Any] = {
        "dataset_slug": bundle["dataset_slug"],
        "selected_model_id": bundle["model_id"],
        "selected_model_family": bundle["model_family"],
        "model_state_fingerprint": bundle["model_state_fingerprint"],
        "feature_order": features,
        "target_encoding": bundle["target_encoding"],
        "positive_class": bundle["positive_class"],
        "educational_threshold": bundle["educational_decision_threshold"],
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "final_test_evaluation_completed": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_sealed_at_input": True,
        "test_partition_evaluated": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "test_partition_used_for_model_selection": False,
        "test_partition_used_for_threshold_selection": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
        "operational_threshold": "unresolved",
        "api_implemented": False,
        "final_references": {
            "model_artifact": {
                "path": bundle["model_artifact_path"],
                "byte_sha256": bundle["model_artifact_sha256"],
                "semantic_sha256": bundle["model_state_fingerprint"],
            }
        },
    }
    manifest: dict[str, Any] = {
        "dataset_slug": bundle["dataset_slug"],
        "selected_model_id": bundle["model_id"],
        "selected_model_family": bundle["model_family"],
        "fitted_state_semantic_fingerprint": bundle["model_state_fingerprint"],
        "feature_columns": features,
        "target_encoding": bundle["target_encoding"],
        "educational_threshold": bundle["educational_decision_threshold"],
        "model_artifact_path": bundle["model_artifact_path"],
        "model_artifact_byte_sha256": bundle["model_artifact_sha256"],
        "fitted_state_descriptor": {
            "steps": ["preprocess", "model"],
            "feature_order": features,
            "transformed_feature_names": bundle["transformed_feature_names"],
            "categorical_vocabularies": bundle["fitted_categorical_vocabularies"],
        },
    }
    return pipeline, bundle, handoff, manifest


@pytest.fixture()
def valid_row() -> dict[str, Any]:
    return {
        "category": "A",
        "SeniorCitizen": 0,
        "tenure": 2,
        "MonthlyCharges": 30.0,
        "TotalCharges": 60.0,
    }


def _component(report: smoke.RuntimeCompatibilityReport, name: str) -> smoke.RuntimeComponentReport:
    return next(component for component in report.components if component.component == name)


def _write_artifact_set(
    tmp_path: Path,
    bundle: dict[str, Any],
    handoff: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    updated_bundle = deepcopy(bundle)
    updated_handoff = deepcopy(handoff)
    updated_manifest = deepcopy(manifest)
    model_path = tmp_path / Path(updated_bundle["model_artifact_path"])
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"synthetic trusted artifact")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    updated_bundle["model_artifact_sha256"] = digest
    updated_handoff["final_references"]["model_artifact"]["byte_sha256"] = digest
    updated_manifest["model_artifact_byte_sha256"] = digest
    manifest_path = tmp_path / "artifacts/models/synthetic/final-model-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        __import__("json").dumps(updated_manifest, sort_keys=True), encoding="utf-8"
    )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    updated_handoff["final_references"]["final_model_manifest"] = {
        "path": "artifacts/models/synthetic/final-model-manifest.json",
        "byte_sha256": manifest_hash,
        "semantic_sha256": "not-used-by-this-unit-test",
    }
    return updated_bundle, updated_handoff, updated_manifest


# A. Runtime

def test_runtime_exact_compatible() -> None:
    versions = {"python": "3.13.13", "pandas": "3.0.5", "scikit_learn": "1.9.0", "joblib": "1.5.3"}
    report = smoke.validate_runtime_compatibility(versions, observed_versions=versions, mode="exact")
    assert report.compatible
    assert all(component.status == "compatible" for component in report.components)


def test_runtime_load_safe_compatible() -> None:
    versions = smoke.current_runtime_versions()
    assert smoke.validate_runtime_compatibility(versions, observed_versions=versions, mode="load_safe").compatible


def test_runtime_python_patch_difference_warns_and_is_safe() -> None:
    expected = {"python": "3.13.13", "pandas": "3.0.5", "scikit_learn": "1.9.0", "joblib": "1.5.3"}
    observed = dict(expected, python="3.13.5")
    with pytest.warns(smoke.RuntimeCompatibilityWarning):
        report = smoke.validate_runtime_compatibility(expected, observed_versions=observed, mode="load_safe")
    assert report.compatible
    assert _component(report, "python").status == "warning"


@pytest.mark.parametrize(
    ("component", "observed"),
    [("python", "3.12.9"), ("pandas", "3.0.4"), ("scikit_learn", "1.8.0"), ("joblib", "1.5.2")],
)
def test_runtime_load_safe_blocks_material_mismatch(component: str, observed: str) -> None:
    expected = {"python": "3.13.13", "pandas": "3.0.5", "scikit_learn": "1.9.0", "joblib": "1.5.3"}
    actual = dict(expected)
    actual[component] = observed
    with pytest.raises(smoke.RuntimeCompatibilityError) as exc:
        smoke.validate_runtime_compatibility(expected, observed_versions=actual, mode="load_safe")
    detail = _component(exc.value.report, component)
    assert detail.expected == expected[component]
    assert detail.observed == observed


def test_runtime_exact_reports_without_raising_by_default() -> None:
    expected = {"python": "3.13.13", "pandas": "3.0.5", "scikit_learn": "1.9.0", "joblib": "1.5.3"}
    report = smoke.validate_runtime_compatibility(expected, observed_versions=dict(expected, pandas="2.2.3"), mode="exact")
    assert not report.compatible
    assert _component(report, "pandas").status == "incompatible"


def test_runtime_does_not_install_warning_filters() -> None:
    source = inspect.getsource(smoke)
    assert "filterwarnings" not in source
    assert "simplefilter" not in source
    assert "catch_warnings" not in source


# B. Handoff and bundle

def test_readiness_contract_accepts_expected_state(fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    _, bundle, handoff, _ = fitted_contract
    smoke.validate_inference_readiness(handoff, bundle)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("educational_inference_demo_ready", False),
        ("model_artifact_materialized", False),
        ("model_bundle_materialized", False),
        ("operational_modeling_ready", True),
        ("operational_validity", "confirmed"),
        ("operational_threshold", 0.5),
    ],
)
def test_readiness_rejects_handoff_divergence(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], field: str, bad_value: Any
) -> None:
    _, bundle, handoff, _ = fitted_contract
    changed = deepcopy(handoff)
    changed[field] = bad_value
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_inference_readiness(changed, bundle)


@pytest.mark.parametrize(
    ("handoff_field", "bundle_field"),
    [
        ("dataset_slug", "dataset_slug"),
        ("model_state_fingerprint", "model_state_fingerprint"),
        ("feature_order", "feature_columns"),
        ("target_encoding", "target_encoding"),
        ("educational_threshold", "educational_decision_threshold"),
    ],
)
def test_alignment_rejects_divergence(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    handoff_field: str,
    bundle_field: str,
) -> None:
    _, bundle, handoff, manifest = fitted_contract
    changed = deepcopy(bundle)
    value = changed[bundle_field]
    changed[bundle_field] = ["different"] if isinstance(value, list) else "different"
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_bundle_handoff_alignment(handoff, changed, manifest=manifest)


# C. Security and loading

def test_relative_posix_and_windows_paths_are_accepted(
    tmp_path: Path, fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, bundle, handoff, manifest = fitted_contract
    bundle, handoff, manifest = _write_artifact_set(tmp_path, bundle, handoff, manifest)
    assert smoke.validate_model_artifact_before_load(project_root=tmp_path, bundle=bundle, handoff=handoff, manifest=manifest).is_file()
    win_bundle = deepcopy(bundle)
    win_handoff = deepcopy(handoff)
    win_bundle["model_artifact_path"] = bundle["model_artifact_path"].replace("/", "\\")
    win_handoff["final_references"]["model_artifact"]["path"] = win_bundle["model_artifact_path"]
    assert smoke.validate_model_artifact_before_load(project_root=tmp_path, bundle=win_bundle, handoff=win_handoff, manifest=None).is_file()


@pytest.mark.parametrize("path", ["/tmp/model.joblib", "C:\\models\\model.joblib", "../model.joblib", "artifacts/../model.joblib"])
def test_unsafe_model_paths_are_rejected(
    tmp_path: Path,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    path: str,
) -> None:
    _, bundle, handoff, _ = fitted_contract
    changed = deepcopy(bundle)
    changed["model_artifact_path"] = path
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_model_artifact_before_load(project_root=tmp_path, bundle=changed, handoff=handoff)


def test_missing_and_hash_divergent_artifacts_are_rejected(
    tmp_path: Path, fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, bundle, handoff, _ = fitted_contract
    with pytest.raises(FileNotFoundError):
        smoke.validate_model_artifact_before_load(project_root=tmp_path, bundle=bundle, handoff=handoff)
    path = tmp_path / Path(bundle["model_artifact_path"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    with pytest.raises(smoke.TrustedModelSourceError):
        smoke.validate_model_artifact_before_load(project_root=tmp_path, bundle=bundle, handoff=handoff)


def test_loader_not_called_after_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pipeline, bundle, handoff, manifest = fitted_contract
    bundle, handoff, _ = _write_artifact_set(tmp_path, bundle, handoff, manifest)
    monkeypatch.setattr(smoke, "load_and_validate_final_model_handoff", lambda **_: deepcopy(handoff))
    monkeypatch.setattr(smoke, "load_and_validate_inference_bundle", lambda **_: deepcopy(bundle))
    calls = 0
    def loader(**_: Any) -> Pipeline:
        nonlocal calls
        calls += 1
        return pipeline
    observed = dict(bundle["runtime_version_requirements"], pandas="0.0.0")
    with pytest.raises(smoke.RuntimeCompatibilityError):
        smoke.load_validated_inference_pipeline(
            project_root=tmp_path,
            handoff_path="handoff.json",
            bundle_path="bundle.json",
            trusted_source=True,
            observed_runtime_versions=observed,
            loader=loader,
        )
    assert calls == 0


def test_trusted_source_required_after_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pipeline, bundle, handoff, manifest = fitted_contract
    bundle, handoff, _ = _write_artifact_set(tmp_path, bundle, handoff, manifest)
    monkeypatch.setattr(smoke, "load_and_validate_final_model_handoff", lambda **_: deepcopy(handoff))
    monkeypatch.setattr(smoke, "load_and_validate_inference_bundle", lambda **_: deepcopy(bundle))
    calls = 0
    def loader(**_: Any) -> Pipeline:
        nonlocal calls
        calls += 1
        return pipeline
    with pytest.raises(smoke.TrustedModelSourceError):
        smoke.load_validated_inference_pipeline(
            project_root=tmp_path,
            handoff_path="handoff.json",
            bundle_path="bundle.json",
            trusted_source=False,
            observed_runtime_versions=bundle["runtime_version_requirements"],
            loader=loader,
        )
    assert calls == 0


def test_validated_loader_delegates_once_after_all_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pipeline, bundle, handoff, manifest = fitted_contract
    bundle, handoff, _ = _write_artifact_set(tmp_path, bundle, handoff, manifest)
    monkeypatch.setattr(smoke, "load_and_validate_final_model_handoff", lambda **_: deepcopy(handoff))
    monkeypatch.setattr(smoke, "load_and_validate_inference_bundle", lambda **_: deepcopy(bundle))
    calls = 0
    def loader(**_: Any) -> Pipeline:
        nonlocal calls
        calls += 1
        return pipeline
    loaded, _, _, report = smoke.load_validated_inference_pipeline(
        project_root=tmp_path,
        handoff_path="handoff.json",
        bundle_path="bundle.json",
        trusted_source=True,
        observed_runtime_versions=bundle["runtime_version_requirements"],
        loader=loader,
    )
    assert loaded is pipeline
    assert report.compatible
    assert calls == 1


def test_invalid_pipeline_shapes_are_rejected(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    pipeline, bundle, _, manifest = fitted_contract
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_loaded_pipeline_contract(pipeline.named_steps["model"], bundle=bundle)
    wrong_steps = Pipeline([("model", deepcopy(pipeline.named_steps["model"]))])
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_loaded_pipeline_contract(wrong_steps, bundle=bundle)
    unfitted = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numerical", "passthrough", bundle["numerical_features"]),
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop=None,
                                sparse_output=False,
                            ),
                            bundle["categorical_features"],
                        ),
                    ],
                    remainder="drop",
                    sparse_threshold=0.0,
                ),
            ),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]
    )
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_loaded_pipeline_contract(unfitted, bundle=bundle)
    wrong_classes = deepcopy(pipeline)
    wrong_classes.named_steps["model"].classes_ = np.array([1, 0])
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_loaded_pipeline_contract(wrong_classes, bundle=bundle, manifest=manifest)


# D/E/F. Input, missing values, categories
@pytest.mark.parametrize("kind", ["mapping", "series", "dataframe"])
def test_input_types_are_supported(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
    kind: str,
) -> None:
    _, bundle, _, _ = fitted_contract
    value: Any = valid_row
    if kind == "series":
        value = pd.Series(valid_row, name="row-7")
    elif kind == "dataframe":
        value = pd.DataFrame([valid_row], index=pd.Index([77], name="case"))
    result = smoke.normalize_inference_input(value, bundle=bundle)
    assert list(result.dataframe.columns) == bundle["feature_columns"]
    if kind == "series":
        assert result.dataframe.index.tolist() == ["row-7"]
    if kind == "dataframe":
        assert result.dataframe.index.tolist() == [77]


def test_input_is_defensively_copied_and_not_mutated(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    original = pd.DataFrame([valid_row])
    snapshot = original.copy(deep=True)
    result = smoke.normalize_inference_input(original, bundle=bundle)
    pd.testing.assert_frame_equal(original, snapshot)
    result.dataframe.loc[0, "category"] = "CHANGED"
    pd.testing.assert_frame_equal(original, snapshot)


def test_missing_duplicate_extra_and_prohibited_columns_fail(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    missing = dict(valid_row)
    missing.pop("category")
    with pytest.raises(smoke.InferenceInputError, match="Missing required"):
        smoke.normalize_inference_input(missing, bundle=bundle)
    extra = dict(valid_row, extra="x")
    with pytest.raises(smoke.InferenceInputError, match="Unexpected"):
        smoke.normalize_inference_input(extra, bundle=bundle)
    for prohibited in ("customerID", "Churn"):
        bad = dict(valid_row)
        bad[prohibited] = "x"
        with pytest.raises(smoke.InferenceInputError, match="Prohibited"):
            smoke.normalize_inference_input(bad, bundle=bundle)
    duplicated = pd.DataFrame([["A", "B", 0, 2, 30.0, 60.0]], columns=["category", "category", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"])
    with pytest.raises(smoke.InferenceInputError, match="duplicate"):
        smoke.normalize_inference_input(duplicated, bundle=bundle)
    with pytest.raises(smoke.InferenceInputError, match="at least one row"):
        smoke.normalize_inference_input(pd.DataFrame(columns=bundle["feature_columns"]), bundle=bundle)


def test_order_and_explicit_dtype_coercion(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    reversed_row = {key: valid_row[key] for key in reversed(list(valid_row))}
    reversed_row["SeniorCitizen"] = np.float64(0.0)
    reversed_row["tenure"] = "2"
    reversed_row["MonthlyCharges"] = np.float32(30.5)
    reversed_row["TotalCharges"] = " 61.0 "
    result = smoke.normalize_inference_input(reversed_row, bundle=bundle)
    assert list(result.dataframe.columns) == bundle["feature_columns"]
    assert str(result.dataframe["category"].dtype) == "string"
    assert str(result.dataframe["SeniorCitizen"].dtype) == "int64"
    assert str(result.dataframe["tenure"].dtype) == "int64"
    assert str(result.dataframe["MonthlyCharges"].dtype) == "float64"
    assert result.dataframe.iloc[0]["TotalCharges"] == 61.0


def test_declared_total_charges_blank_materialization_and_report(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    row = dict(valid_row, tenure=0, TotalCharges="   ")
    result = smoke.normalize_inference_input(row, bundle=bundle)
    assert result.dataframe.iloc[0]["TotalCharges"] == 0.0
    assert result.materializations_dict() == {"TotalCharges": 1}


def test_declared_blank_materialization_supports_pandas_string_dtype(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
) -> None:
    _, bundle, _, _ = fitted_contract
    original = pd.DataFrame([dict(valid_row, tenure=0, TotalCharges="   ")])
    original["TotalCharges"] = original["TotalCharges"].astype("string")
    snapshot = original.copy(deep=True)

    result = smoke.normalize_inference_input(original, bundle=bundle)

    pd.testing.assert_frame_equal(original, snapshot)
    assert str(result.dataframe["TotalCharges"].dtype) == "float64"
    assert result.dataframe.iloc[0]["TotalCharges"] == 0.0
    assert result.materializations_dict() == {"TotalCharges": 1}


@pytest.mark.parametrize("bad_value", ["", "not-a-number", None, np.nan])
def test_invalid_total_charges_values_fail(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
    bad_value: Any,
) -> None:
    _, bundle, _, _ = fitted_contract
    row = dict(valid_row, tenure=2, TotalCharges=bad_value)
    with pytest.raises(smoke.InferenceInputError):
        smoke.normalize_inference_input(row, bundle=bundle)


def test_missing_values_are_not_generically_imputed(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    with pytest.raises(smoke.InferenceInputError):
        smoke.normalize_inference_input(dict(valid_row, MonthlyCharges=np.nan), bundle=bundle)


def test_known_and_unknown_categories_are_deterministic_and_preserved(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    vocab_snapshot = deepcopy(bundle["fitted_categorical_vocabularies"])
    known = smoke.normalize_inference_input(valid_row, bundle=bundle)
    assert known.unknown_categories_dict() == {}
    frame = pd.DataFrame([
        dict(valid_row, category="Z"),
        dict(valid_row, category="Y"),
        dict(valid_row, category="Z"),
    ])
    unknown = smoke.normalize_inference_input(frame, bundle=bundle)
    assert unknown.dataframe["category"].tolist() == ["Z", "Y", "Z"]
    assert unknown.unknown_categories_dict() == {"category": ["Y", "Z"]}
    assert unknown.dataframe["SeniorCitizen"].dtype == "int64"
    assert bundle["fitted_categorical_vocabularies"] == vocab_snapshot


# G/H. Inference and output

def test_positive_probability_column_uses_classes(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    assert smoke.resolve_positive_probability_column(pipeline, bundle=bundle) == 1


def test_predict_proba_called_once_per_batch_and_order_preserved(
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    original = pipeline.predict_proba
    calls = 0
    def spy(value: pd.DataFrame) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(value)
    monkeypatch.setattr(pipeline, "predict_proba", spy)
    frame = pd.DataFrame([valid_row, dict(valid_row, category="B")], index=["first", "second"])
    output = smoke.predict_educational_batch(pipeline, frame, bundle=bundle)
    assert calls == 1
    assert output.index.tolist() == ["first", "second"]
    assert output["positive_class_probability"].map(lambda value: 0.0 <= float(value) <= 1.0).all()


def test_threshold_and_reverse_label_come_from_bundle(
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    monkeypatch.setattr(pipeline, "predict_proba", lambda _: np.array([[0.6, 0.4]]))
    at_threshold = smoke.predict_educational(pipeline, valid_row, bundle=bundle)
    assert at_threshold["educational_threshold"] == 0.4
    assert at_threshold["educational_prediction_encoded"] == 1
    assert at_threshold["educational_prediction_label"] == "Yes"
    monkeypatch.setattr(pipeline, "predict_proba", lambda _: np.array([[0.61, 0.39]]))
    below = smoke.predict_educational(pipeline, valid_row, bundle=bundle)
    assert below["educational_prediction_encoded"] == 0
    assert below["educational_prediction_label"] == "No"


def test_single_output_contract_uses_python_types_and_is_non_operational(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    runtime = smoke.validate_runtime_compatibility(bundle["runtime_version_requirements"], observed_versions=bundle["runtime_version_requirements"], mode="load_safe")
    output = smoke.predict_educational(pipeline, valid_row, bundle=bundle, runtime_report=runtime)
    assert isinstance(output["positive_class_probability"], float)
    assert isinstance(output["educational_prediction_encoded"], int)
    assert isinstance(output["educational_prediction_label"], str)
    assert output["operational_prediction_available"] is False
    assert output["runtime_compatibility_confirmed"] is True
    assert "customerID" not in output and "Churn" not in output


def test_prediction_does_not_fit_transform_persist_or_mutate(
    monkeypatch: pytest.MonkeyPatch,
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]],
    valid_row: dict[str, Any],
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    frame = pd.DataFrame([valid_row])
    snapshot = frame.copy(deep=True)
    monkeypatch.setattr(pipeline, "fit", lambda *_args, **_kwargs: pytest.fail("fit called"))
    monkeypatch.setattr(
        pipeline.named_steps["preprocess"],
        "fit_transform",
        lambda *_args, **_kwargs: pytest.fail("fit_transform called"),
    )
    smoke.predict_educational_batch(pipeline, frame, bundle=bundle)
    pd.testing.assert_frame_equal(frame, snapshot)
    source = inspect.getsource(smoke)
    assert ".to_csv(" not in source
    assert ".to_json(" not in source
    assert ".write_text(" not in source


def test_single_rejects_batch(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    pipeline, bundle, _, _ = fitted_contract
    with pytest.raises(smoke.InferenceInputError):
        smoke.predict_educational(pipeline, pd.DataFrame([valid_row, valid_row]), bundle=bundle)


# I/J. Independence and compatibility

def test_module_has_no_dataset_reads_network_or_dataset_specific_contract() -> None:
    source = inspect.getsource(smoke)
    assert "read_csv" not in source
    assert "prepared.csv" not in source
    assert "train.csv" not in source
    assert "validation.csv" not in source
    assert "test.csv" not in source
    assert "requests" not in source
    assert "urllib" not in source
    assert "telco-customer-churn" not in source


def test_deterministic_runtime_and_unknown_reports(
    fitted_contract: tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]], valid_row: dict[str, Any]
) -> None:
    _, bundle, _, _ = fitted_contract
    runtime_1 = smoke.validate_runtime_compatibility(bundle["runtime_version_requirements"], observed_versions=bundle["runtime_version_requirements"], mode="exact")
    runtime_2 = smoke.validate_runtime_compatibility(bundle["runtime_version_requirements"], observed_versions=bundle["runtime_version_requirements"], mode="exact")
    assert runtime_1.as_dict() == runtime_2.as_dict()
    frame = pd.DataFrame([dict(valid_row, category="Z"), dict(valid_row, category="Y")])
    assert smoke.normalize_inference_input(frame, bundle=bundle).unknown_categories_report == smoke.normalize_inference_input(frame, bundle=bundle).unknown_categories_report


def test_current_runtime_reports_required_components() -> None:
    versions = smoke.current_runtime_versions()
    assert versions == {
        "python": __import__("platform").python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
