from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import scripts.smoke_predict as smoke


FEATURES = tuple(f"feature_{index:02d}" for index in range(1, 17))
OUTPUT_CLASSES = ("zeta", "alpha", "eta", "beta", "gamma", "delta", "epsilon")


@pytest.fixture()
def multiclass_contract(
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for class_position, label in enumerate(OUTPUT_CLASSES):
        for row_position in range(14):
            rows.append(
                {
                    feature: float(
                        class_position * 20
                        + row_position * 0.2
                        + feature_position * 0.01
                    )
                    for feature_position, feature in enumerate(FEATURES)
                }
            )
            labels.append(label)
    frame = pd.DataFrame(rows, columns=FEATURES)
    preprocess = ColumnTransformer(
        [("numerical", "passthrough", list(FEATURES))],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        class_weight=None,
        l2_regularization=0.0,
        learning_rate=0.1,
        max_iter=20,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)]).fit(
        frame, labels
    )
    estimator_order = [
        value.item() if isinstance(value, np.generic) else value
        for value in pipeline.named_steps["model"].classes_.tolist()
    ]
    runtime = smoke.current_runtime_versions()
    selected_hyperparameters = {
        "model__class_weight": None,
        "model__l2_regularization": 0.0,
        "model__learning_rate": 0.1,
        "model__max_iter": 20,
        "model__max_leaf_nodes": 7,
        "model__min_samples_leaf": 2,
    }
    preprocessing_contract = {
        "categorical_processing": "not_applicable",
        "feature_projection": list(FEATURES),
        "learned_preprocessing_in_notebook_02": False,
        "numerical_scaling": "none",
        "pipeline": "sklearn.pipeline.Pipeline",
        "scaling_fit_scope": "inside_training_fold_or_final_training_only",
    }
    imbalance_policy = {
        "class_weight": None,
        "resampling": "none",
        "strategy": "none",
    }
    descriptor = {
        "pipeline_class": "sklearn.pipeline.Pipeline",
        "steps": ["preprocess", "model"],
        "preprocess_class": (
            "sklearn.compose._column_transformer.ColumnTransformer"
        ),
        "model_class": (
            "sklearn.ensemble._hist_gradient_boosting.gradient_boosting."
            "HistGradientBoostingClassifier"
        ),
        "selected_hyperparameters": selected_hyperparameters,
        "random_state": 42,
        "feature_order": list(FEATURES),
        "preprocessing_contract": preprocessing_contract,
        "imbalance_policy": imbalance_policy,
        "transformed_feature_names": list(
            pipeline.named_steps["preprocess"].get_feature_names_out()
        ),
        "estimator_class_order": estimator_order,
        "output_class_order": list(OUTPUT_CLASSES),
        "fitted_state": {
            "n_iter": int(pipeline.named_steps["model"].n_iter_),
            "do_early_stopping": bool(
                pipeline.named_steps["model"].do_early_stopping_
            ),
            "is_fitted": True,
        },
        "runtime_versions": runtime,
    }
    fingerprint = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected_dtypes = {feature: "numeric" for feature in FEATURES}
    bundle: dict[str, Any] = {
        "schema_version": "inference-bundle.v2",
        "artifact_type": "inference_bundle",
        "dataset_slug": "synthetic-multiclass",
        "problem_type": "multiclass_classification",
        "model_id": "synthetic_hgb",
        "model_family": "HistGradientBoostingClassifier",
        "model_artifact_path": (
            "artifacts/models/synthetic-multiclass/final-pipeline.joblib"
        ),
        "model_artifact_sha256": "placeholder",
        "model_state_fingerprint": fingerprint,
        "model_state_descriptor": descriptor,
        "selected_hyperparameters": selected_hyperparameters,
        "estimator_random_state": 42,
        "feature_policy": "all_features",
        "feature_columns": list(FEATURES),
        "required_input_columns": list(FEATURES),
        "numerical_features": list(FEATURES),
        "categorical_features": [],
        "identifier_columns_excluded": [],
        "expected_input_dtypes": expected_dtypes,
        "expected_input_schema": [
            {
                "name": feature,
                "role": "numerical",
                "required": True,
                "expected_dtype": "numeric",
                "missing_value_behavior": "reject",
            }
            for feature in FEATURES
        ],
        "prohibited_input_columns": ["target"],
        "missing_value_policy": {
            "strategy": "reject_missing_required_values",
            "learned_imputation_in_final_pipeline": False,
            "prepared_training_missing_value_count": 0,
        },
        "target_column": "target",
        "target_classes": list(OUTPUT_CLASSES),
        "target_semantics": "nominal_unordered",
        "estimator_class_order": estimator_order,
        "output_class_order": list(OUTPUT_CLASSES),
        "decision_rule": "argmax_class_score_or_probability",
        "preprocessing_contract": preprocessing_contract,
        "imbalance_policy": imbalance_policy,
        "runtime_version_requirements": runtime,
        "inference_output_contract": {
            "predicted_class": {
                "type": "string",
                "allowed_values": list(OUTPUT_CLASSES),
            },
            "class_order": list(OUTPUT_CLASSES),
            "class_probabilities": {
                "type": "array",
                "length": 7,
                "aligned_to": "class_order",
                "finite": True,
                "row_sum": 1.0,
            },
            "decision_rule": "argmax_class_score_or_probability",
            "binary_threshold": "not_applicable",
            "operational_prediction_available": False,
        },
        "readiness": {
            "educational_inference_demo_ready": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "serialization_reload_validated": True,
            "inference_smoke_test_completed": True,
            "operational_modeling_ready": False,
        },
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
    }
    handoff: dict[str, Any] = {
        "schema_version": "final-model-handoff.v2",
        "artifact_type": "final_model_handoff",
        "dataset_slug": bundle["dataset_slug"],
        "problem_type": bundle["problem_type"],
        "selected_model_id": bundle["model_id"],
        "selected_model_family": bundle["model_family"],
        "model_state_fingerprint": fingerprint,
        "feature_policy": bundle["feature_policy"],
        "feature_order": list(FEATURES),
        "target_column": bundle["target_column"],
        "target_classes": list(OUTPUT_CLASSES),
        "target_semantics": bundle["target_semantics"],
        "selected_hyperparameters": selected_hyperparameters,
        "preprocessing": preprocessing_contract,
        "imbalance_policy": imbalance_policy,
        "decision_rule": bundle["decision_rule"],
        "estimator_class_order": estimator_order,
        "output_class_order": list(OUTPUT_CLASSES),
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "final_test_evaluation_completed": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "no_model_selection_decision_changed_after_test": True,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "api_implemented": False,
        "final_references": {
            "model_artifact": {
                "path": bundle["model_artifact_path"],
                "byte_sha256": bundle["model_artifact_sha256"],
                "semantic_sha256": fingerprint,
            }
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": "final-model-manifest.v2",
        "artifact_type": "final_model_manifest",
        "dataset_slug": bundle["dataset_slug"],
        "problem_type": bundle["problem_type"],
        "selected_model_id": bundle["model_id"],
        "selected_model_family": bundle["model_family"],
        "model_state_fingerprint": fingerprint,
        "model_state_descriptor": descriptor,
        "feature_columns": list(FEATURES),
        "target_column": bundle["target_column"],
        "target_classes": list(OUTPUT_CLASSES),
        "selected_hyperparameters": selected_hyperparameters,
        "preprocessing_contract": preprocessing_contract,
        "imbalance_policy": imbalance_policy,
        "decision_rule": bundle["decision_rule"],
        "estimator_class_order": estimator_order,
        "output_class_order": list(OUTPUT_CLASSES),
        "model_artifact_path": bundle["model_artifact_path"],
        "model_artifact_byte_sha256": bundle["model_artifact_sha256"],
        "runtime_versions": runtime,
    }
    return pipeline, bundle, handoff, manifest, frame


@pytest.fixture()
def valid_row(multiclass_contract) -> dict[str, float]:
    return multiclass_contract[4].iloc[0].to_dict()


def _materialize_loader_inputs(
    tmp_path: Path,
    pipeline: Pipeline,
    bundle: dict[str, Any],
    handoff: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = deepcopy(bundle)
    handoff = deepcopy(handoff)
    manifest = deepcopy(manifest)
    model_path = tmp_path / bundle["model_artifact_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle["model_artifact_sha256"] = digest
    handoff["final_references"]["model_artifact"]["byte_sha256"] = digest
    manifest["model_artifact_byte_sha256"] = digest
    manifest_path = model_path.parent / "final-model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    handoff["final_references"]["final_model_manifest"] = {
        "path": manifest_path.relative_to(tmp_path).as_posix(),
        "byte_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "semantic_sha256": "synthetic-manifest",
    }
    return bundle, handoff, manifest


def test_v2_readiness_and_schema_dispatch(multiclass_contract) -> None:
    _, bundle, handoff, _, _ = multiclass_contract
    smoke.validate_inference_readiness(handoff, bundle)
    smoke.validate_multiclass_inference_readiness(handoff, bundle)
    changed = deepcopy(bundle)
    changed["schema_version"] = "inference-bundle.v99"
    with pytest.raises(smoke.InferenceContractError, match="Unsupported"):
        smoke.validate_inference_readiness(handoff, changed)
    changed_handoff = deepcopy(handoff)
    changed_handoff["schema_version"] = "final-model-handoff.v99"
    with pytest.raises(smoke.InferenceContractError, match="Unsupported"):
        smoke.validate_inference_readiness(changed_handoff, bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("educational_final_model_completed", False),
        ("test_partition_evaluation_count", 2),
        ("test_partition_used_for_adjustment", True),
        ("no_model_selection_decision_changed_after_test", False),
        ("operational_modeling_ready", True),
        ("operational_validity", "confirmed"),
        ("api_implemented", True),
    ],
)
def test_v2_readiness_rejects_divergence(
    multiclass_contract, field: str, value: Any
) -> None:
    _, bundle, handoff, _, _ = multiclass_contract
    changed = deepcopy(handoff)
    changed[field] = value
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_inference_readiness(changed, bundle)


def test_v2_bundle_handoff_manifest_alignment(multiclass_contract) -> None:
    _, bundle, handoff, manifest, _ = multiclass_contract
    smoke.validate_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    changed = deepcopy(bundle)
    changed["output_class_order"] = list(reversed(OUTPUT_CLASSES))
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_bundle_handoff_alignment(handoff, changed, manifest=manifest)


@pytest.mark.parametrize("kind", ["mapping", "series", "dataframe"])
def test_v2_mapping_series_dataframe_and_non_mutation(
    multiclass_contract, valid_row, kind: str
) -> None:
    pipeline, bundle, _, _, _ = multiclass_contract
    if kind == "mapping":
        value: Any = dict(valid_row)
        snapshot = deepcopy(value)
    elif kind == "series":
        value = pd.Series(valid_row, name="single-case")
        snapshot = value.copy(deep=True)
    else:
        value = pd.DataFrame([valid_row], index=pd.Index([17], name="case"))
        snapshot = value.copy(deep=True)
    normalized = smoke.normalize_inference_input(value, bundle=bundle)
    prediction = smoke.predict_multiclass(pipeline, value, bundle=bundle)
    assert list(normalized.dataframe.columns) == list(FEATURES)
    assert normalized.unknown_categories_report == ()
    assert prediction["predicted_class"] in OUTPUT_CLASSES
    if kind == "mapping":
        assert value == snapshot
    elif kind == "series":
        pd.testing.assert_series_equal(value, snapshot)
        assert normalized.dataframe.index.tolist() == ["single-case"]
    else:
        pd.testing.assert_frame_equal(value, snapshot)
        assert normalized.dataframe.index.tolist() == [17]


def test_v2_zero_categorical_requires_no_vocabularies(
    multiclass_contract, valid_row
) -> None:
    _, bundle, _, _, _ = multiclass_contract
    assert "fitted_categorical_vocabularies" not in bundle
    result = smoke.normalize_inference_input(valid_row, bundle=bundle)
    assert result.unknown_categories_dict() == {}


def test_v2_reordered_and_numeric_string_inputs_are_normalized(
    multiclass_contract, valid_row
) -> None:
    _, bundle, _, _, _ = multiclass_contract
    reordered = {key: str(valid_row[key]) for key in reversed(FEATURES)}
    result = smoke.normalize_inference_input(reordered, bundle=bundle)
    assert list(result.dataframe.columns) == list(FEATURES)
    assert all(str(dtype) == "float64" for dtype in result.dataframe.dtypes)


@pytest.mark.parametrize("bad_kind", ["missing", "extra", "target", "text", "nan", "inf"])
def test_v2_invalid_inputs_are_rejected(
    multiclass_contract, valid_row, bad_kind: str
) -> None:
    _, bundle, _, _, _ = multiclass_contract
    changed = dict(valid_row)
    if bad_kind == "missing":
        changed.pop(FEATURES[0])
    elif bad_kind == "extra":
        changed["unexpected"] = 1.0
    elif bad_kind == "target":
        changed["target"] = OUTPUT_CLASSES[0]
    elif bad_kind == "text":
        changed[FEATURES[0]] = "not-a-number"
    elif bad_kind == "nan":
        changed[FEATURES[0]] = np.nan
    else:
        changed[FEATURES[0]] = np.inf
    with pytest.raises(smoke.InferenceInputError):
        smoke.normalize_inference_input(changed, bundle=bundle)


def test_v2_duplicate_columns_are_rejected(multiclass_contract, valid_row) -> None:
    _, bundle, _, _, _ = multiclass_contract
    values = [valid_row[feature] for feature in FEATURES]
    frame = pd.DataFrame(
        [[values[0], *values]],
        columns=[FEATURES[0], *FEATURES],
    )
    with pytest.raises(smoke.InferenceInputError, match="duplicate"):
        smoke.normalize_inference_input(frame, bundle=bundle)


def test_v2_pipeline_contract_and_class_order(multiclass_contract) -> None:
    pipeline, bundle, _, manifest, _ = multiclass_contract
    smoke.validate_loaded_pipeline_contract(pipeline, bundle=bundle, manifest=manifest)
    assert list(pipeline.named_steps["model"].classes_) == bundle[
        "estimator_class_order"
    ]
    assert bundle["estimator_class_order"] != bundle["output_class_order"]
    changed = deepcopy(bundle)
    changed["estimator_class_order"] = list(reversed(changed["estimator_class_order"]))
    with pytest.raises(smoke.InferenceContractError, match="class order"):
        smoke.validate_loaded_pipeline_contract(pipeline, bundle=changed)


def test_v2_unfitted_or_wrong_pipeline_is_rejected(multiclass_contract) -> None:
    _, bundle, _, _, _ = multiclass_contract
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_loaded_pipeline_contract(
            HistGradientBoostingClassifier(), bundle=bundle
        )
    unfitted = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numerical", "passthrough", list(FEATURES))],
                    remainder="drop",
                    sparse_threshold=0.0,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    class_weight=None,
                    l2_regularization=0.0,
                    learning_rate=0.1,
                    max_iter=20,
                    max_leaf_nodes=7,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )
    with pytest.raises(smoke.InferenceContractError, match="fitted"):
        smoke.validate_loaded_pipeline_contract(unfitted, bundle=bundle)


def test_v2_probability_remapping_argmax_and_sum(
    multiclass_contract, valid_row
) -> None:
    pipeline, bundle, _, _, _ = multiclass_contract
    normalized = smoke.normalize_inference_input(valid_row, bundle=bundle).dataframe
    raw = pipeline.predict_proba(normalized)
    estimator_order = bundle["estimator_class_order"]
    expected = raw[
        0,
        [estimator_order.index(label) for label in bundle["output_class_order"]],
    ]
    result = smoke.predict_multiclass(pipeline, valid_row, bundle=bundle)
    assert result["class_order"] == bundle["output_class_order"]
    assert len(result["class_probabilities"]) == 7
    assert np.allclose(result["class_probabilities"], expected, rtol=0.0, atol=0.0)
    assert all(np.isfinite(result["class_probabilities"]))
    assert sum(result["class_probabilities"]) == pytest.approx(1.0)
    assert result["predicted_class"] == result["class_order"][
        int(np.argmax(result["class_probabilities"]))
    ]
    assert result["operational_prediction_available"] is False


def test_v2_batch_index_order_single_equivalence_and_determinism(
    multiclass_contract
) -> None:
    pipeline, bundle, _, _, frame = multiclass_contract
    batch = frame.iloc[[0, 21, 55, 83]].copy(deep=True)
    batch.index = pd.Index(["case-d", "case-a", "case-c", "case-b"], name="case")
    snapshot = batch.copy(deep=True)
    first = smoke.predict_multiclass_batch(pipeline, batch, bundle=bundle)
    second = smoke.predict_multiclass_batch(pipeline, batch, bundle=bundle)
    pd.testing.assert_frame_equal(batch, snapshot)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.tolist() == batch.index.tolist()
    assert len(first) == len(batch)
    single = smoke.predict_multiclass(
        pipeline, batch.iloc[2], bundle=bundle
    )
    assert single["predicted_class"] == first.iloc[2]["predicted_class"]
    assert single["class_order"] == first.iloc[2]["class_order"]
    assert np.allclose(
        single["class_probabilities"],
        first.iloc[2]["class_probabilities"],
        rtol=0.0,
        atol=0.0,
    )
    presentation = smoke.multiclass_output_to_frame(first)
    assert presentation.index.tolist() == batch.index.tolist()
    assert list(presentation) == [
        "predicted_class",
        *(f"probability_{label}" for label in OUTPUT_CLASSES),
    ]


def test_v2_invalid_probability_rows_are_rejected(
    multiclass_contract, valid_row, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, bundle, _, _, _ = multiclass_contract
    monkeypatch.setattr(
        pipeline,
        "predict_proba",
        lambda _: np.full((1, 7), 0.2, dtype=float),
    )
    with pytest.raises(smoke.InferenceContractError, match="sum to one"):
        smoke.predict_multiclass(
            pipeline, valid_row, bundle=bundle, validate_pipeline=False
        )


def test_v2_model_path_missing_hash_and_unsafe_are_rejected(
    tmp_path: Path, multiclass_contract
) -> None:
    _, bundle, handoff, _, _ = multiclass_contract
    with pytest.raises(FileNotFoundError):
        smoke.validate_model_artifact_before_load(
            project_root=tmp_path, bundle=bundle, handoff=handoff
        )
    changed = deepcopy(bundle)
    changed["model_artifact_path"] = "../unsafe.joblib"
    with pytest.raises(smoke.InferenceContractError):
        smoke.validate_model_artifact_before_load(
            project_root=tmp_path, bundle=changed, handoff=handoff
        )
    path = tmp_path / bundle["model_artifact_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wrong")
    with pytest.raises(smoke.TrustedModelSourceError):
        smoke.validate_model_artifact_before_load(
            project_root=tmp_path, bundle=bundle, handoff=handoff
        )


@pytest.mark.parametrize("failure", ["runtime", "trust", "success"])
def test_v2_validated_loader_order_and_single_deserialization(
    tmp_path: Path,
    multiclass_contract,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    pipeline, bundle, handoff, manifest, _ = multiclass_contract
    bundle, handoff, manifest = _materialize_loader_inputs(
        tmp_path, pipeline, bundle, handoff, manifest
    )
    monkeypatch.setattr(
        smoke,
        "load_and_validate_final_model_handoff",
        lambda **_: deepcopy(handoff),
    )
    monkeypatch.setattr(
        smoke,
        "load_and_validate_inference_bundle",
        lambda **_: deepcopy(bundle),
    )
    monkeypatch.setattr(
        smoke,
        "load_and_validate_final_model_manifest",
        lambda **_: deepcopy(manifest),
    )
    calls = 0

    def loader(**_: Any) -> Pipeline:
        nonlocal calls
        calls += 1
        return pipeline

    observed = dict(bundle["runtime_version_requirements"])
    trusted = True
    if failure == "runtime":
        observed["scikit_learn"] = "0.0.0"
    elif failure == "trust":
        trusted = False
    if failure == "runtime":
        with pytest.raises(smoke.RuntimeCompatibilityError):
            smoke.load_validated_inference_pipeline(
                project_root=tmp_path,
                handoff_path="handoff.json",
                bundle_path="bundle.json",
                trusted_source=trusted,
                observed_runtime_versions=observed,
                loader=loader,
            )
        assert calls == 0
    elif failure == "trust":
        with pytest.raises(smoke.TrustedModelSourceError):
            smoke.load_validated_inference_pipeline(
                project_root=tmp_path,
                handoff_path="handoff.json",
                bundle_path="bundle.json",
                trusted_source=trusted,
                observed_runtime_versions=observed,
                loader=loader,
            )
        assert calls == 0
    else:
        loaded, loaded_handoff, loaded_bundle, report = (
            smoke.load_validated_inference_pipeline(
                project_root=tmp_path,
                handoff_path="handoff.json",
                bundle_path="bundle.json",
                trusted_source=True,
                observed_runtime_versions=observed,
                loader=loader,
            )
        )
        assert loaded is pipeline
        assert loaded_handoff == handoff
        assert loaded_bundle == bundle
        assert report.compatible
        assert calls == 1


def test_v2_consumer_helpers_have_no_fit_dataset_reads_metrics_or_constants() -> None:
    source = inspect.getsource(smoke)
    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert ".partial_fit(" not in source
    assert "read_csv" not in source
    assert "prepared.csv" not in source
    assert "train.csv" not in source
    assert "validation.csv" not in source
    assert "test.csv" not in source
    assert "confusion_matrix" not in source
    assert "classification_report" not in source
    assert "accuracy_score" not in source
    assert "dry-bean" not in source
    assert "SEKER" not in source


def test_notebook_05_is_clean_and_respects_inference_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/05_inference_demo.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in code_cells
    )
    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    prohibited = (
        ".fit(",
        ".fit_transform(",
        ".partial_fit(",
        "read_csv",
        "test.csv",
        "f1_score",
        "accuracy_score",
        "balanced_accuracy_score",
        "recall_score",
        "classification_report",
        "confusion_matrix",
        "log_loss",
        "roc_auc_score",
        "average_precision_score",
        "positive_class_probability",
    )
    assert not any(token in code for token in prohibited)
    assert "predict_continuous_batch" in code
    assert "predict_multiclass" not in code
    assert "trusted_source=True" in code
