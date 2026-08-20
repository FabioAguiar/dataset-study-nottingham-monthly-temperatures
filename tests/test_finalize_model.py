from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from scripts.finalize_model import (
    ArtifactConflictError,
    DuplicateTestEvaluationError,
    EvaluationGuard,
    FinalizationContractError,
    FrozenFinalizationContract,
    SerializationValidationError,
    TestAccessError as FinalTestAccessError,
    UntrustedArtifactError,
    assemble_final_training_data,
    build_final_test_evidence,
    canonical_json_bytes,
    compute_fitted_model_fingerprint,
    compute_generalization_deltas,
    compute_probability_metrics,
    describe_fitted_pipeline,
    evaluate_final_model_once,
    evaluate_fixed_threshold,
    freeze_finalization_decisions,
    inspect_final_artifact_set,
    load_and_validate_final_model_handoff,
    load_and_validate_inference_bundle,
    load_test_partition_after_freeze,
    load_trusted_pipeline_from_bundle,
    reconstruct_selected_pipeline,
    report_unknown_categories,
    runtime_versions,
    semantic_fingerprint,
    serialize_pipeline_to_staging,
    sha256_file,
    validate_existing_finalization_equivalence,
    validate_final_partition_roles,
    validate_finalization_contract,
    validate_frozen_model_contract,
    validate_serialized_pipeline,
    validate_test_access_gate,
    verify_pipeline_contract,
    write_final_model_artifacts,
)


@pytest.fixture
def selection_handoff() -> dict:
    return {
        "schema_version": "model-selection-handoff.v1",
        "artifact_type": "model_selection_handoff",
        "dataset_slug": "synthetic",
        "feature_columns": ["num", "cat"],
        "numerical_features": ["num"],
        "categorical_features": ["cat"],
        "target_encoding": {"No": 0, "Yes": 1},
        "positive_class": "Yes",
        "selected_model_id": "hist_gradient_boosting",
        "selected_model_family": "HistGradientBoostingClassifier",
        "selected_hyperparameters": {
            "model__l2_regularization": 0.1,
            "model__learning_rate": 0.1,
            "model__max_depth": 2,
            "model__max_iter": 20,
            "model__max_leaf_nodes": 5,
            "model__min_samples_leaf": 2,
        },
        "selected_preprocessing_contract": {
            "categorical_strategy": "one_hot",
            "unknown_category_policy": "ignore_and_report",
            "drop_category": None,
            "numerical_scaling_for_selected_family": "none",
        },
        "selected_educational_threshold": {
            "threshold": 0.35,
            "scenario_id": "minimum_recall_0_80",
            "precision": 0.6,
            "recall": 0.8,
        },
        "selected_validation_metrics": {
            "average_precision": 0.75,
            "roc_auc": 0.8,
            "log_loss": 0.5,
            "brier_score": 0.17,
            "precision": 0.65,
            "recall": 0.6,
        },
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_model_trained": False,
        "model_artifact": None,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "readiness": {
            "final_model_training_ready": True,
        },
    }


@pytest.fixture
def contract(selection_handoff: dict) -> FrozenFinalizationContract:
    return freeze_finalization_decisions(
        dataset_slug="synthetic",
        model_selection_handoff=selection_handoff,
        identifier_columns=["id"],
        target_column="target",
        target_classes=["No", "Yes"],
        estimator_random_state=7,
    )


def make_frame(start: int, rows: int, *, unknown: bool = False) -> pd.DataFrame:
    values = []
    for i in range(start, start + rows):
        label = "Yes" if i % 3 == 0 or i % 5 == 0 else "No"
        category = "C" if unknown and i == start else ("A" if i % 2 == 0 else "B")
        values.append({"id": f"c-{i}", "num": float(i % 11), "cat": category, "target": label})
    frame = pd.DataFrame(values)
    frame["cat"] = frame["cat"].astype("string")
    frame["target"] = frame["target"].astype("string")
    return frame


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_frame(0, 36), make_frame(36, 18), make_frame(54, 18, unknown=True)


@pytest.fixture
def fitted(contract: FrozenFinalizationContract, frames):
    train, validation, _ = frames
    data = assemble_final_training_data(train=train, validation=validation, contract=contract)
    pipeline = reconstruct_selected_pipeline(
        estimator=HistGradientBoostingClassifier(random_state=7), contract=contract
    )
    pipeline.fit(data.features, data.target)
    return pipeline, data


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "bad.v1"),
        ("artifact_type", "bad"),
        ("test_partition_sealed", False),
        ("test_partition_evaluated", True),
        ("final_model_trained", True),
        ("model_artifact", "model.joblib"),
        ("model_artifact_materialized", True),
        ("model_bundle_materialized", True),
        ("operational_modeling_ready", True),
        ("operational_validity", "confirmed"),
        ("operational_threshold", 0.5),
    ],
)
def test_invalid_upstream_contract_fields(selection_handoff, field, value):
    payload = copy.deepcopy(selection_handoff)
    payload[field] = value
    with pytest.raises(FinalizationContractError):
        validate_finalization_contract(payload)


def test_final_training_ready_required(selection_handoff):
    selection_handoff["readiness"]["final_model_training_ready"] = False
    with pytest.raises(FinalizationContractError):
        validate_finalization_contract(selection_handoff)


def test_valid_contract(selection_handoff, contract):
    validate_finalization_contract(selection_handoff)
    validate_frozen_model_contract(contract, expected_model_id="hist_gradient_boosting")
    assert contract.educational_threshold == 0.35
    assert contract.threshold_selection_partition == "validation"


def test_absolute_path_rejected(selection_handoff):
    selection_handoff["model_artifact_path"] = "/tmp/model.joblib"
    with pytest.raises(FinalizationContractError):
        validate_finalization_contract(selection_handoff)


def test_feature_and_target_roles(contract, frames):
    train, validation, _ = frames
    before_train = train.copy(deep=True)
    before_validation = validation.copy(deep=True)
    final = assemble_final_training_data(train=train, validation=validation, contract=contract)
    assert list(final.features.columns) == ["num", "cat"]
    assert "id" not in final.features and "target" not in final.features
    assert final.row_count == 54
    assert set(final.target.unique()) == {0, 1}
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(validation, before_validation)


def test_partition_order_and_missing_feature_rejected(contract, frames):
    train, _, _ = frames
    with pytest.raises(FinalizationContractError):
        validate_final_partition_roles(train[["id", "cat", "num", "target"]], contract=contract, partition_name="train")
    with pytest.raises(FinalizationContractError):
        validate_final_partition_roles(train.drop(columns="cat"), contract=contract, partition_name="train")


def test_target_encoding_and_positive_class_validation(contract):
    bad = copy.deepcopy(contract)
    object.__setattr__(bad, "target_encoding", (("No", 1), ("Yes", 0)))
    with pytest.raises(FinalizationContractError):
        validate_frozen_model_contract(bad)




def test_model_family_mismatch_rejected(contract):
    with pytest.raises(FinalizationContractError):
        reconstruct_selected_pipeline(
            estimator=DecisionTreeClassifier(random_state=7), contract=contract
        )


def test_nontrivial_indices_preserved_on_inputs(contract, frames):
    train, validation, _ = frames
    train = train.copy(deep=True)
    validation = validation.copy(deep=True)
    train.index = range(1000, 1000 + len(train))
    validation.index = range(5000, 5000 + len(validation))
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)
    final = assemble_final_training_data(
        train=train, validation=validation, contract=contract
    )
    assert final.features.index.tolist() == list(range(len(final.features)))
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_integer_categories_remain_categorical(contract, frames):
    train, validation, test = (frame.copy(deep=True) for frame in frames)
    mapping = {"A": 0, "B": 1, "C": 2}
    for frame in (train, validation, test):
        frame["cat"] = frame["cat"].map(mapping).astype("int64")
    data = assemble_final_training_data(
        train=train, validation=validation, contract=contract
    )
    pipeline = reconstruct_selected_pipeline(
        estimator=HistGradientBoostingClassifier(random_state=7), contract=contract
    )
    pipeline.fit(data.features, data.target)
    report = report_unknown_categories(
        fitted_pipeline=pipeline,
        features=test[["num", "cat"]],
        categorical_features=["cat"],
    )
    assert report == {"cat": [2]}

def test_pipeline_reconstruction_and_fitted_state(contract, fitted):
    pipeline = reconstruct_selected_pipeline(
        estimator=HistGradientBoostingClassifier(random_state=7), contract=contract
    )
    assert isinstance(pipeline, Pipeline)
    verify_pipeline_contract(pipeline, contract=contract, require_fitted=False)
    assert pipeline.named_steps["model"].max_iter == 20
    assert pipeline.named_steps["model"].random_state == 7
    fitted_pipeline, _ = fitted
    verify_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    encoder = fitted_pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    assert encoder.handle_unknown == "ignore"
    assert encoder.drop is None
    assert list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())


def test_pipeline_input_not_mutated(contract, frames):
    train, validation, _ = frames
    final = assemble_final_training_data(train=train, validation=validation, contract=contract)
    x = final.features
    before = x.copy(deep=True)
    pipeline = reconstruct_selected_pipeline(estimator=HistGradientBoostingClassifier(random_state=7), contract=contract)
    pipeline.fit(x, final.target)
    pd.testing.assert_frame_equal(x, before)


def test_unknown_categories_reported(fitted, frames):
    pipeline, _ = fitted
    _, _, test = frames
    report = report_unknown_categories(fitted_pipeline=pipeline, features=test[["num", "cat"]], categorical_features=["cat"])
    assert report == {"cat": ["C"]}
    probabilities = pipeline.predict_proba(test[["num", "cat"]])
    assert probabilities.shape == (18, 2)


def write_test_csv(tmp_path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    path = tmp_path / "data" / "test.csv"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    return "data/test.csv", sha256_file(path)


def test_test_access_requires_fit(tmp_path, contract, frames):
    _, _, test = frames
    relative, digest = write_test_csv(tmp_path, test)
    unfitted = reconstruct_selected_pipeline(estimator=HistGradientBoostingClassifier(random_state=7), contract=contract)
    with pytest.raises(FinalizationContractError):
        validate_test_access_gate(contract=contract, fitted_pipeline=unfitted, test_path=relative, expected_sha256=digest, project_root=tmp_path)


def test_test_hash_verified_before_read(tmp_path, contract, fitted, frames):
    pipeline, _ = fitted
    _, _, test = frames
    relative, _ = write_test_csv(tmp_path, test)
    with pytest.raises(FinalTestAccessError):
        load_test_partition_after_freeze(project_root=tmp_path, test_path=relative, expected_sha256="0" * 64, fitted_pipeline=pipeline, contract=contract)


def test_single_probability_evaluation_and_metrics(tmp_path, contract, fitted, frames, selection_handoff, monkeypatch):
    pipeline, _ = fitted
    _, _, test = frames
    relative, digest = write_test_csv(tmp_path, test)
    loaded = load_test_partition_after_freeze(project_root=tmp_path, test_path=relative, expected_sha256=digest, fitted_pipeline=pipeline, contract=contract)
    original = pipeline.predict_proba
    calls = {"count": 0}

    def counted(x):
        calls["count"] += 1
        return original(x)

    monkeypatch.setattr(pipeline, "predict_proba", counted)
    guard = EvaluationGuard()
    result = evaluate_final_model_once(
        fitted_pipeline=pipeline,
        x_test=loaded.features,
        y_test=loaded.target,
        educational_threshold=contract.educational_threshold,
        educational_recall_target=0.8,
        validation_metrics=selection_handoff["selected_validation_metrics"],
        validation_educational_threshold=selection_handoff["selected_educational_threshold"],
        categorical_features=contract.categorical_features,
        guard=guard,
    )
    assert calls["count"] == 1
    assert guard.probability_call_count == 1
    assert set(result.probability_metrics) == {"average_precision", "roc_auc", "log_loss", "brier_score"}
    assert result.threshold_default["threshold"] == 0.5
    assert result.threshold_educational["threshold"] == 0.35
    assert sum(result.threshold_default[k] for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]) == loaded.row_count
    assert result.unknown_categories_report == {"cat": ["C"]}
    with pytest.raises(DuplicateTestEvaluationError):
        evaluate_final_model_once(
            fitted_pipeline=pipeline,
            x_test=loaded.features,
            y_test=loaded.target,
            educational_threshold=0.35,
            educational_recall_target=0.8,
            validation_metrics=selection_handoff["selected_validation_metrics"],
            validation_educational_threshold=selection_handoff["selected_educational_threshold"],
            categorical_features=contract.categorical_features,
            guard=guard,
        )


def test_fixed_threshold_metrics_and_zero_division():
    result = evaluate_fixed_threshold(y_true=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.3, 0.4], threshold=0.99)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["predicted_positive_count"] == 0
    assert result["true_negatives"] == 2
    assert result["false_negatives"] == 2


def test_probability_metric_orientation():
    metrics = compute_probability_metrics(y_true=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.8, 0.9])
    assert metrics["log_loss"] >= 0
    assert metrics["brier_score"] >= 0
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_generalization_deltas_are_descriptive():
    deltas = compute_generalization_deltas(
        validation_metrics={"average_precision": 0.7, "roc_auc": 0.8, "brier_score": 0.2, "log_loss": 0.5, "precision": 0.6, "recall": 0.5},
        test_probability_metrics={"average_precision": 0.6, "roc_auc": 0.75, "brier_score": 0.22, "log_loss": 0.55},
        test_default_threshold={"precision": 0.55, "recall": 0.45},
        test_educational_threshold={"precision": 0.5, "recall": 0.78},
        validation_educational_threshold={"precision": 0.52, "recall": 0.8},
    )
    assert deltas["average_precision_test_minus_validation"] == pytest.approx(-0.1)
    assert deltas["brier_score_test_minus_validation"] == pytest.approx(0.02)


def test_serialization_round_trip_and_fingerprint(tmp_path, contract, fitted):
    pipeline, data = fitted
    descriptor = describe_fitted_pipeline(pipeline=pipeline, contract=contract, training_data=data, train_sha256="a" * 64, validation_sha256="b" * 64)
    fingerprint = compute_fitted_model_fingerprint(descriptor)
    assert len(fingerprint) == 64
    path = tmp_path / "pipeline.joblib"
    digest = serialize_pipeline_to_staging(pipeline=pipeline, staging_path=path)
    loaded = validate_serialized_pipeline(staging_path=path, expected_sha256=digest, contract=contract, reference_pipeline=pipeline, validation_sample=data.features.iloc[:5])
    assert isinstance(loaded, Pipeline)
    assert loaded.named_steps["model"].classes_.tolist() == [0, 1]


def test_unfitted_or_isolated_estimator_rejected(tmp_path, contract):
    estimator = HistGradientBoostingClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    with pytest.raises(SerializationValidationError):
        serialize_pipeline_to_staging(pipeline=estimator, staging_path=tmp_path / "bad.joblib")


def test_corrupted_joblib_or_hash_rejected(tmp_path, contract, fitted):
    pipeline, data = fitted
    path = tmp_path / "pipeline.joblib"
    digest = serialize_pipeline_to_staging(pipeline=pipeline, staging_path=path)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(SerializationValidationError):
        validate_serialized_pipeline(staging_path=path, expected_sha256=digest, contract=contract, reference_pipeline=pipeline, validation_sample=data.features.iloc[:3])


def artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff):
    pipeline, data = fitted
    _, _, test = frames
    relative, digest = write_test_csv(tmp_path, test)
    loaded = load_test_partition_after_freeze(project_root=tmp_path, test_path=relative, expected_sha256=digest, fitted_pipeline=pipeline, contract=contract)
    evaluation = evaluate_final_model_once(
        fitted_pipeline=pipeline,
        x_test=loaded.features,
        y_test=loaded.target,
        educational_threshold=contract.educational_threshold,
        educational_recall_target=0.8,
        validation_metrics=selection_handoff["selected_validation_metrics"],
        validation_educational_threshold=selection_handoff["selected_educational_threshold"],
        categorical_features=contract.categorical_features,
        guard=EvaluationGuard(),
    )
    kwargs = dict(
        project_root=tmp_path,
        output_directory="artifacts/models/synthetic",
        pipeline=pipeline,
        contract=contract,
        training_data=data,
        train_sha256="a" * 64,
        validation_sha256="b" * 64,
        test_partition=loaded,
        evaluation=evaluation,
        fit_duration_seconds=0.01,
        upstream_references={
            "preparation": {"manifest_path": "artifacts/preparation/manifest.json", "sha256": "c" * 64},
            "model_selection": {"handoff_path": "artifacts/model-selection/handoff.json", "sha256": "d" * 64},
        },
        preparation_handoff_references={"path": "artifacts/preparation/manifest.json", "sha256": "c" * 64},
        model_selection_handoff_references={"path": "artifacts/model-selection/handoff.json", "sha256": "d" * 64},
        validation_metrics=selection_handoff["selected_validation_metrics"],
        validation_educational_threshold=selection_handoff["selected_educational_threshold"],
        expected_input_dtypes={"num": "numeric", "cat": "string"},
        missing_value_policy={"strategy": "upstream_prepared_contract"},
    )
    return kwargs


def test_atomic_materialization_bundle_and_handoff(tmp_path, contract, fitted, frames, selection_handoff):
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    result = write_final_model_artifacts(**kwargs)
    assert result.idempotent is False
    assert set(result.created) == {
        "final-pipeline.joblib",
        "final-model-manifest.json",
        "final-test-evidence.json",
        "inference-bundle.json",
        "final-model-handoff.json",
    }
    output = tmp_path / "artifacts/models/synthetic"
    assert inspect_final_artifact_set(output) == "complete"
    bundle = load_and_validate_inference_bundle(project_root=tmp_path, bundle_path="artifacts/models/synthetic/inference-bundle.json")
    handoff = load_and_validate_final_model_handoff(project_root=tmp_path, handoff_path="artifacts/models/synthetic/final-model-handoff.json")
    loaded = load_trusted_pipeline_from_bundle(project_root=tmp_path, bundle=bundle)
    assert isinstance(loaded, Pipeline)
    assert handoff["test_partition_evaluation_count"] == 1
    assert handoff["operational_validity"] == "unconfirmed"
    assert bundle["operational_threshold"] == "unresolved"
    assert bundle["preprocessing_embedded"] is True
    assert bundle["output_contract"]["operational_prediction_available"] is False
    evidence = json.loads((output / "final-test-evidence.json").read_text())
    rendered = canonical_json_bytes(evidence)
    assert b"c-54" not in rendered
    assert b"customerID" not in rendered
    assert evidence["test_probability_evaluation_count"] == 1
    assert evidence["no_post_test_adjustment"] is True
    assert not list(output.parent.glob(".final-model-staging-*"))
    assert not list(output.parent.glob(".final-model-backup-*"))


def test_idempotent_rerun_does_not_replace(tmp_path, contract, fitted, frames, selection_handoff):
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    first = write_final_model_artifacts(**kwargs)
    before = dict(first.byte_sha256)
    second = write_final_model_artifacts(**kwargs)
    assert second.idempotent is True
    assert second.created == () and second.replaced == ()
    assert dict(second.byte_sha256) == before
    assert validate_existing_finalization_equivalence(output_directory=tmp_path / "artifacts/models/synthetic", contract=contract)


def test_partial_set_rejected(tmp_path, contract, fitted, frames, selection_handoff):
    output = tmp_path / "artifacts/models/synthetic"
    output.mkdir(parents=True)
    (output / "final-test-evidence.json").write_text("{}")
    assert inspect_final_artifact_set(output) == "partial"
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    with pytest.raises(ArtifactConflictError):
        write_final_model_artifacts(**kwargs)


def test_semantic_conflict_rejected(tmp_path, contract, fitted, frames, selection_handoff):
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    write_final_model_artifacts(**kwargs)
    handoff_path = tmp_path / "artifacts/models/synthetic/final-model-handoff.json"
    handoff = json.loads(handoff_path.read_text())
    handoff["selected_model_id"] = "different"
    handoff_path.write_text(json.dumps(handoff))
    with pytest.raises((ArtifactConflictError, FinalizationContractError)):
        write_final_model_artifacts(**kwargs)


def test_bundle_hash_tamper_blocks_loading(tmp_path, contract, fitted, frames, selection_handoff):
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    write_final_model_artifacts(**kwargs)
    bundle = load_and_validate_inference_bundle(project_root=tmp_path, bundle_path="artifacts/models/synthetic/inference-bundle.json")
    model = tmp_path / bundle["model_artifact_path"]
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(UntrustedArtifactError):
        load_trusted_pipeline_from_bundle(project_root=tmp_path, bundle=bundle)


def test_final_evidence_contains_aggregate_curves_only(tmp_path, contract, fitted, frames, selection_handoff):
    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    evidence = build_final_test_evidence(
        contract=contract,
        test_partition=kwargs["test_partition"],
        evaluation=kwargs["evaluation"],
        validation_metrics=kwargs["validation_metrics"],
        validation_educational_threshold=kwargs["validation_educational_threshold"],
    )
    assert "precision_recall_curve" in evidence
    assert "roc_curve" in evidence
    assert "calibration_curve" in evidence
    assert evidence["individual_rows_persisted"] is False
    rendered = canonical_json_bytes(evidence).decode()
    assert "identifier_columns" not in rendered
    assert "c-54" not in rendered


def test_runtime_and_semantic_fingerprints_are_deterministic():
    assert "joblib" in runtime_versions()
    left = {"a": 1, "timestamp": "one"}
    right = {"a": 1, "timestamp": "two"}
    assert semantic_fingerprint(left) == semantic_fingerprint(right)


def test_atomic_rollback_preserves_previous_set(
    tmp_path, contract, fitted, frames, selection_handoff, monkeypatch
):
    import scripts.finalize_model as module

    kwargs = artifact_inputs(tmp_path, contract, fitted, frames, selection_handoff)
    first = write_final_model_artifacts(**kwargs)
    output = tmp_path / "artifacts/models/synthetic"
    before = {name: sha256_file(output / name) for name in first.byte_sha256}
    original_replace = module.os.replace
    failed = {"value": False}

    def flaky_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed["value"]
            and ".final-model-staging-" in source_path.as_posix()
            and destination_path.name == "final-test-evidence.json"
        ):
            failed["value"] = True
            raise OSError("synthetic promotion failure")
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="synthetic promotion failure"):
        write_final_model_artifacts(**kwargs, overwrite=True)
    after = {name: sha256_file(output / name) for name in before}
    assert after == before
    assert inspect_final_artifact_set(output) == "complete"
    assert not list(output.parent.glob(".final-model-staging-*"))
    assert not list(output.parent.glob(".final-model-backup-*"))
