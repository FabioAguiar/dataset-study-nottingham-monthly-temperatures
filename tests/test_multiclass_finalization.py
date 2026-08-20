from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.finalize_model import (
    ArtifactConflictError,
    DuplicateTestEvaluationError,
    EvaluationGuard,
    FinalizationContractError,
    MulticlassFrozenFinalizationContract,
    TestAccessError as MulticlassTestAccessError,
    UntrustedArtifactError,
    assemble_multiclass_final_training_data,
    build_multiclass_final_model_handoff,
    compare_multiclass_confusion_pairs,
    compute_fitted_model_fingerprint,
    compute_multiclass_generalization_review,
    describe_multiclass_fitted_pipeline,
    evaluate_multiclass_final_model_once,
    freeze_multiclass_finalization_decisions,
    inspect_final_artifact_set,
    load_and_validate_final_model_handoff,
    load_and_validate_final_model_manifest,
    load_and_validate_final_test_evidence,
    load_and_validate_inference_bundle,
    load_multiclass_test_partition_after_fit,
    load_trusted_pipeline_from_bundle,
    reconstruct_multiclass_selected_pipeline,
    serialize_multiclass_pipeline_to_staging,
    sha256_file,
    smoke_predict_multiclass_bundle,
    validate_existing_multiclass_finalization_equivalence,
    validate_finalization_contract,
    validate_multiclass_finalization_contract,
    validate_multiclass_frozen_model_contract,
    validate_multiclass_inference_input,
    validate_multiclass_serialized_pipeline,
    validate_multiclass_test_access_gate,
    verify_multiclass_pipeline_contract,
    write_multiclass_final_model_artifacts,
)
from scripts.select_models import compute_multiclass_metrics


CLASSES = ("SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA")
FEATURES = ("f1", "f2", "ShapeFactor2")
DRY_BEAN_FEATURES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)


def make_frame(rows_per_class: int, *, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for class_index, label in enumerate(CLASSES):
        for index in range(rows_per_class):
            rows.append(
                {
                    "f1": float(class_index * 10 + index / 10 + offset),
                    "f2": float(class_index * 3 + (index % 4) / 7 + offset),
                    "ShapeFactor2": float(class_index / 10 + (index % 3) / 100),
                    "Class": label,
                }
            )
    return pd.DataFrame(rows, columns=[*FEATURES, "Class"])


@pytest.fixture
def base_contract() -> MulticlassFrozenFinalizationContract:
    return MulticlassFrozenFinalizationContract(
        dataset_slug="synthetic-dry-bean",
        problem_type="multiclass_classification",
        model_selection_handoff_path="artifacts/model-selection/synthetic/model-selection-handoff.json",
        model_selection_handoff_sha256="a" * 64,
        model_id="hist_gradient_boosting__all_features",
        model_family="HistGradientBoostingClassifier",
        hyperparameters=(
            ("model__class_weight", None),
            ("model__l2_regularization", 0.0),
            ("model__learning_rate", 0.1),
            ("model__max_iter", 12),
            ("model__max_leaf_nodes", 7),
            ("model__min_samples_leaf", 2),
        ),
        random_state=42,
        feature_policy="all_features",
        feature_columns=FEATURES,
        numerical_features=FEATURES,
        categorical_features=(),
        identifier_columns=(),
        target_column="Class",
        target_classes=CLASSES,
        target_encoding=tuple((label, index) for index, label in enumerate(CLASSES)),
        target_semantics="nominal_unordered",
        preprocessing_contract=(
            ("categorical_processing", "not_applicable"),
            ("feature_projection", list(FEATURES)),
            ("learned_preprocessing_in_notebook_02", False),
            ("numerical_scaling", "none"),
            ("pipeline", "sklearn.pipeline.Pipeline"),
            ("scaling_fit_scope", "inside_training_fold_or_final_training_only"),
        ),
        imbalance_policy=(("class_weight", None), ("resampling", "none"), ("strategy", "none")),
        decision_rule="argmax_class_score_or_probability",
        training_partitions=("train", "validation"),
        evaluation_partition="test",
        test_partition_path="data/test.csv",
        test_partition_sha256="b" * 64,
        test_partition_row_count=21,
    )


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_frame(10), make_frame(5, offset=0.02), make_frame(3, offset=0.04)


def validation_evidence(frame: pd.DataFrame) -> dict:
    probabilities = np.zeros((len(frame), len(CLASSES)), dtype=float)
    for index, label in enumerate(frame["Class"]):
        probabilities[index, CLASSES.index(label)] = 1.0
    return compute_multiclass_metrics(
        y_true=frame["Class"],
        y_pred=frame["Class"],
        target_classes=CLASSES,
        probabilities=probabilities,
        probability_class_order=CLASSES,
    )


@pytest.fixture
def fitted(base_contract, frames):
    train, validation, _ = frames
    data = assemble_multiclass_final_training_data(
        train=train, validation=validation, contract=base_contract
    )
    pipeline = reconstruct_multiclass_selected_pipeline(contract=base_contract)
    pipeline.fit(data.features, data.target)
    verify_multiclass_pipeline_contract(pipeline, contract=base_contract, require_fitted=True)
    return pipeline, data


def persist_test(tmp_path: Path, frame: pd.DataFrame, contract):
    path = tmp_path / "data/test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return replace(
        contract,
        test_partition_sha256=sha256_file(path),
        test_partition_row_count=len(frame),
    )


def evaluated_inputs(tmp_path, base_contract, fitted, frames):
    pipeline, data = fitted
    _, validation, test = frames
    contract = persist_test(tmp_path, test, base_contract)
    loaded = load_multiclass_test_partition_after_fit(
        project_root=tmp_path, fitted_pipeline=pipeline, contract=contract
    )
    evaluation = evaluate_multiclass_final_model_once(
        fitted_pipeline=pipeline,
        test_partition=loaded,
        final_training_features=data.features,
        contract=contract,
        validation_evidence=validation_evidence(validation),
        guard=EvaluationGuard(),
    )
    return contract, pipeline, data, loaded, evaluation, validation_evidence(validation)


def artifact_kwargs(tmp_path, base_contract, fitted, frames):
    contract, pipeline, data, loaded, evaluation, frozen_validation = evaluated_inputs(
        tmp_path, base_contract, fitted, frames
    )
    return {
        "project_root": tmp_path,
        "output_directory": "artifacts/models/synthetic-dry-bean",
        "pipeline": pipeline,
        "contract": contract,
        "training_data": data,
        "test_partition": loaded,
        "evaluation": evaluation,
        "validation_evidence": frozen_validation,
        "fit_duration_seconds": 0.01,
        "upstream_references": {
            "preparation": {
                "path": "artifacts/preparation/synthetic/preparation-handoff.json",
                "byte_sha256": "c" * 64,
            },
            "model_selection": {
                "path": contract.model_selection_handoff_path,
                "byte_sha256": contract.model_selection_handoff_sha256,
            },
        },
        "expected_input_dtypes": {column: "numeric" for column in FEATURES},
        "missing_value_policy": {
            "strategy": "reject_missing_required_values",
            "learned_imputation_in_final_pipeline": False,
        },
        "analysis_conclusions": {
            "shape_factor_2": {
                "selected_feature_policy": "all_features",
                "provenance_status": "unresolved",
                "interpretation": "Predictive sensitivity does not resolve or invent source provenance.",
            },
            "confirmed_derived_feature_ablation": {"frozen": True},
        },
    }


def _current_multiclass_selection_handoff(**changes) -> dict:
    handoff = {
        "schema_version": "model-selection-handoff.v2",
        "artifact_type": "model_selection_handoff",
        "dataset_slug": "dry-bean",
        "problem_type": "multiclass_classification",
        "target_column": "Class",
        "target_classes": list(CLASSES),
        "target_encoding": {label: index for index, label in enumerate(CLASSES)},
        "target_semantics": "nominal_unordered",
        "positive_class": None,
        "binary_threshold": {"status": "not_applicable", "value": None},
        "operational_threshold": {"status": "not_applicable", "value": None},
        "decision_rule": "argmax_class_score_or_probability",
        "available_feature_columns": list(DRY_BEAN_FEATURES),
        "selected_feature_columns": list(DRY_BEAN_FEATURES),
        "selected_feature_policy": "all_features",
        "selected_model_id": "hist_gradient_boosting__all_features",
        "selected_model_family": "HistGradientBoostingClassifier",
        "selected_hyperparameters": {
            "model__class_weight": None,
            "model__l2_regularization": 0.0,
            "model__learning_rate": 0.05,
            "model__max_iter": 250,
            "model__max_leaf_nodes": 15,
            "model__min_samples_leaf": 40,
        },
        "selected_preprocessing_contract": {
            "pipeline": "sklearn.pipeline.Pipeline",
            "numerical_scaling": "none",
            "categorical_processing": "not_applicable",
            "feature_projection": list(DRY_BEAN_FEATURES),
            "learned_preprocessing_in_notebook_02": False,
            "scaling_fit_scope": "inside_training_fold_or_final_training_only",
        },
        "selected_imbalance_policy": {
            "strategy": "none",
            "class_weight": None,
            "resampling": "none",
        },
        "random_seeds": {"estimators": 42},
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_model_trained": False,
        "model_artifact": None,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "readiness": {
            "preparation_handoff_validated": True,
            "selected_candidate_frozen": True,
            "imbalance_policy_frozen": True,
            "multiclass_decision_rule_frozen": True,
            "final_model_training_ready": True,
            "test_partition_sealed": True,
        },
        "final_training_instructions": {
            "reconstruct_pipeline_from_contract": True,
            "fit_partitions": ["train", "validation"],
            "final_evaluation_partition": "test",
            "access_test_only_after_contract_freeze_and_final_fit": True,
            "evaluate_test_once": True,
            "do_not_retune": True,
            "do_not_change_feature_policy": True,
            "do_not_change_imbalance_policy": True,
            "do_not_change_hyperparameters": True,
            "decision_rule": "argmax_class_score_or_probability",
        },
    }
    handoff.update(changes)
    return handoff


def _current_feature_manifest() -> dict:
    return {
        "feature_columns": list(DRY_BEAN_FEATURES),
        "numerical_features": list(DRY_BEAN_FEATURES),
        "categorical_features": [],
        "identifier_columns": [],
        "target_column": "Class",
        "target_classes": list(CLASSES),
        "target_encoding_contract": {label: index for index, label in enumerate(CLASSES)},
    }


def _current_split_manifest() -> dict:
    return {
        "partition_paths": {
            "train": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/train.csv",
            "validation": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/validation.csv",
            "test": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/test.csv",
        },
        "partition_sha256": {"test": "b" * 64},
        "row_counts": {"test": 2042},
    }


def test_model_selection_handoff_v2_is_accepted_from_current_contract():
    validate_multiclass_finalization_contract(_current_multiclass_selection_handoff())


def test_binary_v1_validation_is_preserved():
    payload = {
        "schema_version": "model-selection-handoff.v1",
        "artifact_type": "model_selection_handoff",
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_model_trained": False,
        "model_artifact": None,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "readiness": {"final_model_training_ready": True},
    }
    validate_finalization_contract(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("positive_class", "SEKER"),
        ("test_partition_sealed", False),
        ("test_partition_evaluated", True),
        ("operational_modeling_ready", True),
        ("operational_validity", "confirmed"),
    ],
)
def test_multiclass_upstream_binary_or_readiness_drift_is_rejected(field, value):
    payload = _current_multiclass_selection_handoff()
    payload[field] = value
    with pytest.raises(FinalizationContractError):
        validate_multiclass_finalization_contract(payload)


def test_real_frozen_contract_exact_values_without_test_read():
    selection = _current_multiclass_selection_handoff()
    contract = freeze_multiclass_finalization_decisions(
        dataset_slug="dry-bean",
        model_selection_handoff=selection,
        feature_manifest=_current_feature_manifest(),
        split_manifest=_current_split_manifest(),
        model_selection_handoff_path="artifacts/model-selection/dry-bean/model-selection-handoff.json",
        model_selection_handoff_sha256="a" * 64,
    )
    assert contract.model_family == "HistGradientBoostingClassifier"
    assert contract.random_state == 42
    assert contract.feature_policy == "all_features"
    assert len(contract.feature_columns) == 16 and "ShapeFactor2" in contract.feature_columns
    assert dict(contract.hyperparameters) == selection["selected_hyperparameters"]
    assert dict(contract.preprocessing_contract)["numerical_scaling"] == "none"
    assert dict(contract.imbalance_policy) == {"class_weight": None, "resampling": "none", "strategy": "none"}
    assert contract.test_partition_row_count == 2042


def test_multiclass_contract_has_no_positive_class_or_threshold(base_contract):
    frozen = base_contract.as_dict()
    assert frozen["positive_class"] == "not_applicable"
    assert frozen["binary_threshold"] == "not_applicable"
    assert frozen["operational_threshold"] == "not_applicable"


def test_pipeline_reconstruction_exact_seed_parameters_and_no_scaling(base_contract):
    pipeline = reconstruct_multiclass_selected_pipeline(contract=base_contract)
    verify_multiclass_pipeline_contract(pipeline, contract=base_contract, require_fitted=False)
    model = pipeline.named_steps["model"]
    assert model.random_state == 42
    assert model.class_weight is None
    assert pipeline.named_steps["preprocess"].transformers == [
        ("numerical", "passthrough", list(FEATURES))
    ]


def test_train_validation_assembly_preserves_order_and_inputs(base_contract, frames):
    train, validation, _ = frames
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)
    data = assemble_multiclass_final_training_data(
        train=train, validation=validation, contract=base_contract
    )
    assert data.row_count == len(train) + len(validation)
    assert list(data.features.columns) == list(FEATURES)
    assert list(dict(data.class_counts)) == list(CLASSES)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_synthetic_contract_shape_assembles_11569_rows(base_contract):
    labels = np.resize(np.asarray(CLASSES, dtype=object), 11569)
    frame = pd.DataFrame({column: np.arange(11569, dtype=float) for column in FEATURES})
    frame["Class"] = labels
    train = frame.iloc[:9527].reset_index(drop=True)
    validation = frame.iloc[9527:].reset_index(drop=True)
    data = assemble_multiclass_final_training_data(
        train=train, validation=validation, contract=base_contract
    )
    assert data.row_count == 11569


def test_test_unavailable_before_final_fit(tmp_path, base_contract, frames):
    _, _, test = frames
    contract = persist_test(tmp_path, test, base_contract)
    pipeline = reconstruct_multiclass_selected_pipeline(contract=contract)
    with pytest.raises(FinalizationContractError):
        validate_multiclass_test_access_gate(
            contract=contract, fitted_pipeline=pipeline, project_root=tmp_path
        )


def test_test_sha_verified_before_csv_read(tmp_path, base_contract, fitted, frames, monkeypatch):
    _, _, test = frames
    contract = persist_test(tmp_path, test, base_contract)
    contract = replace(contract, test_partition_sha256="0" * 64)
    called = {"read": False}

    def forbidden(*args, **kwargs):
        called["read"] = True
        raise AssertionError("CSV parsing must not occur")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(MulticlassTestAccessError):
        load_multiclass_test_partition_after_fit(
            project_root=tmp_path, fitted_pipeline=fitted[0], contract=contract
        )
    assert called["read"] is False


def test_single_evaluation_guard_probability_order_and_metrics(tmp_path, base_contract, fitted, frames):
    contract, pipeline, data, loaded, evaluation, frozen_validation = evaluated_inputs(
        tmp_path, base_contract, fitted, frames
    )
    assert evaluation.test_probability_evaluation_count == 1
    assert evaluation.output_class_order == CLASSES
    assert set(evaluation.estimator_class_order) == set(CLASSES)
    assert evaluation.metrics["row_count"] == len(frames[2])
    assert set(evaluation.metrics) == {
        "macro_f1", "balanced_accuracy", "macro_recall", "weighted_f1",
        "accuracy", "minimum_per_class_recall", "log_loss", "row_count",
    }
    guard = EvaluationGuard(evaluated=True, probability_call_count=1)
    with pytest.raises(DuplicateTestEvaluationError):
        evaluate_multiclass_final_model_once(
            fitted_pipeline=pipeline,
            test_partition=loaded,
            final_training_features=data.features,
            contract=contract,
            validation_evidence=frozen_validation,
            guard=guard,
        )


def test_fixed_order_confusion_and_per_class_contract(tmp_path, base_contract, fitted, frames):
    *_, evaluation, _ = evaluated_inputs(tmp_path, base_contract, fitted, frames)
    assert evaluation.confusion_matrix["class_order"] == list(CLASSES)
    assert np.asarray(evaluation.confusion_matrix["counts"]).shape == (7, 7)
    assert [row["class"] for row in evaluation.per_class] == list(CLASSES)


def test_generalization_review_is_descriptive():
    frame = make_frame(2)
    evidence = validation_evidence(frame)
    review = compute_multiclass_generalization_review(
        validation_evidence=evidence, test_evidence=evidence
    )
    assert all(value == pytest.approx(0.0) for value in review["aggregate_deltas"].values())
    assert review["selection_reopened"] is False


def test_confusion_pair_comparison_tracks_frozen_pairs():
    frame = make_frame(2)
    evidence = validation_evidence(frame)
    comparison = compare_multiclass_confusion_pairs(
        validation_confusion=evidence["confusion_matrix"],
        test_confusion=evidence["confusion_matrix"],
        target_classes=CLASSES,
    )
    assert [row["class_pair"] for row in comparison["focal_pair_comparisons"]] == [
        ["DERMASON", "SIRA"], ["BARBUNYA", "CALI"]
    ]
    assert all(row["pattern_direction"] == "persisted" for row in comparison["focal_pair_comparisons"])


def test_repeated_profile_sensitivity_is_non_destructive(tmp_path, base_contract, fitted, frames):
    train, validation, test = frames
    test = test.copy(deep=True)
    test.loc[0, list(FEATURES)] = train.loc[0, list(FEATURES)]
    custom_frames = (train, validation, test)
    before = test.copy(deep=True)
    *_, evaluation, _ = evaluated_inputs(tmp_path, base_contract, fitted, custom_frames)
    sensitivity = evaluation.repeated_profile_sensitivity
    assert sensitivity["repeated_profile_test_row_count"] >= 1
    assert sensitivity["official_full_test_row_count"] == len(test)
    assert "does not prove" in sensitivity["interpretation"]
    pd.testing.assert_frame_equal(test, before)


def test_serialization_round_trip_sha_state_and_equivalence(tmp_path, base_contract, fitted):
    pipeline, data = fitted
    descriptor = describe_multiclass_fitted_pipeline(
        pipeline=pipeline, contract=base_contract
    )
    assert len(compute_fitted_model_fingerprint(descriptor)) == 64
    path = tmp_path / "model.joblib"
    digest = serialize_multiclass_pipeline_to_staging(
        pipeline=pipeline, staging_path=path
    )
    loaded = validate_multiclass_serialized_pipeline(
        staging_path=path,
        expected_sha256=digest,
        contract=base_contract,
        reference_pipeline=pipeline,
        validation_sample=data.features.iloc[:10],
    )
    assert sha256_file(path) == digest
    assert np.array_equal(
        pipeline.predict(data.features.iloc[:5]), loaded.predict(data.features.iloc[:5])
    )


def test_inference_input_rejects_missing_extra_order_and_non_numeric(base_contract, fitted):
    pipeline, data = fitted
    descriptor = describe_multiclass_fitted_pipeline(pipeline=pipeline, contract=base_contract)
    from scripts.finalize_model import build_multiclass_inference_bundle

    bundle = build_multiclass_inference_bundle(
        contract=base_contract,
        fitted_pipeline=pipeline,
        model_artifact_path="artifacts/models/synthetic/final-pipeline.joblib",
        model_artifact_sha256="d" * 64,
        model_state_fingerprint=compute_fitted_model_fingerprint(descriptor),
        model_state_descriptor=descriptor,
        expected_input_dtypes={column: "numeric" for column in FEATURES},
        missing_value_policy={"strategy": "reject_missing_required_values"},
        upstream_references={
            "model_selection": {"path": base_contract.model_selection_handoff_path, "byte_sha256": "a" * 64}
        },
        final_artifact_paths={
            "final-model-manifest.json": "artifacts/models/synthetic/final-model-manifest.json",
            "final-test-evidence.json": "artifacts/models/synthetic/final-test-evidence.json",
            "final-model-handoff.json": "artifacts/models/synthetic/final-model-handoff.json",
        },
    )
    valid = data.features.iloc[:2]
    assert list(validate_multiclass_inference_input(valid, bundle=bundle)) == list(FEATURES)
    for invalid in (
        valid.drop(columns="f2"),
        valid[["f2", "f1", "ShapeFactor2"]],
        valid.assign(f1=np.nan),
        valid.assign(f1="bad"),
    ):
        with pytest.raises(FinalizationContractError):
            validate_multiclass_inference_input(invalid, bundle=bundle)


def test_smoke_output_has_seven_ordered_probabilities(tmp_path, base_contract, fitted, frames):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    write_multiclass_final_model_artifacts(**kwargs)
    bundle = load_and_validate_inference_bundle(
        project_root=tmp_path,
        bundle_path="artifacts/models/synthetic-dry-bean/inference-bundle.json",
    )
    pipeline = load_trusted_pipeline_from_bundle(project_root=tmp_path, bundle=bundle)
    output = smoke_predict_multiclass_bundle(
        pipeline, kwargs["training_data"].features.iloc[:4], bundle=bundle
    )
    assert output["class_order"].tolist() == [list(CLASSES)] * 4
    assert all(len(values) == 7 for values in output["class_probabilities"])
    assert all(sum(values) == pytest.approx(1.0) for values in output["class_probabilities"])


def test_atomic_v2_artifact_set_loaders_and_readiness(tmp_path, base_contract, fitted, frames):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    result = write_multiclass_final_model_artifacts(**kwargs)
    assert result.idempotent is False
    assert set(result.created) == set(name for name in result.byte_sha256)
    output = tmp_path / "artifacts/models/synthetic-dry-bean"
    assert inspect_final_artifact_set(output) == "complete"
    handoff = load_and_validate_final_model_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/models/synthetic-dry-bean/final-model-handoff.json",
    )
    manifest = load_and_validate_final_model_manifest(
        project_root=tmp_path,
        manifest_path="artifacts/models/synthetic-dry-bean/final-model-manifest.json",
    )
    evidence = load_and_validate_final_test_evidence(
        project_root=tmp_path,
        evidence_path="artifacts/models/synthetic-dry-bean/final-test-evidence.json",
    )
    assert handoff["schema_version"] == "final-model-handoff.v2"
    assert manifest["schema_version"] == "final-model-manifest.v2"
    assert evidence["schema_version"] == "final-test-evidence.v2"
    assert handoff["test_partition_evaluation_count"] == 1
    assert handoff["no_model_selection_decision_changed_after_test"] is True
    assert handoff["operational_modeling_ready"] is False
    assert handoff["operational_validity"] == "unconfirmed"


def test_complete_equivalent_rerun_is_idempotent_and_does_not_read_test(
    tmp_path, base_contract, fitted, frames, monkeypatch
):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    first = write_multiclass_final_model_artifacts(**kwargs)
    evidence_path = first.output_directory / "final-test-evidence.json"
    original_evidence_text = evidence_path.read_text()
    evidence_path.write_text(
        original_evidence_text.replace(
            "Repeated-profile evidence does not prove duplicate identity or leakage",
            "repeated profiles do not prove duplicate identity or leakage",
        )
    )
    assert validate_existing_multiclass_finalization_equivalence(
        output_directory=first.output_directory, contract=kwargs["contract"]
    )
    evidence_path.write_text(original_evidence_text)

    def forbidden(*args, **kwargs):
        raise AssertionError("Equivalent reuse must not reopen test")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    assert validate_existing_multiclass_finalization_equivalence(
        output_directory=first.output_directory, contract=kwargs["contract"]
    )
    second = write_multiclass_final_model_artifacts(**kwargs)
    assert second.idempotent is True and second.created == () and second.replaced == ()


def test_final_evidence_metric_tamper_still_fails_equivalence(
    tmp_path, base_contract, fitted, frames
):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    result = write_multiclass_final_model_artifacts(**kwargs)
    evidence_path = result.output_directory / "final-test-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["metrics"]["macro_f1"] = 0.0
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    with pytest.raises(ArtifactConflictError):
        validate_existing_multiclass_finalization_equivalence(
            output_directory=result.output_directory,
            contract=kwargs["contract"],
        )


def test_partial_set_fails_closed(tmp_path, base_contract):
    output = tmp_path / "artifacts/models/synthetic-dry-bean"
    output.mkdir(parents=True)
    (output / "final-test-evidence.json").write_text("{}")
    assert inspect_final_artifact_set(output) == "partial"
    with pytest.raises(ArtifactConflictError):
        validate_existing_multiclass_finalization_equivalence(
            output_directory=output, contract=base_contract
        )


def test_semantic_conflict_fails_closed(tmp_path, base_contract, fitted, frames):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    write_multiclass_final_model_artifacts(**kwargs)
    divergent = replace(kwargs["contract"], feature_policy="without_shape_factor_2")
    with pytest.raises((ArtifactConflictError, FinalizationContractError)):
        validate_existing_multiclass_finalization_equivalence(
            output_directory=tmp_path / "artifacts/models/synthetic-dry-bean",
            contract=divergent,
        )


@pytest.mark.parametrize(
    "filename",
    ["final-pipeline.joblib", "final-test-evidence.json", "inference-bundle.json", "final-model-manifest.json"],
)
def test_sibling_tamper_invalidates_final_handoff(
    tmp_path, base_contract, fitted, frames, filename
):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    write_multiclass_final_model_artifacts(**kwargs)
    output = tmp_path / "artifacts/models/synthetic-dry-bean"
    path = output / filename
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises((ArtifactConflictError, FinalizationContractError, json.JSONDecodeError)):
        load_and_validate_final_model_handoff(
            project_root=tmp_path,
            handoff_path="artifacts/models/synthetic-dry-bean/final-model-handoff.json",
        )


def test_model_byte_tamper_blocks_trusted_load(tmp_path, base_contract, fitted, frames):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    write_multiclass_final_model_artifacts(**kwargs)
    bundle_path = tmp_path / "artifacts/models/synthetic-dry-bean/inference-bundle.json"
    bundle = json.loads(bundle_path.read_text())
    model_path = tmp_path / bundle["model_artifact_path"]
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    with pytest.raises(UntrustedArtifactError):
        load_trusted_pipeline_from_bundle(project_root=tmp_path, bundle=bundle)


def test_handoff_builder_preserves_shape_factor2_and_no_post_test_change(
    tmp_path, base_contract, fitted, frames
):
    kwargs = artifact_kwargs(tmp_path, base_contract, fitted, frames)
    contract = kwargs["contract"]
    evaluation = kwargs["evaluation"]
    handoff = build_multiclass_final_model_handoff(
        contract=contract,
        upstream_references=kwargs["upstream_references"],
        final_references={
            "model_artifact": {"path": "a.joblib", "byte_sha256": "a" * 64, "semantic_sha256": "b" * 64},
            "final_model_manifest": {"path": "a.json", "byte_sha256": "a" * 64, "semantic_sha256": "b" * 64},
            "final_test_evidence": {"path": "b.json", "byte_sha256": "a" * 64, "semantic_sha256": "b" * 64},
            "inference_bundle": {"path": "c.json", "byte_sha256": "a" * 64, "semantic_sha256": "b" * 64},
        },
        evaluation=evaluation,
        analysis_conclusions=kwargs["analysis_conclusions"],
        runtime_requirements={"python": "test"},
    )
    assert "ShapeFactor2" in handoff["feature_order"]
    assert handoff["analysis_conclusions"]["shape_factor_2"]["provenance_status"] == "unresolved"
    assert handoff["test_partition_used_for_feature_selection"] is False
    assert handoff["no_model_selection_decision_changed_after_test"] is True


def test_multiclass_finalization_source_uses_argmax_without_binary_thresholds():
    text = "\n".join(
        (
            inspect.getsource(evaluate_multiclass_final_model_once),
            inspect.getsource(validate_multiclass_finalization_contract),
        )
    ).lower()
    assert "evaluate_multiclass_final_model_once" in text
    assert "argmax_class_score_or_probability" in text
    assert "precision_recall_curve" not in text
    assert "roc_curve" not in text
    assert "educational_threshold" not in text
    assert "positive_class_probability" not in text
