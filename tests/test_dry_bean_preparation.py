"""Multiclass, identifier-free preparation contract tests for Dry Bean."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_data import (
    ArtifactConflictError,
    ClassificationSplitPolicy,
    DatasetValidationError,
    HandoffValidationError,
    analyze_repeated_profiles_across_partitions,
    build_feature_manifest,
    build_preparation_handoff_manifest,
    build_preparation_manifest,
    build_quality_evidence,
    build_split_manifest,
    fingerprint_dataframe,
    fingerprint_dataframe_csv,
    fingerprint_file,
    load_and_validate_preparation_handoff,
    prepare_tabular_dataset,
    separate_dataset_roles,
    split_classification_dataset,
    validate_dataset_partitions,
    validate_prepared_dataset,
    validate_raw_dataset,
    validate_source_against_exploration_handoff,
    write_preparation_artifacts,
)


FEATURES = (
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
CLASSES = (
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SIRA",
)
TARGET = "Class"
COLUMN_ORDER = (*FEATURES, TARGET)
EXPECTED_TYPES = {**{feature: "numeric" for feature in FEATURES}, TARGET: "string"}


def make_dry_bean_frame(rows_per_class: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_index, target_class in enumerate(CLASSES):
        for occurrence in range(rows_per_class):
            base = class_index * 1000 + occurrence + 1
            row = {
                feature: float(base + feature_index / 100)
                for feature_index, feature in enumerate(FEATURES)
            }
            row[TARGET] = target_class
            rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMN_ORDER)
    frame.iloc[1] = frame.iloc[0]
    frame.iloc[2] = frame.iloc[0]
    return frame


def split_policy() -> ClassificationSplitPolicy:
    return ClassificationSplitPolicy(
        evaluation_mode="stratified_random_snapshot",
        purpose="educational_benchmark",
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        stratify_by=TARGET,
        random_seed=42,
        shuffle=True,
        educational_justification=(
            "Use the source-released static snapshot for a reproducible "
            "educational multiclass benchmark."
        ),
        operational_validity="unconfirmed",
        temporal_contract_status="resolved_static_snapshot",
        feature_inference_availability="unconfirmed",
    )


def exploration_handoff(source: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "schema_version": "exploration-handoff.v1",
        "artifact_type": "exploration_handoff",
        "dataset_slug": "dry-bean",
        "source": {
            "repository": "UCI Machine Learning Repository",
            "dataset_id": 602,
            "path": "data/raw/dry-bean/dataset.csv",
            "sha256": fingerprint_file(source),
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "column_order": list(frame.columns),
        },
        "prediction_contract": {
            "problem_type": "multiclass_classification",
            "target_column": TARGET,
            "target_classes": list(CLASSES),
            "class_semantics": "Nominal / unordered",
            "positive_class": None,
        },
        "feature_contract": {
            "identifier_columns": [],
            "feature_columns": list(FEATURES),
            "numerical_features": list(FEATURES),
            "categorical_features": [],
            "baseline_feature_count": len(FEATURES),
        },
        "preparation_contract": {
            "split_policy": {
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "stratification_field": TARGET,
                "random_seed": 42,
                "temporal_policy_status": "Resolved snapshot fallback",
            }
        },
        "continuation": {},
        "readiness": {
            "notebook_01_complete": True,
            "deterministic_preparation_ready": True,
            "split_execution_ready": True,
        },
    }


def materialize_uci_source(root: Path, frame: pd.DataFrame):
    raw = root / "data/raw/dry-bean"
    raw.mkdir(parents=True)
    source = raw / "dataset.csv"
    frame.to_csv(source, index=False)
    metadata = raw / "metadata.json"
    metadata.write_text('{"uci_id": 602}\n', encoding="utf-8")
    variables = raw / "variables.csv"
    pd.DataFrame(
        {
            "name": [*FEATURES, TARGET],
            "role": [*["Feature"] * len(FEATURES), "Target"],
        }
    ).to_csv(variables, index=False)
    return source, metadata, variables


def build_runtime_bundle(root: Path, frame: pd.DataFrame):
    raw_validation = validate_raw_dataset(
        frame,
        column_order=COLUMN_ORDER,
        identifier_columns=(),
        feature_columns=FEATURES,
        target_column=TARGET,
        target_classes=CLASSES,
        categorical_expected_values={},
        expected_types=EXPECTED_TYPES,
    )
    preparation = prepare_tabular_dataset(frame)
    prepared = preparation.dataframe
    prepared_validation = validate_prepared_dataset(
        frame,
        prepared,
        column_order=COLUMN_ORDER,
        identifier_columns=(),
        feature_columns=FEATURES,
        target_column=TARGET,
        target_classes=CLASSES,
        categorical_expected_values={},
        expected_types=EXPECTED_TYPES,
        authorized_changed_columns=(),
        expected_row_count=len(frame),
        expected_materialized_counts={},
        observed_materialized_counts=dict(preparation.materialized_counts),
    )
    policy = split_policy()
    partitions = split_classification_dataset(
        prepared,
        policy=policy,
        identifier_columns=(),
        target_classes=CLASSES,
    )
    partition_validation = validate_dataset_partitions(
        prepared,
        partitions,
        identifier_columns=(),
        target_column=TARGET,
        target_classes=CLASSES,
        prevalence_tolerance=0.03,
    )
    repeated = analyze_repeated_profiles_across_partitions(
        prepared,
        partitions,
        feature_columns=FEATURES,
        target_column=TARGET,
        identifier_columns=(),
    )
    prepared_path = Path("data/processed/dry-bean/prepared.csv")
    split_root = Path(
        "data/processed/dry-bean/splits/stratified-70-15-15-seed-42"
    )
    partition_paths = {
        name: split_root / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    manifest_root = Path("artifacts/preparation/dry-bean")
    manifest_paths = {
        "preparation_manifest": manifest_root / "preparation-manifest.json",
        "feature_manifest": manifest_root / "feature-manifest.json",
        "split_manifest": manifest_root / "split-manifest.json",
        "quality_evidence": manifest_root / "quality-evidence.json",
    }
    readiness = {
        "notebook_01_handoff_validated": True,
        "source_independently_revalidated": True,
        "prepared_dataset_materialized": True,
        "benchmark_partitions_materialized": True,
        "partition_integrity_validated": True,
        "preparation_handoff_reloadable": True,
        "educational_model_selection_ready": True,
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "model_selected": False,
        "final_model_trained": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "resolved_static_snapshot",
        "feature_inference_availability": "unconfirmed",
    }
    source = root / "data/raw/dry-bean/dataset.csv"
    prep_manifest = build_preparation_manifest(
        dataset_slug="dry-bean",
        source_path="data/raw/dry-bean/dataset.csv",
        source_sha256=fingerprint_file(source),
        prepared_path=prepared_path,
        prepared_sha256=fingerprint_dataframe_csv(prepared),
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        preparation=preparation,
        raw_fingerprint_before=fingerprint_dataframe(frame),
        raw_fingerprint_after=fingerprint_dataframe(frame),
        source_sha256_after=fingerprint_file(source),
        deterministic_rules=[],
        readiness=readiness,
    )
    feature_manifest = build_feature_manifest(
        dataset_slug="dry-bean",
        identifier_columns=(),
        feature_columns=FEATURES,
        numerical_features=FEATURES,
        categorical_features=(),
        categorical_expected_values={},
        target_column=TARGET,
        target_classes=CLASSES,
        expected_dtypes={"raw": EXPECTED_TYPES, "prepared": EXPECTED_TYPES},
        preprocessing_contract={
            "numerical_scaling": "model_specific_inside_training_fold",
            "learned_transformations_fitted_in_notebook_02": False,
            "test_partition_sealed": True,
        },
        prohibited_predictors=(TARGET, "technical membership tokens"),
        positive_target_class=None,
        target_encoding={value: index for index, value in enumerate(CLASSES)},
        problem_type="multiclass_classification",
        target_semantics="nominal_unordered",
    )
    partition_sha = {
        name: fingerprint_dataframe_csv(partition)
        for name, partition in partitions.as_mapping().items()
    }
    split_manifest = build_split_manifest(
        dataset_slug="dry-bean",
        policy=policy,
        partitions=partitions,
        validation=partition_validation,
        partition_paths=partition_paths,
        partition_sha256=partition_sha,
        repeated_profile_evidence=repeated,
    )
    quality = build_quality_evidence(
        dataset_slug="dry-bean",
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        partition_report=partition_validation,
        preparation=preparation,
        fingerprints={
            "prepared_sha256": fingerprint_dataframe_csv(prepared),
            "partition_sha256": partition_sha,
        },
        readiness=readiness,
        preservation_checks={
            "rows_removed": 0,
            "values_changed": 0,
            "source_file_unchanged": True,
        },
        repeated_profile_evidence=repeated,
    )
    components = {
        "preparation_manifest": prep_manifest,
        "feature_manifest": feature_manifest,
        "split_manifest": split_manifest,
        "quality_evidence": quality,
    }
    handoff_path = manifest_root / "preparation-handoff.json"
    handoff = build_preparation_handoff_manifest(
        dataset_slug="dry-bean",
        component_paths=manifest_paths,
        component_payloads=components,
        readiness=readiness,
    )
    csv_artifacts = {
        prepared_path: prepared,
        **{
            partition_paths[name]: partition
            for name, partition in partitions.as_mapping().items()
        },
    }
    json_artifacts = {
        **{manifest_paths[name]: payload for name, payload in components.items()},
        handoff_path: handoff,
    }
    return csv_artifacts, json_artifacts, handoff_path, repeated


def test_zero_identifier_multiclass_schema_and_role_separation() -> None:
    frame = make_dry_bean_frame()
    report = validate_raw_dataset(
        frame,
        column_order=COLUMN_ORDER,
        identifier_columns=(),
        feature_columns=FEATURES,
        target_column=TARGET,
        target_classes=CLASSES,
        categorical_expected_values={},
        expected_types=EXPECTED_TYPES,
    )
    roles = separate_dataset_roles(
        frame,
        identifier_columns=(),
        feature_columns=FEATURES,
        target_column=TARGET,
    )

    assert report.is_valid
    assert roles.lineage.shape == (len(frame), 0)
    assert tuple(roles.features.columns) == FEATURES
    assert set(roles.target) == set(CLASSES)


def test_zero_rule_projection_is_defensive_and_unchanged() -> None:
    frame = make_dry_bean_frame()
    original = frame.copy(deep=True)
    result = prepare_tabular_dataset(frame)

    pd.testing.assert_frame_equal(result.dataframe, original)
    pd.testing.assert_frame_equal(frame, original)
    assert result.rules == ()
    assert result.materialized_counts == ()


def test_identifier_free_split_preserves_duplicates_and_is_reproducible() -> None:
    frame = make_dry_bean_frame()
    first = split_classification_dataset(
        frame,
        policy=split_policy(),
        identifier_columns=(),
        target_classes=CLASSES,
    )
    second = split_classification_dataset(
        frame,
        policy=split_policy(),
        identifier_columns=(),
        target_classes=CLASSES,
    )
    validation = validate_dataset_partitions(
        frame,
        first,
        identifier_columns=(),
        target_column=TARGET,
        target_classes=CLASSES,
        prevalence_tolerance=0.03,
    )

    assert validation.is_valid
    assert validation.membership_kind == "technical_row_occurrence"
    assert validation.entity_disjointness_status == "not_claimed_without_source_identifiers"
    assert first.membership_mapping() == second.membership_mapping()
    assert sum(len(value) for value in first.as_mapping().values()) == len(frame)
    assert sum((part == frame.iloc[0]).all(axis=1).sum() for part in first.as_mapping().values()) == 3
    for partition in first.as_mapping().values():
        assert set(partition[TARGET]) == set(CLASSES)


def test_repeated_profile_evidence_is_epistemically_bounded() -> None:
    frame = make_dry_bean_frame()
    partitions = split_classification_dataset(
        frame,
        policy=split_policy(),
        identifier_columns=(),
        target_classes=CLASSES,
    )
    evidence = analyze_repeated_profiles_across_partitions(
        frame,
        partitions,
        feature_columns=FEATURES,
        target_column=TARGET,
    )

    assert evidence["source_exact_row_equality_group_count"] == 1
    assert evidence["source_exact_row_equality_row_count"] == 3
    assert evidence["exact_row_multiplicity_preserved"] is True
    assert evidence["proven_duplicate_identity"] is False
    assert "not proven" in evidence["identity_interpretation"]


def test_multiclass_feature_manifest_has_no_positive_class() -> None:
    manifest = build_feature_manifest(
        dataset_slug="dry-bean",
        identifier_columns=(),
        feature_columns=FEATURES,
        numerical_features=FEATURES,
        categorical_features=(),
        categorical_expected_values={},
        target_column=TARGET,
        target_classes=CLASSES,
        expected_dtypes={"raw": EXPECTED_TYPES, "prepared": EXPECTED_TYPES},
        preprocessing_contract={
            "input_feature_type": "numerical_only",
            "categorical_strategy": "not_applicable",
            "numerical_scaling": "model_specific",
            "learned_fit_scope": "inside_training_data_or_training_fold_only",
            "learned_transformations_fitted_in_notebook_02": False,
        },
        prohibited_predictors=(TARGET,),
        positive_target_class=None,
        target_encoding={value: index for index, value in enumerate(CLASSES)},
        problem_type="multiclass_classification",
        target_semantics="nominal_unordered",
    )

    assert manifest["schema_version"] == "feature-manifest.v2"
    assert manifest["positive_target_class"] is None
    assert manifest["problem_type"] == "multiclass_classification"
    assert manifest["target_contract"]["semantics"] == "nominal_unordered"
    assert manifest["target_contract"]["persisted_labels_remain_readable"] is True
    assert manifest["preprocessing_contract"]["input_feature_type"] == "numerical_only"
    assert manifest["preprocessing_contract"]["categorical_strategy"] == "not_applicable"
    assert manifest["preprocessing_contract"]["learned_transformations_fitted_in_notebook_02"] is False


def test_source_identity_gate_validates_uci_materialization(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    source, metadata, variables = materialize_uci_source(tmp_path, frame)
    handoff = exploration_handoff(source, frame)

    report = validate_source_against_exploration_handoff(
        pd.read_csv(source),
        handoff=handoff,
        source_file=source,
        project_root=tmp_path,
        dataset_slug="dry-bean",
        source_repository="UCI Machine Learning Repository",
        source_dataset_id=602,
        metadata_file=metadata,
        variables_file=variables,
    )

    assert report.is_valid
    assert report.identifier_columns == ()
    assert report.feature_columns == FEATURES


def test_source_identity_gate_fails_on_handoff_mismatch(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    source, metadata, variables = materialize_uci_source(tmp_path, frame)
    handoff = exploration_handoff(source, frame)
    tampered = copy.deepcopy(handoff)
    tampered["source"]["row_count"] = len(frame) + 1

    with pytest.raises(DatasetValidationError, match="row_count"):
        validate_source_against_exploration_handoff(
            pd.read_csv(source),
            handoff=tampered,
            source_file=source,
            project_root=tmp_path,
            dataset_slug="dry-bean",
            source_repository="UCI Machine Learning Repository",
            source_dataset_id=602,
            metadata_file=metadata,
            variables_file=variables,
        )


def test_persist_reload_handoff_preserves_multiset_and_seals_test(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    materialize_uci_source(tmp_path, frame)
    csv_artifacts, json_artifacts, handoff_path, repeated = build_runtime_bundle(
        tmp_path, frame
    )
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )

    reloaded = load_and_validate_preparation_handoff(
        project_root=tmp_path,
        preparation_handoff_path=handoff_path,
    )

    assert len(reloaded.prepared) == len(frame)
    assert tuple(reloaded.prepared.columns) == COLUMN_ORDER
    assert tuple(reloaded.manifests["feature_manifest"]["feature_columns"]) == FEATURES
    assert tuple(reloaded.manifests["feature_manifest"]["target_classes"]) == CLASSES
    assert reloaded.manifests["feature_manifest"]["positive_target_class"] is None
    assert reloaded.manifests["split_manifest"]["test_holdout_policy"]
    assert reloaded.manifests["quality_evidence"]["readiness"]["test_partition_evaluated"] is False
    assert repeated["exact_row_multiplicity_preserved"] is True


def test_handoff_component_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    materialize_uci_source(tmp_path, frame)
    csv_artifacts, json_artifacts, handoff_path, _ = build_runtime_bundle(
        tmp_path, frame
    )
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    feature_path = tmp_path / "artifacts/preparation/dry-bean/feature-manifest.json"
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    payload["feature_columns"] = payload["feature_columns"][:-1]
    feature_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HandoffValidationError, match="component fingerprint"):
        load_and_validate_preparation_handoff(
            project_root=tmp_path,
            preparation_handoff_path=handoff_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["target_contract"].__setitem__("semantics", "ordinal"), "semantics"),
        (lambda payload: payload["target_contract"].__setitem__("ordered_class_contract", list(reversed(CLASSES))), "ordered class"),
        (lambda payload: payload.__setitem__("target_classes", list(reversed(CLASSES))), "ordered class"),
        (lambda payload: payload.__setitem__("positive_target_class", CLASSES[0]), "positive class"),
        (lambda payload: payload.__setitem__("dataset_slug", "other"), "dataset_slug"),
    ],
)
def test_multiclass_handoff_contract_mutations_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    frame = make_dry_bean_frame()
    materialize_uci_source(tmp_path, frame)
    csv_artifacts, json_artifacts, handoff_path, _ = build_runtime_bundle(tmp_path, frame)
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    feature_path = tmp_path / "artifacts/preparation/dry-bean/feature-manifest.json"
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    mutation(feature)
    feature_path.write_text(json.dumps(feature, sort_keys=True) + "\n", encoding="utf-8")
    handoff_file = tmp_path / handoff_path
    handoff = json.loads(handoff_file.read_text(encoding="utf-8"))
    handoff["components"]["feature_manifest"]["sha256"] = fingerprint_file(feature_path)
    handoff_file.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(HandoffValidationError, match=message):
        load_and_validate_preparation_handoff(
            project_root=tmp_path,
            preparation_handoff_path=handoff_path,
        )


def test_identifier_free_runtime_artifacts_are_semantically_idempotent(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    materialize_uci_source(tmp_path, frame)
    csv_artifacts, json_artifacts, _, _ = build_runtime_bundle(tmp_path, frame)
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    result = write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )

    assert set(dict(result.statuses).values()) == {"reused_equivalent"}


def test_identifier_free_runtime_conflict_fails_closed(tmp_path: Path) -> None:
    frame = make_dry_bean_frame()
    materialize_uci_source(tmp_path, frame)
    csv_artifacts, json_artifacts, _, _ = build_runtime_bundle(tmp_path, frame)
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    changed = copy.deepcopy(json_artifacts)
    quality_path = Path("artifacts/preparation/dry-bean/quality-evidence.json")
    changed[quality_path]["readiness"]["educational_model_selection_ready"] = False

    with pytest.raises(ArtifactConflictError, match="divergent"):
        write_preparation_artifacts(
            project_root=tmp_path,
            csv_artifacts=csv_artifacts,
            json_artifacts=changed,
        )
