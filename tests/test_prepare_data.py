from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_data import (
    ArtifactConflictError,
    ClassificationSplitPolicy,
    ConditionalMaterializationError,
    ConditionalNumericRule,
    DatasetValidationError,
    HandoffValidationError,
    PartitionValidationError,
    PreparedDataset,
    SplitPolicyError,
    build_feature_manifest,
    build_preparation_manifest,
    build_quality_evidence,
    build_split_manifest,
    dataframe_csv_bytes,
    fingerprint_dataframe,
    fingerprint_dataframe_csv,
    fingerprint_file,
    load_and_validate_preparation_handoff,
    materialize_conditional_numeric_values,
    prepare_tabular_dataset,
    semantically_equivalent,
    separate_dataset_roles,
    split_classification_dataset,
    validate_dataset_partitions,
    validate_prepared_dataset,
    validate_raw_dataset,
    validate_split_policy,
    write_preparation_artifacts,
)


COLUMN_ORDER = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)
IDENTIFIER_COLUMNS = ("customerID",)
FEATURE_COLUMNS = COLUMN_ORDER[1:-1]
TARGET_COLUMN = "Churn"
TARGET_CLASSES = ("No", "Yes")
NUMERICAL_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
CATEGORICAL_EXPECTED_VALUES = {
    "gender": ("Female", "Male"),
    "SeniorCitizen": (0, 1),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No", "Yes", "No internet service"),
    "OnlineBackup": ("No", "Yes", "No internet service"),
    "DeviceProtection": ("No", "Yes", "No internet service"),
    "TechSupport": ("No", "Yes", "No internet service"),
    "StreamingTV": ("No", "Yes", "No internet service"),
    "StreamingMovies": ("No", "Yes", "No internet service"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("No", "Yes"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
RAW_EXPECTED_TYPES = {
    "customerID": "string",
    "gender": "string",
    "SeniorCitizen": "integer",
    "Partner": "string",
    "Dependents": "string",
    "tenure": "integer",
    "PhoneService": "string",
    "MultipleLines": "string",
    "InternetService": "string",
    "OnlineSecurity": "string",
    "OnlineBackup": "string",
    "DeviceProtection": "string",
    "TechSupport": "string",
    "StreamingTV": "string",
    "StreamingMovies": "string",
    "Contract": "string",
    "PaperlessBilling": "string",
    "PaymentMethod": "string",
    "MonthlyCharges": "numeric",
    "TotalCharges": "numeric",
    "Churn": "string",
}
PREPARED_EXPECTED_TYPES = dict(RAW_EXPECTED_TYPES)
RULE = ConditionalNumericRule(
    column="TotalCharges",
    condition_column="tenure",
    condition_value=0,
    blank_replacement=0.0,
)


def make_telco_frame(row_count: int = 60, *, blank_count: int = 3) -> pd.DataFrame:
    assert row_count >= 30
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        tenure = 0 if index < blank_count else (index % 71) + 1
        total = "   " if index < blank_count else f" {20.25 + index * 8.5:.2f} "
        rows.append(
            {
                "customerID": f"C{index:05d}",
                "gender": CATEGORICAL_EXPECTED_VALUES["gender"][index % 2],
                "SeniorCitizen": index % 2,
                "Partner": CATEGORICAL_EXPECTED_VALUES["Partner"][(index // 2) % 2],
                "Dependents": CATEGORICAL_EXPECTED_VALUES["Dependents"][(index // 3) % 2],
                "tenure": tenure,
                "PhoneService": CATEGORICAL_EXPECTED_VALUES["PhoneService"][(index // 4) % 2],
                "MultipleLines": CATEGORICAL_EXPECTED_VALUES["MultipleLines"][index % 3],
                "InternetService": CATEGORICAL_EXPECTED_VALUES["InternetService"][(index // 2) % 3],
                "OnlineSecurity": CATEGORICAL_EXPECTED_VALUES["OnlineSecurity"][index % 3],
                "OnlineBackup": CATEGORICAL_EXPECTED_VALUES["OnlineBackup"][(index + 1) % 3],
                "DeviceProtection": CATEGORICAL_EXPECTED_VALUES["DeviceProtection"][(index + 2) % 3],
                "TechSupport": CATEGORICAL_EXPECTED_VALUES["TechSupport"][(index // 2) % 3],
                "StreamingTV": CATEGORICAL_EXPECTED_VALUES["StreamingTV"][(index // 3) % 3],
                "StreamingMovies": CATEGORICAL_EXPECTED_VALUES["StreamingMovies"][(index // 4) % 3],
                "Contract": CATEGORICAL_EXPECTED_VALUES["Contract"][index % 3],
                "PaperlessBilling": CATEGORICAL_EXPECTED_VALUES["PaperlessBilling"][index % 2],
                "PaymentMethod": CATEGORICAL_EXPECTED_VALUES["PaymentMethod"][index % 4],
                "MonthlyCharges": 18.5 + index * 0.75,
                "TotalCharges": total,
                "Churn": "Yes" if index % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(rows, columns=COLUMN_ORDER)


def raw_report(frame: pd.DataFrame):
    return validate_raw_dataset(
        frame,
        column_order=COLUMN_ORDER,
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
        categorical_expected_values=CATEGORICAL_EXPECTED_VALUES,
        expected_types=RAW_EXPECTED_TYPES,
        numeric_text_columns=("TotalCharges",),
    )


def prepared_result(frame: pd.DataFrame) -> PreparedDataset:
    return prepare_tabular_dataset(
        frame,
        conditional_numeric_rules=(RULE,),
    )


def prepared_report(frame: pd.DataFrame, result: PreparedDataset):
    return validate_prepared_dataset(
        frame,
        result.dataframe,
        column_order=COLUMN_ORDER,
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
        categorical_expected_values=CATEGORICAL_EXPECTED_VALUES,
        expected_types=PREPARED_EXPECTED_TYPES,
        authorized_changed_columns=("TotalCharges",),
        expected_row_count=len(frame),
        expected_materialized_counts={"TotalCharges": 3},
        observed_materialized_counts=dict(result.materialized_counts),
    )


def split_policy(**changes) -> ClassificationSplitPolicy:
    values = {
        "evaluation_mode": "stratified_random_snapshot",
        "purpose": "educational_benchmark",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "stratify_by": "Churn",
        "random_seed": 42,
        "shuffle": True,
        "educational_justification": (
            "Advance a reproducible educational benchmark without claiming "
            "future-customer or production validity."
        ),
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }
    values.update(changes)
    return ClassificationSplitPolicy(**values)


def partition_bundle(frame: pd.DataFrame):
    result = prepared_result(frame)
    prepared = result.dataframe
    policy = split_policy()
    partitions = split_classification_dataset(
        prepared,
        policy=policy,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    validation = validate_dataset_partitions(
        prepared,
        partitions,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_column=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
        prevalence_tolerance=0.08,
    )
    return result, prepared, policy, partitions, validation


# A. Immutability

def test_prepare_does_not_modify_source_dataframe() -> None:
    frame = make_telco_frame()
    original = frame.copy(deep=True)
    prepare_tabular_dataset(frame, conditional_numeric_rules=(RULE,))
    pd.testing.assert_frame_equal(frame, original)


def test_prepare_does_not_modify_rule_configuration() -> None:
    frame = make_telco_frame()
    original = copy.deepcopy(RULE)
    prepare_tabular_dataset(frame, conditional_numeric_rules=(RULE,))
    assert RULE == original


def test_prepared_dataframe_property_returns_defensive_copy() -> None:
    result = prepared_result(make_telco_frame())
    first = result.dataframe
    first.loc[first.index[0], "TotalCharges"] = 99999.0
    assert result.dataframe.loc[result.dataframe.index[0], "TotalCharges"] == 0.0


def test_role_projections_return_defensive_copies() -> None:
    prepared = prepared_result(make_telco_frame()).dataframe
    roles = separate_dataset_roles(
        prepared,
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    features = roles.features
    features.iloc[0, 0] = "changed"
    assert roles.features.iloc[0, 0] != "changed"


def test_raw_logical_fingerprint_is_preserved_after_preparation() -> None:
    frame = make_telco_frame()
    before = fingerprint_dataframe(frame)
    prepared_result(frame)
    assert fingerprint_dataframe(frame) == before


def test_column_order_is_preserved() -> None:
    result = prepared_result(make_telco_frame())
    assert tuple(result.dataframe.columns) == COLUMN_ORDER


# B. TotalCharges

@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("100.25", 100.25), (" 100.25 ", 100.25), (12, 12.0), (12.5, 12.5)],
)
def test_total_charges_valid_numeric_inputs(raw_value, expected) -> None:
    frame = pd.DataFrame({"tenure": [1], "TotalCharges": [raw_value]})
    prepared, count, invalid = materialize_conditional_numeric_values(frame, RULE)
    assert prepared.loc[0, "TotalCharges"] == expected
    assert count == 0
    assert invalid == 0


def test_total_charges_blank_with_zero_tenure_becomes_zero() -> None:
    frame = pd.DataFrame({"tenure": [0], "TotalCharges": ["   "]})
    prepared, count, invalid = materialize_conditional_numeric_values(frame, RULE)
    assert prepared.loc[0, "TotalCharges"] == 0.0
    assert count == 1
    assert invalid == 0


def test_total_charges_supports_strict_pandas_string_dtype() -> None:
    frame = pd.DataFrame(
        {
            "tenure": pd.Series([0, 1], dtype="int64"),
            "TotalCharges": pd.Series(["   ", " 19.95 "], dtype="string"),
        }
    )

    prepared, count, invalid = materialize_conditional_numeric_values(frame, RULE)

    assert prepared["TotalCharges"].tolist() == [0.0, 19.95]
    assert pd.api.types.is_float_dtype(prepared["TotalCharges"])
    assert frame["TotalCharges"].dtype == pd.StringDtype()
    assert frame["TotalCharges"].tolist() == ["   ", " 19.95 "]
    assert count == 1
    assert invalid == 0


def test_total_charges_blank_with_positive_tenure_is_rejected() -> None:
    frame = pd.DataFrame({"tenure": [3], "TotalCharges": [" "]})
    with pytest.raises(ConditionalMaterializationError, match="blank value"):
        materialize_conditional_numeric_values(frame, RULE)


def test_total_charges_non_numeric_text_is_rejected() -> None:
    frame = pd.DataFrame({"tenure": [2], "TotalCharges": ["not-a-number"]})
    with pytest.raises(ConditionalMaterializationError, match="non-convertible"):
        materialize_conditional_numeric_values(frame, RULE)


def test_total_charges_null_is_not_silently_imputed() -> None:
    frame = pd.DataFrame({"tenure": [0], "TotalCharges": [None]})
    with pytest.raises(ConditionalMaterializationError, match="null values"):
        materialize_conditional_numeric_values(frame, RULE)


def test_total_charges_materialization_count_and_dtype() -> None:
    result = prepared_result(make_telco_frame(blank_count=3))
    assert dict(result.materialized_counts) == {"TotalCharges": 3}
    assert pd.api.types.is_float_dtype(result.dataframe["TotalCharges"])
    assert result.dataframe["TotalCharges"].isna().sum() == 0


def test_total_charges_preparation_removes_no_rows() -> None:
    frame = make_telco_frame()
    result = prepared_result(frame)
    assert len(result.dataframe) == len(frame)
    assert list(result.dataframe["customerID"]) == list(frame["customerID"])


# C. Schema and roles

@pytest.mark.parametrize(
    ("column", "message"),
    [("customerID", "Missing required columns"), ("Churn", "Missing required columns"), ("tenure", "Missing required columns")],
)
def test_schema_rejects_missing_required_columns(column, message) -> None:
    frame = make_telco_frame().drop(columns=[column])
    with pytest.raises(DatasetValidationError, match=message):
        raw_report(frame)


def test_schema_rejects_unexpected_field() -> None:
    frame = make_telco_frame().assign(extra_field="unexpected")
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        raw_report(frame)


def test_schema_rejects_duplicate_identifier() -> None:
    frame = make_telco_frame()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    with pytest.raises(DatasetValidationError, match="duplicate"):
        raw_report(frame)


def test_schema_rejects_blank_identifier() -> None:
    frame = make_telco_frame()
    frame.loc[0, "customerID"] = "  "
    with pytest.raises(DatasetValidationError, match="blank"):
        raw_report(frame)


def test_schema_rejects_unexpected_target() -> None:
    frame = make_telco_frame()
    frame.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DatasetValidationError, match="unexpected classes"):
        raw_report(frame)


def test_schema_rejects_absent_expected_target_class() -> None:
    frame = make_telco_frame()
    frame["Churn"] = "No"
    with pytest.raises(DatasetValidationError, match="missing expected classes"):
        raw_report(frame)


def test_schema_rejects_unexpected_category() -> None:
    frame = make_telco_frame()
    frame.loc[0, "InternetService"] = "Satellite"
    with pytest.raises(DatasetValidationError, match="unexpected values"):
        raw_report(frame)


def test_structural_categories_are_preserved() -> None:
    frame = make_telco_frame()
    result = prepared_result(frame)
    assert "No internet service" in set(result.dataframe["OnlineSecurity"])
    assert "No phone service" in set(result.dataframe["MultipleLines"])


def test_senior_citizen_remains_categorical_contract() -> None:
    report = raw_report(make_telco_frame())
    observed = report.as_dict()["observed_categories"]
    assert observed["SeniorCitizen"] == [0, 1]
    assert "SeniorCitizen" in CATEGORICAL_FEATURES
    assert "SeniorCitizen" not in NUMERICAL_FEATURES


def test_customer_id_and_target_are_excluded_from_x() -> None:
    prepared = prepared_result(make_telco_frame()).dataframe
    roles = separate_dataset_roles(
        prepared,
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    assert "customerID" not in roles.features.columns
    assert "Churn" not in roles.features.columns
    assert tuple(roles.features.columns) == FEATURE_COLUMNS


def test_prepared_validation_rejects_unauthorized_change() -> None:
    frame = make_telco_frame()
    result = prepared_result(frame)
    changed = result.dataframe
    changed.loc[0, "gender"] = "Male" if changed.loc[0, "gender"] == "Female" else "Female"
    with pytest.raises(DatasetValidationError, match="Unauthorized"):
        validate_prepared_dataset(
            frame,
            changed,
            column_order=COLUMN_ORDER,
            identifier_columns=IDENTIFIER_COLUMNS,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            target_classes=TARGET_CLASSES,
            categorical_expected_values=CATEGORICAL_EXPECTED_VALUES,
            expected_types=PREPARED_EXPECTED_TYPES,
            authorized_changed_columns=("TotalCharges",),
        )


# D. Split policy

def test_valid_split_policy() -> None:
    checks = validate_split_policy(split_policy(), known_columns=COLUMN_ORDER)
    assert all(checks.values())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evaluation_mode": "temporal"}, "evaluation_mode"),
        ({"purpose": "production"}, "purpose"),
        ({"educational_justification": ""}, "justification"),
        ({"train_fraction": -0.1}, "greater than zero"),
        ({"validation_fraction": 0.0}, "greater than zero"),
        ({"test_fraction": 0.2}, "sum to 1.0"),
        ({"stratify_by": "unknown"}, "Unknown stratification"),
        ({"random_seed": 42.0}, "integer"),
        ({"shuffle": False}, "shuffle"),
        ({"operational_validity": "confirmed"}, "operational_validity"),
        ({"temporal_contract_status": "resolved"}, "temporal_contract_status"),
        ({"feature_inference_availability": "confirmed"}, "feature_inference_availability"),
    ],
)
def test_invalid_split_policy(changes, message) -> None:
    with pytest.raises(SplitPolicyError, match=message):
        validate_split_policy(split_policy(**changes), known_columns=COLUMN_ORDER)


# E. Partitions

def test_split_is_reproducible_with_same_seed() -> None:
    prepared = prepared_result(make_telco_frame(120)).dataframe
    first = split_classification_dataset(
        prepared,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    second = split_classification_dataset(
        prepared,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    for name in ("train", "validation", "test"):
        assert list(first.as_mapping()[name]["customerID"]) == list(
            second.as_mapping()[name]["customerID"]
        )


def test_split_membership_is_independent_of_input_row_order() -> None:
    prepared = prepared_result(make_telco_frame(120)).dataframe
    shuffled = prepared.sample(frac=1.0, random_state=999)
    first = split_classification_dataset(
        prepared,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    second = split_classification_dataset(
        shuffled,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    for name in ("train", "validation", "test"):
        assert set(first.as_mapping()[name]["customerID"]) == set(
            second.as_mapping()[name]["customerID"]
        )


def test_partition_validation_confirms_classes_prevalence_isolation_and_coverage() -> None:
    _, prepared, _, partitions, validation = partition_bundle(make_telco_frame(120))
    assert validation.is_valid
    checks = validation.as_dict()["checks"]
    assert checks["all_classes_present"]
    assert checks["class_prevalence_within_tolerance"]
    assert checks["full_coverage"]
    assert sum(validation.as_dict()["row_counts"].values()) == len(prepared)


def test_partition_membership_and_customer_ids_are_disjoint() -> None:
    _, _, _, partitions, _ = partition_bundle(make_telco_frame(120))
    mappings = partitions.as_mapping()
    memberships = {name: set(frame["customerID"]) for name, frame in mappings.items()}
    assert memberships["train"].isdisjoint(memberships["validation"])
    assert memberships["train"].isdisjoint(memberships["test"])
    assert memberships["validation"].isdisjoint(memberships["test"])


def test_partition_rows_are_written_in_stable_source_order() -> None:
    _, prepared, _, partitions, _ = partition_bundle(make_telco_frame(120))
    positions = {identifier: index for index, identifier in enumerate(prepared["customerID"])}
    for frame in partitions.as_mapping().values():
        observed = [positions[value] for value in frame["customerID"]]
        assert observed == sorted(observed)


def test_partition_properties_are_defensive_copies() -> None:
    _, _, _, partitions, _ = partition_bundle(make_telco_frame(120))
    train = partitions.train
    original = partitions.train.loc[partitions.train.index[0], "customerID"]
    train.loc[train.index[0], "customerID"] = "changed"
    assert partitions.train.loc[partitions.train.index[0], "customerID"] == original


def test_partition_validation_detects_overlap() -> None:
    _, prepared, _, partitions, _ = partition_bundle(make_telco_frame(120))
    broken = type(partitions)(
        _train=partitions.train,
        _validation=partitions.validation,
        _test=pd.concat([partitions.test, partitions.train.iloc[[0]]]),
        split_method=partitions.split_method,
        rounding_method=partitions.rounding_method,
    )
    with pytest.raises(PartitionValidationError):
        validate_dataset_partitions(
            prepared,
            broken,
            identifier_columns=IDENTIFIER_COLUMNS,
            target_column=TARGET_COLUMN,
            target_classes=TARGET_CLASSES,
            prevalence_tolerance=0.20,
        )


def test_partition_counts_follow_two_stage_rounding() -> None:
    prepared = prepared_result(make_telco_frame(101)).dataframe
    partitions = split_classification_dataset(
        prepared,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    assert len(partitions.train) == 70
    assert len(partitions.validation) == 15
    assert len(partitions.test) == 16
    assert "ceil" in partitions.rounding_method


# F. Fingerprints and manifests

def test_file_sha256_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"alpha")
    first = fingerprint_file(path)
    assert first == hashlib.sha256(b"alpha").hexdigest()
    assert fingerprint_file(path) == first
    path.write_bytes(b"beta")
    assert fingerprint_file(path) != first


def test_dataframe_fingerprint_supports_string_integer_float_and_nontrivial_index() -> None:
    frame = pd.DataFrame(
        {
            "text": pd.Series(["a", "b"], dtype="string"),
            "integer": pd.Series([1, 2], dtype="Int64"),
            "float": [1.5, 2.5],
        },
        index=pd.Index([10, 20], name="row_id"),
    )
    first = fingerprint_dataframe(frame)
    assert len(first) == 64
    assert fingerprint_dataframe(frame.copy(deep=True)) == first
    changed = frame.copy(deep=True)
    changed.iloc[0, 2] = 9.5
    assert fingerprint_dataframe(changed) != first


def test_csv_fingerprint_matches_serialized_bytes() -> None:
    frame = prepared_result(make_telco_frame()).dataframe
    assert fingerprint_dataframe_csv(frame) == hashlib.sha256(
        dataframe_csv_bytes(frame)
    ).hexdigest()


def build_manifests(frame: pd.DataFrame):
    result, prepared, policy, partitions, partition_validation = partition_bundle(frame)
    raw_validation = raw_report(frame)
    prepared_validation = prepared_report(frame, result)
    source_sha = "1" * 64
    prepared_path = "data/processed/telco/prepared.csv"
    split_paths = {
        "train": "data/processed/telco/splits/policy/train.csv",
        "validation": "data/processed/telco/splits/policy/validation.csv",
        "test": "data/processed/telco/splits/policy/test.csv",
    }
    partition_sha = {
        name: fingerprint_dataframe_csv(part)
        for name, part in partitions.as_mapping().items()
    }
    readiness = {
        "deterministic_preparation_ready": True,
        "prepared_dataset_materialized": True,
        "benchmark_split_ready": True,
        "benchmark_partitions_materialized": True,
        "educational_model_selection_ready": True,
        "operational_modeling_ready": False,
    }
    prep_manifest = build_preparation_manifest(
        dataset_slug="telco",
        source_path="data/raw/telco/source.csv",
        source_sha256=source_sha,
        prepared_path=prepared_path,
        prepared_sha256=fingerprint_dataframe_csv(prepared),
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        preparation=result,
        raw_fingerprint_before=fingerprint_dataframe(frame),
        raw_fingerprint_after=fingerprint_dataframe(frame),
        source_sha256_after=source_sha,
        deterministic_rules=[RULE.as_dict()],
        readiness=readiness,
    )
    feature_manifest = build_feature_manifest(
        dataset_slug="telco",
        identifier_columns=IDENTIFIER_COLUMNS,
        feature_columns=FEATURE_COLUMNS,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        categorical_expected_values=CATEGORICAL_EXPECTED_VALUES,
        target_column=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
        positive_target_class="Yes",
        target_encoding={"No": 0, "Yes": 1},
        expected_dtypes={
            "raw": RAW_EXPECTED_TYPES,
            "prepared": PREPARED_EXPECTED_TYPES,
        },
        preprocessing_contract={
            "categorical_strategy": "one_hot",
            "categorical_fit_scope": "inside_training_fold",
            "unknown_category_policy": "ignore_and_report",
            "drop_category": None,
            "numerical_scaling": "model_specific",
            "numerical_fit_scope": "inside_training_fold",
        },
        prohibited_predictors=("customerID", "Churn"),
    )
    split_manifest = build_split_manifest(
        dataset_slug="telco",
        policy=policy,
        partitions=partitions,
        validation=partition_validation,
        partition_paths=split_paths,
        partition_sha256=partition_sha,
    )
    quality = build_quality_evidence(
        dataset_slug="telco",
        raw_report=raw_validation,
        prepared_report=prepared_validation,
        partition_report=partition_validation,
        preparation=result,
        fingerprints={
            "source_sha256_before": source_sha,
            "source_sha256_after": source_sha,
            "prepared_sha256": fingerprint_dataframe_csv(prepared),
            "partition_sha256": partition_sha,
        },
        readiness=readiness,
        preservation_checks={
            "raw_unchanged": True,
            "rows_removed": 0,
            "identifiers_changed": 0,
            "categories_changed": 0,
        },
    )
    return (
        result,
        prepared,
        partitions,
        prep_manifest,
        feature_manifest,
        split_manifest,
        quality,
    )


def test_manifests_preserve_schema_versions_feature_order_and_operational_block() -> None:
    _, _, _, prep, feature, split, quality = build_manifests(make_telco_frame(120))
    assert prep["schema_version"] == "preparation-manifest.v1"
    assert feature["feature_columns"] == list(FEATURE_COLUMNS)
    assert split["schema_version"] == "split-manifest.v1"
    assert split["operational_validity"] == "unconfirmed"
    assert split["operational_modeling_ready"] is False
    assert quality["operational_block"]["operational_modeling_ready"] is False


def test_manifest_paths_reject_absolute_paths() -> None:
    frame = make_telco_frame()
    result = prepared_result(frame)
    with pytest.raises(ValueError, match="project-relative"):
        build_preparation_manifest(
            dataset_slug="telco",
            source_path=Path("/secret/source.csv"),
            source_sha256="1" * 64,
            prepared_path="data/processed/prepared.csv",
            prepared_sha256="2" * 64,
            raw_report=raw_report(frame),
            prepared_report=prepared_report(frame, result),
            preparation=result,
            raw_fingerprint_before=fingerprint_dataframe(frame),
            raw_fingerprint_after=fingerprint_dataframe(frame),
            source_sha256_after="1" * 64,
            deterministic_rules=[RULE.as_dict()],
            readiness={},
        )


def test_split_manifest_counts_prevalence_membership_and_hashes_are_coherent() -> None:
    _, _, partitions, _, _, split, _ = build_manifests(make_telco_frame(120))
    for name, frame in partitions.as_mapping().items():
        assert split["row_counts"][name] == len(frame)
        assert split["class_counts"][name]["No"] + split["class_counts"][name]["Yes"] == len(frame)
        assert len(split["membership"][name]) == len(frame)
        assert split["partition_sha256"][name] == fingerprint_dataframe_csv(frame)


def test_timestamp_is_ignored_only_for_semantic_equivalence() -> None:
    left = {"schema_version": "x.v1", "generated_at_utc": "2026-01-01", "value": 1}
    right = {"schema_version": "x.v1", "generated_at_utc": "2026-02-01", "value": 1}
    assert semantically_equivalent(left, right)
    right["value"] = 2
    assert not semantically_equivalent(left, right)


# G. Persistence and handoff

def artifact_payloads(frame: pd.DataFrame):
    result, prepared, partitions, prep, feature, split, quality = build_manifests(frame)
    csv_artifacts = {
        prep["prepared_path"]: prepared,
        split["partition_paths"]["train"]: partitions.train,
        split["partition_paths"]["validation"]: partitions.validation,
        split["partition_paths"]["test"]: partitions.test,
    }
    json_artifacts = {
        "artifacts/preparation/telco/preparation-manifest.json": prep,
        "artifacts/preparation/telco/feature-manifest.json": feature,
        "artifacts/preparation/telco/split-manifest.json": split,
        "artifacts/preparation/telco/quality-evidence.json": quality,
    }
    return csv_artifacts, json_artifacts


def test_persistence_creates_directories_and_complete_artifact_set(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    result = write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    assert set(dict(result.statuses).values()) == {"created"}
    for relative in (*csv_artifacts, *json_artifacts):
        assert (tmp_path / relative).is_file()
    assert not list(tmp_path.glob(".preparation-staging-*"))
    assert not list(tmp_path.glob(".preparation-backup-*"))


def test_persistence_equivalent_rerun_is_idempotent(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    second = write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    assert set(dict(second.statuses).values()) == {"reused_equivalent"}


def test_persistence_accepts_timestamp_only_json_difference(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    first_payloads = copy.deepcopy(json_artifacts)
    first_payloads["artifacts/preparation/telco/quality-evidence.json"]["generated_at_utc"] = "2026-01-01"
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=first_payloads,
    )
    second_payloads = copy.deepcopy(json_artifacts)
    second_payloads["artifacts/preparation/telco/quality-evidence.json"]["generated_at_utc"] = "2026-02-01"
    result = write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=second_payloads,
    )
    assert dict(result.statuses)["artifacts/preparation/telco/quality-evidence.json"] == "reused_equivalent"


def test_persistence_rejects_semantic_conflict_without_overwrite(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    changed = copy.deepcopy(json_artifacts)
    changed["artifacts/preparation/telco/quality-evidence.json"]["readiness"]["benchmark_split_ready"] = False
    with pytest.raises(ArtifactConflictError, match="divergent"):
        write_preparation_artifacts(
            project_root=tmp_path,
            csv_artifacts=csv_artifacts,
            json_artifacts=changed,
        )


def test_persistence_explicit_overwrite_replaces_conflict(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    changed = copy.deepcopy(json_artifacts)
    target = "artifacts/preparation/telco/quality-evidence.json"
    changed[target]["readiness"]["benchmark_split_ready"] = False
    result = write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=changed,
        overwrite=True,
    )
    assert dict(result.statuses)[target] == "overwritten"
    assert json.loads((tmp_path / target).read_text())["readiness"]["benchmark_split_ready"] is False


def test_conflict_detection_happens_before_any_new_file_is_promoted(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    conflict_payloads = copy.deepcopy(json_artifacts)
    target = "artifacts/preparation/telco/feature-manifest.json"
    conflict_payloads[target]["target_column"] = "Other"
    conflict_payloads["artifacts/preparation/telco/new.json"] = {"schema_version": "new.v1"}
    with pytest.raises(ArtifactConflictError):
        write_preparation_artifacts(
            project_root=tmp_path,
            csv_artifacts=csv_artifacts,
            json_artifacts=conflict_payloads,
        )
    assert not (tmp_path / "artifacts/preparation/telco/new.json").exists()


def test_handoff_loads_and_validates_without_resplitting(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    handoff = load_and_validate_preparation_handoff(
        project_root=tmp_path,
        preparation_manifest_path="artifacts/preparation/telco/preparation-manifest.json",
        feature_manifest_path="artifacts/preparation/telco/feature-manifest.json",
        split_manifest_path="artifacts/preparation/telco/split-manifest.json",
        quality_evidence_path="artifacts/preparation/telco/quality-evidence.json",
    )
    assert len(handoff.prepared) == 120
    assert len(handoff.train) + len(handoff.validation) + len(handoff.test) == 120
    assert handoff.manifests["split_manifest"]["operational_modeling_ready"] is False


def test_handoff_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    prepared_path = next(path for path in csv_artifacts if path.endswith("prepared.csv"))
    with (tmp_path / prepared_path).open("a", encoding="utf-8") as target:
        target.write("corruption\n")
    with pytest.raises(HandoffValidationError, match="fingerprint"):
        load_and_validate_preparation_handoff(
            project_root=tmp_path,
            preparation_manifest_path="artifacts/preparation/telco/preparation-manifest.json",
            feature_manifest_path="artifacts/preparation/telco/feature-manifest.json",
            split_manifest_path="artifacts/preparation/telco/split-manifest.json",
            quality_evidence_path="artifacts/preparation/telco/quality-evidence.json",
        )


def test_handoff_returns_defensive_dataframes(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    handoff = load_and_validate_preparation_handoff(
        project_root=tmp_path,
        preparation_manifest_path="artifacts/preparation/telco/preparation-manifest.json",
        feature_manifest_path="artifacts/preparation/telco/feature-manifest.json",
        split_manifest_path="artifacts/preparation/telco/split-manifest.json",
        quality_evidence_path="artifacts/preparation/telco/quality-evidence.json",
    )
    copy_frame = handoff.prepared
    copy_frame.loc[0, "customerID"] = "changed"
    assert handoff.prepared.loc[0, "customerID"] != "changed"


# H. Cross-platform and package-boundary behavior

def test_artifact_paths_use_posix_relative_strings() -> None:
    _, _, _, prep, _, split, _ = build_manifests(make_telco_frame(120))
    assert "\\" not in prep["prepared_path"]
    assert not Path(prep["prepared_path"]).is_absolute()
    assert all("\\" not in value for value in split["partition_paths"].values())


def test_runtime_artifacts_are_only_written_to_supplied_paths(tmp_path: Path) -> None:
    csv_artifacts, json_artifacts = artifact_payloads(make_telco_frame(120))
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts=csv_artifacts,
        json_artifacts=json_artifacts,
    )
    files = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert files == set(csv_artifacts) | set(json_artifacts)


def test_raw_validation_does_not_modify_configuration_structures() -> None:
    frame = make_telco_frame()
    categories = copy.deepcopy(CATEGORICAL_EXPECTED_VALUES)
    types = copy.deepcopy(RAW_EXPECTED_TYPES)
    raw_report(frame)
    assert categories == CATEGORICAL_EXPECTED_VALUES
    assert types == RAW_EXPECTED_TYPES


def test_split_supports_nontrivial_indices_and_preserves_source_order() -> None:
    frame = prepared_result(make_telco_frame(120)).dataframe
    frame.index = pd.Index(range(1000, 1120), name="source_row")
    partitions = split_classification_dataset(
        frame,
        policy=split_policy(),
        identifier_columns=IDENTIFIER_COLUMNS,
        target_classes=TARGET_CLASSES,
    )
    source_positions = {identifier: position for position, identifier in enumerate(frame["customerID"])}
    for partition in partitions.as_mapping().values():
        observed = [source_positions[value] for value in partition["customerID"]]
        assert observed == sorted(observed)


def test_mid_promotion_failure_rolls_back_complete_new_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.prepare_data as prepare_module

    csv_artifacts = {"data/processed/example/prepared.csv": pd.DataFrame({"value": [1, 2]})}
    json_artifacts = {"artifacts/preparation/example/manifest.json": {"schema_version": "example.v1"}}
    real_replace = prepare_module.os.replace
    promotion_calls = 0

    def flaky_replace(source, destination):
        nonlocal promotion_calls
        if ".preparation-staging-" in str(source):
            promotion_calls += 1
            if promotion_calls == 2:
                raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(prepare_module.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated promotion failure"):
        write_preparation_artifacts(
            project_root=tmp_path,
            csv_artifacts=csv_artifacts,
            json_artifacts=json_artifacts,
        )

    assert not (tmp_path / "data/processed/example/prepared.csv").exists()
    assert not (tmp_path / "artifacts/preparation/example/manifest.json").exists()
    assert not list(tmp_path.glob(".preparation-staging-*"))
    assert not list(tmp_path.glob(".preparation-backup-*"))
