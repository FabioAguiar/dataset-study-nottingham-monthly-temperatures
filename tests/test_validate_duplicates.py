"""Tests for reusable duplicate-record analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_duplicates import (
    DuplicateValidationError,
    analyze_duplicate_records,
)


def _analyze(dataframe: pd.DataFrame, **kwargs: object):
    return analyze_duplicate_records(
        dataframe,
        identifiers=("customerID",),
        feature_columns=("tenure", "Contract"),
        target="Churn",
        **kwargs,
    )


def test_clean_dataset_has_no_duplicate_findings() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [1, 2, 3],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn": ["No", "No", "Yes"],
        }
    )

    report = _analyze(dataframe)

    assert report.row_count == 3
    assert not report.has_exact_duplicates
    assert not report.has_duplicate_identifiers
    assert not report.has_conflicting_identifiers
    assert not report.has_repeated_profiles
    assert not report.has_target_conflicts
    assert not report.has_quality_issues
    assert report.issues_frame().empty
    report.raise_if_invalid()


def test_exact_duplicate_is_also_an_identical_identifier_duplicate() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "A", "B"],
            "tenure": [12, 12, 3],
            "Contract": ["One year", "One year", "Month-to-month"],
            "Churn": ["No", "No", "Yes"],
        },
        index=[10, 20, 30],
    )

    report = _analyze(dataframe)

    assert report.exact_duplicate_group_count == 1
    assert report.exact_duplicate_row_count == 2
    assert report.duplicate_identifier_group_count == 1
    assert report.duplicate_identifier_row_count == 2
    assert not report.has_conflicting_identifiers
    assert not report.has_repeated_profiles

    exact = report.exact_duplicates_frame()
    assert list(exact["Row position"]) == [0, 1]
    assert list(exact["Row index"]) == [10, 20]

    identifiers = report.identifier_duplicates_frame()
    assert identifiers.iloc[0]["Classification"] == "Repeated identical record"
    assert identifiers.iloc[0]["Distinct record count"] == 1


def test_repeated_identifier_with_different_content_is_conflicting() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "A", "B"],
            "tenure": [12, 13, 3],
            "Contract": ["One year", "One year", "Month-to-month"],
            "Churn": ["No", "Yes", "Yes"],
        }
    )

    report = _analyze(dataframe)

    assert not report.has_exact_duplicates
    assert report.has_duplicate_identifiers
    assert report.has_conflicting_identifiers
    assert report.conflicting_identifier_group_count == 1
    assert report.conflicting_identifier_row_count == 2
    assert (
        report.identifier_duplicates_frame().iloc[0]["Classification"]
        == "Conflicting records"
    )


def test_repeated_profile_with_same_target_is_valid_repetition() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [12, 12, 3],
            "Contract": ["One year", "One year", "Month-to-month"],
            "Churn": ["No", "No", "Yes"],
        }
    )

    report = _analyze(dataframe)

    assert report.has_repeated_profiles
    assert not report.has_target_conflicts
    assert not report.has_quality_issues
    assert report.repeated_profile_group_count == 1
    assert report.repeated_profile_row_count == 2
    assert report.same_target_profile_group_count == 1
    assert report.same_target_profile_row_count == 2

    profiles = report.repeated_profiles_frame()
    assert profiles.iloc[0]["Distinct identifier count"] == 2
    assert profiles.iloc[0]["Target values"] == "'No'"
    assert profiles.iloc[0]["Classification"] == "Same target"


def test_repeated_profile_with_different_targets_is_analytical_ambiguity() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [12, 12, 3],
            "Contract": ["One year", "One year", "Month-to-month"],
            "Churn": ["No", "Yes", "Yes"],
        }
    )

    report = _analyze(dataframe)

    assert report.has_repeated_profiles
    assert report.has_target_conflicts
    assert not report.has_quality_issues
    assert report.target_conflict_group_count == 1
    assert report.target_conflict_row_count == 2

    conflicts = report.target_conflicts_frame()
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["Target values"] == "'No', 'Yes'"
    assert conflicts.iloc[0]["Classification"] == "Target disagreement"


def test_exact_duplicate_does_not_count_as_distinct_identifier_profile() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "A"],
            "tenure": [12, 12],
            "Contract": ["One year", "One year"],
            "Churn": ["No", "No"],
        }
    )

    report = _analyze(dataframe)

    assert report.has_exact_duplicates
    assert report.has_duplicate_identifiers
    assert not report.has_repeated_profiles


def test_multiple_identifier_columns_are_supported() -> None:
    dataframe = pd.DataFrame(
        {
            "account": ["A", "A", "A"],
            "snapshot": [1, 2, 2],
            "value": [10, 10, 11],
            "target": ["No", "No", "Yes"],
        }
    )

    report = analyze_duplicate_records(
        dataframe,
        identifiers=("account", "snapshot"),
        feature_columns=("value",),
        target="target",
    )

    assert report.duplicate_identifier_group_count == 1
    assert report.has_conflicting_identifiers
    duplicate_group = report.identifier_duplicates_frame().iloc[0]
    assert duplicate_group["account"] == "A"
    assert duplicate_group["snapshot"] == 2


def test_group_samples_are_deterministic_and_capped() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D", "E", "F"],
            "tenure": [1, 1, 2, 2, 3, 3],
            "Contract": ["One year"] * 6,
            "Churn": ["No"] * 6,
        }
    )

    report = _analyze(dataframe, max_group_samples=2)

    assert report.repeated_profile_group_count == 3
    sampled = report.repeated_profiles_frame()
    assert len(sampled) == 2
    assert list(sampled["Row positions"]) == [(0, 1), (2, 3)]


def test_summary_and_issue_frames_use_stable_contracts() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "A"],
            "tenure": [1, 2],
            "Contract": ["One year", "One year"],
            "Churn": ["No", "Yes"],
        }
    )

    report = _analyze(dataframe)

    summary = report.summary_frame()
    assert list(summary.columns) == [
        "Metric",
        "Group count",
        "Row count",
        "Interpretation",
    ]
    assert list(summary["Metric"]) == [
        "Exact duplicate records",
        "Duplicate identifiers",
        "Conflicting duplicate identifiers",
        "Repeated feature profiles",
        "Repeated profiles with the same target",
        "Repeated profiles with target disagreement",
    ]

    issues = report.issues_frame()
    assert list(issues["Issue"]) == [
        "Duplicate identifiers",
        "Conflicting duplicate identifiers",
    ]


def test_raise_if_invalid_can_allow_exploratory_repeated_profiles() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [12, 12],
            "Contract": ["One year", "One year"],
            "Churn": ["No", "Yes"],
        }
    )
    report = _analyze(dataframe)

    report.raise_if_invalid()

    with pytest.raises(
        DuplicateValidationError,
        match="repeated feature profile groups found: 1",
    ):
        report.raise_if_invalid(require_no_repeated_profiles=True)

    with pytest.raises(
        DuplicateValidationError,
        match="target disagreement found: 1",
    ):
        report.raise_if_invalid(require_no_target_conflicts=True)


def test_raise_if_invalid_reports_duplicate_quality_failures() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "A"],
            "tenure": [1, 1],
            "Contract": ["One year", "One year"],
            "Churn": ["No", "No"],
        }
    )
    report = _analyze(dataframe)

    with pytest.raises(
        DuplicateValidationError,
        match="exact duplicate record groups found: 1",
    ):
        report.raise_if_invalid()

    report.raise_if_invalid(
        require_no_exact_duplicates=False,
        require_unique_identifiers=False,
        require_no_conflicting_identifiers=False,
    )


def test_invalid_configuration_is_rejected() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A"],
            "tenure": [1],
            "Contract": ["One year"],
            "Churn": ["No"],
        }
    )

    with pytest.raises(KeyError, match="missing from the dataset: missing"):
        analyze_duplicate_records(
            dataframe,
            identifiers=("customerID",),
            feature_columns=("tenure", "missing"),
            target="Churn",
        )

    with pytest.raises(
        DuplicateValidationError,
        match="identifier and feature columns overlap",
    ):
        analyze_duplicate_records(
            dataframe,
            identifiers=("customerID",),
            feature_columns=("customerID", "tenure"),
            target="Churn",
        )

    with pytest.raises(
        DuplicateValidationError,
        match="cannot also be a feature",
    ):
        analyze_duplicate_records(
            dataframe,
            identifiers=("customerID",),
            feature_columns=("tenure", "Churn"),
            target="Churn",
        )

    with pytest.raises(ValueError, match="at least 1"):
        _analyze(dataframe, max_group_samples=0)


def test_duplicate_column_labels_are_rejected() -> None:
    dataframe = pd.DataFrame(
        [["A", 1, 2, "No"]],
        columns=["customerID", "tenure", "tenure", "Churn"],
    )

    with pytest.raises(
        DuplicateValidationError,
        match="dataframe columns must be unique",
    ):
        analyze_duplicate_records(
            dataframe,
            identifiers=("customerID",),
            feature_columns=("tenure",),
            target="Churn",
        )


def test_analysis_does_not_modify_the_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [12, 12],
            "Contract": ["One year", "One year"],
            "Churn": ["No", "Yes"],
        },
        index=[5, 9],
    )
    before = dataframe.copy(deep=True)

    report = _analyze(dataframe)
    sampled = report.repeated_profiles_frame()
    sampled.at[0, "Classification"] = "changed"

    pd.testing.assert_frame_equal(dataframe, before)


def test_dataset_without_source_identifiers_is_supported() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
            "Class": ["A", "B", "C"],
        }
    )

    report = analyze_duplicate_records(
        dataframe,
        identifiers=(),
        feature_columns=("feature_a", "feature_b"),
        target="Class",
    )

    assert not report.has_source_identifiers
    assert not report.has_exact_duplicates
    assert not report.has_duplicate_identifiers
    assert not report.has_repeated_profiles
    assert not report.has_quality_issues
    report.raise_if_invalid()


def test_exact_row_matches_without_identifiers_require_review_not_removal() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 1.0, 2.0],
            "feature_b": [10.0, 10.0, 20.0],
            "Class": ["A", "A", "B"],
        }
    )

    report = analyze_duplicate_records(
        dataframe,
        identifiers=None,
        feature_columns=("feature_a", "feature_b"),
        target="Class",
    )

    assert report.has_exact_duplicates
    assert report.exact_duplicate_group_count == 1
    assert report.exact_duplicate_row_count == 2
    assert report.has_repeated_profiles
    assert report.same_target_profile_group_count == 1
    assert not report.has_quality_issues
    assert report.issues_frame().empty

    summary = report.summary_frame().set_index("Metric")
    assert (
        summary.loc["Exact duplicate records", "Interpretation"]
        == "Review required; source identity unavailable"
    )
    assert (
        summary.loc["Duplicate identifiers", "Interpretation"]
        == "Not applicable; source identifier unavailable"
    )

    report.raise_if_invalid(
        require_no_exact_duplicates=False,
        require_unique_identifiers=False,
        require_no_conflicting_identifiers=False,
    )


def test_repeated_profiles_without_identifiers_can_expose_target_disagreement() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 1.0, 2.0],
            "feature_b": [10.0, 10.0, 20.0],
            "Class": ["A", "B", "C"],
        }
    )

    report = analyze_duplicate_records(
        dataframe,
        identifiers=(),
        feature_columns=("feature_a", "feature_b"),
        target="Class",
    )

    assert not report.has_exact_duplicates
    assert report.has_repeated_profiles
    assert report.has_target_conflicts
    assert report.target_conflict_group_count == 1
    assert report.target_conflict_row_count == 2

    profiles = report.repeated_profiles_frame()
    assert pd.isna(profiles.iloc[0]["Distinct identifier count"])
    assert profiles.iloc[0]["Classification"] == "Target disagreement"
    assert profiles.iloc[0]["Target values"] == "'A', 'B'"
