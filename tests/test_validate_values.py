"""Tests for reusable missing and invalid value validation."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_values import (
    ValueValidationError,
    analyze_missing_and_invalid_values,
    analyze_source_backed_missing_and_invalid_values,
)


def test_report_detects_missing_blank_and_invalid_values() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", None, "   ", "D"],
            "churn": ["No", "Yes", "Maybe", "No"],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "customer_id": {
                "required": True,
                "allow_blank": False,
            },
            "churn": {
                "required": True,
                "allowed_values": ("No", "Yes"),
            },
        },
    )

    assert report.has_missing_values
    assert report.has_blank_values
    assert report.has_invalid_values
    assert report.has_issues
    assert report.affected_columns == ("customer_id", "churn")

    checks = report.column_frame().set_index("Column")
    assert checks.loc["customer_id", "Missing count"] == 1
    assert checks.loc["customer_id", "Blank count"] == 1
    assert checks.loc["customer_id", "Affected count"] == 2
    assert checks.loc["churn", "Invalid count"] == 1
    assert checks.loc["churn", "Issue types"] == "Unexpected value"


def test_report_detects_all_supported_pandas_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "value": [None, float("nan"), pd.NA, pd.NaT, "present"],
        },
        dtype="object",
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {"value": {"required": True}},
    )

    checks = report.column_frame().iloc[0]
    assert checks["Missing count"] == 4
    assert checks["Affected count"] == 4

    issues = report.issues_frame()
    assert list(issues["Issue"]) == ["Missing value"]
    assert issues.iloc[0]["Raw value"] == "<missing>"
    assert issues.iloc[0]["Count"] == 4


def test_numeric_validation_distinguishes_blank_and_non_numeric_values() -> None:
    dataframe = pd.DataFrame(
        {
            "TotalCharges": ["1889.50", " ", "invalid"],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "TotalCharges": {
                "required": True,
                "numeric": True,
                "minimum": 0,
            }
        },
    )

    checks = report.column_frame().iloc[0]
    assert checks["Blank count"] == 1
    assert checks["Invalid count"] == 2
    assert checks["Affected count"] == 2
    assert checks["Issue types"] == "Blank value, Non-numeric value"

    issues = report.issues_frame()
    assert list(issues["Issue"]) == [
        "Blank value",
        "Non-numeric value",
    ]
    assert list(issues["Raw value"]) == ["' '", "'invalid'"]


def test_numeric_validation_checks_integer_and_inclusive_limits() -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": [0, 12, 2.5, -1, 73],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "tenure": {
                "required": True,
                "integer": True,
                "minimum": 0,
                "maximum": 72,
            }
        },
    )

    checks = report.column_frame().iloc[0]
    assert checks["Invalid count"] == 3
    assert checks["Affected count"] == 3
    assert checks["Issue types"] == (
        "Non-integer value, Below minimum, Above maximum"
    )


def test_text_normalization_reports_inconsistencies_separately() -> None:
    dataframe = pd.DataFrame(
        {
            "churn": ["Yes", " yes ", "NO", "Maybe"],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "churn": {
                "required": True,
                "allowed_values": ("No", "Yes"),
                "strip_strings": True,
                "case_sensitive": False,
            }
        },
    )

    checks = report.column_frame().iloc[0]
    assert checks["Inconsistent count"] == 2
    assert checks["Invalid count"] == 1
    assert checks["Affected count"] == 3
    assert report.has_inconsistent_values

    issues = report.issues_frame()
    assert set(issues["Issue"]) == {
        "Inconsistent text",
        "Unexpected value",
    }


def test_allowed_missing_and_blank_values_are_counted_but_not_issues() -> None:
    dataframe = pd.DataFrame(
        {
            "optional_note": [None, "", "present"],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "optional_note": {
                "required": False,
                "allow_blank": True,
            }
        },
    )

    checks = report.column_frame().iloc[0]
    assert checks["Missing count"] == 1
    assert checks["Blank count"] == 1
    assert checks["Affected count"] == 0
    assert checks["Status"] == "Valid"
    assert report.has_missing_values
    assert report.has_blank_values
    assert not report.has_issues


def test_report_tracks_unassessed_and_missing_rule_columns_in_order() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A"],
            "value": [1],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "customer_id": {"required": True},
            "future_column": {"required": True},
        },
    )

    assert report.unassessed_columns == ("value",)
    assert report.missing_rule_columns == ("future_column",)
    assert not report.is_fully_assessed
    assert list(report.column_frame()["Column"]) == [
        "customer_id",
        "value",
        "future_column",
    ]


def test_report_raise_if_invalid_can_allow_exploratory_findings() -> None:
    dataframe = pd.DataFrame({"value": [1, -1]})
    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "value": {
                "numeric": True,
                "minimum": 0,
            }
        },
    )

    report.raise_if_invalid(require_no_quality_issues=False)

    with pytest.raises(
        ValueValidationError,
        match="columns requiring value-quality review: value",
    ):
        report.raise_if_invalid()


def test_report_raise_if_invalid_checks_rule_coverage() -> None:
    dataframe = pd.DataFrame({"value": [1], "unassessed": [2]})
    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "value": {"numeric": True},
            "missing": {"required": True},
        },
    )

    with pytest.raises(
        ValueValidationError,
        match="without value validation rules: unassessed",
    ):
        report.raise_if_invalid(require_no_quality_issues=False)

    report.raise_if_invalid(
        require_all_columns_assessed=False,
        require_rule_columns_present=False,
        require_no_quality_issues=False,
    )


def test_invalid_rule_configuration_is_rejected() -> None:
    dataframe = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError, match="Unsupported validation rule field"):
        analyze_missing_and_invalid_values(
            dataframe,
            {"value": {"unknown": True}},
        )

    with pytest.raises(TypeError, match="must be a boolean"):
        analyze_missing_and_invalid_values(
            dataframe,
            {"value": {"required": "yes"}},
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        analyze_missing_and_invalid_values(
            dataframe,
            {"value": {"allowed_values": []}},
        )

    with pytest.raises(ValueError, match="without enabling 'numeric'"):
        analyze_missing_and_invalid_values(
            dataframe,
            {"value": {"minimum": 0}},
        )

    with pytest.raises(ValueError, match="minimum greater than maximum"):
        analyze_missing_and_invalid_values(
            dataframe,
            {
                "value": {
                    "numeric": True,
                    "minimum": 10,
                    "maximum": 1,
                }
            },
        )


def test_issue_samples_are_capped_and_deterministic() -> None:
    dataframe = pd.DataFrame(
        {
            "category": ["z", "a", "z", "b", "a", "c"],
        }
    )

    report = analyze_missing_and_invalid_values(
        dataframe,
        {"category": {"allowed_values": ("valid",)}},
        max_issue_samples=2,
    )

    issues = report.issues_frame()
    assert list(issues["Raw value"]) == ["'a'", "'z'"]
    assert list(issues["Count"]) == [2, 2]


def test_analysis_and_report_frames_do_not_modify_inputs() -> None:
    dataframe = pd.DataFrame(
        {
            "value": ["1", "invalid"],
        }
    )
    original_dataframe = dataframe.copy(deep=True)
    rules = {
        "value": {
            "required": True,
            "numeric": True,
        }
    }
    original_rules = {
        column: dict(rule)
        for column, rule in rules.items()
    }

    report = analyze_missing_and_invalid_values(dataframe, rules)
    external_checks = report.column_frame()
    external_issues = report.issues_frame()
    external_checks.loc[0, "Status"] = "Changed"
    external_issues.loc[0, "Issue"] = "Changed"

    pd.testing.assert_frame_equal(dataframe, original_dataframe)
    assert rules == original_rules
    assert report.column_frame().loc[0, "Status"] == "Review required"
    assert report.issues_frame().loc[0, "Issue"] == "Non-numeric value"



def test_source_backed_analysis_derives_rules_from_metadata_and_target_contract(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": [100, 101, 102],
            "Perimeter": [40.5, float("inf"), 42.0],
            "Class": ["SEKER", " SEKER ", None],
        }
    )
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Perimeter", "Class"],
            "type": ["Integer", "Continuous", "Categorical"],
        }
    ).to_csv(variables_file, index=False)

    report = analyze_source_backed_missing_and_invalid_values(
        dataframe,
        source_variables_file=variables_file,
        target="Class",
        expected_target_values=("SEKER", "BARBUNYA"),
    )

    checks = report.column_frame().set_index("Column")
    assert checks.loc["Area", "Status"] == "Valid"
    assert checks.loc["Perimeter", "Invalid count"] == 1
    assert checks.loc["Perimeter", "Issue types"] == "Non-finite value"
    assert checks.loc["Class", "Missing count"] == 1
    assert checks.loc["Class", "Inconsistent count"] == 1
    assert checks.loc["Class", "Affected count"] == 2

    issues = report.issues_with_impacts_frame()
    assert set(issues["Issue"]) == {
        "Non-finite value",
        "Inconsistent text",
        "Missing value",
    }
    assert issues["Potential analytical impact"].notna().all()


def test_source_backed_analysis_requires_exact_metadata_coverage(tmp_path) -> None:
    dataframe = pd.DataFrame({"Area": [100], "Class": ["SEKER"]})
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Class", "Extra"],
            "type": ["Integer", "Categorical", "Continuous"],
        }
    ).to_csv(variables_file, index=False)

    with pytest.raises(
        ValueValidationError,
        match="metadata contains columns absent from dataset: Extra",
    ):
        analyze_source_backed_missing_and_invalid_values(
            dataframe,
            source_variables_file=variables_file,
            target="Class",
            expected_target_values=("SEKER", "BARBUNYA"),
        )


def test_finite_rule_rejects_infinities_without_reclassifying_missing() -> None:
    dataframe = pd.DataFrame({"value": [1.0, float("inf"), float("-inf"), None]})

    report = analyze_missing_and_invalid_values(
        dataframe,
        {
            "value": {
                "required": True,
                "numeric": True,
                "finite": True,
            }
        },
    )

    checks = report.column_frame().iloc[0]
    assert checks["Missing count"] == 1
    assert checks["Invalid count"] == 3
    assert checks["Affected count"] == 3
    assert checks["Issue types"] == "Missing value, Non-finite value"


def test_finite_rule_requires_numeric_validation() -> None:
    dataframe = pd.DataFrame({"value": [1.0]})

    with pytest.raises(ValueError, match="enables 'finite'.*without enabling 'numeric'"):
        analyze_missing_and_invalid_values(
            dataframe,
            {"value": {"finite": True}},
        )

def test_source_backed_analysis_supports_continuous_regression_target(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Cement": [540.0, 332.5, 198.6],
            "Age": [28, 90, 365],
            "Concrete compressive strength": [79.99, 40.27, float("inf")],
        }
    )
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": [
                "Cement",
                "Age",
                "Concrete compressive strength",
            ],
            "type": ["Continuous", "Integer", "Continuous"],
        }
    ).to_csv(variables_file, index=False)

    report = analyze_source_backed_missing_and_invalid_values(
        dataframe,
        source_variables_file=variables_file,
        target="Concrete compressive strength",
    )

    checks = report.column_frame().set_index("Column")
    assert checks.loc["Cement", "Status"] == "Valid"
    assert checks.loc["Age", "Status"] == "Valid"
    assert checks.loc["Concrete compressive strength", "Invalid count"] == 1
    assert (
        checks.loc["Concrete compressive strength", "Issue types"]
        == "Non-finite value"
    )
    assert report.is_fully_assessed


def test_source_backed_analysis_rejects_empty_target_domain_when_supplied(
    tmp_path,
) -> None:
    dataframe = pd.DataFrame({"Class": ["SEKER"]})
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Class"],
            "type": ["Categorical"],
        }
    ).to_csv(variables_file, index=False)

    with pytest.raises(
        ValueValidationError,
        match="expected_target_values must contain at least one target value",
    ):
        analyze_source_backed_missing_and_invalid_values(
            dataframe,
            source_variables_file=variables_file,
            target="Class",
            expected_target_values=(),
        )

