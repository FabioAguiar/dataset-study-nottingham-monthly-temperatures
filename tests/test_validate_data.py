import pandas as pd
import pytest

from scripts.validate_data import (
    DataValidationError,
    analyze_data_dictionary,
    analyze_data_types,
    analyze_observation_unit,
)


def test_observation_report_for_complete_unique_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
    )

    report = analyze_observation_unit(dataframe, "customer_id")

    assert report.row_count == 3
    assert report.non_null_identifier_count == 3
    assert report.unique_identifier_count == 3
    assert report.missing_identifier_count == 0
    assert report.duplicate_identifier_count == 0
    assert report.duplicated_row_count == 0
    assert report.is_complete
    assert report.is_unique
    report.raise_if_invalid()


def test_observation_report_separates_missing_and_duplicate_counts() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", None],
            "value": [1, 2, 3, 4],
        }
    )

    report = analyze_observation_unit(dataframe, "customer_id")

    assert report.missing_identifier_count == 1
    assert report.duplicate_identifier_count == 1
    assert report.duplicated_row_count == 2
    assert list(report.duplicated_rows["customer_id"]) == ["A", "A"]

    with pytest.raises(DataValidationError, match="missing value"):
        report.raise_if_invalid()


def test_observation_report_does_not_modify_source_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["B", "A", "A"],
            "value": [1, 2, 3],
        }
    )
    original = dataframe.copy(deep=True)

    analyze_observation_unit(dataframe, "customer_id")

    pd.testing.assert_frame_equal(dataframe, original)


def test_observation_report_supports_missing_source_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "class": ["A", "B", "C"],
        }
    )

    report = analyze_observation_unit(dataframe)

    assert report.identifier is None
    assert report.row_count == 3
    assert not report.has_source_identifier
    assert report.non_null_identifier_count is None
    assert report.unique_identifier_count is None
    assert report.missing_identifier_count is None
    assert report.duplicate_identifier_count is None
    assert report.duplicated_row_count is None
    assert report.is_complete is None
    assert report.is_unique is None
    assert report.duplicated_rows.empty

    summary = report.summary_frame()
    assert list(summary["Metric"]) == [
        "Rows",
        "Source identifier",
        "Identifier completeness",
        "Identifier uniqueness",
    ]
    assert list(summary["Value"]) == [
        3,
        "Not provided",
        "Not assessable",
        "Not assessable",
    ]


def test_observation_report_without_identifier_cannot_validate_key_rules() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2]})
    report = analyze_observation_unit(dataframe)

    with pytest.raises(
        DataValidationError,
        match="does not provide an observation identifier",
    ):
        report.raise_if_invalid()


def test_data_type_report_matches_semantic_and_exact_types() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "tenure": pd.Series([1, 2], dtype="int64"),
            "charges": pd.Series([10.5, 20.0], dtype="float64"),
            "active": pd.Series([True, False], dtype="bool"),
        }
    )

    report = analyze_data_types(
        dataframe,
        {
            "customer_id": "string",
            "tenure": ("integer", "Int64"),
            "charges": "numeric",
            "active": "boolean",
        },
    )

    assert report.all_observed_types_match
    assert report.is_fully_declared
    assert not report.has_mismatches
    assert report.mismatched_columns == ()
    assert list(report.column_frame()["Status"]) == [
        "Match",
        "Match",
        "Match",
        "Match",
    ]


def test_data_type_report_highlights_numeric_column_loaded_as_text() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "total_charges": ["10.5", "", "31.0"],
        }
    )

    report = analyze_data_types(
        dataframe,
        {
            "customer_id": "string",
            "total_charges": "numeric",
        },
    )

    assert report.has_mismatches
    assert report.mismatched_columns == ("total_charges",)

    issues = report.issues_frame()
    assert list(issues["Column"]) == ["total_charges"]
    assert issues.iloc[0]["Observed dtype"] == str(
        dataframe["total_charges"].dtype
    )
    assert issues.iloc[0]["Observed type"] == "string"
    assert issues.iloc[0]["Expected type"] == "numeric"
    assert issues.iloc[0]["Status"] == "Mismatch"


def test_data_type_report_tracks_undeclared_and_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "value": [1, 2],
        }
    )

    report = analyze_data_types(
        dataframe,
        {
            "customer_id": "string",
            "missing_expected": "numeric",
        },
    )

    assert report.undeclared_columns == ("value",)
    assert report.missing_expected_columns == ("missing_expected",)
    assert not report.is_fully_declared

    checks = report.column_frame()
    assert list(checks["Column"]) == [
        "customer_id",
        "value",
        "missing_expected",
    ]
    assert list(checks["Status"]) == [
        "Match",
        "Not declared",
        "Missing column",
    ]


def test_data_type_report_raise_if_invalid_can_allow_discovered_mismatches() -> None:
    dataframe = pd.DataFrame(
        {
            "total_charges": ["10.5", "20.0"],
        }
    )

    report = analyze_data_types(
        dataframe,
        {"total_charges": "numeric"},
    )

    report.raise_if_invalid(require_matching_types=False)

    with pytest.raises(
        DataValidationError,
        match="incompatible observed types: total_charges",
    ):
        report.raise_if_invalid()


def test_data_type_report_rejects_invalid_expectation_configuration() -> None:
    dataframe = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(ValueError, match="Unsupported expected type"):
        analyze_data_types(dataframe, {"value": "numerci"})

    with pytest.raises(ValueError, match="cannot be empty"):
        analyze_data_types(dataframe, {"value": []})


def test_data_type_analysis_does_not_modify_source_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "value": [1, 2],
        }
    )
    original = dataframe.copy(deep=True)

    report = analyze_data_types(
        dataframe,
        {
            "customer_id": "string",
            "value": "integer",
        },
    )

    report_checks = report.column_frame()
    report_checks.loc[0, "Status"] = "Changed outside report"

    pd.testing.assert_frame_equal(dataframe, original)
    assert report.column_frame().loc[0, "Status"] == "Match"



def test_data_dictionary_report_documents_roles_and_preserves_order() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "tenure": [1, 2],
            "churn": ["No", "Yes"],
        }
    )

    report = analyze_data_dictionary(
        dataframe,
        {
            "customer_id": {
                "description": "Unique customer identifier.",
                "expected_values": "Unique non-empty string.",
                "unit": None,
            },
            "tenure": {
                "description": "Customer relationship duration.",
                "expected_values": "Non-negative whole number.",
                "unit": "months",
            },
            "churn": {
                "description": "Whether the customer left.",
                "expected_values": ("No", "Yes"),
                "unit": None,
            },
        },
        target="churn",
        identifiers=["customer_id"],
    )

    assert report.is_complete
    assert report.undocumented_columns == ()
    assert report.missing_documented_columns == ()

    checks = report.column_frame()
    assert list(checks["Column"]) == [
        "customer_id",
        "tenure",
        "churn",
    ]
    assert list(checks["Analytical role"]) == [
        "Identifier",
        "Candidate feature",
        "Target",
    ]
    assert list(checks["Expected values"]) == [
        "Unique non-empty string.",
        "Non-negative whole number.",
        "No, Yes",
    ]
    assert list(checks["Status"]) == [
        "Documented",
        "Documented",
        "Documented",
    ]


def test_data_dictionary_report_tracks_coverage_issues() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "value": [1, 2],
            "target": [0, 1],
        }
    )

    report = analyze_data_dictionary(
        dataframe,
        {
            "customer_id": {
                "description": "Unique identifier.",
                "expected_values": "Unique string.",
            },
            "target": {
                "description": "Observed outcome.",
                "expected_values": (0, 1),
            },
            "missing_documented": {
                "description": "Documented but unavailable field.",
                "expected_values": "Any value.",
            },
        },
        target="target",
        identifiers=["customer_id"],
    )

    assert report.undocumented_columns == ("value",)
    assert report.missing_documented_columns == (
        "missing_documented",
    )
    assert not report.is_complete

    issues = report.issues_frame()
    assert list(issues["Column"]) == [
        "value",
        "missing_documented",
    ]
    assert list(issues["Status"]) == [
        "Not documented",
        "Missing column",
    ]

    with pytest.raises(
        DataValidationError,
        match="without data dictionary entries: value",
    ):
        report.raise_if_invalid()


def test_data_dictionary_report_can_validate_selected_requirements() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A"],
            "value": [1],
            "target": [0],
        }
    )

    report = analyze_data_dictionary(
        dataframe,
        {
            "customer_id": {
                "description": "Unique identifier.",
                "expected_values": "Unique string.",
            },
            "target": {
                "description": "Observed outcome.",
                "expected_values": (0, 1),
            },
            "future_column": {
                "description": "Future documented field.",
                "expected_values": "Any value.",
            },
        },
        target="target",
        identifiers=["customer_id"],
    )

    report.raise_if_invalid(
        require_all_columns_documented=False,
        require_documented_columns_present=False,
    )

    with pytest.raises(
        DataValidationError,
        match="missing from the dataset: future_column",
    ):
        report.raise_if_invalid(
            require_all_columns_documented=False,
        )


def test_data_dictionary_rejects_invalid_metadata_and_roles() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A"],
            "target": [0],
        }
    )

    with pytest.raises(ValueError, match="must define 'description'"):
        analyze_data_dictionary(
            dataframe,
            {
                "customer_id": {
                    "expected_values": "Unique string.",
                },
                "target": {
                    "description": "Outcome.",
                    "expected_values": (0, 1),
                },
            },
            target="target",
            identifiers=["customer_id"],
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        analyze_data_dictionary(
            dataframe,
            {
                "customer_id": {
                    "description": "",
                    "expected_values": "Unique string.",
                },
                "target": {
                    "description": "Outcome.",
                    "expected_values": (0, 1),
                },
            },
            target="target",
            identifiers=["customer_id"],
        )

    with pytest.raises(
        ValueError,
        match="target column cannot also be an identifier",
    ):
        analyze_data_dictionary(
            dataframe,
            {},
            target="target",
            identifiers=["target"],
        )

    with pytest.raises(
        KeyError,
        match="Declared analytical role columns were not found",
    ):
        analyze_data_dictionary(
            dataframe,
            {},
            target="missing_target",
            identifiers=["customer_id"],
        )


def test_data_dictionary_analysis_does_not_modify_inputs() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "target": [0, 1],
        }
    )
    original_dataframe = dataframe.copy(deep=True)
    dictionary = {
        "customer_id": {
            "description": "Unique identifier.",
            "expected_values": "Unique string.",
            "unit": None,
        },
        "target": {
            "description": "Observed outcome.",
            "expected_values": (0, 1),
            "unit": None,
        },
    }
    original_dictionary = {
        column: dict(metadata)
        for column, metadata in dictionary.items()
    }

    report = analyze_data_dictionary(
        dataframe,
        dictionary,
        target="target",
        identifiers=["customer_id"],
    )

    external_checks = report.column_frame()
    external_checks.loc[0, "Description"] = "Changed outside report"

    pd.testing.assert_frame_equal(dataframe, original_dataframe)
    assert dictionary == original_dictionary
    assert (
        report.column_frame().loc[0, "Description"]
        == "Unique identifier."
    )
