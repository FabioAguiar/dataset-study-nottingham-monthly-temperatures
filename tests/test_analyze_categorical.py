"""Tests for reusable categorical-feature exploration."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_categorical import (
    CategoricalAnalysisError,
    analyze_categorical_features,
)


def _analyze(
    values: list[object] | pd.Series,
    **kwargs: object,
):
    dataframe = pd.DataFrame({"Category": values})
    return analyze_categorical_features(
        dataframe,
        features=("Category",),
        expected_values={"Category": ("No", "Yes")},
        rare_count_threshold=0,
        rare_share_threshold=0,
        **kwargs,
    )


def test_binary_feature_reports_counts_shares_and_mode() -> None:
    report = _analyze(["No", "No", "No", "Yes"])

    summary = report.feature_summary_frame().iloc[0]
    assert summary["Cardinality"] == 2
    assert summary["Mode"] == "No"
    assert summary["Mode count"] == 3
    assert summary["Mode share"] == pytest.approx(0.75)

    frequencies = report.frequency_frame()
    assert list(frequencies["Category"]) == ["No", "Yes"]
    assert list(frequencies["Count"]) == [3, 1]
    assert list(frequencies["Share"]) == pytest.approx([0.75, 0.25])
    assert list(frequencies["Rank"]) == [1, 2]
    assert list(frequencies["Dominant"]) == [True, False]


def test_multiclass_feature_preserves_deterministic_frequency_order() -> None:
    dataframe = pd.DataFrame(
        {"Plan": ["B", "A", "C", "A", "B", "A"]}
    )

    report = analyze_categorical_features(
        dataframe,
        features=("Plan",),
        expected_values={"Plan": ("A", "B", "C")},
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    frequencies = report.frequency_frame()
    assert list(frequencies["Category"]) == ["A", "B", "C"]
    assert list(frequencies["Count"]) == [3, 2, 1]


def test_integer_categories_are_supported() -> None:
    dataframe = pd.DataFrame({"SeniorCitizen": [0, 1, 0, 0]})

    report = analyze_categorical_features(
        dataframe,
        features=("SeniorCitizen",),
        expected_values={"SeniorCitizen": (0, 1)},
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    frequencies = report.frequency_frame()
    assert list(frequencies["Category"]) == [0, 1]
    assert list(frequencies["Count"]) == [3, 1]
    assert list(frequencies["Expected"]) == [True, True]


def test_dominant_tie_is_reported_without_arbitrary_mode() -> None:
    report = _analyze(["No", "Yes", "No", "Yes"])

    summary = report.feature_summary_frame().iloc[0]
    assert summary["Mode"] == ("No", "Yes")
    assert summary["Mode count"] == 2
    assert summary["Mode share"] == pytest.approx(0.5)
    assert list(report.frequency_frame()["Dominant"]) == [True, True]


def test_rare_category_can_be_triggered_by_count() -> None:
    dataframe = pd.DataFrame({"Plan": ["A"] * 100 + ["B"] * 4})

    report = analyze_categorical_features(
        dataframe,
        features=("Plan",),
        expected_values={"Plan": ("A", "B")},
        rare_count_threshold=5,
        rare_share_threshold=0,
    )

    rare = report.rare_categories_frame()
    assert list(rare["Category"]) == ["B"]
    assert rare.iloc[0]["Trigger"] == "count"
    assert report.has_rare_categories


def test_rare_category_can_be_triggered_by_share() -> None:
    dataframe = pd.DataFrame({"Plan": ["A"] * 199 + ["B"]})

    report = analyze_categorical_features(
        dataframe,
        features=("Plan",),
        expected_values={"Plan": ("A", "B")},
        rare_count_threshold=0,
        rare_share_threshold=0.01,
    )

    rare = report.rare_categories_frame()
    assert list(rare["Category"]) == ["B"]
    assert rare.iloc[0]["Trigger"] == "share"


def test_rare_category_can_trigger_both_rules() -> None:
    dataframe = pd.DataFrame({"Plan": ["A"] * 199 + ["B"]})

    report = analyze_categorical_features(
        dataframe,
        features=("Plan",),
        expected_values={"Plan": ("A", "B")},
        rare_count_threshold=5,
        rare_share_threshold=0.01,
    )

    assert report.rare_categories_frame().iloc[0]["Trigger"] == (
        "count, share"
    )


def test_missing_and_blank_values_are_excluded_from_shares() -> None:
    report = _analyze(["No", None, " ", "Yes"])

    summary = report.feature_summary_frame().iloc[0]
    assert summary["Valid category count"] == 2
    assert summary["Missing count"] == 1
    assert summary["Blank count"] == 1
    assert report.has_missing_values
    assert report.has_blank_values
    assert list(report.frequency_frame()["Share"]) == pytest.approx(
        [0.5, 0.5]
    )


def test_pandas_string_dtype_variants_are_reported() -> None:
    values = pd.Series(
        ["Yes", " yes ", "YES", "No"],
        dtype="string",
    )
    dataframe = pd.DataFrame({"Partner": values})
    original = dataframe.copy(deep=True)

    report = analyze_categorical_features(
        dataframe,
        features=("Partner",),
        expected_values={"Partner": ("No", "Yes")},
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    issues = report.label_issues_frame()
    assert len(issues) == 1
    assert issues.iloc[0]["Normalized category"] == "yes"
    assert issues.iloc[0]["Raw variants"] == (
        "Yes",
        " yes ",
        "YES",
    )
    assert issues.iloc[0]["Rows"] == 3
    assert report.has_label_inconsistencies
    assert not report.has_unexpected_categories
    pd.testing.assert_frame_equal(dataframe, original)


def test_case_variant_alone_is_considered_expected() -> None:
    report = _analyze(["NO", "YES"])

    assert not report.has_unexpected_categories
    assert not report.has_missing_expected_categories
    assert list(report.frequency_frame()["Expected"]) == [True, True]


def test_unexpected_category_is_reported() -> None:
    report = _analyze(["No", "Yes", "Unknown"])

    assert report.has_unexpected_categories
    issues = report.category_contract_issues_frame()
    assert issues.iloc[0]["Issue"] == "Unexpected categories"
    assert issues.iloc[0]["Values"] == ("Unknown",)
    assert list(report.frequency_frame()["Expected"]) == [
        True,
        True,
        False,
    ]

    with pytest.raises(
        CategoricalAnalysisError,
        match="unexpected_categories",
    ):
        report.raise_if_invalid()


def test_missing_expected_category_is_reported() -> None:
    report = _analyze(["No", "No"])

    assert report.has_missing_expected_categories
    issues = report.category_contract_issues_frame()
    assert issues.iloc[0]["Issue"] == "Missing expected categories"
    assert issues.iloc[0]["Values"] == ("Yes",)

    with pytest.raises(
        CategoricalAnalysisError,
        match="missing_expected_categories",
    ):
        report.raise_if_invalid()


def test_feature_without_expected_contract_is_still_analyzed() -> None:
    dataframe = pd.DataFrame({"Category": ["B", "A", "B"]})

    report = analyze_categorical_features(
        dataframe,
        features=("Category",),
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    assert not report.has_unexpected_categories
    assert not report.has_missing_expected_categories
    assert report.feature_summary_frame().iloc[0][
        "Expected cardinality"
    ] is None


def test_constant_feature_is_reported() -> None:
    report = _analyze(["No", "No", "No"])

    assert report.constant_features == ("Category",)


def test_high_cardinality_feature_is_reported() -> None:
    dataframe = pd.DataFrame(
        {"Category": [f"value-{index}" for index in range(5)]}
    )

    report = analyze_categorical_features(
        dataframe,
        features=("Category",),
        rare_count_threshold=0,
        rare_share_threshold=0,
        high_cardinality_threshold=4,
    )

    assert report.has_high_cardinality_features
    assert report.high_cardinality_features == ("Category",)
    with pytest.raises(
        CategoricalAnalysisError,
        match="high_cardinality_features:Category",
    ):
        report.raise_if_invalid(
            require_no_high_cardinality=True,
            require_expected_categories_present=False,
        )


def test_missing_feature_is_reported_in_declared_order() -> None:
    dataframe = pd.DataFrame({"A": ["x"]})

    report = analyze_categorical_features(
        dataframe,
        features=("MissingOne", "A", "MissingTwo"),
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    assert report.available_features == ("A",)
    assert report.missing_features == ("MissingOne", "MissingTwo")
    with pytest.raises(
        CategoricalAnalysisError,
        match="missing_features",
    ):
        report.raise_if_invalid(
            require_expected_categories_present=False,
        )


def test_duplicate_requested_features_are_rejected() -> None:
    dataframe = pd.DataFrame({"A": ["x"]})

    with pytest.raises(
        CategoricalAnalysisError,
        match="must be unique",
    ):
        analyze_categorical_features(
            dataframe,
            features=("A", "A"),
        )


def test_duplicated_dataframe_labels_are_rejected() -> None:
    dataframe = pd.DataFrame(
        [["x", "y"]],
        columns=["A", "A"],
    )

    with pytest.raises(
        CategoricalAnalysisError,
        match="duplicated column labels",
    ):
        analyze_categorical_features(
            dataframe,
            features=("A",),
        )


def test_expected_values_must_be_unique_after_normalization() -> None:
    dataframe = pd.DataFrame({"A": ["Yes"]})

    with pytest.raises(
        CategoricalAnalysisError,
        match="duplicate normalized category",
    ):
        analyze_categorical_features(
            dataframe,
            features=("A",),
            expected_values={"A": ("Yes", " yes ")},
        )


def test_expected_values_cannot_reference_undeclared_feature() -> None:
    dataframe = pd.DataFrame({"A": ["x"]})

    with pytest.raises(
        CategoricalAnalysisError,
        match="undeclared features",
    ):
        analyze_categorical_features(
            dataframe,
            features=("A",),
            expected_values={"B": ("x",)},
        )


def test_complete_grouping_reports_group_frequencies() -> None:
    dataframe = pd.DataFrame(
        {
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ]
        }
    )
    expected = (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    )

    report = analyze_categorical_features(
        dataframe,
        features=("PaymentMethod",),
        expected_values={"PaymentMethod": expected},
        category_groupings={
            "PaymentMethod": {
                "Automatic": (
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ),
                "Manual": (
                    "Electronic check",
                    "Mailed check",
                ),
            }
        },
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    grouping = report.grouping_frame()
    assert list(grouping["Group"]) == ["Automatic", "Manual"]
    assert list(grouping["Count"]) == [2, 3]
    assert list(grouping["Share"]) == pytest.approx([0.4, 0.6])
    assert not report.has_grouping_issues


def test_unknown_grouping_category_is_reported() -> None:
    report = analyze_categorical_features(
        pd.DataFrame({"Category": ["No", "Yes"]}),
        features=("Category",),
        expected_values={"Category": ("No", "Yes")},
        category_groupings={
            "Category": {
                "Known": ("No", "Yes", "Unknown"),
            }
        },
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    issues = report.grouping_issues_frame()
    assert "Unknown grouping categories" in set(issues["Issue"])
    with pytest.raises(
        CategoricalAnalysisError,
        match="invalid_groupings",
    ):
        report.raise_if_invalid()


def test_category_assigned_to_multiple_groups_is_reported() -> None:
    report = analyze_categorical_features(
        pd.DataFrame({"Category": ["No", "Yes"]}),
        features=("Category",),
        expected_values={"Category": ("No", "Yes")},
        category_groupings={
            "Category": {
                "First": ("No", "Yes"),
                "Second": ("Yes",),
            }
        },
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    issues = report.grouping_issues_frame()
    assert "Categories assigned to multiple groups" in set(
        issues["Issue"]
    )
    grouping = report.grouping_frame()
    unassigned = grouping.loc[grouping["Group"] == "Unassigned"]
    assert unassigned.iloc[0]["Count"] == 1


def test_unassigned_expected_category_is_reported() -> None:
    report = analyze_categorical_features(
        pd.DataFrame({"Category": ["No", "Yes"]}),
        features=("Category",),
        expected_values={"Category": ("No", "Yes")},
        category_groupings={
            "Category": {
                "Negative": ("No",),
            }
        },
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    issues = report.grouping_issues_frame()
    row = issues.loc[issues["Issue"] == "Ungrouped categories"].iloc[0]
    assert row["Values"] == ("Yes",)


def test_raw_projection_and_source_dataframe_are_not_mutated() -> None:
    dataframe = pd.DataFrame(
        {
            "A": pd.Series(["Yes", "No"], dtype="string"),
            "B": [0, 1],
        }
    )
    original = dataframe.copy(deep=True)

    report = analyze_categorical_features(
        dataframe,
        features=("A", "B"),
        expected_values={
            "A": ("No", "Yes"),
            "B": (0, 1),
        },
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    projection = report.categorical_frame()
    projection.loc[0, "A"] = "Changed"

    pd.testing.assert_frame_equal(dataframe, original)
    assert report.categorical_frame().loc[0, "A"] == "Yes"


def test_feature_order_is_preserved_in_all_summary_outputs() -> None:
    dataframe = pd.DataFrame(
        {
            "B": ["x", "y"],
            "A": ["m", "n"],
        }
    )

    report = analyze_categorical_features(
        dataframe,
        features=("A", "B"),
        rare_count_threshold=0,
        rare_share_threshold=0,
    )

    assert list(report.feature_summary_frame()["Feature"]) == ["A", "B"]
    assert list(dict.fromkeys(report.frequency_frame()["Feature"])) == [
        "A",
        "B",
    ]


def test_validation_flags_can_allow_observed_issues() -> None:
    report = _analyze(["No", None, "Unknown"])

    report.raise_if_invalid(
        require_no_unexpected_categories=False,
        require_expected_categories_present=False,
        require_no_inconsistent_labels=False,
        require_no_missing_values=False,
    )


def test_invalid_thresholds_are_rejected() -> None:
    dataframe = pd.DataFrame({"A": ["x"]})

    with pytest.raises(ValueError, match="rare_count_threshold"):
        analyze_categorical_features(
            dataframe,
            features=("A",),
            rare_count_threshold=-1,
        )

    with pytest.raises(ValueError, match="rare_share_threshold"):
        analyze_categorical_features(
            dataframe,
            features=("A",),
            rare_share_threshold=1.1,
        )

    with pytest.raises(ValueError, match="high_cardinality_threshold"):
        analyze_categorical_features(
            dataframe,
            features=("A",),
            high_cardinality_threshold=0,
        )
