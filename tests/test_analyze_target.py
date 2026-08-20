"""Tests for reusable categorical target-distribution analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_target import (
    TargetAnalysisError,
    analyze_target_distribution,
)


def _analyze(
    values: list[object],
    **kwargs: object,
):
    dataframe = pd.DataFrame({"Churn": values})
    return analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
        **kwargs,
    )


def test_balanced_binary_target_reports_tie() -> None:
    report = _analyze(["No", "Yes", "No", "Yes"])

    assert report.row_count == 4
    assert report.non_missing_count == 4
    assert report.missing_count == 0
    assert report.class_count == 2
    assert report.majority_classes == ("No", "Yes")
    assert report.minority_classes == ("No", "Yes")
    assert report.majority_class is None
    assert report.minority_class is None
    assert report.has_majority_tie
    assert report.has_minority_tie
    assert report.imbalance_ratio == pytest.approx(1.0)
    assert report.majority_baseline_accuracy == pytest.approx(0.5)
    assert report.positive_class_share == pytest.approx(0.5)

    distribution = report.distribution_frame()
    assert list(distribution["Class"]) == ["No", "Yes"]
    assert list(distribution["Count"]) == [2, 2]
    assert list(distribution["Role"]) == [
        "Tied / Negative",
        "Tied / Positive",
    ]


def test_imbalanced_binary_target_reports_majority_and_minority() -> None:
    report = _analyze(["No", "No", "No", "Yes"])

    assert report.majority_class == "No"
    assert report.minority_class == "Yes"
    assert report.majority_share == pytest.approx(0.75)
    assert report.minority_share == pytest.approx(0.25)
    assert report.imbalance_ratio == pytest.approx(3.0)
    assert report.positive_class_count == 1
    assert report.positive_class_share == pytest.approx(0.25)
    assert report.majority_baseline_accuracy == pytest.approx(0.75)

    distribution = report.distribution_frame()
    assert list(distribution["Percentage"]) == [0.75, 0.25]
    assert list(distribution["Role"]) == [
        "Majority / Negative",
        "Minority / Positive",
    ]


def test_multiclass_target_supports_intermediate_class() -> None:
    dataframe = pd.DataFrame(
        {"Outcome": ["A", "A", "A", "B", "B", "C"]}
    )

    report = analyze_target_distribution(
        dataframe,
        target="Outcome",
        expected_classes=("A", "B", "C"),
        positive_class="C",
    )

    assert report.class_count == 3
    assert report.majority_class == "A"
    assert report.minority_class == "C"
    assert report.imbalance_ratio == pytest.approx(3.0)
    assert list(report.distribution_frame()["Role"]) == [
        "Majority / Negative",
        "Intermediate / Negative",
        "Minority / Positive",
    ]


def test_expected_class_order_is_preserved() -> None:
    report = _analyze(["Yes", "No", "Yes"])

    assert list(report.distribution_frame()["Class"]) == ["No", "Yes"]


def test_observed_order_is_preserved_without_expected_classes() -> None:
    dataframe = pd.DataFrame({"Outcome": ["B", "A", "C", "A"]})

    report = analyze_target_distribution(
        dataframe,
        target="Outcome",
    )

    assert list(report.distribution_frame()["Class"]) == ["B", "A", "C"]
    assert list(report.distribution_frame()["Count"]) == [1, 2, 1]


def test_missing_target_values_are_reported_and_excluded_from_shares() -> None:
    report = _analyze(["No", None, "Yes", pd.NA])

    assert report.row_count == 4
    assert report.non_missing_count == 2
    assert report.missing_count == 2
    assert report.has_missing_values
    assert report.positive_class_share == pytest.approx(0.5)

    issues = report.issues_frame()
    assert list(issues["Issue"]) == ["Missing target values"]
    assert issues.iloc[0]["Count"] == 2

    with pytest.raises(
        TargetAnalysisError,
        match="missing_target_values:2",
    ):
        report.raise_if_invalid()


def test_missing_expected_class_is_reported_with_zero_count() -> None:
    report = _analyze(["No", "No"])

    assert report.has_missing_expected_classes
    assert report.missing_expected_classes == ("Yes",)
    assert report.positive_class_count == 0
    assert report.positive_class_share == pytest.approx(0.0)

    distribution = report.distribution_frame()
    yes_row = distribution.loc[distribution["Class"] == "Yes"].iloc[0]
    assert yes_row["Count"] == 0
    assert yes_row["Role"] == "Absent expected / Positive"

    with pytest.raises(
        TargetAnalysisError,
        match="missing_expected_classes:'Yes'",
    ):
        report.raise_if_invalid()


def test_unexpected_class_is_appended_and_reported() -> None:
    report = _analyze(["No", "Yes", "Unknown"])

    assert report.has_unexpected_classes
    assert report.unexpected_classes == ("Unknown",)
    assert list(report.distribution_frame()["Class"]) == [
        "No",
        "Yes",
        "Unknown",
    ]

    issues = report.issues_frame()
    assert issues.iloc[0]["Issue"] == "Unexpected classes"
    assert issues.iloc[0]["Count"] == 1

    with pytest.raises(
        TargetAnalysisError,
        match="unexpected_classes:'Unknown'",
    ):
        report.raise_if_invalid()


def test_validation_flags_can_allow_observed_issues() -> None:
    report = _analyze(["No", None, "Unknown"])

    report.raise_if_invalid(
        require_no_missing_target=False,
        require_expected_classes_present=False,
        require_no_unexpected_classes=False,
    )


def test_empty_dataframe_produces_an_empty_observed_distribution() -> None:
    dataframe = pd.DataFrame({"Churn": pd.Series(dtype="object")})

    report = analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
    )

    assert report.row_count == 0
    assert report.non_missing_count == 0
    assert report.class_count == 0
    assert report.majority_class is None
    assert report.minority_class is None
    assert report.majority_share is None
    assert report.minority_share is None
    assert report.imbalance_ratio is None
    assert report.positive_class_share is None
    assert list(report.distribution_frame()["Count"]) == [0, 0]
    assert report.missing_expected_classes == ("No", "Yes")


def test_target_column_must_exist() -> None:
    dataframe = pd.DataFrame({"Outcome": ["No"]})

    with pytest.raises(KeyError, match="Target column not found"):
        analyze_target_distribution(
            dataframe,
            target="Churn",
        )


def test_duplicated_column_labels_are_rejected() -> None:
    dataframe = pd.DataFrame(
        [["No", "Yes"]],
        columns=["Churn", "Churn"],
    )

    with pytest.raises(
        TargetAnalysisError,
        match="duplicated column labels",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
        )


def test_expected_classes_must_be_unique_and_non_missing() -> None:
    dataframe = pd.DataFrame({"Churn": ["No", "Yes"]})

    with pytest.raises(
        TargetAnalysisError,
        match="duplicate value",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", "No"),
        )

    with pytest.raises(
        TargetAnalysisError,
        match="missing values",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", None),
        )


def test_positive_class_must_belong_to_expected_classes() -> None:
    dataframe = pd.DataFrame({"Churn": ["No", "Yes"]})

    with pytest.raises(
        TargetAnalysisError,
        match="positive_class must be included",
    ):
        analyze_target_distribution(
            dataframe,
            target="Churn",
            expected_classes=("No", "Yes"),
            positive_class="Maybe",
        )


def test_summary_and_distribution_frames_are_defensive_copies() -> None:
    report = _analyze(["No", "No", "Yes"])

    first = report.distribution_frame()
    first.loc[0, "Count"] = 999

    assert report.distribution_frame().loc[0, "Count"] == 2
    assert list(report.summary_frame().columns) == [
        "Metric",
        "Value",
        "Interpretation",
    ]


def test_analysis_does_not_modify_the_source_dataframe() -> None:
    dataframe = pd.DataFrame(
        {"Churn": ["No", "Yes", "No"]},
        index=[10, 20, 30],
    )
    before = dataframe.copy(deep=True)

    analyze_target_distribution(
        dataframe,
        target="Churn",
        expected_classes=("No", "Yes"),
        positive_class="Yes",
    )

    pd.testing.assert_frame_equal(dataframe, before)



def test_multiclass_target_without_positive_class_uses_neutral_roles() -> None:
    dataframe = pd.DataFrame(
        {"Class": ["A", "A", "A", "B", "B", "C"]}
    )

    report = analyze_target_distribution(
        dataframe,
        target="Class",
        expected_classes=("A", "B", "C"),
    )

    assert report.positive_class is None
    assert report.positive_class_share is None
    assert list(report.distribution_frame()["Role"]) == [
        "Majority",
        "Intermediate",
        "Minority",
    ]
    assert "Positive class" not in set(report.summary_frame()["Metric"])
    assert "Positive-class prevalence" not in set(
        report.summary_frame()["Metric"]
    )


def test_normalized_class_entropy_is_one_for_equal_class_shares() -> None:
    dataframe = pd.DataFrame(
        {"Class": ["A", "B", "C", "A", "B", "C"]}
    )
    report = analyze_target_distribution(
        dataframe,
        target="Class",
        expected_classes=("A", "B", "C"),
    )

    assert report.normalized_class_entropy == pytest.approx(1.0)


def test_normalized_class_entropy_reflects_multiclass_imbalance() -> None:
    dataframe = pd.DataFrame(
        {"Class": ["A"] * 8 + ["B"] * 3 + ["C"]}
    )
    report = analyze_target_distribution(
        dataframe,
        target="Class",
        expected_classes=("A", "B", "C"),
    )

    assert report.normalized_class_entropy is not None
    assert 0.0 < report.normalized_class_entropy < 1.0


def test_distribution_frame_can_format_percentages_for_display() -> None:
    dataframe = pd.DataFrame({"Class": ["A", "A", "B", "C"]})
    report = analyze_target_distribution(
        dataframe,
        target="Class",
        expected_classes=("A", "B", "C"),
    )

    formatted = report.distribution_frame(format_percentages=True)

    assert list(formatted["Percentage"]) == [
        "50.00%",
        "25.00%",
        "25.00%",
    ]
    assert report.distribution_frame().iloc[0]["Percentage"] == pytest.approx(
        0.5
    )


def test_target_distribution_plot_uses_report_classes() -> None:
    from matplotlib import pyplot as plt

    from scripts.analyze_target import plot_target_distribution

    dataframe = pd.DataFrame({"Class": ["A", "A", "B", "C"]})
    report = analyze_target_distribution(
        dataframe,
        target="Class",
        expected_classes=("A", "B", "C"),
    )

    figure = plot_target_distribution(
        report,
        title="Example target distribution",
    )

    try:
        axis = figure.axes[0]
        assert axis.get_title() == "Example target distribution"
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "A",
            "B",
            "C",
        ]
    finally:
        plt.close(figure)


def test_continuous_target_reports_distribution_range_and_extremes() -> None:
    from scripts.analyze_target import analyze_continuous_target_distribution

    dataframe = pd.DataFrame(
        {"Strength": [10.0, 12.0, 14.0, 16.0, 18.0, 100.0]}
    )
    report = analyze_continuous_target_distribution(
        dataframe,
        target="Strength",
        unit="MPa",
    )

    assert report.row_count == 6
    assert report.finite_count == 6
    assert report.missing_count == 0
    assert report.non_finite_count == 0
    assert report.minimum == pytest.approx(10.0)
    assert report.maximum == pytest.approx(100.0)
    assert report.observed_range == pytest.approx(90.0)
    assert report.median == pytest.approx(15.0)
    assert report.iqr is not None
    assert report.upper_extreme_count == 1
    assert report.lower_extreme_count == 0
    assert report.extreme_count == 1
    assert report.extreme_share == pytest.approx(1 / 6)


def test_continuous_target_quantiles_and_summary_are_deterministic() -> None:
    from scripts.analyze_target import analyze_continuous_target_distribution

    dataframe = pd.DataFrame({"Strength": [1.0, 2.0, 3.0, 4.0, 5.0]})
    report = analyze_continuous_target_distribution(
        dataframe,
        target="Strength",
        unit="MPa",
    )

    summary = report.summary_frame()
    quantiles = report.quantiles_frame()
    extremes = report.extremes_frame()

    assert list(summary.columns) == ["Metric", "Value", "Interpretation"]
    assert list(quantiles["Quantile"]) == [
        "1%", "5%", "25%", "50%", "75%", "95%", "99%"
    ]
    assert set(quantiles["Unit"]) == {"MPa"}
    assert list(extremes["Side"]) == ["Lower", "Upper"]


def test_continuous_target_reports_missing_and_non_finite_values() -> None:
    from scripts.analyze_target import (
        TargetAnalysisError,
        analyze_continuous_target_distribution,
    )

    dataframe = pd.DataFrame(
        {"Strength": [10.0, None, float("inf"), 20.0]}
    )
    report = analyze_continuous_target_distribution(
        dataframe,
        target="Strength",
    )

    assert report.missing_count == 1
    assert report.non_finite_count == 1
    assert report.finite_count == 2
    assert list(report.issues_frame()["Issue"]) == [
        "Missing target values",
        "Non-finite target values",
    ]

    with pytest.raises(
        TargetAnalysisError,
        match="missing_target_values:1; non_finite_target_values:1",
    ):
        report.raise_if_invalid()


def test_continuous_target_rejects_non_numeric_values() -> None:
    from scripts.analyze_target import (
        TargetAnalysisError,
        analyze_continuous_target_distribution,
    )

    dataframe = pd.DataFrame({"Strength": [10.0, "bad", 20.0]})

    with pytest.raises(
        TargetAnalysisError,
        match="non-numeric non-missing values: 1",
    ):
        analyze_continuous_target_distribution(
            dataframe,
            target="Strength",
        )


def test_continuous_target_constant_values_are_invalid_by_default() -> None:
    from scripts.analyze_target import (
        TargetAnalysisError,
        analyze_continuous_target_distribution,
    )

    report = analyze_continuous_target_distribution(
        pd.DataFrame({"Strength": [5.0, 5.0, 5.0]}),
        target="Strength",
    )

    assert not report.has_variation
    with pytest.raises(TargetAnalysisError, match="constant_target"):
        report.raise_if_invalid()


def test_continuous_target_analysis_does_not_modify_dataframe() -> None:
    from scripts.analyze_target import analyze_continuous_target_distribution

    dataframe = pd.DataFrame(
        {"Strength": [10.0, 20.0, 30.0]},
        index=[3, 6, 9],
    )
    before = dataframe.copy(deep=True)

    analyze_continuous_target_distribution(
        dataframe,
        target="Strength",
        unit="MPa",
    )

    pd.testing.assert_frame_equal(dataframe, before)


def test_continuous_target_plot_uses_original_scale_label() -> None:
    from matplotlib import pyplot as plt

    from scripts.analyze_target import (
        analyze_continuous_target_distribution,
        plot_continuous_target_distribution,
    )

    report = analyze_continuous_target_distribution(
        pd.DataFrame({"Strength": [10.0, 20.0, 30.0, 40.0]}),
        target="Strength",
        unit="MPa",
    )
    figure = plot_continuous_target_distribution(
        report,
        title="Concrete target distribution",
        bins=4,
    )

    try:
        axis = figure.axes[0]
        assert axis.get_title() == "Concrete target distribution"
        assert axis.get_xlabel() == "Strength (MPa)"
        assert axis.get_ylabel() == "Observation count"
    finally:
        plt.close(figure)
