"""Tests for reusable numerical-feature exploration."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_numerical import (
    NumericalAnalysisError,
    analyze_numerical_features,
)


def test_descriptive_statistics_are_calculated() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Minimum"] == pytest.approx(1.0)
    assert row["Q1"] == pytest.approx(2.0)
    assert row["Median"] == pytest.approx(3.0)
    assert row["Mean"] == pytest.approx(3.0)
    assert row["Q3"] == pytest.approx(4.0)
    assert row["Maximum"] == pytest.approx(5.0)
    assert row["Range"] == pytest.approx(4.0)
    assert row["Standard deviation"] == pytest.approx(1.58113883)
    assert row["Variance"] == pytest.approx(2.5)
    assert row["IQR"] == pytest.approx(2.0)
    assert row["Skewness"] == pytest.approx(0.0)


def test_numeric_text_is_projected_without_mutating_source() -> None:
    dataframe = pd.DataFrame({"amount": [" 1.5 ", "2.5", "3"]})
    before = dataframe.copy(deep=True)

    report = analyze_numerical_features(
        dataframe,
        features=("amount",),
        materialization_rules={
            "amount": {"strip_strings": True},
        },
    )

    assert list(report.numeric_frame()["amount"]) == [1.5, 2.5, 3.0]
    assert report.is_analysis_ready
    pd.testing.assert_frame_equal(dataframe, before)


def test_telco_structural_blank_is_materialized_only_when_tenure_is_zero() -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": [0, 10, 20],
            "TotalCharges": [" ", "500.00", "1200.00"],
        }
    )

    report = analyze_numerical_features(
        dataframe,
        features=("tenure", "TotalCharges"),
        materialization_rules={
            "TotalCharges": {
                "strip_strings": True,
                "blank_replacement": 0.0,
                "condition_column": "tenure",
                "condition_value": 0,
            }
        },
    )

    assert list(report.numeric_frame()["TotalCharges"]) == [
        0.0,
        500.0,
        1200.0,
    ]
    total_row = report.statistics_frame().set_index("Feature").loc[
        "TotalCharges"
    ]
    assert total_row["Blank count"] == 1
    assert total_row["Materialized count"] == 1
    assert total_row["Non-numeric count"] == 0
    assert report.materialized_value_count == 1
    report.raise_if_invalid()


def test_materializes_blank_from_pandas_string_dtype() -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": pd.Series(
                [0, 10, 20],
                dtype="int64",
            ),
            "TotalCharges": pd.Series(
                [
                    " ",
                    "500.00",
                    "1200.00",
                ],
                dtype="string",
            ),
        }
    )
    before = dataframe.copy(deep=True)

    report = analyze_numerical_features(
        dataframe=dataframe,
        features=(
            "tenure",
            "TotalCharges",
        ),
        materialization_rules={
            "TotalCharges": {
                "strip_strings": True,
                "blank_replacement": 0.0,
                "condition_column": "tenure",
                "condition_value": 0,
            },
        },
    )

    assert list(report.numeric_frame()["TotalCharges"]) == [
        0.0,
        500.0,
        1200.0,
    ]
    assert report.materialized_value_count == 1
    pd.testing.assert_frame_equal(dataframe, before)


def test_structural_blank_condition_failure_is_reported() -> None:
    dataframe = pd.DataFrame(
        {
            "tenure": [5],
            "TotalCharges": [" "],
        }
    )

    report = analyze_numerical_features(
        dataframe,
        features=("TotalCharges",),
        materialization_rules={
            "TotalCharges": {
                "strip_strings": True,
                "blank_replacement": 0.0,
                "condition_column": "tenure",
                "condition_value": 0,
            }
        },
    )

    issues = report.conversion_issues_frame()
    assert list(issues["Issue"]) == [
        "Materialization condition not satisfied"
    ]
    assert issues.iloc[0]["Count"] == 1
    assert pd.isna(report.numeric_frame().iloc[0, 0])

    with pytest.raises(
        NumericalAnalysisError,
        match="materialization condition failures found: 1",
    ):
        report.raise_if_invalid()


def test_non_numeric_values_are_reported() -> None:
    dataframe = pd.DataFrame({"amount": ["1", "invalid", "3"]})

    report = analyze_numerical_features(
        dataframe,
        features=("amount",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Valid numeric count"] == 2
    assert row["Non-numeric count"] == 1
    assert report.has_conversion_issues
    assert report.conversion_issues_frame().iloc[0]["Issue"] == (
        "Non-numeric value"
    )

    with pytest.raises(
        NumericalAnalysisError,
        match="numeric conversion issues found: 1",
    ):
        report.raise_if_invalid()


def test_real_missing_and_blank_values_are_counted_separately() -> None:
    dataframe = pd.DataFrame({"value": [1, None, " ", 4]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Missing count"] == 1
    assert row["Blank count"] == 1
    assert row["Valid numeric count"] == 2
    assert report.has_missing_values
    assert report.has_blank_values

    with pytest.raises(
        NumericalAnalysisError,
        match="missing numerical values found: 1",
    ):
        report.raise_if_invalid(
            require_numeric_conversion=False,
            require_materialization_conditions=False,
            require_no_missing_values=True,
        )


def test_zero_negative_and_unique_counts_are_reported() -> None:
    dataframe = pd.DataFrame({"value": [-1, 0, 0, 2]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Negative count"] == 1
    assert row["Zero count"] == 2
    assert row["Unique count"] == 3
    assert report.features_with_zero_values == ("value",)


def test_iqr_outliers_are_identified_and_sampled() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 2, 3, 100]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
        max_outlier_samples=10,
    )

    summary = report.outlier_summary_frame().iloc[0]
    assert summary["Upper outlier count"] == 1
    assert summary["Lower outlier count"] == 0
    assert summary["Outlier count"] == 1
    assert report.has_outliers
    assert report.features_with_outliers == ("value",)

    outlier = report.outliers_frame().iloc[0]
    assert outlier["Direction"] == "Upper"
    assert outlier["Value"] == pytest.approx(100.0)
    assert outlier["Row position"] == 4

    with pytest.raises(
        NumericalAnalysisError,
        match="candidate outliers found: 1",
    ):
        report.raise_if_invalid(require_no_outliers=True)


def test_lower_outlier_is_identified() -> None:
    dataframe = pd.DataFrame({"value": [-100, 1, 2, 2, 3]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    summary = report.outlier_summary_frame().iloc[0]
    assert summary["Lower outlier count"] == 1
    assert report.outliers_frame().iloc[0]["Direction"] == "Lower"


def test_outlier_sample_limit_does_not_change_total_count() -> None:
    dataframe = pd.DataFrame(
        {"value": [0] * 20 + [100, 101, 102]}
    )

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
        max_outlier_samples=2,
    )

    assert report.outlier_summary_frame().iloc[0]["Outlier count"] == 3
    assert len(report.outliers_frame()) == 2


def test_constant_feature_has_zero_variability_and_no_outliers() -> None:
    dataframe = pd.DataFrame({"value": [5, 5, 5]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Standard deviation"] == pytest.approx(0.0)
    assert row["Variance"] == pytest.approx(0.0)
    assert row["IQR"] == pytest.approx(0.0)
    assert row["Skewness"] is None
    assert not report.has_outliers


def test_single_value_feature_has_undefined_sample_variability() -> None:
    dataframe = pd.DataFrame({"value": [7]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Standard deviation"] is None
    assert row["Variance"] is None
    assert row["Skewness"] is None
    assert row["Minimum"] == pytest.approx(7.0)


def test_empty_feature_values_produce_empty_statistics() -> None:
    dataframe = pd.DataFrame({"value": [None, None]})

    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    row = report.statistics_frame().iloc[0]
    assert row["Valid numeric count"] == 0
    assert row["Minimum"] is None
    assert row["Lower fence"] is None
    assert not report.has_outliers


def test_missing_declared_feature_is_reported() -> None:
    dataframe = pd.DataFrame({"value": [1, 2]})

    report = analyze_numerical_features(
        dataframe,
        features=("value", "missing"),
    )

    assert report.available_features == ("value",)
    assert report.missing_features == ("missing",)
    assert report.has_missing_features

    with pytest.raises(
        NumericalAnalysisError,
        match="missing numerical features: missing",
    ):
        report.raise_if_invalid()


def test_feature_order_is_preserved_in_outputs() -> None:
    dataframe = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    report = analyze_numerical_features(
        dataframe,
        features=("c", "a"),
    )

    assert list(report.statistics_frame()["Feature"]) == ["c", "a"]
    assert list(report.numeric_frame().columns) == ["c", "a"]


def test_duplicate_feature_names_are_rejected() -> None:
    dataframe = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError, match="duplicate column names"):
        analyze_numerical_features(
            dataframe,
            features=("value", "value"),
        )


def test_duplicated_dataframe_labels_are_rejected() -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["value", "value"])

    with pytest.raises(
        NumericalAnalysisError,
        match="duplicated column labels",
    ):
        analyze_numerical_features(
            dataframe,
            features=("value",),
        )


def test_materialization_rule_configuration_is_validated() -> None:
    dataframe = pd.DataFrame({"tenure": [0], "amount": [" "]})

    with pytest.raises(
        NumericalAnalysisError,
        match="requires both condition_column and condition_value",
    ):
        analyze_numerical_features(
            dataframe,
            features=("amount",),
            materialization_rules={
                "amount": {"blank_replacement": 0.0},
            },
        )

    with pytest.raises(KeyError, match="condition column not found"):
        analyze_numerical_features(
            dataframe,
            features=("amount",),
            materialization_rules={
                "amount": {
                    "blank_replacement": 0.0,
                    "condition_column": "missing",
                    "condition_value": 0,
                }
            },
        )


def test_frames_are_defensive_copies() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    report = analyze_numerical_features(
        dataframe,
        features=("value",),
    )

    first = report.statistics_frame()
    first.loc[0, "Mean"] = 999
    projected = report.numeric_frame()
    projected.loc[0, "value"] = 999

    assert report.statistics_frame().loc[0, "Mean"] == pytest.approx(2.0)
    assert report.numeric_frame().loc[0, "value"] == pytest.approx(1.0)
    assert list(report.summary_frame().columns) == [
        "Metric",
        "Value",
        "Interpretation",
    ]


def test_exploration_frame_is_compact_and_interprets_skewness() -> None:
    dataframe = pd.DataFrame(
        {
            "symmetric": [1, 2, 3, 4, 5],
            "right_skewed": [1, 1, 1, 1, 10],
        }
    )

    report = analyze_numerical_features(
        dataframe,
        features=("symmetric", "right_skewed"),
    )

    frame = report.exploration_frame(skewness_review_threshold=0.5)

    assert list(frame.columns) == [
        "Feature",
        "Minimum",
        "Q1",
        "Median",
        "Mean",
        "Q3",
        "Maximum",
        "Standard deviation",
        "IQR",
        "Skewness",
        "Distribution shape",
        "Outlier count",
        "Outlier percent",
    ]
    assert frame.loc[0, "Distribution shape"] == "Approximately symmetric"
    assert frame.loc[1, "Distribution shape"] == "Right-skewed"
    assert frame.loc[0, "Outlier percent"].endswith("%")


def test_exploration_frame_can_preserve_numeric_outlier_percent() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 2, 3, 100]})
    report = analyze_numerical_features(dataframe, features=("value",))

    frame = report.exploration_frame(format_percentages=False)

    assert frame.iloc[0]["Outlier percent"] == pytest.approx(0.2)


def test_exploration_frame_validates_skewness_threshold() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    report = analyze_numerical_features(dataframe, features=("value",))

    with pytest.raises(ValueError, match="zero or greater"):
        report.exploration_frame(skewness_review_threshold=-0.1)


def test_distribution_overview_plot_covers_all_features() -> None:
    from matplotlib import pyplot as plt

    from scripts.analyze_numerical import plot_numerical_distributions

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2, 3, 4],
            "c": [3, 4, 5],
        }
    )
    report = analyze_numerical_features(
        dataframe,
        features=("a", "b", "c"),
    )

    figure = plot_numerical_distributions(report, columns=2)

    try:
        visible_axes = [axis for axis in figure.axes if axis.get_visible()]
        assert [axis.get_title() for axis in visible_axes] == ["a", "b", "c"]
    finally:
        plt.close(figure)


def test_boxplot_overview_plot_covers_all_features() -> None:
    from matplotlib import pyplot as plt

    from scripts.analyze_numerical import plot_numerical_boxplots

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2, 3, 100],
        }
    )
    report = analyze_numerical_features(
        dataframe,
        features=("a", "b"),
    )

    figure = plot_numerical_boxplots(report, columns=2)

    try:
        assert [axis.get_title() for axis in figure.axes] == ["a", "b"]
    finally:
        plt.close(figure)
