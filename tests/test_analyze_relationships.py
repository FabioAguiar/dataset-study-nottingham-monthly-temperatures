"""Tests for reusable feature-to-target relationship analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_target_relationships import (
    FeatureTargetAnalysisError,
    analyze_feature_target_relationships,
)


def _analyze(
    numerical: pd.DataFrame | None = None,
    categorical: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    **kwargs: object,
):
    numerical_frame = (
        numerical
        if numerical is not None
        else pd.DataFrame({"amount": [1.0, 2.0, 8.0, 9.0]})
    )
    categorical_frame = (
        categorical
        if categorical is not None
        else pd.DataFrame({"group": ["A", "A", "B", "B"]})
    )
    target_series = (
        target
        if target is not None
        else pd.Series(["No", "No", "Yes", "Yes"], name="Churn")
    )
    options = {
        "numerical_bin_count": 2,
        "minimum_group_count": 2,
    }
    options.update(kwargs)
    return analyze_feature_target_relationships(
        numerical_frame=numerical_frame,
        categorical_frame=categorical_frame,
        target=target_series,
        numerical_features=tuple(numerical_frame.columns),
        categorical_features=tuple(categorical_frame.columns),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        **options,
    )


def test_valid_analysis_produces_all_report_tables() -> None:
    report = _analyze()

    assert report.is_analysis_ready
    assert len(report.numerical_relationships_frame()) == 1
    assert len(report.numerical_class_statistics_frame()) == 2
    assert len(report.numerical_bins_frame()) == 2
    assert len(report.categorical_relationships_frame()) == 1
    assert len(report.categorical_rates_frame()) == 2
    assert report.issues_frame().empty


def test_positive_numerical_separation_preserves_metric_direction() -> None:
    report = _analyze(
        numerical=pd.DataFrame(
            {"amount": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]}
        ),
        categorical=pd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"]}
        ),
        target=pd.Series(["No", "No", "No", "Yes", "Yes", "Yes"]),
    )

    row = report.numerical_relationships_frame().iloc[0]
    assert row["Mean difference"] == pytest.approx(7.0)
    assert row["Point-biserial correlation"] > 0
    assert row["Cohen's d"] > 0
    assert row["Eta squared"] > 0
    assert bool(row["Review flag"])


def test_negative_numerical_separation_preserves_metric_direction() -> None:
    report = _analyze(
        numerical=pd.DataFrame({"amount": [8.0, 9.0, 10.0, 1.0, 2.0, 3.0]}),
        categorical=pd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"]}
        ),
        target=pd.Series(["No", "No", "No", "Yes", "Yes", "Yes"]),
    )

    row = report.numerical_relationships_frame().iloc[0]
    assert row["Mean difference"] == pytest.approx(-7.0)
    assert row["Point-biserial correlation"] < 0
    assert row["Cohen's d"] < 0


def test_no_numerical_difference_reports_limited_separation() -> None:
    report = _analyze(
        numerical=pd.DataFrame({"amount": [1.0, 2.0, 1.0, 2.0]}),
    )

    row = report.numerical_relationships_frame().iloc[0]
    assert row["Mean difference"] == pytest.approx(0.0)
    assert row["Point-biserial correlation"] == pytest.approx(0.0)
    assert row["Cohen's d"] == pytest.approx(0.0)
    assert not bool(row["Review flag"])


def test_class_statistics_preserve_expected_target_order() -> None:
    report = _analyze()
    statistics = report.numerical_class_statistics_frame()

    assert list(statistics["Target class"]) == ["No", "Yes"]
    assert list(statistics["Mean"]) == [1.5, 8.5]


def test_numerical_missing_values_are_excluded_pairwise() -> None:
    report = _analyze(
        numerical=pd.DataFrame({"amount": [1.0, None, 8.0, 9.0]}),
    )

    row = report.numerical_relationships_frame().iloc[0]
    assert row["Valid paired rows"] == 3
    assert row["Missing paired rows"] == 1
    statistics = report.numerical_class_statistics_frame()
    assert list(statistics["Valid numeric count"]) == [1, 2]


def test_constant_numerical_feature_is_reported() -> None:
    report = _analyze(
        numerical=pd.DataFrame({"amount": [1.0, 1.0, 1.0, 1.0]}),
    )

    assert report.has_constant_features
    row = report.numerical_relationships_frame().iloc[0]
    assert row["Point-biserial correlation"] is None
    assert row["Cohen's d"] is None
    assert report.numerical_bins_frame().empty


def test_quantile_bins_report_rates_lift_and_support() -> None:
    numerical = pd.DataFrame({"amount": list(range(1, 9))})
    categorical = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4})
    target = pd.Series(["No"] * 4 + ["Yes"] * 4)
    report = analyze_feature_target_relationships(
        numerical,
        categorical,
        target,
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        numerical_bin_count=4,
        minimum_group_count=3,
    )

    bins = report.numerical_bins_frame()
    assert list(bins["Bin"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(bins["Positive-class rate"]) == [0.0, 0.0, 1.0, 1.0]
    assert list(bins["Lift"]) == [0.0, 0.0, 2.0, 2.0]
    assert bins["Low-support flag"].all()


def test_quantile_bins_handle_duplicated_edges() -> None:
    report = _analyze(
        numerical=pd.DataFrame({"amount": [0, 0, 0, 1, 1, 1]}),
        categorical=pd.DataFrame({"group": ["A"] * 3 + ["B"] * 3}),
        target=pd.Series(["No", "No", "Yes", "No", "Yes", "Yes"]),
    )

    bins = report.numerical_bins_frame()
    assert 1 <= len(bins) <= 2
    assert bins["Row count"].sum() == 6


def test_categorical_perfect_association_reports_high_values() -> None:
    report = _analyze()
    row = report.categorical_relationships_frame().iloc[0]

    assert row["Cramer's V"] == pytest.approx(1.0)
    assert row["U(Target | Feature)"] == pytest.approx(1.0)
    assert row["Positive-class rate spread"] == pytest.approx(1.0)
    assert bool(row["Review flag"])


def test_categorical_independence_reports_zero_association() -> None:
    report = _analyze(
        categorical=pd.DataFrame({"group": ["A", "B", "A", "B"]}),
    )
    row = report.categorical_relationships_frame().iloc[0]

    assert row["Cramer's V"] == pytest.approx(0.0)
    assert row["U(Target | Feature)"] == pytest.approx(0.0)
    assert row["Positive-class rate spread"] == pytest.approx(0.0)
    assert not bool(row["Review flag"])


def test_categorical_rates_include_rate_difference_and_lift() -> None:
    rates = _analyze().categorical_rates_frame()
    a = rates.loc[rates["Category"].eq("A")].iloc[0]
    b = rates.loc[rates["Category"].eq("B")].iloc[0]

    assert a["Positive-class rate"] == pytest.approx(0.0)
    assert a["Rate difference"] == pytest.approx(-0.5)
    assert a["Lift"] == pytest.approx(0.0)
    assert b["Positive-class rate"] == pytest.approx(1.0)
    assert b["Rate difference"] == pytest.approx(0.5)
    assert b["Lift"] == pytest.approx(2.0)


def test_odds_ratio_uses_zero_cell_correction() -> None:
    rates = _analyze().categorical_rates_frame()

    assert all(
        value is not None and value > 0
        for value in rates["Odds ratio versus remaining categories"]
    )


def test_wilson_intervals_are_bounded_and_contain_observed_rate() -> None:
    rates = _analyze().categorical_rates_frame()

    for _, row in rates.iterrows():
        assert 0 <= row["Wilson interval lower"] <= 1
        assert 0 <= row["Wilson interval upper"] <= 1
        assert (
            row["Wilson interval lower"]
            <= row["Positive-class rate"]
            <= row["Wilson interval upper"]
        )


def test_expected_category_order_and_absent_category_are_preserved() -> None:
    report = analyze_feature_target_relationships(
        pd.DataFrame({"amount": [1, 2, 3, 4]}),
        pd.DataFrame({"group": ["B", "A", "B", "A"]}),
        pd.Series(["No", "No", "Yes", "Yes"]),
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        expected_category_values={"group": ("A", "B", "C")},
        numerical_bin_count=2,
        minimum_group_count=1,
    )

    rates = report.categorical_rates_frame()
    assert list(rates["Category"]) == ["A", "B", "C"]
    assert list(rates["Expected category"]) == [True, True, True]
    assert rates.iloc[2]["Row count"] == 0
    assert pd.isna(rates.iloc[2]["Positive-class rate"])


def test_unexpected_observed_category_is_appended_after_expected_values() -> None:
    report = analyze_feature_target_relationships(
        pd.DataFrame({"amount": [1, 2, 3, 4]}),
        pd.DataFrame({"group": ["A", "Other", "A", "Other"]}),
        pd.Series(["No", "No", "Yes", "Yes"]),
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        expected_category_values={"group": ("A", "B")},
        numerical_bin_count=2,
        minimum_group_count=1,
    )

    rates = report.categorical_rates_frame()
    assert list(rates["Category"]) == ["A", "B", "Other"]
    assert list(rates["Expected category"]) == [True, True, False]


def test_integer_category_is_supported() -> None:
    report = _analyze(
        categorical=pd.DataFrame(
            {"SeniorCitizen": pd.Series([0, 0, 1, 1], dtype="int64")}
        ),
    )

    rates = report.categorical_rates_frame()
    assert list(rates["Category"]) == [0, 1]
    assert list(rates["Positive-class rate"]) == [0.0, 1.0]


def test_pandas_string_dtype_and_blanks_are_supported() -> None:
    report = _analyze(
        categorical=pd.DataFrame(
            {
                "group": pd.Series(
                    [" A ", "", "B", "B"],
                    dtype="string",
                )
            }
        ),
    )

    row = report.categorical_relationships_frame().iloc[0]
    assert row["Valid paired rows"] == 3
    assert row["Missing paired rows"] == 1
    assert list(report.categorical_rates_frame()["Category"]) == ["A", "B"]


def test_constant_categorical_feature_is_reported() -> None:
    report = _analyze(
        categorical=pd.DataFrame({"group": ["A", "A", "A", "A"]}),
    )

    assert report.has_constant_features
    row = report.categorical_relationships_frame().iloc[0]
    assert row["Cramer's V"] is None
    assert row["U(Target | Feature)"] is None


def test_low_support_categories_are_reported() -> None:
    report = analyze_feature_target_relationships(
        pd.DataFrame({"amount": [1, 2, 3, 4, 5]}),
        pd.DataFrame({"group": ["A", "A", "A", "A", "B"]}),
        pd.Series(["No", "No", "Yes", "Yes", "Yes"]),
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        numerical_bin_count=2,
        minimum_group_count=2,
    )

    assert report.has_low_support_groups
    relationship = report.categorical_relationships_frame().iloc[0]
    assert relationship["Low-support category count"] == 1


def test_missing_target_values_block_analysis_and_are_reported() -> None:
    report = _analyze(
        target=pd.Series(["No", None, "Yes", "Yes"]),
    )

    assert report.has_missing_target_values
    assert not report.is_analysis_ready
    assert report.numerical_relationships_frame().empty
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="missing_target_values:1",
    ):
        report.raise_if_invalid()


def test_unexpected_target_class_blocks_analysis() -> None:
    report = _analyze(
        target=pd.Series(["No", "No", "Yes", "Unknown"]),
    )

    assert report.has_unexpected_target_classes
    assert report.unexpected_target_classes == ("Unknown",)
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="unexpected_target_classes:'Unknown'",
    ):
        report.raise_if_invalid()


def test_missing_expected_target_class_is_reported() -> None:
    report = _analyze(
        target=pd.Series(["No", "No", "No", "No"]),
    )

    assert report.has_missing_expected_target_classes
    assert report.missing_expected_target_classes == ("Yes",)
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="positive_class_not_observed",
    ):
        report.raise_if_invalid()


def test_non_binary_target_contract_is_rejected_by_validation() -> None:
    report = analyze_feature_target_relationships(
        pd.DataFrame({"amount": [1, 2, 3]}),
        pd.DataFrame({"group": ["A", "B", "C"]}),
        pd.Series(["A", "B", "C"]),
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("A", "B", "C"),
        positive_class="C",
        numerical_bin_count=2,
        minimum_group_count=1,
    )

    with pytest.raises(
        FeatureTargetAnalysisError,
        match="target_contract_is_not_binary",
    ):
        report.raise_if_invalid()


def test_indices_must_align_even_when_lengths_match() -> None:
    report = _analyze(
        numerical=pd.DataFrame(
            {"amount": [1.0, 2.0, 8.0, 9.0]},
            index=[0, 1, 2, 3],
        ),
        categorical=pd.DataFrame(
            {"group": ["A", "A", "B", "B"]},
            index=[0, 1, 2, 3],
        ),
        target=pd.Series(
            ["No", "No", "Yes", "Yes"],
            index=[1, 2, 3, 4],
        ),
    )

    assert report.has_alignment_issues
    assert report.numerical_relationships_frame().empty
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="projection_indices_not_aligned",
    ):
        report.raise_if_invalid()


def test_missing_requested_features_are_reported() -> None:
    report = analyze_feature_target_relationships(
        pd.DataFrame({"amount": [1, 2, 3, 4]}),
        pd.DataFrame({"group": ["A", "A", "B", "B"]}),
        pd.Series(["No", "No", "Yes", "Yes"]),
        numerical_features=("amount", "missing_number"),
        categorical_features=("group", "missing_category"),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        numerical_bin_count=2,
        minimum_group_count=1,
    )

    assert report.has_missing_features
    assert report.missing_numerical_features == ("missing_number",)
    assert report.missing_categorical_features == ("missing_category",)
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="missing_numerical_features:missing_number",
    ):
        report.raise_if_invalid()


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="contains duplicate names",
    ):
        analyze_feature_target_relationships(
            pd.DataFrame({"amount": [1, 2]}),
            pd.DataFrame({"group": ["A", "B"]}),
            pd.Series(["No", "Yes"]),
            numerical_features=("amount", "amount"),
            categorical_features=("group",),
            expected_target_classes=("No", "Yes"),
            positive_class="Yes",
        )


def test_duplicated_dataframe_columns_are_reported() -> None:
    numerical = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        columns=["amount", "amount"],
    )
    report = analyze_feature_target_relationships(
        numerical,
        pd.DataFrame({"group": ["A", "A", "B", "B"]}),
        pd.Series(["No", "No", "Yes", "Yes"]),
        numerical_features=("amount",),
        categorical_features=("group",),
        expected_target_classes=("No", "Yes"),
        positive_class="Yes",
        numerical_bin_count=2,
        minimum_group_count=1,
    )

    with pytest.raises(
        FeatureTargetAnalysisError,
        match="duplicated_column_labels",
    ):
        report.raise_if_invalid()


def test_positive_class_must_be_declared() -> None:
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="must belong",
    ):
        analyze_feature_target_relationships(
            pd.DataFrame({"amount": [1, 2]}),
            pd.DataFrame({"group": ["A", "B"]}),
            pd.Series(["No", "Yes"]),
            numerical_features=("amount",),
            categorical_features=("group",),
            expected_target_classes=("No", "Yes"),
            positive_class="Maybe",
        )


def test_invalid_thresholds_and_counts_are_rejected() -> None:
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="numerical_bin_count must be at least 2",
    ):
        _analyze(numerical_bin_count=1)

    with pytest.raises(
        FeatureTargetAnalysisError,
        match="minimum_group_count must be at least 1",
    ):
        _analyze(minimum_group_count=0)

    with pytest.raises(
        FeatureTargetAnalysisError,
        match="must be at most 1",
    ):
        _analyze(rate_difference_review_threshold=1.1)


def test_expected_category_contract_rejects_undeclared_feature() -> None:
    with pytest.raises(
        FeatureTargetAnalysisError,
        match="undeclared feature",
    ):
        _analyze(expected_category_values={"other": ("A", "B")})


def test_inputs_are_not_mutated_and_outputs_are_defensive_copies() -> None:
    numerical = pd.DataFrame({"amount": [1.0, 2.0, 8.0, 9.0]})
    categorical = pd.DataFrame(
        {"group": pd.Series([" A ", "A", "B", "B"], dtype="string")}
    )
    target = pd.Series(["No", "No", "Yes", "Yes"], name="Churn")
    numerical_before = numerical.copy(deep=True)
    categorical_before = categorical.copy(deep=True)
    target_before = target.copy(deep=True)

    report = _analyze(
        numerical=numerical,
        categorical=categorical,
        target=target,
    )
    first = report.categorical_rates_frame()
    first.loc[:, "Row count"] = -1
    second = report.categorical_rates_frame()

    pd.testing.assert_frame_equal(numerical, numerical_before)
    pd.testing.assert_frame_equal(categorical, categorical_before)
    pd.testing.assert_series_equal(target, target_before)
    assert not second["Row count"].eq(-1).any()


def test_results_are_deterministic() -> None:
    first = _analyze()
    second = _analyze()

    pd.testing.assert_frame_equal(
        first.numerical_relationships_frame(),
        second.numerical_relationships_frame(),
    )
    pd.testing.assert_frame_equal(
        first.categorical_rates_frame(),
        second.categorical_rates_frame(),
    )


# ---------------------------------------------------------------------------
# Multiclass numerical feature-to-target analysis
# ---------------------------------------------------------------------------

from scripts.analyze_target_relationships import (
    analyze_multiclass_numerical_target_relationships,
    plot_multiclass_feature_target_associations,
)


def _multiclass_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "separated": [1.0, 1.2, 1.1, 5.0, 5.2, 5.1, 9.0, 9.2, 9.1],
            "overlap": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "Class": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        }
    )


def test_multiclass_analysis_uses_unordered_target_without_positive_class() -> None:
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated", "overlap"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
        association_review_threshold=0.10,
    )

    assert report.is_analysis_ready
    assert report.expected_target_classes == ("A", "B", "C")
    assert report.has_review_candidates
    assert "Positive-class count" not in report.relationships_frame().columns


def test_multiclass_eta_squared_identifies_mean_separation() -> None:
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated", "overlap"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
        association_review_threshold=0.10,
    )
    relationships = report.relationships_frame().set_index("Feature")

    assert relationships.loc["separated", "Eta squared"] > 0.99
    assert relationships.loc["overlap", "Eta squared"] == pytest.approx(0.0)
    assert bool(relationships.loc["separated", "Review flag"])
    assert not bool(relationships.loc["overlap", "Review flag"])


def test_multiclass_rank_eta_squared_is_bounded() -> None:
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated",),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )
    value = report.relationships_frame().iloc[0]["Rank eta squared"]
    assert 0.0 <= value <= 1.0


def test_multiclass_class_statistics_preserve_contract_order() -> None:
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated",),
        target="Class",
        expected_target_classes=("C", "A", "B"),
    )
    statistics = report.class_statistics_frame()
    assert list(statistics["Target class"]) == ["C", "A", "B"]


def test_multiclass_missing_feature_blocks_validation() -> None:
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated", "missing"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )
    assert report.missing_features == ("missing",)
    with pytest.raises(FeatureTargetAnalysisError, match="missing_features"):
        report.raise_if_invalid()


def test_multiclass_unexpected_target_class_blocks_validation() -> None:
    frame = _multiclass_frame()
    frame.loc[0, "Class"] = "OTHER"
    report = analyze_multiclass_numerical_target_relationships(
        frame,
        features=("separated",),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )
    assert report.has_unexpected_target_classes
    with pytest.raises(FeatureTargetAnalysisError, match="unexpected_target_classes"):
        report.raise_if_invalid()


def test_multiclass_constant_feature_can_be_required_to_vary() -> None:
    frame = _multiclass_frame().assign(constant=1.0)
    report = analyze_multiclass_numerical_target_relationships(
        frame,
        features=("constant",),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )
    assert report.has_constant_features
    with pytest.raises(FeatureTargetAnalysisError, match="constant_features"):
        report.raise_if_invalid(require_sufficient_variation=True)


def test_multiclass_plot_returns_figure_without_mutating_report() -> None:
    pytest.importorskip("matplotlib")
    report = analyze_multiclass_numerical_target_relationships(
        _multiclass_frame(),
        features=("separated", "overlap"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )
    before = report.relationships_frame()
    figure = plot_multiclass_feature_target_associations(report)
    try:
        assert len(figure.axes) == 1
        pd.testing.assert_frame_equal(before, report.relationships_frame())
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)

# ---------------------------------------------------------------------------
# Continuous-regression numerical feature-to-target analysis
# ---------------------------------------------------------------------------

from scripts.analyze_target_relationships import (
    ContinuousFeatureTargetRelationshipReport,
    analyze_continuous_numerical_target_relationships,
    plot_continuous_feature_target_associations,
)


def _continuous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "positive": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "negative": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "weak": [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            "strength": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_continuous_analysis_preserves_signed_pearson_and_spearman() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("positive", "negative"),
        target="strength",
        unit="MPa",
        association_review_threshold=0.30,
    )
    relationships = report.relationships_frame().set_index("Feature")

    assert report.is_analysis_ready
    assert relationships.loc["positive", "Pearson correlation"] == pytest.approx(1.0)
    assert relationships.loc["positive", "Spearman correlation"] == pytest.approx(1.0)
    assert relationships.loc["negative", "Pearson correlation"] == pytest.approx(-1.0)
    assert relationships.loc["negative", "Spearman correlation"] == pytest.approx(-1.0)


def test_continuous_relationships_are_ranked_by_maximum_absolute_association() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("weak", "positive"),
        target="strength",
        association_review_threshold=0.30,
    )

    relationships = report.relationships_frame()
    assert relationships.iloc[0]["Feature"] == "positive"
    assert relationships.iloc[0]["Maximum absolute association"] == pytest.approx(1.0)


def test_continuous_review_flag_uses_maximum_absolute_correlation() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [2.0, 1.0, 4.0, 3.0, 5.0],
        }
    )
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("feature",),
        target="target",
        association_review_threshold=0.70,
    )
    row = report.relationships_frame().iloc[0]

    assert row["Maximum absolute association"] == pytest.approx(
        max(
            abs(row["Pearson correlation"]),
            abs(row["Spearman correlation"]),
        )
    )
    assert bool(row["Review flag"]) == (
        row["Maximum absolute association"] >= 0.70
    )


def test_continuous_summary_preserves_target_unit() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("positive",),
        target="strength",
        unit="MPa",
    )
    summary = report.summary_frame().set_index("Metric")

    assert summary.loc["Target", "Value"] == "strength (MPa)"
    assert summary.loc["Target unique values", "Value"] == 6


def test_continuous_missing_feature_blocks_validation() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("positive", "missing"),
        target="strength",
    )

    assert report.missing_features == ("missing",)
    with pytest.raises(FeatureTargetAnalysisError, match="missing_features:missing"):
        report.raise_if_invalid()


def test_continuous_missing_target_blocks_validation() -> None:
    frame = _continuous_frame()
    frame.loc[0, "strength"] = None
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("positive",),
        target="strength",
    )

    assert report.has_missing_target_values
    assert not report.is_analysis_ready
    with pytest.raises(FeatureTargetAnalysisError, match="missing_target_values:1"):
        report.raise_if_invalid()


def test_continuous_non_numeric_and_non_finite_target_values_are_invalid() -> None:
    frame = _continuous_frame().astype(object)
    frame.loc[0, "strength"] = "not-a-number"
    frame.loc[1, "strength"] = float("inf")
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("positive",),
        target="strength",
    )

    assert report.invalid_target_count == 2
    assert not report.is_analysis_ready
    with pytest.raises(FeatureTargetAnalysisError, match="invalid_target_values:2"):
        report.raise_if_invalid()


def test_continuous_constant_target_is_rejected() -> None:
    frame = _continuous_frame()
    frame.loc[:, "strength"] = 10.0
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("positive",),
        target="strength",
    )

    assert report.has_constant_target
    with pytest.raises(FeatureTargetAnalysisError, match="constant_target_detected"):
        report.raise_if_invalid()


def test_continuous_constant_feature_is_reported_and_optionally_rejected() -> None:
    frame = _continuous_frame()
    frame.loc[:, "positive"] = 1.0
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("positive",),
        target="strength",
    )
    row = report.relationships_frame().iloc[0]

    assert report.has_constant_features
    assert pd.isna(row["Pearson correlation"])
    assert pd.isna(row["Spearman correlation"])
    with pytest.raises(FeatureTargetAnalysisError, match="constant_features_detected"):
        report.raise_if_invalid(require_sufficient_feature_variation=True)


def test_continuous_inputs_and_report_frames_are_defensive() -> None:
    frame = _continuous_frame()
    before = frame.copy(deep=True)
    report = analyze_continuous_numerical_target_relationships(
        frame,
        features=("positive", "negative"),
        target="strength",
    )

    first = report.relationships_frame()
    first.loc[:, "Review flag"] = False
    second = report.relationships_frame()

    pd.testing.assert_frame_equal(frame, before)
    assert second["Review flag"].any()


def test_continuous_plot_returns_figure_without_mutating_report() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("positive", "negative", "weak"),
        target="strength",
    )
    before = report.relationships_frame()

    figure = plot_continuous_feature_target_associations(report)

    assert figure.axes
    pd.testing.assert_frame_equal(report.relationships_frame(), before)


def test_continuous_invalid_threshold_and_unit_are_rejected() -> None:
    with pytest.raises(FeatureTargetAnalysisError, match="must be at most 1"):
        analyze_continuous_numerical_target_relationships(
            _continuous_frame(),
            features=("positive",),
            target="strength",
            association_review_threshold=1.1,
        )

    with pytest.raises(FeatureTargetAnalysisError, match="unit must be"):
        analyze_continuous_numerical_target_relationships(
            _continuous_frame(),
            features=("positive",),
            target="strength",
            unit="",
        )


def test_continuous_report_type_is_explicit() -> None:
    report = analyze_continuous_numerical_target_relationships(
        _continuous_frame(),
        features=("positive",),
        target="strength",
    )

    assert isinstance(report, ContinuousFeatureTargetRelationshipReport)

