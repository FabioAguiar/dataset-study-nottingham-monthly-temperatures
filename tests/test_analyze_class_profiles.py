"""Tests for multiclass class-profile, separation, and overlap exploration."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_class_profiles import (
    ClassProfileAnalysisError,
    analyze_multiclass_class_profiles,
    plot_class_pca_projection,
    plot_standardized_class_profiles,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0, 1.5, 0.8, 1.0, 1.4, 1.8, 8.0, 8.5, 9.0, 9.5],
            "y": [0.0, 0.2, 0.4, 0.6, 0.25, 0.4, 0.65, 0.85, 8.0, 8.2, 8.4, 8.6],
            "Class": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        }
    )


def _report():
    return analyze_multiclass_class_profiles(
        _frame(),
        features=("x", "y"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )


def test_report_contains_all_unique_class_pairs() -> None:
    report = _report()

    assert len(report.pairwise_overlap_frame()) == 3
    assert set(
        tuple(row)
        for row in report.pairwise_overlap_frame()[["Class A", "Class B"]].to_numpy()
    ) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_standardized_profile_preserves_contract_and_feature_order() -> None:
    frame = _report().standardized_profile_frame()

    assert list(frame.index) == ["A", "B", "C"]
    assert list(frame.columns) == ["x", "y"]
    assert frame.loc["C", "x"] > frame.loc["A", "x"]


def test_pairwise_overlap_orders_most_overlapping_pair_first() -> None:
    pairwise = _report().pairwise_overlap_frame()
    first = pairwise.iloc[0]

    assert {first["Class A"], first["Class B"]} == {"A", "B"}
    assert first["Mean IQR overlap coefficient"] > 0
    far_pair = pairwise.loc[
        pairwise[["Class A", "Class B"]].apply(
            lambda row: set(row) == {"A", "C"}, axis=1
        )
    ].iloc[0]
    assert far_pair["Mean IQR overlap coefficient"] == pytest.approx(0.0)


def test_pairwise_gap_is_smaller_for_nearby_classes() -> None:
    pairwise = _report().pairwise_overlap_frame()
    ab = pairwise.loc[
        pairwise[["Class A", "Class B"]].apply(
            lambda row: set(row) == {"A", "B"}, axis=1
        )
    ].iloc[0]
    ac = pairwise.loc[
        pairwise[["Class A", "Class B"]].apply(
            lambda row: set(row) == {"A", "C"}, axis=1
        )
    ].iloc[0]

    assert ab["RMS robust median gap"] < ac["RMS robust median gap"]


def test_pca_projection_covers_every_row_and_reports_variance() -> None:
    report = _report()
    projection = report.pca_projection_frame()

    assert len(projection) == len(_frame())
    assert list(projection.columns) == ["PC1", "PC2", "Class"]
    pc1, pc2 = report.pca_explained_variance_ratio
    assert 0 <= pc1 <= 1
    assert 0 <= pc2 <= 1
    assert pc1 + pc2 <= 1.0 + 1e-12


def test_missing_expected_class_is_reported_and_can_block() -> None:
    report = analyze_multiclass_class_profiles(
        _frame().loc[lambda frame: frame["Class"].ne("C")],
        features=("x", "y"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )

    assert report.has_missing_expected_target_classes
    with pytest.raises(ClassProfileAnalysisError, match="missing_expected_target_classes"):
        report.raise_if_invalid()


def test_non_numeric_or_missing_feature_values_are_reported() -> None:
    frame = _frame()
    frame.loc[0, "x"] = None
    report = analyze_multiclass_class_profiles(
        frame,
        features=("x", "y"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )

    assert report.has_non_numeric_or_missing_features
    with pytest.raises(
        ClassProfileAnalysisError,
        match="non_numeric_missing_or_constant_features_detected",
    ):
        report.raise_if_invalid()


def test_constant_feature_is_reported() -> None:
    frame = _frame()
    frame["x"] = 1.0
    report = analyze_multiclass_class_profiles(
        frame,
        features=("x", "y"),
        target="Class",
        expected_target_classes=("A", "B", "C"),
    )

    assert report.has_non_numeric_or_missing_features
    assert "Constant numerical feature" in set(report.issues_frame()["Issue"])


def test_pairwise_limit_requires_positive_integer() -> None:
    report = _report()

    assert len(report.pairwise_overlap_frame(limit=2)) == 2
    with pytest.raises(ClassProfileAnalysisError, match="positive integer"):
        report.pairwise_overlap_frame(limit=0)


def test_profile_and_pca_plots_are_generated_without_mutating_report() -> None:
    report = _report()
    before_profiles = report.standardized_profile_frame()
    before_projection = report.pca_projection_frame()

    profile_figure = plot_standardized_class_profiles(report)
    pca_figure = plot_class_pca_projection(
        report,
        max_points_per_class=2,
        random_state=42,
    )

    assert profile_figure.axes
    assert pca_figure.axes
    pd.testing.assert_frame_equal(before_profiles, report.standardized_profile_frame())
    pd.testing.assert_frame_equal(before_projection, report.pca_projection_frame())

    import matplotlib.pyplot as plt

    plt.close(profile_figure)
    plt.close(pca_figure)
