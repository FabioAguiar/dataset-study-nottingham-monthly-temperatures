"""Tests for continuous-regression structure diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_regression_structure import (
    RegressionStructureAnalysisError,
    analyze_regression_structure,
    plot_interaction_signals,
    plot_nonlinearity_signals,
)


def _quadratic_frame() -> pd.DataFrame:
    x = np.linspace(-3.0, 3.0, 61)
    z = np.linspace(-2.0, 2.0, 61)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "target": 4.0 + 2.5 * x**2 + 0.2 * z,
        }
    )


def _interaction_frame() -> pd.DataFrame:
    x = np.tile(np.linspace(-2.0, 2.0, 15), 15)
    z = np.repeat(np.linspace(-2.0, 2.0, 15), 15)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "target": 3.0 + x + 0.5 * z + 4.0 * x * z,
        }
    )


def test_quadratic_signal_is_detected() -> None:
    report = analyze_regression_structure(
        _quadratic_frame(),
        features=("x", "z"),
        target="target",
        nonlinearity_review_threshold=0.02,
    )
    row = report.nonlinearity_frame().set_index("Feature").loc["x"]

    assert row["Quadratic adjusted R squared"] > 0.99
    assert row["Adjusted R squared gain"] > 0.5
    assert bool(row["Nonlinearity signal"])


def test_linear_feature_is_not_forced_to_nonlinear_signal() -> None:
    x = np.linspace(-3.0, 3.0, 80)
    z = np.linspace(4.0, 8.0, 80)
    frame = pd.DataFrame({"x": x, "z": z, "target": 2.0 + 5.0 * x})
    report = analyze_regression_structure(
        frame,
        features=("x", "z"),
        target="target",
        nonlinearity_review_threshold=0.02,
    )
    row = report.nonlinearity_frame().set_index("Feature").loc["x"]

    assert row["Linear adjusted R squared"] == pytest.approx(1.0)
    assert float(row["Adjusted R squared gain"]) <= 1e-10
    assert not bool(row["Nonlinearity signal"])


def test_product_interaction_signal_is_detected_beyond_quadratic_main_effects() -> None:
    report = analyze_regression_structure(
        _interaction_frame(),
        features=("x", "z"),
        target="target",
        interaction_review_threshold=0.02,
    )
    row = report.interaction_frame().iloc[0]

    assert row["With interaction adjusted R squared"] > 0.99
    assert row["Adjusted R squared gain"] > 0.5
    assert bool(row["Interaction signal"])


def test_additive_relationship_does_not_force_interaction_signal() -> None:
    frame = _interaction_frame().copy()
    frame["target"] = 2.0 + 3.0 * frame["x"] - 2.0 * frame["z"]
    report = analyze_regression_structure(
        frame,
        features=("x", "z"),
        target="target",
        interaction_review_threshold=0.02,
    )
    row = report.interaction_frame().iloc[0]

    assert row["Quadratic main-effects adjusted R squared"] == pytest.approx(1.0)
    assert float(row["Adjusted R squared gain"]) <= 1e-10
    assert not bool(row["Interaction signal"])


def test_pair_count_matches_unique_feature_combinations() -> None:
    x = np.linspace(0.0, 4.0, 30)
    frame = pd.DataFrame(
        {
            "a": x,
            "b": x**2 + 1.0,
            "c": np.sin(x) + x,
            "target": 2.0 * x + x**2,
        }
    )
    report = analyze_regression_structure(
        frame,
        features=("a", "b", "c"),
        target="target",
    )

    assert len(report.interaction_frame()) == 3


def test_missing_feature_is_reported_and_can_block() -> None:
    report = analyze_regression_structure(
        _quadratic_frame(),
        features=("x", "missing"),
        target="target",
    )

    assert report.has_missing_features
    with pytest.raises(RegressionStructureAnalysisError, match="missing_features"):
        report.raise_if_invalid()


def test_missing_target_values_are_reported() -> None:
    frame = _quadratic_frame()
    frame.loc[0, "target"] = np.nan
    report = analyze_regression_structure(
        frame,
        features=("x", "z"),
        target="target",
    )

    assert report.has_missing_target_values
    with pytest.raises(
        RegressionStructureAnalysisError,
        match="missing_target_values",
    ):
        report.raise_if_invalid()


def test_constant_feature_can_block_complete_numeric_requirement() -> None:
    frame = _quadratic_frame()
    frame["z"] = 1.0
    report = analyze_regression_structure(
        frame,
        features=("x", "z"),
        target="target",
    )

    assert report.has_feature_issues
    with pytest.raises(
        RegressionStructureAnalysisError,
        match="numeric_feature_issues_detected",
    ):
        report.raise_if_invalid()


def test_thresholds_must_be_non_negative_finite_numbers() -> None:
    with pytest.raises(RegressionStructureAnalysisError, match="non-negative"):
        analyze_regression_structure(
            _quadratic_frame(),
            features=("x", "z"),
            target="target",
            nonlinearity_review_threshold=-0.1,
        )
    with pytest.raises(RegressionStructureAnalysisError, match="finite"):
        analyze_regression_structure(
            _quadratic_frame(),
            features=("x", "z"),
            target="target",
            interaction_review_threshold=float("inf"),
        )


def test_interaction_frame_limit_is_validated() -> None:
    report = analyze_regression_structure(
        _interaction_frame(),
        features=("x", "z"),
        target="target",
    )

    assert len(report.interaction_frame(limit=1)) == 1
    with pytest.raises(RegressionStructureAnalysisError, match="positive integer"):
        report.interaction_frame(limit=0)


def test_plots_are_generated_without_mutating_report() -> None:
    report = analyze_regression_structure(
        _interaction_frame(),
        features=("x", "z"),
        target="target",
    )
    before_nonlinearity = report.nonlinearity_frame()
    before_interactions = report.interaction_frame()

    nonlinearity_figure = plot_nonlinearity_signals(report)
    interaction_figure = plot_interaction_signals(report)

    assert nonlinearity_figure.axes
    assert interaction_figure.axes
    pd.testing.assert_frame_equal(before_nonlinearity, report.nonlinearity_frame())
    pd.testing.assert_frame_equal(before_interactions, report.interaction_frame())

    import matplotlib.pyplot as plt

    plt.close(nonlinearity_figure)
    plt.close(interaction_figure)
