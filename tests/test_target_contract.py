"""Tests for reusable classification target-contract validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.target_contract import (
    TargetContractError,
    define_univariate_forecasting_target_contract,
    define_classification_target_contract,
    define_continuous_regression_target_contract,
)


DRY_BEAN_CLASSES = (
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SIRA",
)


def _multiclass_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": range(7),
            "Class": DRY_BEAN_CLASSES,
        }
    )


def test_multiclass_contract_validates_declared_labels() -> None:
    report = define_classification_target_contract(
        _multiclass_frame(),
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
    )

    assert report.target == "Class"
    assert report.problem_type == "multiclass_classification"
    assert report.class_count == 7
    assert report.positive_class is None
    assert report.class_semantics == "Nominal / unordered"
    assert report.expected_classes == DRY_BEAN_CLASSES
    assert report.classes_frame()["Observed"].all()


def test_contract_rejects_unexpected_label() -> None:
    dataframe = _multiclass_frame()
    dataframe.loc[0, "Class"] = "UNKNOWN"

    with pytest.raises(TargetContractError, match="unexpected target labels"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
        )


def test_contract_rejects_declared_label_not_observed() -> None:
    dataframe = _multiclass_frame().loc[lambda frame: frame["Class"] != "SIRA"]

    with pytest.raises(
        TargetContractError,
        match="declared target labels not observed",
    ):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
        )


def test_missing_targets_do_not_change_label_contract() -> None:
    dataframe = pd.concat(
        [
            _multiclass_frame(),
            pd.DataFrame({"Area": [10], "Class": [None]}),
        ],
        ignore_index=True,
    )

    report = define_classification_target_contract(
        dataframe,
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
    )

    assert report.class_count == 7


def test_multiclass_requires_at_least_three_classes() -> None:
    dataframe = pd.DataFrame({"Class": ["A", "B"]})

    with pytest.raises(TargetContractError, match="at least three"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=("A", "B"),
            problem_type="multiclass_classification",
        )


def test_binary_contract_requires_exactly_two_classes() -> None:
    dataframe = pd.DataFrame({"Class": ["A", "B", "C"]})

    with pytest.raises(TargetContractError, match="exactly two"):
        define_classification_target_contract(
            dataframe,
            target="Class",
            expected_classes=("A", "B", "C"),
            problem_type="binary_classification",
        )


def test_source_variables_must_declare_target_role(tmp_path: Path) -> None:
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Class"],
            "role": ["Feature", "Target"],
        }
    ).to_csv(variables_file, index=False)

    report = define_classification_target_contract(
        _multiclass_frame(),
        target="Class",
        expected_classes=DRY_BEAN_CLASSES,
        problem_type="multiclass_classification",
        source_variables_file=variables_file,
    )

    assert report.source_role == "Target"


def test_source_variables_reject_wrong_target_role(tmp_path: Path) -> None:
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Class"],
            "role": ["Feature", "Feature"],
        }
    ).to_csv(variables_file, index=False)

    with pytest.raises(TargetContractError, match="does not declare"):
        define_classification_target_contract(
            _multiclass_frame(),
            target="Class",
            expected_classes=DRY_BEAN_CLASSES,
            problem_type="multiclass_classification",
            source_variables_file=variables_file,
        )



def _continuous_regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Cement": [540.0, 332.5, 198.6],
            "Age": [28, 56, 90],
            "Concrete compressive strength": [79.99, 39.29, 44.30],
        }
    )


def _write_continuous_variables_file(path: Path) -> Path:
    pd.DataFrame(
        {
            "name": [
                "Cement",
                "Age",
                "Concrete compressive strength",
            ],
            "role": ["Feature", "Feature", "Target"],
            "type": ["Continuous", "Integer", "Continuous"],
            "units": ["kg/m^3", "day", "MPa"],
        }
    ).to_csv(path, index=False)
    return path


def test_continuous_regression_contract_validates_numeric_target(
    tmp_path: Path,
) -> None:
    variables_file = _write_continuous_variables_file(
        tmp_path / "variables.csv"
    )

    report = define_continuous_regression_target_contract(
        _continuous_regression_frame(),
        target="Concrete compressive strength",
        problem_type="continuous_regression",
        target_semantics="Continuous / quantitative",
        expected_unit="MPa",
        expected_source_type="Continuous",
        source_variables_file=variables_file,
    )

    assert report.target == "Concrete compressive strength"
    assert report.problem_type == "continuous_regression"
    assert report.target_semantics == "Continuous / quantitative"
    assert report.prediction_output == (
        "Continuous numeric value on the original target scale"
    )
    assert report.source_role == "Target"
    assert report.source_type == "Continuous"
    assert report.source_unit == "MPa"
    assert report.summary_frame().loc[
        lambda frame: frame["Contract item"].eq("Contract status"),
        "Value",
    ].item() == "Valid"


def test_continuous_regression_contract_rejects_non_numeric_target() -> None:
    dataframe = _continuous_regression_frame()
    dataframe["Concrete compressive strength"] = ["high", "medium", "low"]

    with pytest.raises(TargetContractError, match="numeric target"):
        define_continuous_regression_target_contract(
            dataframe,
            target="Concrete compressive strength",
            problem_type="continuous_regression",
        )


def test_continuous_regression_contract_rejects_wrong_problem_type() -> None:
    with pytest.raises(
        TargetContractError,
        match="continuous_regression",
    ):
        define_continuous_regression_target_contract(
            _continuous_regression_frame(),
            target="Concrete compressive strength",
            problem_type="multiclass_classification",  # type: ignore[arg-type]
        )


def test_continuous_regression_contract_validates_source_type(
    tmp_path: Path,
) -> None:
    variables_file = _write_continuous_variables_file(
        tmp_path / "variables.csv"
    )

    with pytest.raises(TargetContractError, match="target type"):
        define_continuous_regression_target_contract(
            _continuous_regression_frame(),
            target="Concrete compressive strength",
            problem_type="continuous_regression",
            expected_source_type="Integer",
            source_variables_file=variables_file,
        )


def test_continuous_regression_contract_validates_target_unit(
    tmp_path: Path,
) -> None:
    variables_file = _write_continuous_variables_file(
        tmp_path / "variables.csv"
    )

    with pytest.raises(TargetContractError, match="target unit"):
        define_continuous_regression_target_contract(
            _continuous_regression_frame(),
            target="Concrete compressive strength",
            problem_type="continuous_regression",
            expected_unit="psi",
            source_variables_file=variables_file,
        )


def _univariate_forecasting_series() -> pd.Series:
    return pd.Series(
        [40.6, 40.8, 44.4],
        index=pd.period_range("1920-01", periods=3, freq="M", name="period"),
        name="value",
    )


def test_univariate_forecasting_contract_validates_period_indexed_target() -> None:
    report = define_univariate_forecasting_target_contract(
        _univariate_forecasting_series(),
        target="temperature",
        problem_type="time_series_forecasting",
        forecasting_mode="univariate",
        target_semantics="Monthly average air temperature at Nottingham Castle",
        target_unit="degrees Fahrenheit",
        expected_frequency="M",
        source_exogenous_predictors=0,
    )

    assert report.target == "temperature"
    assert report.source_value_name == "value"
    assert report.problem_type == "time_series_forecasting"
    assert report.forecasting_mode == "univariate"
    assert report.target_unit == "degrees Fahrenheit"
    assert report.index_type == "PeriodIndex"
    assert report.index_name == "period"
    assert report.frequency == "M"
    assert report.source_exogenous_predictors == 0
    assert report.summary_frame().loc[
        lambda frame: frame["Contract item"].eq("Contract status"),
        "Value",
    ].item() == "Valid"


def test_univariate_forecasting_contract_rejects_tabular_regression_type() -> None:
    with pytest.raises(
        TargetContractError,
        match="time_series_forecasting",
    ):
        define_univariate_forecasting_target_contract(
            _univariate_forecasting_series(),
            target="temperature",
            problem_type="continuous_regression",  # type: ignore[arg-type]
            forecasting_mode="univariate",
            target_semantics="Monthly average air temperature",
            target_unit="degrees Fahrenheit",
            expected_frequency="M",
        )


def test_univariate_forecasting_contract_requires_period_index() -> None:
    series = pd.Series([40.6, 40.8, 44.4], name="value")

    with pytest.raises(TargetContractError, match="PeriodIndex"):
        define_univariate_forecasting_target_contract(
            series,
            target="temperature",
            problem_type="time_series_forecasting",
            forecasting_mode="univariate",
            target_semantics="Monthly average air temperature",
            target_unit="degrees Fahrenheit",
            expected_frequency="M",
        )


def test_univariate_forecasting_contract_rejects_non_numeric_target() -> None:
    series = pd.Series(
        ["cold", "mild", "warm"],
        index=pd.period_range("1920-01", periods=3, freq="M"),
        name="value",
    )

    with pytest.raises(TargetContractError, match="numeric endogenous target"):
        define_univariate_forecasting_target_contract(
            series,
            target="temperature",
            problem_type="time_series_forecasting",
            forecasting_mode="univariate",
            target_semantics="Monthly average air temperature",
            target_unit="degrees Fahrenheit",
            expected_frequency="M",
        )


def test_univariate_forecasting_contract_validates_frequency() -> None:
    series = pd.Series(
        [40.6, 40.8, 44.4],
        index=pd.period_range("1920-01-01", periods=3, freq="D"),
        name="value",
    )

    with pytest.raises(TargetContractError, match="frequency"):
        define_univariate_forecasting_target_contract(
            series,
            target="temperature",
            problem_type="time_series_forecasting",
            forecasting_mode="univariate",
            target_semantics="Monthly average air temperature",
            target_unit="degrees Fahrenheit",
            expected_frequency="M",
        )
