"""Reusable validation and presentation of prediction target contracts.

The notebook remains responsible for declaring study-specific prediction
semantics. This module performs non-mutating structural validation without
analyzing target prevalence, distribution, range, or outliers and without
encoding or transforming target values for modeling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd


ClassificationProblemType = Literal[
    "binary_classification",
    "multiclass_classification",
]

ContinuousRegressionProblemType = Literal["continuous_regression"]
ForecastingProblemType = Literal["time_series_forecasting"]
ForecastingMode = Literal["univariate"]

_SUMMARY_COLUMNS: Final[list[str]] = [
    "Contract item",
    "Value",
    "Interpretation",
]

_CLASS_COLUMNS: Final[list[str]] = [
    "Class label",
    "Declared",
    "Observed",
]


class TargetContractError(ValueError):
    """Raised when a prediction target contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class ClassificationTargetContract:
    """Validated, non-mutating classification target contract."""

    target: str
    problem_type: ClassificationProblemType
    expected_classes: tuple[object, ...]
    observed_classes: tuple[object, ...]
    source_role: str | None

    @property
    def class_count(self) -> int:
        """Return the number of classes declared by the contract."""
        return len(self.expected_classes)

    @property
    def positive_class(self) -> None:
        """Return no positive class for the neutral target contract layer."""
        return None

    @property
    def class_semantics(self) -> str:
        """Describe classification labels as nominal rather than ordinal."""
        return "Nominal / unordered"

    def summary_frame(self) -> pd.DataFrame:
        """Return the contract as a compact deterministic table."""
        rows = [
            {
                "Contract item": "Problem type",
                "Value": self.problem_type,
                "Interpretation": "Supervised classification task",
            },
            {
                "Contract item": "Target column",
                "Value": self.target,
                "Interpretation": "Outcome to be predicted",
            },
            {
                "Contract item": "Declared classes",
                "Value": self.class_count,
                "Interpretation": "Expected target cardinality",
            },
            {
                "Contract item": "Class semantics",
                "Value": self.class_semantics,
                "Interpretation": "Labels have no ordinal ranking",
            },
            {
                "Contract item": "Positive class",
                "Value": "Not applicable",
                "Interpretation": "No binary positive/negative semantics",
            },
            {
                "Contract item": "Source variable role",
                "Value": self.source_role or "Not checked",
                "Interpretation": "Role declared by source metadata",
            },
            {
                "Contract item": "Contract status",
                "Value": "Valid",
                "Interpretation": (
                    "Observed non-missing labels match the declared classes"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def classes_frame(self) -> pd.DataFrame:
        """Return declared labels without leaking class-frequency analysis."""
        rows = [
            {
                "Class label": class_value,
                "Declared": True,
                "Observed": _contains_value(
                    self.observed_classes,
                    class_value,
                ),
            }
            for class_value in self.expected_classes
        ]
        return pd.DataFrame(rows, columns=_CLASS_COLUMNS)


@dataclass(frozen=True, slots=True)
class ContinuousRegressionTargetContract:
    """Validated, non-mutating continuous-regression target contract."""

    target: str
    problem_type: ContinuousRegressionProblemType
    target_semantics: str
    expected_unit: str | None
    source_role: str | None
    source_type: str | None
    source_unit: str | None

    @property
    def prediction_output(self) -> str:
        """Describe the expected prediction representation."""
        return "Continuous numeric value on the original target scale"

    def summary_frame(self) -> pd.DataFrame:
        """Return the contract as a compact deterministic table."""
        rows = [
            {
                "Contract item": "Problem type",
                "Value": self.problem_type,
                "Interpretation": "Supervised continuous regression task",
            },
            {
                "Contract item": "Target column",
                "Value": self.target,
                "Interpretation": "Continuous outcome to be predicted",
            },
            {
                "Contract item": "Target semantics",
                "Value": self.target_semantics,
                "Interpretation": "Quantitative target; not a class label",
            },
            {
                "Contract item": "Prediction output",
                "Value": self.prediction_output,
                "Interpretation": "No thresholding or class decoding applies",
            },
            {
                "Contract item": "Target unit",
                "Value": self.expected_unit or self.source_unit or "Not declared",
                "Interpretation": "Unit of the original prediction scale",
            },
            {
                "Contract item": "Source variable role",
                "Value": self.source_role or "Not checked",
                "Interpretation": "Role declared by source metadata",
            },
            {
                "Contract item": "Source variable type",
                "Value": self.source_type or "Not checked",
                "Interpretation": "Type declared by source metadata",
            },
            {
                "Contract item": "Contract status",
                "Value": "Valid",
                "Interpretation": (
                    "Target exists, is numeric, and matches declared metadata"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


@dataclass(frozen=True, slots=True)
class UnivariateForecastingTargetContract:
    """Validated, non-mutating univariate forecasting target contract."""

    target: str
    source_value_name: str | None
    problem_type: ForecastingProblemType
    forecasting_mode: ForecastingMode
    target_semantics: str
    target_unit: str
    index_type: str
    index_name: str | None
    frequency: str
    source_exogenous_predictors: int

    @property
    def prediction_output(self) -> str:
        """Describe the semantic forecasting output without fixing a horizon."""
        return (
            "Future target values indexed by forecast periods on the original "
            "target scale"
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return the frozen target semantics as a deterministic table."""
        rows = [
            {
                "Contract item": "Problem type",
                "Value": self.problem_type,
                "Interpretation": "Temporally ordered forecasting task",
            },
            {
                "Contract item": "Forecasting mode",
                "Value": self.forecasting_mode,
                "Interpretation": "One endogenous target series",
            },
            {
                "Contract item": "Canonical target",
                "Value": self.target,
                "Interpretation": "Quantity to be forecast at future periods",
            },
            {
                "Contract item": "Source value column",
                "Value": self.source_value_name or "Unnamed",
                "Interpretation": "Raw acquired value representation",
            },
            {
                "Contract item": "Target semantics",
                "Value": self.target_semantics,
                "Interpretation": "Scientific meaning of each target value",
            },
            {
                "Contract item": "Target unit",
                "Value": self.target_unit,
                "Interpretation": "Original target and forecast scale",
            },
            {
                "Contract item": "Temporal index",
                "Value": self.index_type,
                "Interpretation": "Canonical observation-period identity",
            },
            {
                "Contract item": "Frequency",
                "Value": self.frequency,
                "Interpretation": "Canonical spacing of forecast periods",
            },
            {
                "Contract item": "Source exogenous predictors",
                "Value": self.source_exogenous_predictors,
                "Interpretation": "Exogenous series present in the source data",
            },
            {
                "Contract item": "Prediction output",
                "Value": self.prediction_output,
                "Interpretation": (
                    "Forecast output; horizon remains a separate contract item"
                ),
            },
            {
                "Contract item": "Contract status",
                "Value": "Valid",
                "Interpretation": (
                    "Target is numeric and uses the declared PeriodIndex frequency"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def define_univariate_forecasting_target_contract(
    series: pd.Series,
    *,
    target: str,
    problem_type: ForecastingProblemType,
    forecasting_mode: ForecastingMode,
    target_semantics: str,
    target_unit: str,
    expected_frequency: str,
    source_exogenous_predictors: int = 0,
) -> UnivariateForecastingTargetContract:
    """Validate structural target semantics for univariate forecasting.

    This layer intentionally does not choose forecast horizon, forecast origin,
    history-window policy, evaluation protocol, final holdout, model family,
    metric, transformation, or multi-step strategy. Missing values, non-finite
    values, duplicates, range, and anomalies remain dedicated data-quality and
    exploratory concerns.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    target_name = _normalize_text(target, field="target")
    semantics = _normalize_text(target_semantics, field="target_semantics")
    unit = _normalize_text(target_unit, field="target_unit")
    frequency = _normalize_text(
        expected_frequency,
        field="expected_frequency",
    )

    if problem_type != "time_series_forecasting":
        raise TargetContractError(
            "problem_type must be 'time_series_forecasting'."
        )

    if forecasting_mode != "univariate":
        raise TargetContractError("forecasting_mode must be 'univariate'.")

    if pd.api.types.is_bool_dtype(series.dtype) or not (
        pd.api.types.is_numeric_dtype(series.dtype)
    ):
        raise TargetContractError(
            "univariate forecasting requires a numeric endogenous target "
            f"series; observed dtype={series.dtype!s}."
        )

    if not isinstance(series.index, pd.PeriodIndex):
        raise TargetContractError(
            "univariate forecasting requires a pandas PeriodIndex at this "
            "contract stage."
        )

    observed_frequency = series.index.freqstr
    if observed_frequency != frequency:
        raise TargetContractError(
            "forecasting target frequency does not match the declared "
            f"contract: expected={frequency!r}, "
            f"observed={observed_frequency!r}."
        )

    if (
        isinstance(source_exogenous_predictors, bool)
        or not isinstance(source_exogenous_predictors, int)
        or source_exogenous_predictors < 0
    ):
        raise TargetContractError(
            "source_exogenous_predictors must be a non-negative integer."
        )

    source_value_name = (
        str(series.name).strip()
        if series.name is not None and str(series.name).strip()
        else None
    )
    index_name = (
        str(series.index.name).strip()
        if series.index.name is not None and str(series.index.name).strip()
        else None
    )

    return UnivariateForecastingTargetContract(
        target=target_name,
        source_value_name=source_value_name,
        problem_type=problem_type,
        forecasting_mode=forecasting_mode,
        target_semantics=semantics,
        target_unit=unit,
        index_type=type(series.index).__name__,
        index_name=index_name,
        frequency=observed_frequency,
        source_exogenous_predictors=source_exogenous_predictors,
    )


def define_continuous_regression_target_contract(
    dataframe: pd.DataFrame,
    *,
    target: str,
    problem_type: ContinuousRegressionProblemType,
    target_semantics: str = "Continuous / quantitative",
    expected_unit: str | None = None,
    expected_source_type: str | None = None,
    source_variables_file: str | Path | None = None,
) -> ContinuousRegressionTargetContract:
    """Validate and return a continuous-regression target contract.

    Distribution, range, missing-value prevalence, non-finite values, and
    outliers are intentionally out of scope for this contract layer. Missing
    values are ignored only while confirming that observed target values are
    represented by a numeric pandas dtype.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_text(target, field="target")
    _require_unique_columns(dataframe)

    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    if problem_type != "continuous_regression":
        raise TargetContractError(
            "problem_type must be 'continuous_regression'."
        )

    semantics = _normalize_text(
        target_semantics,
        field="target_semantics",
    )
    normalized_expected_unit = _normalize_optional_text(
        expected_unit,
        field="expected_unit",
    )
    normalized_expected_source_type = _normalize_optional_text(
        expected_source_type,
        field="expected_source_type",
    )

    target_series = dataframe[target_name]
    if pd.api.types.is_bool_dtype(target_series.dtype) or not (
        pd.api.types.is_numeric_dtype(target_series.dtype)
    ):
        raise TargetContractError(
            "continuous_regression requires a numeric target column; "
            f"observed dtype={target_series.dtype!s}."
        )

    source_role = None
    source_type = None
    source_unit = None
    if source_variables_file is not None:
        source_metadata = _read_source_target_metadata(
            source_variables_file,
            target=target_name,
        )
        source_role = source_metadata["role"]
        source_type = source_metadata["type"]
        source_unit = source_metadata["unit"]

        if source_role.lower() != "target":
            raise TargetContractError(
                f"Source metadata does not declare {target_name!r} as Target; "
                f"observed role={source_role!r}."
            )

        if (
            normalized_expected_source_type is not None
            and source_type.lower() != normalized_expected_source_type.lower()
        ):
            raise TargetContractError(
                "Source metadata target type does not match the declared "
                f"contract: expected={normalized_expected_source_type!r}, "
                f"observed={source_type!r}."
            )

        if (
            normalized_expected_unit is not None
            and source_unit.lower() != normalized_expected_unit.lower()
        ):
            raise TargetContractError(
                "Source metadata target unit does not match the declared "
                f"contract: expected={normalized_expected_unit!r}, "
                f"observed={source_unit!r}."
            )

    return ContinuousRegressionTargetContract(
        target=target_name,
        problem_type=problem_type,
        target_semantics=semantics,
        expected_unit=normalized_expected_unit,
        source_role=source_role,
        source_type=source_type,
        source_unit=source_unit,
    )


def define_classification_target_contract(
    dataframe: pd.DataFrame,
    *,
    target: str,
    expected_classes: Sequence[object],
    problem_type: ClassificationProblemType,
    source_variables_file: str | Path | None = None,
) -> ClassificationTargetContract:
    """Validate and return a binary or multiclass target contract.

    Class counts and imbalance are intentionally out of scope. Missing target
    values are also left to the dedicated data-quality stages; they are ignored
    only while determining which non-missing labels are observed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_text(target, field="target")
    _require_unique_columns(dataframe)

    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    normalized_classes = _normalize_expected_classes(expected_classes)
    _validate_problem_type(problem_type, normalized_classes)

    observed_series = dataframe[target_name].dropna()
    observed_classes = tuple(pd.unique(observed_series))

    unexpected_classes = tuple(
        value
        for value in observed_classes
        if not _contains_value(normalized_classes, value)
    )
    missing_classes = tuple(
        value
        for value in normalized_classes
        if not _contains_value(observed_classes, value)
    )

    failures: list[str] = []
    if unexpected_classes:
        failures.append(
            "unexpected target labels: "
            + ", ".join(repr(value) for value in unexpected_classes)
        )
    if missing_classes:
        failures.append(
            "declared target labels not observed: "
            + ", ".join(repr(value) for value in missing_classes)
        )

    source_role = None
    if source_variables_file is not None:
        source_role = _validate_source_target_role(
            source_variables_file,
            target=target_name,
        )

    if failures:
        raise TargetContractError(
            "Target contract validation failed: " + "; ".join(failures) + "."
        )

    return ClassificationTargetContract(
        target=target_name,
        problem_type=problem_type,
        expected_classes=normalized_classes,
        observed_classes=observed_classes,
        source_role=source_role,
    )


def _validate_problem_type(
    problem_type: ClassificationProblemType,
    classes: tuple[object, ...],
) -> None:
    if problem_type not in {
        "binary_classification",
        "multiclass_classification",
    }:
        raise TargetContractError(
            "problem_type must be 'binary_classification' or "
            "'multiclass_classification'."
        )

    expected_count = 2 if problem_type == "binary_classification" else 3
    if problem_type == "binary_classification" and len(classes) != 2:
        raise TargetContractError(
            "binary_classification requires exactly two declared classes."
        )
    if problem_type == "multiclass_classification" and len(classes) < expected_count:
        raise TargetContractError(
            "multiclass_classification requires at least three declared classes."
        )


def _validate_source_target_role(
    variables_file: str | Path,
    *,
    target: str,
) -> str:
    metadata = _read_source_target_metadata(
        variables_file,
        target=target,
    )
    role = metadata["role"]
    if role.lower() != "target":
        raise TargetContractError(
            f"Source metadata does not declare {target!r} as Target; "
            f"observed role={role!r}."
        )
    return role


def _read_source_target_metadata(
    variables_file: str | Path,
    *,
    target: str,
) -> dict[str, str]:
    path = Path(variables_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Source variables file not found: {path.name!r}"
        )

    try:
        variables = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TargetContractError(
            "Could not read the source variables metadata table."
        ) from exc

    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in variables.columns
    }
    name_column = normalized_columns.get("name")
    role_column = normalized_columns.get("role")

    if name_column is None or role_column is None:
        raise TargetContractError(
            "Source variables metadata must contain 'name' and 'role' columns."
        )

    names = variables[name_column].astype("string").str.strip()
    matches = variables.loc[names.eq(target)]
    if len(matches) != 1:
        raise TargetContractError(
            "Source variables metadata must describe the target exactly once: "
            f"{target!r}."
        )

    row = matches.iloc[0]

    def source_value(column_name: str) -> str:
        source_column = normalized_columns.get(column_name)
        if source_column is None or pd.isna(row[source_column]):
            return ""
        return str(row[source_column]).strip()

    return {
        "role": source_value("role"),
        "type": source_value("type"),
        "unit": source_value("units"),
    }


def _normalize_expected_classes(
    values: Sequence[object],
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "expected_classes must be a sequence of class values, not a string."
        )

    normalized = tuple(values)
    if not normalized:
        raise TargetContractError("expected_classes must not be empty.")

    for value in normalized:
        if _is_missing_scalar(value):
            raise TargetContractError(
                "expected_classes must not contain missing values."
            )

    for index, value in enumerate(normalized):
        if _contains_value(normalized[:index], value):
            raise TargetContractError(
                f"expected_classes contains a duplicate value: {value!r}."
            )

    return normalized


def _normalize_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise TargetContractError(f"{field} must not be empty.")
    return normalized


def _normalize_optional_text(
    value: object | None,
    *,
    field: str,
) -> str | None:
    if value is None:
        return None
    return _normalize_text(value, field=field)


def _require_unique_columns(dataframe: pd.DataFrame) -> None:
    duplicated = dataframe.columns[
        dataframe.columns.duplicated(keep=False)
    ].tolist()
    if duplicated:
        raise TargetContractError(
            "DataFrame contains duplicated column labels: "
            + ", ".join(repr(value) for value in duplicated)
            + "."
        )


def _contains_value(values: Sequence[object], candidate: object) -> bool:
    return any(_values_equal(value, candidate) for value in values)


def _values_equal(left: object, right: object) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if not pd.api.types.is_scalar(result):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _is_missing_scalar(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if not pd.api.types.is_scalar(result):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False
