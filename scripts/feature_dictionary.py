"""Feature-role and source data-dictionary reporting for tabular studies.

Study-specific target and identifier declarations stay in the notebook. This
module validates those analytical roles against a materialized source-variable
table and renders the source descriptions without duplicating them in notebook
code.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd


_REQUIRED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "role",
    "description",
    "units",
)

_COLUMN_OUTPUT: Final[tuple[str, ...]] = (
    "Column",
    "Analytical role",
    "Source role",
    "Description",
    "Unit",
)


class FeatureDictionaryError(ValueError):
    """Raised when source metadata and declared analytical roles disagree."""


@dataclass(frozen=True, slots=True)
class FeatureDictionaryReport:
    """Validated feature roles and source-backed variable documentation."""

    target_column: str
    feature_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    checks: pd.DataFrame

    @property
    def documented_column_count(self) -> int:
        """Return the number of columns with a non-empty source description."""
        descriptions = self.checks["Description"]
        return int((descriptions != "Not documented by source").sum())

    @property
    def column_count(self) -> int:
        """Return the number of dataset columns covered by source metadata."""
        return len(self.checks)

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact notebook-friendly role/documentation summary."""
        return pd.DataFrame(
            {
                "Metric": [
                    "Candidate features",
                    "Identifiers",
                    "Targets",
                    "Columns covered by source metadata",
                    "Columns with source descriptions",
                ],
                "Value": [
                    len(self.feature_columns),
                    len(self.identifier_columns),
                    1,
                    self.column_count,
                    self.documented_column_count,
                ],
            }
        )

    def column_frame(self) -> pd.DataFrame:
        """Return the source-backed per-column data dictionary."""
        return self.checks.copy(deep=True)


def define_feature_roles_and_dictionary(
    dataframe: pd.DataFrame,
    *,
    source_variables_file: str | Path,
    target: str,
    identifiers: Collection[str] | str | None = None,
) -> FeatureDictionaryReport:
    """Validate analytical roles and build a source-backed data dictionary.

    The source-variable table is expected to use the columns materialized by
    ``ucimlrepo`` (``name``, ``role``, ``description``, and ``units`` among
    others). Data types, value ranges, and missing-value analysis are
    intentionally left to later notebook stages.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    dataset_columns = _normalize_dataset_columns(dataframe)
    target_column = _normalize_column_name(target, label="target")
    identifier_columns = _normalize_identifiers(identifiers)

    if target_column not in dataset_columns:
        raise KeyError(f"Target column not found: {target_column!r}")

    missing_identifiers = [
        column for column in identifier_columns if column not in dataset_columns
    ]
    if missing_identifiers:
        raise KeyError(
            "Identifier column(s) not found: " + ", ".join(missing_identifiers)
        )

    if target_column in identifier_columns:
        raise FeatureDictionaryError(
            "The target column cannot also be an identifier."
        )

    variables = _read_source_variables(source_variables_file)
    source_rows = _index_source_rows(variables)

    dataset_set = set(dataset_columns)
    source_set = set(source_rows)
    missing_from_source = [
        column for column in dataset_columns if column not in source_set
    ]
    extra_in_source = [
        column for column in source_rows if column not in dataset_set
    ]

    failures: list[str] = []
    if missing_from_source:
        failures.append(
            "dataset columns missing from source variable metadata: "
            + ", ".join(missing_from_source)
        )
    if extra_in_source:
        failures.append(
            "source variable metadata contains columns absent from dataset: "
            + ", ".join(extra_in_source)
        )
    if failures:
        raise FeatureDictionaryError("; ".join(failures) + ".")

    identifier_set = set(identifier_columns)
    feature_columns: list[str] = []
    rows: list[dict[str, str]] = []
    role_failures: list[str] = []

    for column in dataset_columns:
        source_row = source_rows[column]
        source_role = _normalize_source_role(source_row["role"])

        if column == target_column:
            analytical_role = "Target"
            expected_source_roles = {"target"}
        elif column in identifier_set:
            analytical_role = "Identifier"
            expected_source_roles = {"id", "identifier"}
        else:
            analytical_role = "Candidate feature"
            expected_source_roles = {"feature"}
            feature_columns.append(column)

        if source_role.casefold() not in expected_source_roles:
            role_failures.append(
                f"{column!r} is declared as {analytical_role!r} but source "
                f"metadata role is {source_role!r}"
            )

        rows.append(
            {
                "Column": column,
                "Analytical role": analytical_role,
                "Source role": source_role,
                "Description": _display_optional_text(
                    source_row["description"],
                    fallback="Not documented by source",
                ),
                "Unit": _display_optional_text(
                    source_row["units"],
                    fallback="Not specified",
                ),
            }
        )

    if role_failures:
        raise FeatureDictionaryError(
            "Analytical/source role validation failed: "
            + "; ".join(role_failures)
            + "."
        )

    checks = pd.DataFrame(rows, columns=list(_COLUMN_OUTPUT))

    return FeatureDictionaryReport(
        target_column=target_column,
        feature_columns=tuple(feature_columns),
        identifier_columns=identifier_columns,
        checks=checks,
    )


def _normalize_dataset_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(str(column) for column in dataframe.columns)
    if len(set(columns)) != len(columns):
        raise FeatureDictionaryError(
            "Dataset columns must be unique before feature roles are defined."
        )
    return columns


def _normalize_column_name(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    return normalized


def _normalize_identifiers(
    identifiers: Collection[str] | str | None,
) -> tuple[str, ...]:
    if identifiers is None:
        return ()
    if isinstance(identifiers, str):
        raw_values = [identifiers]
    elif isinstance(identifiers, Collection):
        raw_values = list(identifiers)
    else:
        raise TypeError(
            "identifiers must be None, a string, or a collection of strings."
        )

    normalized = tuple(
        _normalize_column_name(value, label="identifier")
        for value in raw_values
    )
    if len(set(normalized)) != len(normalized):
        raise FeatureDictionaryError("Identifier columns must be unique.")
    return normalized


def _read_source_variables(source_variables_file: str | Path) -> pd.DataFrame:
    path = Path(source_variables_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source variable metadata file not found: {path}")

    variables = pd.read_csv(path)
    missing_columns = [
        column for column in _REQUIRED_SOURCE_COLUMNS if column not in variables.columns
    ]
    if missing_columns:
        raise FeatureDictionaryError(
            "Source variable metadata is missing required column(s): "
            + ", ".join(missing_columns)
            + "."
        )
    return variables


def _index_source_rows(variables: pd.DataFrame) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in variables.to_dict(orient="records"):
        column = _normalize_column_name(row["name"], label="source variable name")
        if column in indexed:
            raise FeatureDictionaryError(
                f"Duplicate source variable metadata row for {column!r}."
            )
        indexed[column] = row
    return indexed


def _normalize_source_role(value: object) -> str:
    if pd.isna(value):
        raise FeatureDictionaryError("Source variable role cannot be missing.")
    normalized = str(value).strip()
    if not normalized:
        raise FeatureDictionaryError("Source variable role cannot be empty.")
    return normalized


def _display_optional_text(value: object, *, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    normalized = str(value).strip()
    return normalized or fallback
