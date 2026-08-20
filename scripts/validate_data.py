"""Reusable, non-mutating validation helpers for tabular datasets."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_dtype_equal,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    pandas_dtype,
)


ExpectedTypeSpec: TypeAlias = str | Collection[str]

_TYPE_FAMILY_ALIASES: Final[dict[str, str]] = {
    "any": "any",
    "bool": "boolean",
    "boolean": "boolean",
    "category": "categorical",
    "categorical": "categorical",
    "complex": "complex",
    "date": "datetime",
    "datetime": "datetime",
    "float": "floating",
    "floating": "floating",
    "int": "integer",
    "integer": "integer",
    "number": "numeric",
    "numeric": "numeric",
    "object": "object",
    "str": "string",
    "string": "string",
    "text": "string",
    "timedelta": "timedelta",
}


class DataValidationError(ValueError):
    """Raised when a declared dataset expectation is not satisfied."""


@dataclass(frozen=True, slots=True)
class ObservationUnitReport:
    """Summarize row identity evidence for a tabular dataset."""

    identifier: str | None
    row_count: int
    non_null_identifier_count: int | None
    unique_identifier_count: int | None
    missing_identifier_count: int | None
    duplicate_identifier_count: int | None
    duplicated_row_count: int | None
    duplicated_rows: pd.DataFrame

    @property
    def has_source_identifier(self) -> bool:
        """Return whether the source dataset provides an observation key."""
        return self.identifier is not None

    @property
    def has_missing_identifiers(self) -> bool | None:
        """Return identifier completeness status when it can be assessed."""
        if self.missing_identifier_count is None:
            return None
        return self.missing_identifier_count > 0

    @property
    def has_duplicates(self) -> bool | None:
        """Return identifier duplication status when it can be assessed."""
        if self.duplicate_identifier_count is None:
            return None
        return self.duplicate_identifier_count > 0

    @property
    def is_complete(self) -> bool | None:
        """Return identifier completeness, or ``None`` when unavailable."""
        if self.has_missing_identifiers is None:
            return None
        return not self.has_missing_identifiers

    @property
    def is_unique(self) -> bool | None:
        """Return identifier uniqueness, or ``None`` when unavailable."""
        if self.has_duplicates is None:
            return None
        return not self.has_duplicates

    def summary_frame(self) -> pd.DataFrame:
        """Return a notebook-friendly observation identity summary."""
        if not self.has_source_identifier:
            return pd.DataFrame(
                {
                    "Metric": [
                        "Rows",
                        "Source identifier",
                        "Identifier completeness",
                        "Identifier uniqueness",
                    ],
                    "Value": [
                        self.row_count,
                        "Not provided",
                        "Not assessable",
                        "Not assessable",
                    ],
                }
            )

        return pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Non-null identifiers",
                    "Unique identifiers",
                    "Missing identifiers",
                    "Duplicate identifiers",
                    "Rows involved in duplication",
                ],
                "Value": [
                    self.row_count,
                    self.non_null_identifier_count,
                    self.unique_identifier_count,
                    self.missing_identifier_count,
                    self.duplicate_identifier_count,
                    self.duplicated_row_count,
                ],
            }
        )

    def raise_if_invalid(
        self,
        *,
        require_complete: bool = True,
        require_unique: bool = True,
    ) -> None:
        """Raise one consolidated error for violated identity expectations."""
        if (
            not self.has_source_identifier
            and (require_complete or require_unique)
        ):
            raise DataValidationError(
                "The source dataset does not provide an observation "
                "identifier, so identifier completeness and uniqueness "
                "cannot be validated."
            )

        failures: list[str] = []

        if require_complete and self.has_missing_identifiers:
            failures.append(
                f"'{self.identifier}' contains "
                f"{self.missing_identifier_count} missing value(s)"
            )

        if require_unique and self.has_duplicates:
            failures.append(
                f"'{self.identifier}' contains "
                f"{self.duplicate_identifier_count} duplicated "
                "identifier value(s) across "
                f"{self.duplicated_row_count} row(s)"
            )

        if failures:
            raise DataValidationError("; ".join(failures) + ".")


@dataclass(frozen=True, slots=True)
class DataTypeReport:
    """Describe expected and observed data types for dataset columns."""

    checks: pd.DataFrame

    @property
    def mismatched_columns(self) -> tuple[str, ...]:
        """Return columns whose observed type does not meet expectations."""
        return self._columns_with_status("Mismatch")

    @property
    def undeclared_columns(self) -> tuple[str, ...]:
        """Return dataset columns without a declared expected type."""
        return self._columns_with_status("Not declared")

    @property
    def missing_expected_columns(self) -> tuple[str, ...]:
        """Return expected columns that are absent from the dataset."""
        return self._columns_with_status("Missing column")

    @property
    def has_mismatches(self) -> bool:
        """Return whether at least one observed type is incompatible."""
        return bool(self.mismatched_columns)

    @property
    def has_undeclared_columns(self) -> bool:
        """Return whether at least one dataset column lacks an expectation."""
        return bool(self.undeclared_columns)

    @property
    def has_missing_expected_columns(self) -> bool:
        """Return whether an expected column is absent from the dataset."""
        return bool(self.missing_expected_columns)

    @property
    def is_fully_declared(self) -> bool:
        """Return whether dataset and expectation columns align exactly."""
        return not (
            self.has_undeclared_columns
            or self.has_missing_expected_columns
        )

    @property
    def all_observed_types_match(self) -> bool:
        """Return whether every assessed observed type matches."""
        return not self.has_mismatches

    def column_frame(self) -> pd.DataFrame:
        """Return a copy of the per-column type comparison table."""
        return self.checks.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return only mismatches and declaration alignment issues."""
        return self.checks.loc[
            self.checks["Status"] != "Match"
        ].reset_index(drop=True).copy(deep=True)

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts grouped by validation status."""
        status_order = [
            "Match",
            "Mismatch",
            "Not declared",
            "Missing column",
        ]
        counts = self.checks["Status"].value_counts()

        return pd.DataFrame(
            {
                "Status": status_order,
                "Column count": [
                    int(counts.get(status, 0))
                    for status in status_order
                ],
            }
        )

    def raise_if_invalid(
        self,
        *,
        require_all_columns_declared: bool = True,
        require_expected_columns_present: bool = True,
        require_matching_types: bool = True,
    ) -> None:
        """Raise one consolidated error for selected type expectations.

        Type mismatches may be intentionally retained during exploratory work.
        Set ``require_matching_types=False`` to validate only that declarations
        cover the current dataset schema without blocking on discovered type
        issues.
        """
        failures: list[str] = []

        if require_all_columns_declared and self.has_undeclared_columns:
            failures.append(
                "dataset columns without declared expected types: "
                + ", ".join(self.undeclared_columns)
            )

        if (
            require_expected_columns_present
            and self.has_missing_expected_columns
        ):
            failures.append(
                "declared columns missing from the dataset: "
                + ", ".join(self.missing_expected_columns)
            )

        if require_matching_types and self.has_mismatches:
            failures.append(
                "columns with incompatible observed types: "
                + ", ".join(self.mismatched_columns)
            )

        if failures:
            raise DataValidationError("; ".join(failures) + ".")

    def _columns_with_status(self, status: str) -> tuple[str, ...]:
        selected = self.checks.loc[
            self.checks["Status"] == status,
            "Column",
        ]
        return tuple(str(column) for column in selected)


@dataclass(frozen=True, slots=True)
class DataDictionaryReport:
    """Describe semantic documentation coverage for dataset columns."""

    checks: pd.DataFrame

    @property
    def undocumented_columns(self) -> tuple[str, ...]:
        """Return dataset columns without dictionary documentation."""
        return self._columns_with_status("Not documented")

    @property
    def missing_documented_columns(self) -> tuple[str, ...]:
        """Return documented columns that are absent from the dataset."""
        return self._columns_with_status("Missing column")

    @property
    def has_undocumented_columns(self) -> bool:
        """Return whether at least one dataset column is undocumented."""
        return bool(self.undocumented_columns)

    @property
    def has_missing_documented_columns(self) -> bool:
        """Return whether documented columns are absent from the dataset."""
        return bool(self.missing_documented_columns)

    @property
    def is_complete(self) -> bool:
        """Return whether dictionary and dataset columns align exactly."""
        return not (
            self.has_undocumented_columns
            or self.has_missing_documented_columns
        )

    def column_frame(self) -> pd.DataFrame:
        """Return a copy of the per-column semantic documentation table."""
        return self.checks.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return only documentation coverage issues."""
        return self.checks.loc[
            self.checks["Status"] != "Documented"
        ].reset_index(drop=True).copy(deep=True)

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts grouped by documentation status."""
        status_order = [
            "Documented",
            "Not documented",
            "Missing column",
        ]
        counts = self.checks["Status"].value_counts()

        return pd.DataFrame(
            {
                "Status": status_order,
                "Column count": [
                    int(counts.get(status, 0))
                    for status in status_order
                ],
            }
        )

    def raise_if_invalid(
        self,
        *,
        require_all_columns_documented: bool = True,
        require_documented_columns_present: bool = True,
    ) -> None:
        """Raise one consolidated error for documentation coverage issues."""
        failures: list[str] = []

        if (
            require_all_columns_documented
            and self.has_undocumented_columns
        ):
            failures.append(
                "dataset columns without data dictionary entries: "
                + ", ".join(self.undocumented_columns)
            )

        if (
            require_documented_columns_present
            and self.has_missing_documented_columns
        ):
            failures.append(
                "data dictionary columns missing from the dataset: "
                + ", ".join(self.missing_documented_columns)
            )

        if failures:
            raise DataValidationError("; ".join(failures) + ".")

    def _columns_with_status(self, status: str) -> tuple[str, ...]:
        selected = self.checks.loc[
            self.checks["Status"] == status,
            "Column",
        ]
        return tuple(str(column) for column in selected)


def analyze_data_dictionary(
    dataframe: pd.DataFrame,
    dictionary: Mapping[str, Mapping[str, object]],
    *,
    target: str,
    identifiers: Collection[str] = (),
) -> DataDictionaryReport:
    """Validate and render semantic documentation for dataset columns.

    Each dictionary entry must provide a non-empty ``description`` and
    ``expected_values`` field. The optional ``unit`` field may be a string or
    ``None``. Analytical roles are derived from ``target`` and ``identifiers``
    so they do not need to be repeated in dataset-specific documentation.

    The input DataFrame and dictionary are never modified. Dataset columns
    retain their original order. Documented columns absent from the DataFrame
    are appended in dictionary declaration order.
    """
    normalized_target = _normalize_declared_column_name(
        target,
        label="target",
    )
    normalized_identifiers = _normalize_identifier_columns(identifiers)

    if normalized_target in normalized_identifiers:
        raise ValueError(
            "The target column cannot also be an identifier."
        )

    observed_columns = tuple(str(column) for column in dataframe.columns)
    observed_column_set = set(observed_columns)

    missing_role_columns = [
        column
        for column in (normalized_target, *normalized_identifiers)
        if column not in observed_column_set
    ]
    if missing_role_columns:
        raise KeyError(
            "Declared analytical role columns were not found in the "
            "dataset: "
            + ", ".join(missing_role_columns)
        )

    normalized_dictionary = _normalize_data_dictionary(dictionary)
    rows: list[dict[str, object]] = []

    for column in observed_columns:
        entry = normalized_dictionary.get(column)
        role = _analytical_role(
            column,
            target=normalized_target,
            identifiers=normalized_identifiers,
        )

        if entry is None:
            rows.append(
                {
                    "Column": column,
                    "Description": "",
                    "Expected values": "",
                    "Unit": "",
                    "Analytical role": role,
                    "Status": "Not documented",
                }
            )
            continue

        rows.append(
            {
                "Column": column,
                "Description": entry["description"],
                "Expected values": entry["expected_values"],
                "Unit": entry["unit"],
                "Analytical role": role,
                "Status": "Documented",
            }
        )

    for column, entry in normalized_dictionary.items():
        if column in observed_column_set:
            continue

        rows.append(
            {
                "Column": column,
                "Description": entry["description"],
                "Expected values": entry["expected_values"],
                "Unit": entry["unit"],
                "Analytical role": _analytical_role(
                    column,
                    target=normalized_target,
                    identifiers=normalized_identifiers,
                ),
                "Status": "Missing column",
            }
        )

    checks = pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Description",
            "Expected values",
            "Unit",
            "Analytical role",
            "Status",
        ],
    )

    return DataDictionaryReport(checks=checks)


def _normalize_data_dictionary(
    dictionary: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    if not isinstance(dictionary, Mapping):
        raise TypeError("dictionary must be a mapping by column name.")

    normalized: dict[str, dict[str, str]] = {}
    allowed_fields = {"description", "expected_values", "unit"}

    for raw_column, raw_entry in dictionary.items():
        column = _normalize_declared_column_name(
            raw_column,
            label="data dictionary column",
        )

        if column in normalized:
            raise ValueError(
                "Duplicate data dictionary column after normalization: "
                f"{column}"
            )

        if not isinstance(raw_entry, Mapping):
            raise TypeError(
                f"Data dictionary entry for '{column}' must be a mapping."
            )

        unexpected_fields = sorted(
            str(field)
            for field in raw_entry.keys()
            if field not in allowed_fields
        )
        if unexpected_fields:
            raise ValueError(
                f"Unsupported data dictionary field(s) for '{column}': "
                + ", ".join(unexpected_fields)
            )

        if "description" not in raw_entry:
            raise ValueError(
                f"Data dictionary entry for '{column}' must define "
                "'description'."
            )
        if "expected_values" not in raw_entry:
            raise ValueError(
                f"Data dictionary entry for '{column}' must define "
                "'expected_values'."
            )

        description = _normalize_non_empty_text(
            raw_entry["description"],
            field="description",
            column=column,
        )
        expected_values = _format_expected_values(
            raw_entry["expected_values"],
            column=column,
        )
        unit = _normalize_optional_text(
            raw_entry.get("unit"),
            field="unit",
            column=column,
        )

        normalized[column] = {
            "description": description,
            "expected_values": expected_values,
            "unit": unit,
        }

    return normalized


def _normalize_declared_column_name(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")

    return normalized


def _normalize_identifier_columns(
    identifiers: Collection[str],
) -> tuple[str, ...]:
    if isinstance(identifiers, str):
        raw_identifiers = [identifiers]
    elif isinstance(identifiers, Collection):
        raw_identifiers = list(identifiers)
    else:
        raise TypeError(
            "identifiers must be a string or a collection of strings."
        )

    normalized: list[str] = []

    for raw_identifier in raw_identifiers:
        identifier = _normalize_declared_column_name(
            raw_identifier,
            label="identifier column",
        )

        if identifier in normalized:
            raise ValueError(
                f"Duplicate identifier declaration: {identifier}"
            )

        normalized.append(identifier)

    return tuple(normalized)


def _normalize_non_empty_text(
    value: object,
    *,
    field: str,
    column: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"Data dictionary field '{field}' for '{column}' "
            "must be a string."
        )

    normalized = value.strip()
    if not normalized:
        raise ValueError(
            f"Data dictionary field '{field}' for '{column}' "
            "cannot be empty."
        )

    return normalized


def _normalize_optional_text(
    value: object,
    *,
    field: str,
    column: str,
) -> str:
    if value is None:
        return ""

    return _normalize_non_empty_text(
        value,
        field=field,
        column=column,
    )


def _format_expected_values(
    value: object,
    *,
    column: str,
) -> str:
    if isinstance(value, str):
        return _normalize_non_empty_text(
            value,
            field="expected_values",
            column=column,
        )

    if isinstance(value, Collection):
        items = list(value)
        if not items:
            raise ValueError(
                f"Data dictionary field 'expected_values' for "
                f"'{column}' cannot be empty."
            )

        rendered: list[str] = []
        for item in items:
            if item is None:
                rendered.append("null")
                continue

            text = str(item).strip()
            if not text:
                raise ValueError(
                    f"Data dictionary expected values for '{column}' "
                    "cannot contain empty items."
                )
            rendered.append(text)

        return ", ".join(rendered)

    raise TypeError(
        f"Data dictionary field 'expected_values' for '{column}' "
        "must be a string or collection."
    )


def _analytical_role(
    column: str,
    *,
    target: str,
    identifiers: tuple[str, ...],
) -> str:
    if column == target:
        return "Target"
    if column in identifiers:
        return "Identifier"
    return "Candidate feature"


def analyze_observation_unit(
    dataframe: pd.DataFrame,
    identifier: str | None = None,
) -> ObservationUnitReport:
    """Describe row identity with or without a source identifier.

    The input DataFrame is never modified. When ``identifier`` is ``None``,
    the report records that key-based completeness and uniqueness cannot be
    assessed; full-row duplicate analysis remains a separate concern.
    """
    if identifier is None:
        return ObservationUnitReport(
            identifier=None,
            row_count=int(len(dataframe)),
            non_null_identifier_count=None,
            unique_identifier_count=None,
            missing_identifier_count=None,
            duplicate_identifier_count=None,
            duplicated_row_count=None,
            duplicated_rows=dataframe.iloc[0:0].copy(),
        )

    normalized_identifier = identifier.strip()

    if not normalized_identifier:
        raise ValueError("Observation identifier cannot be empty.")

    if normalized_identifier not in dataframe.columns:
        raise KeyError(
            f"Observation identifier not found: {normalized_identifier}"
        )

    identifier_series = dataframe[normalized_identifier]
    non_null_identifiers = identifier_series.dropna()
    identifier_counts = non_null_identifiers.value_counts(dropna=True)
    duplicated_identifier_values = identifier_counts[
        identifier_counts > 1
    ].index

    duplicated_mask = (
        identifier_series.notna()
        & identifier_series.duplicated(keep=False)
    )
    duplicated_rows = (
        dataframe.loc[duplicated_mask]
        .sort_values(normalized_identifier, kind="stable")
        .copy()
    )

    return ObservationUnitReport(
        identifier=normalized_identifier,
        row_count=int(len(dataframe)),
        non_null_identifier_count=int(identifier_series.notna().sum()),
        unique_identifier_count=int(
            non_null_identifiers.nunique(dropna=True)
        ),
        missing_identifier_count=int(identifier_series.isna().sum()),
        duplicate_identifier_count=int(
            len(duplicated_identifier_values)
        ),
        duplicated_row_count=int(len(duplicated_rows)),
        duplicated_rows=duplicated_rows,
    )


def analyze_data_types(
    dataframe: pd.DataFrame,
    expected_types: Mapping[str, ExpectedTypeSpec],
) -> DataTypeReport:
    """Compare declared and observed column data types without mutation.

    Expected values may use semantic type families such as ``"numeric"``,
    ``"integer"``, ``"floating"``, ``"boolean"``, ``"string"``,
    ``"categorical"``, ``"datetime"``, ``"timedelta"``, ``"object"``,
    ``"complex"``, or ``"any"``. Exact pandas dtype strings such as
    ``"int64"``, ``"Int64"``, or ``"datetime64[ns]"`` are also accepted.
    Multiple accepted types may be supplied as a collection of strings.

    Dataset columns retain their original order. Declared columns that are
    absent from the DataFrame are appended to the report in declaration order.
    """
    normalized_expectations = _normalize_type_expectations(expected_types)
    rows: list[dict[str, object]] = []

    for column in dataframe.columns:
        column_name = str(column)
        series = dataframe[column]
        observed_dtype = str(series.dtype)
        observed_type = _infer_observed_type(series)
        expected = normalized_expectations.get(column_name)

        if expected is None:
            expected_display = ""
            status = "Not declared"
        else:
            expected_display = " | ".join(expected)
            status = (
                "Match"
                if any(
                    _series_matches_expected_type(series, candidate)
                    for candidate in expected
                )
                else "Mismatch"
            )

        rows.append(
            {
                "Column": column_name,
                "Expected type": expected_display,
                "Observed dtype": observed_dtype,
                "Observed type": observed_type,
                "Status": status,
            }
        )

    observed_column_names = {str(column) for column in dataframe.columns}

    for column, expected in normalized_expectations.items():
        if column in observed_column_names:
            continue

        rows.append(
            {
                "Column": column,
                "Expected type": " | ".join(expected),
                "Observed dtype": "",
                "Observed type": "",
                "Status": "Missing column",
            }
        )

    checks = pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Expected type",
            "Observed dtype",
            "Observed type",
            "Status",
        ],
    )

    return DataTypeReport(checks=checks)


def _normalize_type_expectations(
    expected_types: Mapping[str, ExpectedTypeSpec],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(expected_types, Mapping):
        raise TypeError("expected_types must be a mapping by column name.")

    normalized: dict[str, tuple[str, ...]] = {}

    for raw_column, raw_specification in expected_types.items():
        if not isinstance(raw_column, str):
            raise TypeError("Expected type column names must be strings.")

        column = raw_column.strip()
        if not column:
            raise ValueError("Expected type column names cannot be empty.")

        if column in normalized:
            raise ValueError(
                f"Duplicate expected type declaration after normalization: "
                f"{column}"
            )

        candidates = _normalize_expected_type_specification(
            raw_specification,
            column=column,
        )
        normalized[column] = candidates

    return normalized


def _normalize_expected_type_specification(
    specification: ExpectedTypeSpec,
    *,
    column: str,
) -> tuple[str, ...]:
    if isinstance(specification, str):
        raw_candidates = [specification]
    elif isinstance(specification, Collection):
        raw_candidates = list(specification)
    else:
        raise TypeError(
            f"Expected type declaration for '{column}' must be a string "
            "or a collection of strings."
        )

    if not raw_candidates:
        raise ValueError(
            f"Expected type declaration for '{column}' cannot be empty."
        )

    normalized: list[str] = []

    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, str):
            raise TypeError(
                f"Expected type candidates for '{column}' must be strings."
            )

        candidate = raw_candidate.strip()
        if not candidate:
            raise ValueError(
                f"Expected type candidates for '{column}' cannot be empty."
            )

        _validate_expected_type_candidate(candidate, column=column)

        if candidate not in normalized:
            normalized.append(candidate)

    return tuple(normalized)


def _validate_expected_type_candidate(
    candidate: str,
    *,
    column: str,
) -> None:
    if candidate.lower() in _TYPE_FAMILY_ALIASES:
        return

    try:
        pandas_dtype(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported expected type '{candidate}' for column "
            f"'{column}'."
        ) from exc


def _infer_observed_type(series: pd.Series) -> str:
    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if is_bool_dtype(dtype):
        return "boolean"
    if is_integer_dtype(dtype):
        return "integer"
    if is_float_dtype(dtype):
        return "floating"
    if is_complex_dtype(dtype):
        return "complex"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    if is_timedelta64_dtype(dtype):
        return "timedelta"
    if is_object_dtype(dtype):
        non_null_values = series.dropna()
        if non_null_values.empty:
            return "object"
        if non_null_values.map(
            lambda value: isinstance(value, str)
        ).all():
            return "string"
        return "object"
    if is_string_dtype(dtype):
        return "string"

    return str(dtype)


def _series_matches_expected_type(
    series: pd.Series,
    expected: str,
) -> bool:
    normalized = expected.lower()
    family = _TYPE_FAMILY_ALIASES.get(normalized)
    dtype = series.dtype
    observed_type = _infer_observed_type(series)

    if family == "any":
        return True
    if family == "boolean":
        return bool(is_bool_dtype(dtype))
    if family == "integer":
        return bool(is_integer_dtype(dtype) and not is_bool_dtype(dtype))
    if family == "floating":
        return bool(is_float_dtype(dtype))
    if family == "numeric":
        return bool(is_numeric_dtype(dtype) and not is_bool_dtype(dtype))
    if family == "complex":
        return bool(is_complex_dtype(dtype))
    if family == "string":
        return observed_type == "string"
    if family == "categorical":
        return isinstance(dtype, pd.CategoricalDtype)
    if family == "datetime":
        return bool(is_datetime64_any_dtype(dtype))
    if family == "timedelta":
        return bool(is_timedelta64_dtype(dtype))
    if family == "object":
        return bool(is_object_dtype(dtype))

    return bool(is_dtype_equal(dtype, pandas_dtype(expected)))
