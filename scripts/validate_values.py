"""Reusable, non-mutating validation of missing and invalid values."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Final, TypeAlias

import pandas as pd


RuleSpec: TypeAlias = Mapping[str, object]

_ALLOWED_RULE_FIELDS: Final[set[str]] = {
    "required",
    "allow_blank",
    "allowed_values",
    "numeric",
    "integer",
    "finite",
    "minimum",
    "maximum",
    "strip_strings",
    "case_sensitive",
}

_COLUMN_FRAME_COLUMNS: Final[list[str]] = [
    "Column",
    "Row count",
    "Missing count",
    "Blank count",
    "Inconsistent count",
    "Invalid count",
    "Affected count",
    "Affected percent",
    "Issue types",
    "Status",
]

_ISSUE_FRAME_COLUMNS: Final[list[str]] = [
    "Column",
    "Issue",
    "Raw value",
    "Count",
]

_SOURCE_NUMERIC_TYPES: Final[set[str]] = {
    "integer",
    "continuous",
    "real",
    "numeric",
}

_SOURCE_INTEGER_TYPES: Final[set[str]] = {"integer"}

_SOURCE_TEXT_TYPES: Final[set[str]] = {
    "categorical",
    "binary",
    "string",
}

_ISSUE_IMPACTS: Final[dict[str, str]] = {
    "Missing value": (
        "May reduce the usable sample or require an explicit missing-value "
        "strategy during data preparation."
    ),
    "Blank value": (
        "Represents hidden missingness and may prevent reliable parsing or "
        "completeness assessment."
    ),
    "Inconsistent text": (
        "May fragment one semantic category into multiple encoded levels."
    ),
    "Unexpected value": (
        "May violate the declared target or categorical value contract."
    ),
    "Non-numeric value": (
        "Prevents reliable quantitative analysis and numeric model input."
    ),
    "Non-integer value": (
        "Violates a source-declared integer field and may indicate malformed data."
    ),
    "Non-finite value": (
        "Cannot be treated as an ordinary finite measurement and may destabilize "
        "statistics or model fitting."
    ),
    "Below minimum": (
        "Violates the declared lower bound and may distort downstream analysis."
    ),
    "Above maximum": (
        "Violates the declared upper bound and may distort downstream analysis."
    ),
}


class ValueValidationError(ValueError):
    """Raised when declared value-quality expectations are not satisfied."""


@dataclass(frozen=True, slots=True)
class ValueQualityReport:
    """Summarize missing, blank, inconsistent, and invalid values."""

    checks: pd.DataFrame
    issues: pd.DataFrame

    @property
    def unassessed_columns(self) -> tuple[str, ...]:
        """Return dataset columns without a declared validation rule."""
        return self._columns_with_status("Not assessed")

    @property
    def missing_rule_columns(self) -> tuple[str, ...]:
        """Return rule columns that are absent from the dataset."""
        return self._columns_with_status("Missing column")

    @property
    def affected_columns(self) -> tuple[str, ...]:
        """Return assessed dataset columns containing quality issues."""
        selected = self.checks.loc[
            self.checks["Status"] == "Review required",
            "Column",
        ]
        return tuple(str(column) for column in selected)

    @property
    def has_missing_values(self) -> bool:
        """Return whether assessed columns contain missing values."""
        return self._has_positive_count("Missing count")

    @property
    def has_blank_values(self) -> bool:
        """Return whether assessed columns contain blank text values."""
        return self._has_positive_count("Blank count")

    @property
    def has_inconsistent_values(self) -> bool:
        """Return whether assessed columns contain text inconsistencies."""
        return self._has_positive_count("Inconsistent count")

    @property
    def has_invalid_values(self) -> bool:
        """Return whether assessed columns contain invalid values."""
        return self._has_positive_count("Invalid count")

    @property
    def has_issues(self) -> bool:
        """Return whether any assessed column requires review."""
        return bool(self.affected_columns)

    @property
    def is_fully_assessed(self) -> bool:
        """Return whether rules and dataset columns align exactly."""
        return not self.unassessed_columns and not self.missing_rule_columns

    def column_frame(self) -> pd.DataFrame:
        """Return a copy of the per-column value-quality table."""
        return self.checks.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return a copy of issue samples and their frequencies."""
        return self.issues.copy(deep=True)

    def issues_with_impacts_frame(self) -> pd.DataFrame:
        """Return issue samples enriched with generic analytical impacts."""
        frame = self.issues.copy(deep=True)
        frame["Potential analytical impact"] = frame["Issue"].map(
            _ISSUE_IMPACTS
        )
        return frame

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts grouped by validation status."""
        status_order = [
            "Valid",
            "Review required",
            "Not assessed",
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
        require_all_columns_assessed: bool = True,
        require_rule_columns_present: bool = True,
        require_no_quality_issues: bool = True,
    ) -> None:
        """Raise one consolidated error for selected expectations.

        Exploratory notebooks should normally set
        ``require_no_quality_issues=False``. This validates rule coverage while
        retaining discovered value problems as evidence for data preparation.
        """
        failures: list[str] = []

        if require_all_columns_assessed and self.unassessed_columns:
            failures.append(
                "dataset columns without value validation rules: "
                + ", ".join(self.unassessed_columns)
            )

        if require_rule_columns_present and self.missing_rule_columns:
            failures.append(
                "value validation rule columns missing from the dataset: "
                + ", ".join(self.missing_rule_columns)
            )

        if require_no_quality_issues and self.affected_columns:
            failures.append(
                "columns requiring value-quality review: "
                + ", ".join(self.affected_columns)
            )

        if failures:
            raise ValueValidationError("; ".join(failures) + ".")

    def _columns_with_status(self, status: str) -> tuple[str, ...]:
        selected = self.checks.loc[
            self.checks["Status"] == status,
            "Column",
        ]
        return tuple(str(column) for column in selected)

    def _has_positive_count(self, column: str) -> bool:
        observed = pd.to_numeric(
            self.checks[column],
            errors="coerce",
        ).fillna(0)
        return bool((observed > 0).any())


def analyze_source_backed_missing_and_invalid_values(
    dataframe: pd.DataFrame,
    *,
    source_variables_file: str | Path,
    target: str,
    expected_target_values: Sequence[object] | None = None,
    max_issue_samples: int = 10,
) -> ValueQualityReport:
    """Analyze value quality using source types and the target contract.

    The source-variable table provides structural type semantics. A finite
    ``expected_target_values`` domain may additionally be supplied for
    categorical targets. Continuous regression targets omit that argument and
    are validated through their source-declared numeric type instead. Numeric
    range/domain rules remain the responsibility of the preceding
    domain-validation stage.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_column_name(target)
    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    target_values: tuple[object, ...] | None
    if expected_target_values is None:
        target_values = None
    else:
        target_values = tuple(expected_target_values)
        if not target_values:
            raise ValueValidationError(
                "expected_target_values must contain at least one target value "
                "when supplied."
            )

    source_rows = _load_source_variable_rows(source_variables_file)
    dataset_columns = tuple(str(column) for column in dataframe.columns)
    _validate_source_variable_coverage(dataset_columns, source_rows)

    rules: dict[str, dict[str, object]] = {}
    for column in dataset_columns:
        source_type = source_rows[column].casefold()
        rule: dict[str, object] = {
            "required": True,
            "allow_blank": False,
        }

        if source_type in _SOURCE_NUMERIC_TYPES:
            rule["numeric"] = True
            rule["finite"] = True
            if source_type in _SOURCE_INTEGER_TYPES:
                rule["integer"] = True
        elif source_type not in _SOURCE_TEXT_TYPES:
            raise ValueValidationError(
                f"Unsupported source variable type for {column!r}: "
                f"{source_rows[column]!r}."
            )

        if column == target_name and target_values is not None:
            rule.update(
                {
                    "allowed_values": target_values,
                    "strip_strings": True,
                    "case_sensitive": True,
                }
            )

        rules[column] = rule

    return analyze_missing_and_invalid_values(
        dataframe,
        rules,
        max_issue_samples=max_issue_samples,
    )


def analyze_missing_and_invalid_values(
    dataframe: pd.DataFrame,
    rules: Mapping[str, RuleSpec],
    *,
    max_issue_samples: int = 10,
) -> ValueQualityReport:
    """Analyze missing and invalid values without changing the DataFrame.

    Rules are declared per column and may contain:

    - ``required``: missing values are issues when true;
    - ``allow_blank``: blank strings are accepted when true;
    - ``allowed_values``: finite domain of accepted values;
    - ``numeric``: non-blank values must be numerically parseable;
    - ``integer``: numerically parseable values must be whole numbers;
    - ``finite``: numeric values must exclude positive/negative infinity;
    - ``minimum`` and ``maximum``: inclusive numeric limits;
    - ``strip_strings``: trimmed variants are classified as inconsistent;
    - ``case_sensitive``: case variants are classified as inconsistent when
      false instead of being reported as unexpected values.

    Dataset columns retain their original order. Rule columns absent from the
    DataFrame are appended in declaration order. Issue samples are deterministic
    and capped per issue type and column.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if isinstance(max_issue_samples, bool) or not isinstance(
        max_issue_samples,
        int,
    ):
        raise TypeError("max_issue_samples must be an integer.")
    if max_issue_samples < 1:
        raise ValueError("max_issue_samples must be at least 1.")

    normalized_rules = _normalize_rules(rules)
    observed_columns = tuple(str(column) for column in dataframe.columns)
    observed_column_set = set(observed_columns)

    check_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []

    for column in observed_columns:
        rule = normalized_rules.get(column)

        if rule is None:
            check_rows.append(
                _empty_check_row(
                    column,
                    row_count=len(dataframe),
                    status="Not assessed",
                )
            )
            continue

        column_check, column_issues = _analyze_column(
            dataframe[column],
            column=column,
            rule=rule,
            max_issue_samples=max_issue_samples,
        )
        check_rows.append(column_check)
        issue_rows.extend(column_issues)

    for column in normalized_rules:
        if column in observed_column_set:
            continue
        check_rows.append(
            _empty_check_row(
                column,
                row_count=0,
                status="Missing column",
            )
        )

    checks = pd.DataFrame(check_rows, columns=_COLUMN_FRAME_COLUMNS)
    issues = pd.DataFrame(issue_rows, columns=_ISSUE_FRAME_COLUMNS)

    return ValueQualityReport(checks=checks, issues=issues)


def _analyze_column(
    series: pd.Series,
    *,
    column: str,
    rule: dict[str, object],
    max_issue_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    missing_mask = series.isna()
    blank_mask = series.map(_is_blank_text).fillna(False).astype(bool)
    blank_mask &= ~missing_mask
    candidate_mask = ~(missing_mask | blank_mask)

    inconsistent_mask = pd.Series(False, index=series.index, dtype=bool)
    invalid_masks: list[tuple[str, pd.Series]] = []

    if rule["required"]:
        invalid_masks.append(("Missing value", missing_mask))

    if not rule["allow_blank"]:
        invalid_masks.append(("Blank value", blank_mask))

    allowed_values = rule["allowed_values"]
    if allowed_values is not None:
        exact_allowed_mask = series.isin(allowed_values) & candidate_mask
        domain_candidate_mask = candidate_mask & ~exact_allowed_mask

        if _can_normalize_text_domain(series, allowed_values):
            normalized_allowed = {
                _normalize_text_value(
                    str(value),
                    strip_strings=bool(rule["strip_strings"]),
                    case_sensitive=bool(rule["case_sensitive"]),
                )
                for value in allowed_values
            }
            normalized_matches = series.map(
                lambda value: (
                    isinstance(value, str)
                    and _normalize_text_value(
                        value,
                        strip_strings=bool(rule["strip_strings"]),
                        case_sensitive=bool(rule["case_sensitive"]),
                    )
                    in normalized_allowed
                )
            ).fillna(False).astype(bool)
            inconsistent_mask = domain_candidate_mask & normalized_matches

        unexpected_mask = domain_candidate_mask & ~inconsistent_mask
        invalid_masks.append(("Unexpected value", unexpected_mask))

    numeric_values: pd.Series | None = None
    if rule["numeric"]:
        numeric_values = pd.to_numeric(
            series.where(candidate_mask),
            errors="coerce",
        )
        non_numeric_mask = candidate_mask & numeric_values.isna()
        invalid_masks.append(("Non-numeric value", non_numeric_mask))

        valid_numeric_mask = candidate_mask & numeric_values.notna()

        if rule["finite"]:
            finite_mask = numeric_values.map(
                lambda value: bool(pd.notna(value) and isfinite(float(value)))
            )
            non_finite_mask = valid_numeric_mask & ~finite_mask
            invalid_masks.append(("Non-finite value", non_finite_mask))

        if rule["integer"]:
            non_integer_mask = valid_numeric_mask & (
                numeric_values.mod(1).abs() > 1e-12
            )
            invalid_masks.append(("Non-integer value", non_integer_mask))

        minimum = rule["minimum"]
        if minimum is not None:
            below_minimum_mask = valid_numeric_mask & (
                numeric_values < minimum
            )
            invalid_masks.append(("Below minimum", below_minimum_mask))

        maximum = rule["maximum"]
        if maximum is not None:
            above_maximum_mask = valid_numeric_mask & (
                numeric_values > maximum
            )
            invalid_masks.append(("Above maximum", above_maximum_mask))

    invalid_union = pd.Series(False, index=series.index, dtype=bool)
    issue_types: list[str] = []
    issue_rows: list[dict[str, object]] = []

    if bool(inconsistent_mask.any()):
        issue_types.append("Inconsistent text")
        issue_rows.extend(
            _issue_value_rows(
                series,
                inconsistent_mask,
                column=column,
                issue="Inconsistent text",
                max_issue_samples=max_issue_samples,
            )
        )

    for issue, mask in invalid_masks:
        normalized_mask = mask.fillna(False).astype(bool)
        if not bool(normalized_mask.any()):
            continue

        invalid_union |= normalized_mask
        issue_types.append(issue)
        issue_rows.extend(
            _issue_value_rows(
                series,
                normalized_mask,
                column=column,
                issue=issue,
                max_issue_samples=max_issue_samples,
            )
        )

    affected_mask = invalid_union | inconsistent_mask
    affected_count = int(affected_mask.sum())
    row_count = int(len(series))
    affected_percent = (
        round((affected_count / row_count) * 100, 4)
        if row_count
        else 0.0
    )

    return (
        {
            "Column": column,
            "Row count": row_count,
            "Missing count": int(missing_mask.sum()),
            "Blank count": int(blank_mask.sum()),
            "Inconsistent count": int(inconsistent_mask.sum()),
            "Invalid count": int(invalid_union.sum()),
            "Affected count": affected_count,
            "Affected percent": affected_percent,
            "Issue types": ", ".join(issue_types),
            "Status": "Review required" if affected_count else "Valid",
        },
        issue_rows,
    )


def _empty_check_row(
    column: str,
    *,
    row_count: int,
    status: str,
) -> dict[str, object]:
    return {
        "Column": column,
        "Row count": row_count,
        "Missing count": 0,
        "Blank count": 0,
        "Inconsistent count": 0,
        "Invalid count": 0,
        "Affected count": 0,
        "Affected percent": 0.0,
        "Issue types": "",
        "Status": status,
    }


def _issue_value_rows(
    series: pd.Series,
    mask: pd.Series,
    *,
    column: str,
    issue: str,
    max_issue_samples: int,
) -> list[dict[str, object]]:
    selected = series.loc[mask]

    if issue == "Missing value":
        return [
            {
                "Column": column,
                "Issue": issue,
                "Raw value": "<missing>",
                "Count": int(len(selected)),
            }
        ]

    counts: dict[str, int] = {}
    for value in selected.tolist():
        display_value = _display_raw_value(value)
        counts[display_value] = counts.get(display_value, 0) + 1

    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:max_issue_samples]

    return [
        {
            "Column": column,
            "Issue": issue,
            "Raw value": raw_value,
            "Count": count,
        }
        for raw_value, count in ordered
    ]


def _load_source_variable_rows(
    source_variables_file: str | Path,
) -> dict[str, str]:
    path = Path(source_variables_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Source variables metadata file not found: {path}"
        )

    try:
        variables = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueValidationError(
            "Could not read the source variables metadata table."
        ) from exc

    normalized_columns = {
        str(column).strip().casefold(): str(column)
        for column in variables.columns
    }
    name_column = normalized_columns.get("name")
    type_column = normalized_columns.get("type")
    if name_column is None or type_column is None:
        raise ValueValidationError(
            "Source variables metadata must contain 'name' and 'type' columns."
        )

    rows: dict[str, str] = {}
    for _, row in variables.iterrows():
        name = _normalize_column_name(str(row[name_column]))
        if name in rows:
            raise ValueValidationError(
                f"Source variables metadata describes {name!r} more than once."
            )
        source_type = str(row[type_column]).strip()
        if not source_type or source_type.casefold() == "nan":
            raise ValueValidationError(
                f"Source variable type is missing for {name!r}."
            )
        rows[name] = source_type

    return rows


def _validate_source_variable_coverage(
    dataset_columns: tuple[str, ...],
    source_rows: Mapping[str, str],
) -> None:
    if len(set(dataset_columns)) != len(dataset_columns):
        raise ValueValidationError(
            "Dataset columns must be unique before value-quality analysis."
        )

    dataset_set = set(dataset_columns)
    source_set = set(source_rows)
    missing = [column for column in dataset_columns if column not in source_set]
    extra = [column for column in source_rows if column not in dataset_set]

    failures: list[str] = []
    if missing:
        failures.append(
            "dataset columns missing from source variable metadata: "
            + ", ".join(missing)
        )
    if extra:
        failures.append(
            "source variable metadata contains columns absent from dataset: "
            + ", ".join(extra)
        )
    if failures:
        raise ValueValidationError("; ".join(failures) + ".")


def _normalize_rules(
    rules: Mapping[str, RuleSpec],
) -> dict[str, dict[str, object]]:
    if not isinstance(rules, Mapping):
        raise TypeError("rules must be a mapping by column name.")

    normalized: dict[str, dict[str, object]] = {}

    for raw_column, raw_rule in rules.items():
        column = _normalize_column_name(raw_column)
        if column in normalized:
            raise ValueError(
                "Duplicate validation rule column after normalization: "
                f"{column}"
            )
        if not isinstance(raw_rule, Mapping):
            raise TypeError(
                f"Validation rule for '{column}' must be a mapping."
            )

        unexpected_fields = sorted(
            str(field)
            for field in raw_rule
            if field not in _ALLOWED_RULE_FIELDS
        )
        if unexpected_fields:
            raise ValueError(
                f"Unsupported validation rule field(s) for '{column}': "
                + ", ".join(unexpected_fields)
            )

        required = _normalize_boolean(
            raw_rule.get("required", False),
            field="required",
            column=column,
        )
        allow_blank = _normalize_boolean(
            raw_rule.get("allow_blank", False),
            field="allow_blank",
            column=column,
        )
        numeric = _normalize_boolean(
            raw_rule.get("numeric", False),
            field="numeric",
            column=column,
        )
        integer = _normalize_boolean(
            raw_rule.get("integer", False),
            field="integer",
            column=column,
        )
        finite = _normalize_boolean(
            raw_rule.get("finite", False),
            field="finite",
            column=column,
        )
        strip_strings = _normalize_boolean(
            raw_rule.get("strip_strings", False),
            field="strip_strings",
            column=column,
        )
        case_sensitive = _normalize_boolean(
            raw_rule.get("case_sensitive", True),
            field="case_sensitive",
            column=column,
        )

        allowed_values = _normalize_allowed_values(
            raw_rule.get("allowed_values"),
            column=column,
        )
        minimum = _normalize_numeric_limit(
            raw_rule.get("minimum"),
            field="minimum",
            column=column,
        )
        maximum = _normalize_numeric_limit(
            raw_rule.get("maximum"),
            field="maximum",
            column=column,
        )

        if integer:
            numeric = True
        if finite and not numeric:
            raise ValueError(
                f"Validation rule for '{column}' enables 'finite' "
                "without enabling 'numeric'."
            )
        if (minimum is not None or maximum is not None) and not numeric:
            raise ValueError(
                f"Validation rule for '{column}' defines numeric limits "
                "without enabling 'numeric'."
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"Validation rule for '{column}' has minimum greater "
                "than maximum."
            )

        normalized[column] = {
            "required": required,
            "allow_blank": allow_blank,
            "allowed_values": allowed_values,
            "numeric": numeric,
            "integer": integer,
            "finite": finite,
            "minimum": minimum,
            "maximum": maximum,
            "strip_strings": strip_strings,
            "case_sensitive": case_sensitive,
        }

    return normalized


def _normalize_column_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("validation rule column names must be strings.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("validation rule column names cannot be empty.")
    return normalized


def _normalize_boolean(
    value: object,
    *,
    field: str,
    column: str,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"Validation rule field '{field}' for '{column}' "
            "must be a boolean."
        )
    return value


def _normalize_allowed_values(
    value: object,
    *,
    column: str,
) -> tuple[object, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise TypeError(
            f"Validation rule field 'allowed_values' for '{column}' "
            "must be a non-string collection."
        )

    normalized = tuple(value)
    if not normalized:
        raise ValueError(
            f"Validation rule field 'allowed_values' for '{column}' "
            "cannot be empty."
        )
    return normalized


def _normalize_numeric_limit(
    value: object,
    *,
    field: str,
    column: str,
) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"Validation rule field '{field}' for '{column}' "
            "must be numeric or None."
        )
    return value


def _is_blank_text(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _can_normalize_text_domain(
    series: pd.Series,
    allowed_values: tuple[object, ...],
) -> bool:
    del series
    return all(isinstance(value, str) for value in allowed_values)


def _normalize_text_value(
    value: str,
    *,
    strip_strings: bool,
    case_sensitive: bool,
) -> str:
    normalized = value.strip() if strip_strings else value
    return normalized if case_sensitive else normalized.casefold()


def _display_raw_value(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return repr(value)
