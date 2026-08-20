"""Source-backed data-type, numeric-range, and domain validation.

The notebook declares study-specific domain assumptions. This module derives
expected semantic types from source-variable metadata, inspects observed ranges,
and evaluates numeric/domain constraints without mutating the raw DataFrame.
Missing-value assessment is intentionally left to the dedicated quality stage.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final

import pandas as pd

from scripts.validate_data import DataTypeReport, analyze_data_types


_SOURCE_TYPE_MAP: Final[dict[str, str]] = {
    "integer": "integer",
    "continuous": "numeric",
    "real": "numeric",
    "numeric": "numeric",
    "categorical": "string",
    "binary": "string",
    "string": "string",
}

_COLUMN_COLUMNS: Final[tuple[str, ...]] = (
    "Column",
    "Source type",
    "Expected type",
    "Observed dtype",
    "Observed minimum",
    "Observed maximum",
    "Domain expectation",
    "Violation count",
    "Status",
)

_RELATION_COLUMNS: Final[tuple[str, ...]] = (
    "Relation",
    "Evaluated rows",
    "Violation count",
    "Status",
)

_ALLOWED_RULE_GROUPS: Final[set[str]] = {
    "positive",
    "non_negative",
    "unit_interval",
    "minimums",
    "maximums",
    "relations",
}

_ALLOWED_RELATION_OPERATORS: Final[set[str]] = {">=", "<=", ">", "<", "=="}


class DomainValidationError(ValueError):
    """Raised when source metadata or declared domain rules are inconsistent."""


@dataclass(frozen=True, slots=True)
class TypesRangesDomainReport:
    """Notebook-friendly report for types, ranges, and domain constraints."""

    type_report: DataTypeReport
    columns: pd.DataFrame
    relations: pd.DataFrame

    @property
    def type_mismatch_columns(self) -> tuple[str, ...]:
        return self.type_report.mismatched_columns

    @property
    def domain_violation_columns(self) -> tuple[str, ...]:
        selected = self.columns.loc[
            self.columns["Violation count"] > 0,
            "Column",
        ]
        return tuple(str(value) for value in selected)

    @property
    def violated_relations(self) -> tuple[str, ...]:
        selected = self.relations.loc[
            self.relations["Violation count"] > 0,
            "Relation",
        ]
        return tuple(str(value) for value in selected)

    @property
    def has_type_mismatches(self) -> bool:
        return bool(self.type_mismatch_columns)

    @property
    def has_domain_violations(self) -> bool:
        return bool(self.domain_violation_columns or self.violated_relations)

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact deterministic summary."""
        assessed_columns = int((self.columns["Domain expectation"] != "Not declared").sum())
        return pd.DataFrame(
            {
                "Metric": [
                    "Columns with source-backed type expectations",
                    "Type mismatches",
                    "Columns with declared domain expectations",
                    "Columns with domain violations",
                    "Declared cross-column relations",
                    "Relations with violations",
                ],
                "Value": [
                    len(self.columns),
                    len(self.type_mismatch_columns),
                    assessed_columns,
                    len(self.domain_violation_columns),
                    len(self.relations),
                    len(self.violated_relations),
                ],
            }
        )

    def column_frame(self) -> pd.DataFrame:
        """Return per-column type, range, and domain evidence."""
        return self.columns.copy(deep=True)

    def relation_frame(self) -> pd.DataFrame:
        """Return cross-column relation evidence."""
        return self.relations.copy(deep=True)


def analyze_types_ranges_and_domains(
    dataframe: pd.DataFrame,
    *,
    source_variables_file: str | Path,
    domain_rules: Mapping[str, object] | None = None,
) -> TypesRangesDomainReport:
    """Analyze source-backed types, observed ranges, and declared domains.

    ``domain_rules`` may declare compact groups:

    - ``positive``: columns constrained to values > 0;
    - ``non_negative``: columns constrained to values >= 0;
    - ``unit_interval``: columns constrained to 0 <= value <= 1;
    - ``minimums`` / ``maximums``: inclusive per-column numeric bounds;
    - ``relations``: ``(left, operator, right)`` column comparisons.

    Missing values are excluded from range/domain evaluation. Their presence is
    neither accepted nor rejected here; that belongs to the dedicated missing
    and invalid value stage.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.columns.duplicated().any():
        raise DomainValidationError("Dataset columns must be unique.")

    source_rows = _load_source_rows(source_variables_file)
    dataset_columns = tuple(str(column) for column in dataframe.columns)
    _validate_source_coverage(dataset_columns, source_rows)

    expected_types = {
        column: _expected_type_from_source(source_rows[column]["type"], column=column)
        for column in dataset_columns
    }
    type_report = analyze_data_types(dataframe, expected_types)

    normalized_rules = _normalize_domain_rules(domain_rules or {}, dataset_columns)
    type_rows = type_report.column_frame().set_index("Column").to_dict(orient="index")

    column_rows: list[dict[str, object]] = []
    for column in dataset_columns:
        series = dataframe[column]
        source_type = _display_source_type(source_rows[column]["type"])
        expected_type = expected_types[column]
        type_row = type_rows[column]

        numeric_values = pd.to_numeric(series, errors="coerce")
        observed_non_missing = series.notna()
        numeric_non_missing = numeric_values.notna()
        has_numeric_semantics = expected_type in {"integer", "numeric"}

        observed_minimum: object = "Not applicable"
        observed_maximum: object = "Not applicable"
        if has_numeric_semantics and bool(numeric_non_missing.any()):
            observed_minimum = float(numeric_values.loc[numeric_non_missing].min())
            observed_maximum = float(numeric_values.loc[numeric_non_missing].max())

        checks = normalized_rules["column_rules"].get(column, ())
        violation_mask = pd.Series(False, index=dataframe.index, dtype=bool)
        descriptions: list[str] = []

        for description, predicate in checks:
            descriptions.append(description)
            evaluable = observed_non_missing & numeric_non_missing
            predicate_result = predicate(numeric_values)
            violation_mask |= evaluable & ~predicate_result.fillna(False)

        # A non-null value in a source-declared numeric field that cannot be
        # parsed numerically is a domain/type issue, not a missing-value issue.
        if has_numeric_semantics:
            violation_mask |= observed_non_missing & ~numeric_non_missing

        violation_count = int(violation_mask.sum())
        domain_expectation = "; ".join(descriptions) if descriptions else "Not declared"
        type_status = str(type_row["Status"])
        status = "Review required" if type_status != "Match" or violation_count else "Valid"

        column_rows.append(
            {
                "Column": column,
                "Source type": source_type,
                "Expected type": expected_type,
                "Observed dtype": str(type_row["Observed dtype"]),
                "Observed minimum": observed_minimum,
                "Observed maximum": observed_maximum,
                "Domain expectation": domain_expectation,
                "Violation count": violation_count,
                "Status": status,
            }
        )

    relation_rows = _evaluate_relations(dataframe, normalized_rules["relations"])

    return TypesRangesDomainReport(
        type_report=type_report,
        columns=pd.DataFrame(column_rows, columns=list(_COLUMN_COLUMNS)),
        relations=pd.DataFrame(relation_rows, columns=list(_RELATION_COLUMNS)),
    )


def _load_source_rows(source_variables_file: str | Path) -> dict[str, dict[str, object]]:
    path = Path(source_variables_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source variable metadata file not found: {path}")

    variables = pd.read_csv(path)
    required = {"name", "type"}
    missing = sorted(required.difference(variables.columns))
    if missing:
        raise DomainValidationError(
            "Source variable metadata is missing required column(s): " + ", ".join(missing) + "."
        )

    rows: dict[str, dict[str, object]] = {}
    for raw in variables.to_dict(orient="records"):
        name = _normalize_column(raw["name"], label="source variable name")
        if name in rows:
            raise DomainValidationError(f"Duplicate source metadata row for {name!r}.")
        rows[name] = raw
    return rows


def _validate_source_coverage(
    dataset_columns: tuple[str, ...],
    source_rows: Mapping[str, Mapping[str, object]],
) -> None:
    dataset_set = set(dataset_columns)
    source_set = set(source_rows)
    missing = [column for column in dataset_columns if column not in source_set]
    extra = [column for column in source_rows if column not in dataset_set]
    failures: list[str] = []
    if missing:
        failures.append("dataset columns missing source type metadata: " + ", ".join(missing))
    if extra:
        failures.append("source metadata columns absent from dataset: " + ", ".join(extra))
    if failures:
        raise DomainValidationError("; ".join(failures) + ".")


def _expected_type_from_source(value: object, *, column: str) -> str:
    if pd.isna(value):
        raise DomainValidationError(f"Source type is missing for {column!r}.")
    normalized = str(value).strip().casefold()
    expected = _SOURCE_TYPE_MAP.get(normalized)
    if expected is None:
        raise DomainValidationError(
            f"Unsupported source type {value!r} for {column!r}; add an explicit source-type mapping."
        )
    return expected


def _display_source_type(value: object) -> str:
    return "Not documented" if pd.isna(value) else str(value).strip()


def _normalize_domain_rules(
    rules: Mapping[str, object],
    dataset_columns: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(rules, Mapping):
        raise TypeError("domain_rules must be a mapping.")

    unexpected = sorted(str(key) for key in rules if key not in _ALLOWED_RULE_GROUPS)
    if unexpected:
        raise DomainValidationError("Unsupported domain rule group(s): " + ", ".join(unexpected) + ".")

    dataset_set = set(dataset_columns)
    column_rules: dict[str, list[tuple[str, object]]] = {}

    def add_rule(column: str, description: str, predicate: object) -> None:
        if column not in dataset_set:
            raise DomainValidationError(f"Domain rule references missing column {column!r}.")
        column_rules.setdefault(column, []).append((description, predicate))

    for raw_column in _normalize_column_collection(rules.get("positive", ()), label="positive"):
        add_rule(raw_column, "> 0", lambda values: values > 0)

    for raw_column in _normalize_column_collection(rules.get("non_negative", ()), label="non_negative"):
        add_rule(raw_column, ">= 0", lambda values: values >= 0)

    for raw_column in _normalize_column_collection(rules.get("unit_interval", ()), label="unit_interval"):
        add_rule(raw_column, "0 <= value <= 1", lambda values: (values >= 0) & (values <= 1))

    for column, minimum in _normalize_bound_mapping(rules.get("minimums", {}), label="minimums").items():
        add_rule(column, f">= {minimum:g}", lambda values, bound=minimum: values >= bound)

    for column, maximum in _normalize_bound_mapping(rules.get("maximums", {}), label="maximums").items():
        add_rule(column, f"<= {maximum:g}", lambda values, bound=maximum: values <= bound)

    relations = _normalize_relations(rules.get("relations", ()), dataset_set)
    return {"column_rules": column_rules, "relations": relations}


def _normalize_column_collection(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Collection):
        raise TypeError(f"domain_rules[{label!r}] must be a non-string collection.")
    normalized = tuple(_normalize_column(item, label=f"{label} column") for item in value)
    if len(set(normalized)) != len(normalized):
        raise DomainValidationError(f"domain_rules[{label!r}] contains duplicate columns.")
    return normalized


def _normalize_bound_mapping(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"domain_rules[{label!r}] must be a mapping by column.")
    normalized: dict[str, float] = {}
    for raw_column, raw_bound in value.items():
        column = _normalize_column(raw_column, label=f"{label} column")
        if isinstance(raw_bound, bool) or not isinstance(raw_bound, Real):
            raise TypeError(f"Bound for {column!r} in {label!r} must be numeric.")
        normalized[column] = float(raw_bound)
    return normalized


def _normalize_relations(value: object, dataset_set: set[str]) -> tuple[tuple[str, str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise TypeError("domain_rules['relations'] must be a collection of 3-item relations.")
    normalized: list[tuple[str, str, str]] = []
    for relation in value:
        if isinstance(relation, (str, bytes)) or not isinstance(relation, Sequence) or len(relation) != 3:
            raise DomainValidationError("Each domain relation must be (left_column, operator, right_column).")
        left = _normalize_column(relation[0], label="relation left column")
        operator = str(relation[1]).strip()
        right = _normalize_column(relation[2], label="relation right column")
        if operator not in _ALLOWED_RELATION_OPERATORS:
            raise DomainValidationError(f"Unsupported relation operator {operator!r}.")
        missing = [column for column in (left, right) if column not in dataset_set]
        if missing:
            raise DomainValidationError("Domain relation references missing column(s): " + ", ".join(missing) + ".")
        normalized.append((left, operator, right))
    return tuple(normalized)


def _evaluate_relations(
    dataframe: pd.DataFrame,
    relations: tuple[tuple[str, str, str], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, operator, right in relations:
        left_values = pd.to_numeric(dataframe[left], errors="coerce")
        right_values = pd.to_numeric(dataframe[right], errors="coerce")
        evaluable = left_values.notna() & right_values.notna()
        result = _compare(left_values, operator, right_values)
        violations = evaluable & ~result.fillna(False)
        rows.append(
            {
                "Relation": f"{left} {operator} {right}",
                "Evaluated rows": int(evaluable.sum()),
                "Violation count": int(violations.sum()),
                "Status": "Review required" if bool(violations.any()) else "Valid",
            }
        )
    return rows


def _compare(left: pd.Series, operator: str, right: pd.Series) -> pd.Series:
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    return left == right


def _normalize_column(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    return normalized
