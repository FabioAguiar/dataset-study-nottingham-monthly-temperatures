"""Reusable, non-mutating analysis of duplicate tabular records."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Group count",
    "Row count",
    "Interpretation",
]

_IDENTIFIER_FRAME_COLUMNS: Final[list[str]] = [
    "Group",
    "Group size",
    "Distinct record count",
    "Classification",
    "Row positions",
    "Row indices",
]

_PROFILE_FRAME_COLUMNS: Final[list[str]] = [
    "Profile group",
    "Row count",
    "Distinct identifier count",
    "Target values",
    "Target value count",
    "Classification",
    "Row positions",
    "Row indices",
    "Feature values",
]

_ISSUE_FRAME_COLUMNS: Final[list[str]] = [
    "Issue",
    "Group count",
    "Row count",
    "Potential impact",
]


class DuplicateValidationError(ValueError):
    """Raised when declared duplicate-record expectations are not met."""


@dataclass(frozen=True, slots=True)
class DuplicateRecordReport:
    """Summarize exact duplicates, identifier conflicts, and repeated profiles."""

    row_count: int
    identifier_columns: tuple[str, ...]
    exact_duplicate_group_count: int
    exact_duplicate_row_count: int
    duplicate_identifier_group_count: int
    duplicate_identifier_row_count: int
    conflicting_identifier_group_count: int
    conflicting_identifier_row_count: int
    repeated_profile_group_count: int
    repeated_profile_row_count: int
    same_target_profile_group_count: int
    same_target_profile_row_count: int
    target_conflict_group_count: int
    target_conflict_row_count: int
    exact_duplicates: pd.DataFrame
    identifier_duplicates: pd.DataFrame
    repeated_profiles: pd.DataFrame

    @property
    def has_source_identifiers(self) -> bool:
        """Return whether source-backed observation identifiers are available."""
        return bool(self.identifier_columns)

    @property
    def has_exact_duplicates(self) -> bool:
        """Return whether completely identical rows were found."""
        return self.exact_duplicate_group_count > 0

    @property
    def has_duplicate_identifiers(self) -> bool:
        """Return whether identifiers occur in more than one row."""
        return self.duplicate_identifier_group_count > 0

    @property
    def has_conflicting_identifiers(self) -> bool:
        """Return whether repeated identifiers have different record content."""
        return self.conflicting_identifier_group_count > 0

    @property
    def has_repeated_profiles(self) -> bool:
        """Return whether different identifiers share the same feature profile."""
        return self.repeated_profile_group_count > 0

    @property
    def has_target_conflicts(self) -> bool:
        """Return whether a repeated profile has more than one target value."""
        return self.target_conflict_group_count > 0

    @property
    def has_quality_issues(self) -> bool:
        """Return whether source identity supports a duplicate quality issue."""
        return self.has_duplicate_identifiers or (
            self.has_source_identifiers and self.has_exact_duplicates
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic duplicate-analysis metrics."""
        rows = [
            {
                "Metric": "Exact duplicate records",
                "Group count": self.exact_duplicate_group_count,
                "Row count": self.exact_duplicate_row_count,
                "Interpretation": (
                    "Quality issue"
                    if self.has_source_identifiers
                    else "Review required; source identity unavailable"
                ),
            },
            {
                "Metric": "Duplicate identifiers",
                "Group count": self.duplicate_identifier_group_count,
                "Row count": self.duplicate_identifier_row_count,
                "Interpretation": (
                    "Quality issue"
                    if self.has_source_identifiers
                    else "Not applicable; source identifier unavailable"
                ),
            },
            {
                "Metric": "Conflicting duplicate identifiers",
                "Group count": self.conflicting_identifier_group_count,
                "Row count": self.conflicting_identifier_row_count,
                "Interpretation": (
                    "Quality issue"
                    if self.has_source_identifiers
                    else "Not applicable; source identifier unavailable"
                ),
            },
            {
                "Metric": "Repeated feature profiles",
                "Group count": self.repeated_profile_group_count,
                "Row count": self.repeated_profile_row_count,
                "Interpretation": "Potentially valid repeated records",
            },
            {
                "Metric": "Repeated profiles with the same target",
                "Group count": self.same_target_profile_group_count,
                "Row count": self.same_target_profile_row_count,
                "Interpretation": "Valid repeated profile",
            },
            {
                "Metric": "Repeated profiles with target disagreement",
                "Group count": self.target_conflict_group_count,
                "Row count": self.target_conflict_row_count,
                "Interpretation": "Analytical ambiguity",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def exact_duplicates_frame(self) -> pd.DataFrame:
        """Return sampled rows involved in complete-row duplicate groups."""
        return self.exact_duplicates.copy(deep=True)

    def identifier_duplicates_frame(self) -> pd.DataFrame:
        """Return sampled repeated-identifier groups."""
        return self.identifier_duplicates.copy(deep=True)

    def repeated_profiles_frame(self) -> pd.DataFrame:
        """Return sampled profiles shared by distinct identifiers."""
        return self.repeated_profiles.copy(deep=True)

    def target_conflicts_frame(self) -> pd.DataFrame:
        """Return repeated feature profiles with target disagreement."""
        if self.repeated_profiles.empty:
            return self.repeated_profiles.copy(deep=True)

        return self.repeated_profiles.loc[
            self.repeated_profiles["Classification"]
            == "Target disagreement"
        ].reset_index(drop=True).copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return only duplicate conditions classified as quality issues."""
        rows: list[dict[str, object]] = []

        if self.has_exact_duplicates and self.has_source_identifiers:
            rows.append(
                {
                    "Issue": "Exact duplicate records",
                    "Group count": self.exact_duplicate_group_count,
                    "Row count": self.exact_duplicate_row_count,
                    "Potential impact": (
                        "May overweight repeated observations and bias "
                        "descriptive statistics or model training."
                    ),
                }
            )

        if self.has_duplicate_identifiers:
            rows.append(
                {
                    "Issue": "Duplicate identifiers",
                    "Group count": self.duplicate_identifier_group_count,
                    "Row count": self.duplicate_identifier_row_count,
                    "Potential impact": (
                        "Violates the declared observation unit and may "
                        "represent repeated or conflicting source observations."
                    ),
                }
            )

        if self.has_conflicting_identifiers:
            rows.append(
                {
                    "Issue": "Conflicting duplicate identifiers",
                    "Group count": self.conflicting_identifier_group_count,
                    "Row count": self.conflicting_identifier_row_count,
                    "Potential impact": (
                        "Assigns multiple incompatible records to the same "
                        "observation identifier."
                    ),
                }
            )

        return pd.DataFrame(rows, columns=_ISSUE_FRAME_COLUMNS)

    def raise_if_invalid(
        self,
        *,
        require_no_exact_duplicates: bool = True,
        require_unique_identifiers: bool = True,
        require_no_conflicting_identifiers: bool = True,
        require_no_repeated_profiles: bool = False,
        require_no_target_conflicts: bool = False,
    ) -> None:
        """Raise one consolidated error for selected duplicate expectations."""
        failures: list[str] = []

        if require_no_exact_duplicates and self.has_exact_duplicates:
            failures.append(
                "exact duplicate record groups found: "
                f"{self.exact_duplicate_group_count}"
            )

        if require_unique_identifiers and self.has_duplicate_identifiers:
            failures.append(
                "duplicate identifier groups found: "
                f"{self.duplicate_identifier_group_count}"
            )

        if (
            require_no_conflicting_identifiers
            and self.has_conflicting_identifiers
        ):
            failures.append(
                "conflicting duplicate identifier groups found: "
                f"{self.conflicting_identifier_group_count}"
            )

        if require_no_repeated_profiles and self.has_repeated_profiles:
            failures.append(
                "repeated feature profile groups found: "
                f"{self.repeated_profile_group_count}"
            )

        if require_no_target_conflicts and self.has_target_conflicts:
            failures.append(
                "repeated feature profiles with target disagreement found: "
                f"{self.target_conflict_group_count}"
            )

        if failures:
            raise DuplicateValidationError("; ".join(failures) + ".")


def analyze_duplicate_records(
    dataframe: pd.DataFrame,
    *,
    identifiers: Collection[str] | None,
    feature_columns: Collection[str],
    target: str,
    max_group_samples: int = 10,
) -> DuplicateRecordReport:
    """Analyze duplicate records without changing the source DataFrame.

    The analysis distinguishes:

    - exact duplicate rows;
    - repeated identifiers with identical or conflicting record content when
      source-backed identifiers are available;
    - repeated feature profiles;
    - repeated profiles whose target values agree or disagree.

    When source identifiers are unavailable, exact row matches are evidence for
    review rather than proof that the same real-world observation was duplicated.
    Repeated feature profiles are not quality issues by default because distinct
    observations may legitimately share the same measured attributes.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if isinstance(max_group_samples, bool) or not isinstance(
        max_group_samples,
        int,
    ):
        raise TypeError("max_group_samples must be an integer.")
    if max_group_samples < 1:
        raise ValueError("max_group_samples must be at least 1.")

    if not dataframe.columns.is_unique:
        raise DuplicateValidationError(
            "dataframe columns must be unique for duplicate analysis."
        )

    identifier_columns = _normalize_column_collection(
        () if identifiers is None else identifiers,
        field_name="identifiers",
        require_non_empty=False,
    )
    features = _normalize_column_collection(
        feature_columns,
        field_name="feature_columns",
        require_non_empty=True,
    )
    target_column = _normalize_column_name(target, field_name="target")

    _validate_role_boundaries(
        identifiers=identifier_columns,
        feature_columns=features,
        target=target_column,
    )
    _validate_declared_columns(
        dataframe,
        [*identifier_columns, *features, target_column],
    )

    exact_groups = _duplicate_groups(
        dataframe,
        columns=list(dataframe.columns),
    )
    identifier_groups = (
        _duplicate_groups(
            dataframe,
            columns=list(identifier_columns),
        )
        if identifier_columns
        else []
    )
    profile_groups = _repeated_profile_groups(
        dataframe,
        identifiers=identifier_columns,
        feature_columns=features,
        target=target_column,
    )

    exact_duplicate_group_count = len(exact_groups)
    exact_duplicate_row_count = sum(len(group) for group in exact_groups)

    identifier_duplicate_rows = _build_identifier_duplicate_rows(
        dataframe,
        groups=identifier_groups,
        identifiers=identifier_columns,
    )
    conflicting_identifier_groups = [
        group
        for group in identifier_duplicate_rows
        if group["Classification"] == "Conflicting records"
    ]

    repeated_profile_rows = _build_repeated_profile_rows(
        dataframe,
        groups=profile_groups,
        identifiers=identifier_columns,
        feature_columns=features,
        target=target_column,
    )
    same_target_profiles = [
        group
        for group in repeated_profile_rows
        if group["Classification"] == "Same target"
    ]
    target_conflict_profiles = [
        group
        for group in repeated_profile_rows
        if group["Classification"] == "Target disagreement"
    ]

    exact_duplicates_frame = _build_exact_duplicates_frame(
        dataframe,
        groups=exact_groups[:max_group_samples],
    )
    identifier_duplicates_frame = pd.DataFrame(
        identifier_duplicate_rows[:max_group_samples],
    )
    if identifier_duplicates_frame.empty:
        identifier_duplicates_frame = pd.DataFrame(
            columns=[*identifier_columns, *_IDENTIFIER_FRAME_COLUMNS]
        )
    else:
        identifier_duplicates_frame = identifier_duplicates_frame.loc[
            :,
            [*identifier_columns, *_IDENTIFIER_FRAME_COLUMNS],
        ]

    repeated_profiles_frame = pd.DataFrame(
        repeated_profile_rows[:max_group_samples],
        columns=_PROFILE_FRAME_COLUMNS,
    )

    return DuplicateRecordReport(
        row_count=len(dataframe),
        identifier_columns=identifier_columns,
        exact_duplicate_group_count=exact_duplicate_group_count,
        exact_duplicate_row_count=exact_duplicate_row_count,
        duplicate_identifier_group_count=len(identifier_duplicate_rows),
        duplicate_identifier_row_count=sum(
            int(group["Group size"])
            for group in identifier_duplicate_rows
        ),
        conflicting_identifier_group_count=len(
            conflicting_identifier_groups
        ),
        conflicting_identifier_row_count=sum(
            int(group["Group size"])
            for group in conflicting_identifier_groups
        ),
        repeated_profile_group_count=len(repeated_profile_rows),
        repeated_profile_row_count=sum(
            int(group["Row count"])
            for group in repeated_profile_rows
        ),
        same_target_profile_group_count=len(same_target_profiles),
        same_target_profile_row_count=sum(
            int(group["Row count"])
            for group in same_target_profiles
        ),
        target_conflict_group_count=len(target_conflict_profiles),
        target_conflict_row_count=sum(
            int(group["Row count"])
            for group in target_conflict_profiles
        ),
        exact_duplicates=exact_duplicates_frame,
        identifier_duplicates=identifier_duplicates_frame,
        repeated_profiles=repeated_profiles_frame,
    )


def _normalize_column_collection(
    columns: Collection[str],
    *,
    field_name: str,
    require_non_empty: bool,
) -> tuple[str, ...]:
    if isinstance(columns, str) or not isinstance(columns, Collection):
        raise TypeError(f"{field_name} must be a collection of column names.")

    normalized = tuple(
        _normalize_column_name(column, field_name=field_name)
        for column in columns
    )

    if require_non_empty and not normalized:
        raise ValueError(f"{field_name} must contain at least one column.")

    duplicates = _find_duplicates(normalized)
    if duplicates:
        raise ValueError(
            f"{field_name} contains duplicate column names: "
            + ", ".join(duplicates)
            + "."
        )

    return normalized


def _normalize_column_name(column: object, *, field_name: str) -> str:
    if not isinstance(column, str):
        raise TypeError(f"{field_name} column names must be strings.")

    normalized = column.strip()
    if not normalized:
        raise ValueError(f"{field_name} column names must not be blank.")

    return normalized


def _find_duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return tuple(duplicates)


def _validate_role_boundaries(
    *,
    identifiers: tuple[str, ...],
    feature_columns: tuple[str, ...],
    target: str,
) -> None:
    identifier_set = set(identifiers)
    feature_set = set(feature_columns)

    overlap = identifier_set.intersection(feature_set)
    if overlap:
        raise DuplicateValidationError(
            "identifier and feature columns overlap: "
            + ", ".join(sorted(overlap))
            + "."
        )

    if target in identifier_set:
        raise DuplicateValidationError(
            f"target column {target!r} cannot also be an identifier."
        )

    if target in feature_set:
        raise DuplicateValidationError(
            f"target column {target!r} cannot also be a feature."
        )


def _validate_declared_columns(
    dataframe: pd.DataFrame,
    declared_columns: Collection[str],
) -> None:
    missing = [
        column
        for column in declared_columns
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(
            "declared duplicate-analysis columns missing from the dataset: "
            + ", ".join(missing)
            + "."
        )


def _duplicate_groups(
    dataframe: pd.DataFrame,
    *,
    columns: list[str],
) -> list[list[int]]:
    if dataframe.empty:
        return []

    working = dataframe.loc[:, columns].copy(deep=False)
    working = working.assign(__row_position=range(len(dataframe)))
    groups: list[list[int]] = []

    grouper: str | list[str]
    if len(columns) == 1:
        grouper = columns[0]
    else:
        grouper = columns

    grouped = working.groupby(
        grouper,
        sort=False,
        dropna=False,
        observed=False,
    )
    for _, group in grouped:
        positions = [int(value) for value in group["__row_position"]]
        if len(positions) > 1:
            groups.append(positions)

    return groups


def _repeated_profile_groups(
    dataframe: pd.DataFrame,
    *,
    identifiers: tuple[str, ...],
    feature_columns: tuple[str, ...],
    target: str,
) -> list[list[int]]:
    candidate_groups = _duplicate_groups(
        dataframe,
        columns=list(feature_columns),
    )
    repeated_profiles: list[list[int]] = []

    for positions in candidate_groups:
        if not identifiers:
            repeated_profiles.append(positions)
            continue

        group = dataframe.iloc[positions]
        distinct_identifiers = group.loc[:, list(identifiers)].drop_duplicates()
        if len(distinct_identifiers) > 1:
            repeated_profiles.append(positions)

    return repeated_profiles


def _build_exact_duplicates_frame(
    dataframe: pd.DataFrame,
    *,
    groups: list[list[int]],
) -> pd.DataFrame:
    columns = [
        "Group",
        "Group size",
        "Row position",
        "Row index",
        *[str(column) for column in dataframe.columns],
    ]
    rows: list[dict[str, object]] = []

    for group_number, positions in enumerate(groups, start=1):
        for position in positions:
            row = dataframe.iloc[position]
            rows.append(
                {
                    "Group": group_number,
                    "Group size": len(positions),
                    "Row position": position,
                    "Row index": dataframe.index[position],
                    **{
                        str(column): row[column]
                        for column in dataframe.columns
                    },
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _build_identifier_duplicate_rows(
    dataframe: pd.DataFrame,
    *,
    groups: list[list[int]],
    identifiers: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    non_identifier_columns = [
        column
        for column in dataframe.columns
        if column not in identifiers
    ]

    for group_number, positions in enumerate(groups, start=1):
        group = dataframe.iloc[positions]
        distinct_record_count = len(
            group.loc[:, non_identifier_columns].drop_duplicates()
        )
        first_row = group.iloc[0]
        rows.append(
            {
                **{
                    identifier: first_row[identifier]
                    for identifier in identifiers
                },
                "Group": group_number,
                "Group size": len(positions),
                "Distinct record count": distinct_record_count,
                "Classification": (
                    "Repeated identical record"
                    if distinct_record_count == 1
                    else "Conflicting records"
                ),
                "Row positions": tuple(positions),
                "Row indices": tuple(dataframe.index[position] for position in positions),
            }
        )

    return rows


def _build_repeated_profile_rows(
    dataframe: pd.DataFrame,
    *,
    groups: list[list[int]],
    identifiers: tuple[str, ...],
    feature_columns: tuple[str, ...],
    target: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for group_number, positions in enumerate(groups, start=1):
        group = dataframe.iloc[positions]
        distinct_identifiers: object = (
            len(group.loc[:, list(identifiers)].drop_duplicates())
            if identifiers
            else pd.NA
        )
        target_values = _ordered_display_values(group[target])
        first_row = group.iloc[0]
        rows.append(
            {
                "Profile group": group_number,
                "Row count": len(positions),
                "Distinct identifier count": distinct_identifiers,
                "Target values": ", ".join(target_values),
                "Target value count": len(target_values),
                "Classification": (
                    "Same target"
                    if len(target_values) == 1
                    else "Target disagreement"
                ),
                "Row positions": tuple(positions),
                "Row indices": tuple(dataframe.index[position] for position in positions),
                "Feature values": {
                    feature: first_row[feature]
                    for feature in feature_columns
                },
            }
        )

    return rows


def _ordered_display_values(series: pd.Series) -> tuple[str, ...]:
    values: list[str] = []

    for value in pd.unique(series):
        display_value = "<missing>" if pd.isna(value) else repr(value)
        if display_value not in values:
            values.append(display_value)

    return tuple(values)
