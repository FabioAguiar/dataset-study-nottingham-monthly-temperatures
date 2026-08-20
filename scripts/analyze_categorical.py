"""Reusable, non-mutating exploration of categorical features."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_FEATURE_SUMMARY_COLUMNS: Final[list[str]] = [
    "Feature",
    "Row count",
    "Valid category count",
    "Missing count",
    "Blank count",
    "Cardinality",
    "Expected cardinality",
    "Mode",
    "Mode count",
    "Mode share",
    "Rare category count",
    "Rare row count",
    "Unexpected category count",
    "Missing expected category count",
    "Inconsistent label count",
    "High cardinality",
    "Status",
]

_FREQUENCY_COLUMNS: Final[list[str]] = [
    "Feature",
    "Category",
    "Count",
    "Share",
    "Rank",
    "Expected",
    "Rare",
    "Rare trigger",
    "Dominant",
    "Normalized category",
]

_RARE_CATEGORY_COLUMNS: Final[list[str]] = [
    "Feature",
    "Category",
    "Count",
    "Share",
    "Trigger",
]

_LABEL_ISSUE_COLUMNS: Final[list[str]] = [
    "Feature",
    "Normalized category",
    "Raw variants",
    "Rows",
    "Interpretation",
]

_CONTRACT_ISSUE_COLUMNS: Final[list[str]] = [
    "Feature",
    "Issue",
    "Count",
    "Values",
    "Potential impact",
]

_GROUPING_COLUMNS: Final[list[str]] = [
    "Feature",
    "Group",
    "Count",
    "Share",
    "Categories",
    "Coverage",
]

_GROUPING_ISSUE_COLUMNS: Final[list[str]] = [
    "Feature",
    "Issue",
    "Count",
    "Values",
    "Potential impact",
]


class CategoricalAnalysisError(ValueError):
    """Raised when categorical-analysis configuration or data are invalid."""


@dataclass(frozen=True, slots=True)
class CategoricalFeatureReport:
    """Summarize categorical frequencies, quality, and groupings."""

    requested_features: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    row_count: int
    rare_count_threshold: int
    rare_share_threshold: float
    high_cardinality_threshold: int
    feature_summary: pd.DataFrame
    frequencies: pd.DataFrame
    rare_categories: pd.DataFrame
    label_issues: pd.DataFrame
    contract_issues: pd.DataFrame
    groupings: pd.DataFrame
    grouping_issues: pd.DataFrame
    categorical_projection: pd.DataFrame

    @property
    def has_missing_features(self) -> bool:
        """Return whether declared categorical features are absent."""
        return bool(self.missing_features)

    @property
    def has_missing_values(self) -> bool:
        """Return whether an available feature contains missing values."""
        if self.feature_summary.empty:
            return False
        return bool((self.feature_summary["Missing count"] > 0).any())

    @property
    def has_blank_values(self) -> bool:
        """Return whether an available feature contains blank strings."""
        if self.feature_summary.empty:
            return False
        return bool((self.feature_summary["Blank count"] > 0).any())

    @property
    def has_rare_categories(self) -> bool:
        """Return whether any observed category meets a rarity criterion."""
        return not self.rare_categories.empty

    @property
    def has_label_inconsistencies(self) -> bool:
        """Return whether normalized labels have multiple raw variants."""
        return not self.label_issues.empty

    @property
    def has_unexpected_categories(self) -> bool:
        """Return whether observed categories violate expected contracts."""
        if self.contract_issues.empty:
            return False
        return bool(
            (
                self.contract_issues["Issue"]
                == "Unexpected categories"
            ).any()
        )

    @property
    def has_missing_expected_categories(self) -> bool:
        """Return whether expected categories are absent from the data."""
        if self.contract_issues.empty:
            return False
        return bool(
            (
                self.contract_issues["Issue"]
                == "Missing expected categories"
            ).any()
        )

    @property
    def has_high_cardinality_features(self) -> bool:
        """Return whether cardinality exceeds the configured threshold."""
        return bool(self.high_cardinality_features)

    @property
    def has_grouping_issues(self) -> bool:
        """Return whether candidate grouping definitions are incomplete."""
        return not self.grouping_issues.empty

    @property
    def constant_features(self) -> tuple[str, ...]:
        """Return features containing exactly one observed category."""
        if self.feature_summary.empty:
            return ()
        mask = self.feature_summary["Cardinality"] == 1
        return tuple(self.feature_summary.loc[mask, "Feature"])

    @property
    def high_cardinality_features(self) -> tuple[str, ...]:
        """Return features above the configured cardinality threshold."""
        if self.feature_summary.empty:
            return ()
        mask = (
            self.feature_summary["Cardinality"]
            > self.high_cardinality_threshold
        )
        return tuple(self.feature_summary.loc[mask, "Feature"])

    @property
    def is_analysis_ready(self) -> bool:
        """Return whether strict categorical contracts are satisfied."""
        return not any(
            (
                self.has_missing_features,
                self.has_label_inconsistencies,
                self.has_unexpected_categories,
                self.has_missing_expected_categories,
                self.has_grouping_issues,
            )
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic overall categorical-analysis metrics."""
        summary = self.feature_summary
        rows = [
            {
                "Metric": "Requested categorical features",
                "Value": len(self.requested_features),
                "Interpretation": "Features declared by the study",
            },
            {
                "Metric": "Available categorical features",
                "Value": len(self.available_features),
                "Interpretation": "Features found in the dataset",
            },
            {
                "Metric": "Missing categorical features",
                "Value": len(self.missing_features),
                "Interpretation": (
                    "Requires review"
                    if self.has_missing_features
                    else "All declared features are present"
                ),
            },
            {
                "Metric": "Features with missing values",
                "Value": (
                    0
                    if summary.empty
                    else int((summary["Missing count"] > 0).sum())
                ),
                "Interpretation": "Missing values recognized by pandas",
            },
            {
                "Metric": "Features with blank values",
                "Value": (
                    0
                    if summary.empty
                    else int((summary["Blank count"] > 0).sum())
                ),
                "Interpretation": "Hidden text-based missingness",
            },
            {
                "Metric": "Features with rare categories",
                "Value": (
                    0
                    if summary.empty
                    else int(
                        (summary["Rare category count"] > 0).sum()
                    )
                ),
                "Interpretation": (
                    f"Count < {self.rare_count_threshold} or share < "
                    f"{self.rare_share_threshold:.2%}"
                ),
            },
            {
                "Metric": "Features with inconsistent labels",
                "Value": (
                    0
                    if summary.empty
                    else int(
                        (summary["Inconsistent label count"] > 0).sum()
                    )
                ),
                "Interpretation": "Case or surrounding-space variants",
            },
            {
                "Metric": "Features with unexpected categories",
                "Value": (
                    0
                    if summary.empty
                    else int(
                        (summary["Unexpected category count"] > 0).sum()
                    )
                ),
                "Interpretation": "Observed values outside the contract",
            },
            {
                "Metric": "Features missing expected categories",
                "Value": (
                    0
                    if summary.empty
                    else int(
                        (
                            summary[
                                "Missing expected category count"
                            ]
                            > 0
                        ).sum()
                    )
                ),
                "Interpretation": "Declared categories absent in the data",
            },
            {
                "Metric": "High-cardinality features",
                "Value": len(self.high_cardinality_features),
                "Interpretation": (
                    "Observed cardinality above "
                    f"{self.high_cardinality_threshold}"
                ),
            },
            {
                "Metric": "Candidate grouping issues",
                "Value": len(self.grouping_issues),
                "Interpretation": (
                    "Unknown, duplicated, or unassigned categories"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def feature_summary_frame(self) -> pd.DataFrame:
        """Return one descriptive row per available feature."""
        return self.feature_summary.copy(deep=True)

    def frequency_frame(self) -> pd.DataFrame:
        """Return category counts and shares for every feature."""
        return self.frequencies.copy(deep=True)

    def rare_categories_frame(self) -> pd.DataFrame:
        """Return observed categories meeting configured rarity rules."""
        return self.rare_categories.copy(deep=True)

    def label_issues_frame(self) -> pd.DataFrame:
        """Return raw label variants sharing a normalized representation."""
        return self.label_issues.copy(deep=True)

    def category_contract_issues_frame(self) -> pd.DataFrame:
        """Return unexpected and missing expected category issues."""
        return self.contract_issues.copy(deep=True)

    def grouping_frame(self) -> pd.DataFrame:
        """Return frequencies for candidate within-feature groupings."""
        return self.groupings.copy(deep=True)

    def grouping_issues_frame(self) -> pd.DataFrame:
        """Return candidate grouping coverage and ambiguity issues."""
        return self.grouping_issues.copy(deep=True)

    def categorical_frame(self) -> pd.DataFrame:
        """Return a defensive projection of analyzed raw features."""
        return self.categorical_projection.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_features_present: bool = True,
        require_no_unexpected_categories: bool = True,
        require_expected_categories_present: bool = True,
        require_no_inconsistent_labels: bool = True,
        require_no_missing_values: bool = False,
        require_no_rare_categories: bool = False,
        require_no_high_cardinality: bool = False,
        require_valid_groupings: bool = True,
    ) -> None:
        """Raise one combined error for requested invalid conditions."""
        failures: list[str] = []

        if require_features_present and self.has_missing_features:
            failures.append(
                "missing_features:"
                + ",".join(repr(value) for value in self.missing_features)
            )

        if (
            require_no_unexpected_categories
            and self.has_unexpected_categories
        ):
            values = self.contract_issues.loc[
                self.contract_issues["Issue"] == "Unexpected categories",
                "Values",
            ]
            failures.append(
                "unexpected_categories:"
                + ";".join(str(value) for value in values)
            )

        if (
            require_expected_categories_present
            and self.has_missing_expected_categories
        ):
            values = self.contract_issues.loc[
                self.contract_issues["Issue"]
                == "Missing expected categories",
                "Values",
            ]
            failures.append(
                "missing_expected_categories:"
                + ";".join(str(value) for value in values)
            )

        if (
            require_no_inconsistent_labels
            and self.has_label_inconsistencies
        ):
            failures.append(
                f"inconsistent_labels:{len(self.label_issues)}"
            )

        if require_no_missing_values and (
            self.has_missing_values or self.has_blank_values
        ):
            missing_count = 0
            blank_count = 0
            if not self.feature_summary.empty:
                missing_count = int(
                    self.feature_summary["Missing count"].sum()
                )
                blank_count = int(
                    self.feature_summary["Blank count"].sum()
                )
            failures.append(
                f"missing_or_blank_values:{missing_count + blank_count}"
            )

        if require_no_rare_categories and self.has_rare_categories:
            failures.append(
                f"rare_categories:{len(self.rare_categories)}"
            )

        if (
            require_no_high_cardinality
            and self.has_high_cardinality_features
        ):
            failures.append(
                "high_cardinality_features:"
                + ",".join(self.high_cardinality_features)
            )

        if require_valid_groupings and self.has_grouping_issues:
            failures.append(
                f"invalid_groupings:{len(self.grouping_issues)}"
            )

        if failures:
            raise CategoricalAnalysisError(
                "Categorical feature analysis failed: "
                + " | ".join(failures)
            )


def analyze_categorical_features(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    expected_values: Mapping[str, Sequence[object]] | None = None,
    category_groupings: Mapping[
        str,
        Mapping[str, Sequence[object]],
    ]
    | None = None,
    rare_count_threshold: int = 50,
    rare_share_threshold: float = 0.01,
    high_cardinality_threshold: int = 20,
) -> CategoricalFeatureReport:
    """Analyze categorical features without modifying ``dataframe``."""
    _validate_dataframe(dataframe)
    requested_features = _normalize_features(features)
    _validate_thresholds(
        rare_count_threshold=rare_count_threshold,
        rare_share_threshold=rare_share_threshold,
        high_cardinality_threshold=high_cardinality_threshold,
    )

    normalized_expected = _normalize_expected_values(
        expected_values or {},
        requested_features=requested_features,
    )
    normalized_groupings = _normalize_groupings(
        category_groupings or {},
        requested_features=requested_features,
    )

    available_features = tuple(
        feature
        for feature in requested_features
        if feature in dataframe.columns
    )
    missing_features = tuple(
        feature
        for feature in requested_features
        if feature not in dataframe.columns
    )

    feature_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    rare_rows: list[dict[str, object]] = []
    label_issue_rows: list[dict[str, object]] = []
    contract_issue_rows: list[dict[str, object]] = []
    grouping_rows: list[dict[str, object]] = []
    grouping_issue_rows: list[dict[str, object]] = []

    for feature in available_features:
        result = _analyze_feature(
            dataframe[feature],
            feature=feature,
            expected=normalized_expected.get(feature, ()),
            groupings=normalized_groupings.get(feature, {}),
            rare_count_threshold=rare_count_threshold,
            rare_share_threshold=rare_share_threshold,
            high_cardinality_threshold=high_cardinality_threshold,
        )
        feature_rows.append(result.feature_summary)
        frequency_rows.extend(result.frequencies)
        rare_rows.extend(result.rare_categories)
        label_issue_rows.extend(result.label_issues)
        contract_issue_rows.extend(result.contract_issues)
        grouping_rows.extend(result.groupings)
        grouping_issue_rows.extend(result.grouping_issues)

    projection = dataframe.loc[:, list(available_features)].copy(deep=True)

    return CategoricalFeatureReport(
        requested_features=requested_features,
        available_features=available_features,
        missing_features=missing_features,
        row_count=len(dataframe),
        rare_count_threshold=rare_count_threshold,
        rare_share_threshold=rare_share_threshold,
        high_cardinality_threshold=high_cardinality_threshold,
        feature_summary=pd.DataFrame(
            feature_rows,
            columns=_FEATURE_SUMMARY_COLUMNS,
        ),
        frequencies=pd.DataFrame(
            frequency_rows,
            columns=_FREQUENCY_COLUMNS,
        ),
        rare_categories=pd.DataFrame(
            rare_rows,
            columns=_RARE_CATEGORY_COLUMNS,
        ),
        label_issues=pd.DataFrame(
            label_issue_rows,
            columns=_LABEL_ISSUE_COLUMNS,
        ),
        contract_issues=pd.DataFrame(
            contract_issue_rows,
            columns=_CONTRACT_ISSUE_COLUMNS,
        ),
        groupings=pd.DataFrame(
            grouping_rows,
            columns=_GROUPING_COLUMNS,
        ),
        grouping_issues=pd.DataFrame(
            grouping_issue_rows,
            columns=_GROUPING_ISSUE_COLUMNS,
        ),
        categorical_projection=projection,
    )


@dataclass(frozen=True, slots=True)
class _FeatureAnalysis:
    feature_summary: dict[str, object]
    frequencies: list[dict[str, object]]
    rare_categories: list[dict[str, object]]
    label_issues: list[dict[str, object]]
    contract_issues: list[dict[str, object]]
    groupings: list[dict[str, object]]
    grouping_issues: list[dict[str, object]]


def _analyze_feature(
    series: pd.Series,
    *,
    feature: str,
    expected: tuple[object, ...],
    groupings: Mapping[str, tuple[object, ...]],
    rare_count_threshold: int,
    rare_share_threshold: float,
    high_cardinality_threshold: int,
) -> _FeatureAnalysis:
    raw_values: list[object] = []
    missing_count = 0
    blank_count = 0

    raw_counts: dict[tuple[object, ...], int] = {}
    raw_representatives: dict[tuple[object, ...], object] = {}
    raw_first_positions: dict[tuple[object, ...], int] = {}
    normalized_variants: dict[
        tuple[object, ...],
        list[tuple[object, ...]],
    ] = defaultdict(list)

    for position, value in enumerate(series.tolist()):
        if _is_missing(value):
            missing_count += 1
            continue
        if _is_blank(value):
            blank_count += 1
            continue

        _require_hashable(value, context=f"feature {feature!r}")
        raw_values.append(value)
        raw_key = _raw_key(value)
        normalized_key = _normalized_key(value)

        if raw_key not in raw_counts:
            raw_counts[raw_key] = 0
            raw_representatives[raw_key] = value
            raw_first_positions[raw_key] = position
            normalized_variants[normalized_key].append(raw_key)
        raw_counts[raw_key] += 1

    valid_count = len(raw_values)
    cardinality = len(raw_counts)
    expected_key_to_value = {
        _normalized_key(value): value
        for value in expected
    }
    observed_normalized_keys = {
        _normalized_key(value)
        for value in raw_representatives.values()
    }

    unexpected_raw_keys = [
        raw_key
        for raw_key, value in raw_representatives.items()
        if expected and _normalized_key(value) not in expected_key_to_value
    ]
    missing_expected = tuple(
        value
        for value in expected
        if _normalized_key(value) not in observed_normalized_keys
    )

    max_count = max(raw_counts.values(), default=0)
    dominant_keys = {
        raw_key
        for raw_key, count in raw_counts.items()
        if max_count > 0 and count == max_count
    }

    sorted_raw_keys = sorted(
        raw_counts,
        key=lambda key: (
            -raw_counts[key],
            raw_first_positions[key],
        ),
    )

    frequencies: list[dict[str, object]] = []
    rare_categories: list[dict[str, object]] = []

    for rank, raw_key in enumerate(sorted_raw_keys, start=1):
        category = raw_representatives[raw_key]
        count = raw_counts[raw_key]
        share = count / valid_count if valid_count else 0.0
        triggers = _rarity_triggers(
            count=count,
            share=share,
            rare_count_threshold=rare_count_threshold,
            rare_share_threshold=rare_share_threshold,
        )
        rare = bool(triggers)
        expected_category = (
            not expected
            or _normalized_key(category) in expected_key_to_value
        )

        frequency_row = {
            "Feature": feature,
            "Category": category,
            "Count": count,
            "Share": share,
            "Rank": rank,
            "Expected": expected_category,
            "Rare": rare,
            "Rare trigger": ", ".join(triggers),
            "Dominant": raw_key in dominant_keys,
            "Normalized category": _normalized_display(category),
        }
        frequencies.append(frequency_row)

        if rare:
            rare_categories.append(
                {
                    "Feature": feature,
                    "Category": category,
                    "Count": count,
                    "Share": share,
                    "Trigger": ", ".join(triggers),
                }
            )

    label_issues: list[dict[str, object]] = []
    for normalized_key, variant_keys in normalized_variants.items():
        if normalized_key[0] != "str" or len(variant_keys) <= 1:
            continue
        variants = tuple(
            raw_representatives[raw_key]
            for raw_key in variant_keys
        )
        rows = sum(raw_counts[raw_key] for raw_key in variant_keys)
        label_issues.append(
            {
                "Feature": feature,
                "Normalized category": normalized_key[1],
                "Raw variants": variants,
                "Rows": rows,
                "Interpretation": (
                    "Labels differ only by case or surrounding spaces"
                ),
            }
        )

    contract_issues: list[dict[str, object]] = []
    if unexpected_raw_keys:
        unexpected_values = tuple(
            raw_representatives[key]
            for key in unexpected_raw_keys
        )
        contract_issues.append(
            {
                "Feature": feature,
                "Issue": "Unexpected categories",
                "Count": len(unexpected_values),
                "Values": unexpected_values,
                "Potential impact": (
                    "Unsupported levels may break encoding or inference"
                ),
            }
        )

    if missing_expected:
        contract_issues.append(
            {
                "Feature": feature,
                "Issue": "Missing expected categories",
                "Count": len(missing_expected),
                "Values": missing_expected,
                "Potential impact": (
                    "Expected levels may be absent from fitted encoders or "
                    "evaluation splits"
                ),
            }
        )

    grouping_rows, grouping_issue_rows = _analyze_groupings(
        feature=feature,
        raw_counts=raw_counts,
        raw_representatives=raw_representatives,
        expected=expected,
        groupings=groupings,
        valid_count=valid_count,
    )

    mode_values = tuple(
        raw_representatives[key]
        for key in sorted_raw_keys
        if key in dominant_keys
    )
    mode_display: object
    if not mode_values:
        mode_display = None
    elif len(mode_values) == 1:
        mode_display = mode_values[0]
    else:
        mode_display = mode_values

    mode_share = max_count / valid_count if valid_count else None
    status_parts: list[str] = []
    if missing_count:
        status_parts.append("Missing values")
    if blank_count:
        status_parts.append("Blank values")
    if unexpected_raw_keys:
        status_parts.append("Unexpected categories")
    if missing_expected:
        status_parts.append("Missing expected categories")
    if label_issues:
        status_parts.append("Inconsistent labels")
    if rare_categories:
        status_parts.append("Rare categories")
    if cardinality > high_cardinality_threshold:
        status_parts.append("High cardinality")
    if grouping_issue_rows:
        status_parts.append("Grouping review")

    feature_summary = {
        "Feature": feature,
        "Row count": len(series),
        "Valid category count": valid_count,
        "Missing count": missing_count,
        "Blank count": blank_count,
        "Cardinality": cardinality,
        "Expected cardinality": len(expected) if expected else None,
        "Mode": mode_display,
        "Mode count": max_count,
        "Mode share": mode_share,
        "Rare category count": len(rare_categories),
        "Rare row count": sum(
            int(row["Count"])
            for row in rare_categories
        ),
        "Unexpected category count": len(unexpected_raw_keys),
        "Missing expected category count": len(missing_expected),
        "Inconsistent label count": len(label_issues),
        "High cardinality": cardinality > high_cardinality_threshold,
        "Status": ", ".join(status_parts) if status_parts else "Analyzed",
    }

    return _FeatureAnalysis(
        feature_summary=feature_summary,
        frequencies=frequencies,
        rare_categories=rare_categories,
        label_issues=label_issues,
        contract_issues=contract_issues,
        groupings=grouping_rows,
        grouping_issues=grouping_issue_rows,
    )


def _analyze_groupings(
    *,
    feature: str,
    raw_counts: Mapping[tuple[object, ...], int],
    raw_representatives: Mapping[tuple[object, ...], object],
    expected: tuple[object, ...],
    groupings: Mapping[str, tuple[object, ...]],
    valid_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not groupings:
        return [], []

    expected_keys = {
        _normalized_key(value): value
        for value in expected
    }
    observed_keys = {
        _normalized_key(value): value
        for value in raw_representatives.values()
    }
    reference_keys = expected_keys or observed_keys

    assignments: dict[tuple[object, ...], list[str]] = defaultdict(list)
    grouped_values: dict[str, tuple[object, ...]] = {}
    unknown_values: list[object] = []

    for group, categories in groupings.items():
        grouped_values[group] = categories
        for category in categories:
            key = _normalized_key(category)
            assignments[key].append(group)
            if reference_keys and key not in reference_keys:
                unknown_values.append(category)

    duplicated_values = tuple(
        reference_keys.get(key, key[1])
        for key, groups in assignments.items()
        if len(groups) > 1
    )
    unassigned_values = tuple(
        value
        for key, value in reference_keys.items()
        if key not in assignments
    )

    issue_rows: list[dict[str, object]] = []
    if unknown_values:
        unique_unknown = _unique_values(unknown_values)
        issue_rows.append(
            {
                "Feature": feature,
                "Issue": "Unknown grouping categories",
                "Count": len(unique_unknown),
                "Values": unique_unknown,
                "Potential impact": (
                    "Grouping references categories outside the declared "
                    "or observed feature contract"
                ),
            }
        )
    if duplicated_values:
        issue_rows.append(
            {
                "Feature": feature,
                "Issue": "Categories assigned to multiple groups",
                "Count": len(duplicated_values),
                "Values": duplicated_values,
                "Potential impact": (
                    "Grouped frequencies would double-count observations"
                ),
            }
        )
    if unassigned_values:
        issue_rows.append(
            {
                "Feature": feature,
                "Issue": "Ungrouped categories",
                "Count": len(unassigned_values),
                "Values": unassigned_values,
                "Potential impact": (
                    "Candidate grouping does not cover the complete "
                    "categorical domain"
                ),
            }
        )

    valid_assignment = {
        key: groups[0]
        for key, groups in assignments.items()
        if len(groups) == 1 and key in reference_keys
    }
    group_counts = {group: 0 for group in groupings}
    unassigned_observed: list[object] = []
    unassigned_count = 0

    for raw_key, count in raw_counts.items():
        value = raw_representatives[raw_key]
        key = _normalized_key(value)
        group = valid_assignment.get(key)
        if group is None:
            unassigned_count += count
            unassigned_observed.append(value)
        else:
            group_counts[group] += count

    rows: list[dict[str, object]] = []
    for group, categories in grouped_values.items():
        count = group_counts[group]
        rows.append(
            {
                "Feature": feature,
                "Group": group,
                "Count": count,
                "Share": count / valid_count if valid_count else 0.0,
                "Categories": categories,
                "Coverage": "Defined",
            }
        )

    if unassigned_count:
        rows.append(
            {
                "Feature": feature,
                "Group": "Unassigned",
                "Count": unassigned_count,
                "Share": (
                    unassigned_count / valid_count
                    if valid_count
                    else 0.0
                ),
                "Categories": _unique_values(unassigned_observed),
                "Coverage": "Requires review",
            }
        )

    return rows, issue_rows


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated:
        raise CategoricalAnalysisError(
            "Categorical analysis does not support duplicated column "
            f"labels: {duplicated!r}"
        )


def _normalize_features(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise TypeError("features must be a sequence of column names")
    normalized = tuple(features)
    if not normalized:
        raise CategoricalAnalysisError(
            "At least one categorical feature must be declared"
        )
    for feature in normalized:
        if not isinstance(feature, str) or not feature.strip():
            raise CategoricalAnalysisError(
                "Categorical feature names must be non-empty strings"
            )
    if len(set(normalized)) != len(normalized):
        raise CategoricalAnalysisError(
            "Categorical feature names must be unique"
        )
    return normalized


def _normalize_expected_values(
    expected_values: Mapping[str, Sequence[object]],
    *,
    requested_features: tuple[str, ...],
) -> dict[str, tuple[object, ...]]:
    unknown_features = tuple(
        feature
        for feature in expected_values
        if feature not in requested_features
    )
    if unknown_features:
        raise CategoricalAnalysisError(
            "Expected-value contracts reference undeclared features: "
            f"{unknown_features!r}"
        )

    result: dict[str, tuple[object, ...]] = {}
    for feature, values in expected_values.items():
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"expected_values[{feature!r}] must be a sequence"
            )
        normalized = tuple(values)
        keys: set[tuple[object, ...]] = set()
        for value in normalized:
            if _is_missing(value) or _is_blank(value):
                raise CategoricalAnalysisError(
                    f"expected_values[{feature!r}] contains a missing "
                    "or blank category"
                )
            _require_hashable(value, context=f"expected_values[{feature!r}]")
            key = _normalized_key(value)
            if key in keys:
                raise CategoricalAnalysisError(
                    f"expected_values[{feature!r}] contains duplicate "
                    f"normalized category {value!r}"
                )
            keys.add(key)
        result[feature] = normalized
    return result


def _normalize_groupings(
    category_groupings: Mapping[
        str,
        Mapping[str, Sequence[object]],
    ],
    *,
    requested_features: tuple[str, ...],
) -> dict[str, dict[str, tuple[object, ...]]]:
    unknown_features = tuple(
        feature
        for feature in category_groupings
        if feature not in requested_features
    )
    if unknown_features:
        raise CategoricalAnalysisError(
            "Category groupings reference undeclared features: "
            f"{unknown_features!r}"
        )

    result: dict[str, dict[str, tuple[object, ...]]] = {}
    for feature, groups in category_groupings.items():
        if not isinstance(groups, Mapping):
            raise TypeError(
                f"category_groupings[{feature!r}] must be a mapping"
            )
        normalized_groups: dict[str, tuple[object, ...]] = {}
        for group, values in groups.items():
            if not isinstance(group, str) or not group.strip():
                raise CategoricalAnalysisError(
                    "Grouping names must be non-empty strings"
                )
            if isinstance(values, (str, bytes)):
                raise TypeError(
                    f"Grouping {group!r} for {feature!r} must contain "
                    "a sequence of categories"
                )
            normalized_values = tuple(values)
            if not normalized_values:
                raise CategoricalAnalysisError(
                    f"Grouping {group!r} for {feature!r} is empty"
                )
            for value in normalized_values:
                if _is_missing(value) or _is_blank(value):
                    raise CategoricalAnalysisError(
                        f"Grouping {group!r} for {feature!r} contains a "
                        "missing or blank category"
                    )
                _require_hashable(
                    value,
                    context=f"category_groupings[{feature!r}][{group!r}]",
                )
            normalized_groups[group] = normalized_values
        result[feature] = normalized_groups
    return result


def _validate_thresholds(
    *,
    rare_count_threshold: int,
    rare_share_threshold: float,
    high_cardinality_threshold: int,
) -> None:
    if isinstance(rare_count_threshold, bool) or not isinstance(
        rare_count_threshold,
        int,
    ):
        raise TypeError("rare_count_threshold must be an integer")
    if rare_count_threshold < 0:
        raise ValueError("rare_count_threshold must be non-negative")

    if isinstance(rare_share_threshold, bool) or not isinstance(
        rare_share_threshold,
        (int, float),
    ):
        raise TypeError("rare_share_threshold must be numeric")
    if not 0 <= float(rare_share_threshold) <= 1:
        raise ValueError("rare_share_threshold must be between 0 and 1")

    if isinstance(high_cardinality_threshold, bool) or not isinstance(
        high_cardinality_threshold,
        int,
    ):
        raise TypeError("high_cardinality_threshold must be an integer")
    if high_cardinality_threshold < 1:
        raise ValueError(
            "high_cardinality_threshold must be at least one"
        )


def _rarity_triggers(
    *,
    count: int,
    share: float,
    rare_count_threshold: int,
    rare_share_threshold: float,
) -> tuple[str, ...]:
    triggers: list[str] = []
    if count < rare_count_threshold:
        triggers.append("count")
    if share < rare_share_threshold:
        triggers.append("share")
    return tuple(triggers)


def _raw_key(value: object) -> tuple[object, ...]:
    return ("raw", type(value).__qualname__, value)


def _normalized_key(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return ("str", value.strip().casefold())
    return ("typed", type(value).__qualname__, value)


def _normalized_display(value: object) -> object:
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def _is_blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _is_missing(value: object) -> bool:
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


def _require_hashable(value: object, *, context: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise CategoricalAnalysisError(
            f"Unhashable category {value!r} in {context}"
        ) from exc


def _unique_values(values: Sequence[object]) -> tuple[object, ...]:
    observed: set[tuple[object, ...]] = set()
    result: list[object] = []
    for value in values:
        key = _raw_key(value)
        if key not in observed:
            observed.add(key)
            result.append(value)
    return tuple(result)
