"""Reusable, non-mutating exploration of numerical features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_STATISTICS_COLUMNS: Final[list[str]] = [
    "Feature",
    "Row count",
    "Valid numeric count",
    "Missing count",
    "Blank count",
    "Materialized count",
    "Non-numeric count",
    "Unique count",
    "Zero count",
    "Negative count",
    "Minimum",
    "Q1",
    "Median",
    "Mean",
    "Q3",
    "Maximum",
    "Range",
    "Standard deviation",
    "Variance",
    "IQR",
    "Skewness",
    "Lower fence",
    "Upper fence",
    "Lower outlier count",
    "Upper outlier count",
    "Outlier count",
    "Outlier percent",
    "Status",
]

_OUTLIER_SUMMARY_COLUMNS: Final[list[str]] = [
    "Feature",
    "Lower fence",
    "Upper fence",
    "Lower outlier count",
    "Upper outlier count",
    "Outlier count",
    "Outlier percent",
    "Interpretation",
]

_EXPLORATION_COLUMNS: Final[list[str]] = [
    "Feature",
    "Minimum",
    "Q1",
    "Median",
    "Mean",
    "Q3",
    "Maximum",
    "Standard deviation",
    "IQR",
    "Skewness",
    "Distribution shape",
    "Outlier count",
    "Outlier percent",
]

_OUTLIER_COLUMNS: Final[list[str]] = [
    "Feature",
    "Direction",
    "Value",
    "Row position",
    "Row index",
    "Lower fence",
    "Upper fence",
]

_CONVERSION_ISSUE_COLUMNS: Final[list[str]] = [
    "Feature",
    "Issue",
    "Count",
    "Raw values",
    "Row positions",
    "Row indices",
    "Potential impact",
]

_ALLOWED_RULE_KEYS: Final[set[str]] = {
    "strip_strings",
    "blank_replacement",
    "condition_column",
    "condition_value",
}


class NumericalAnalysisError(ValueError):
    """Raised when numerical-analysis configuration or data are invalid."""


@dataclass(frozen=True, slots=True)
class NumericalFeatureReport:
    """Summarize numerical projections, statistics, and candidate outliers."""

    requested_features: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    row_count: int
    iqr_multiplier: float
    statistics: pd.DataFrame
    outlier_summary: pd.DataFrame
    outliers: pd.DataFrame
    conversion_issues: pd.DataFrame
    numeric_projection: pd.DataFrame

    @property
    def has_missing_features(self) -> bool:
        """Return whether declared numerical features are absent."""
        return bool(self.missing_features)

    @property
    def has_conversion_issues(self) -> bool:
        """Return whether values could not be safely materialized as numeric."""
        return not self.conversion_issues.empty

    @property
    def has_missing_values(self) -> bool:
        """Return whether any available feature contains real missing values."""
        if self.statistics.empty:
            return False
        return bool((self.statistics["Missing count"] > 0).any())

    @property
    def has_blank_values(self) -> bool:
        """Return whether any available feature contains blank strings."""
        if self.statistics.empty:
            return False
        return bool((self.statistics["Blank count"] > 0).any())

    @property
    def has_outliers(self) -> bool:
        """Return whether the IQR rule identified candidate outliers."""
        if self.outlier_summary.empty:
            return False
        return bool((self.outlier_summary["Outlier count"] > 0).any())

    @property
    def features_with_outliers(self) -> tuple[str, ...]:
        """Return numerical features containing IQR outlier candidates."""
        if self.outlier_summary.empty:
            return ()
        mask = self.outlier_summary["Outlier count"] > 0
        return tuple(self.outlier_summary.loc[mask, "Feature"])

    @property
    def features_with_zero_values(self) -> tuple[str, ...]:
        """Return available features containing at least one zero."""
        if self.statistics.empty:
            return ()
        mask = self.statistics["Zero count"] > 0
        return tuple(self.statistics.loc[mask, "Feature"])

    @property
    def materialized_value_count(self) -> int:
        """Return the number of values materialized only in the projection."""
        if self.statistics.empty:
            return 0
        return int(self.statistics["Materialized count"].sum())

    @property
    def is_analysis_ready(self) -> bool:
        """Return whether all features exist and convert without issues."""
        return not self.has_missing_features and not self.has_conversion_issues

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic overall numerical-analysis metrics."""
        statistics = self.statistics
        rows = [
            {
                "Metric": "Requested numerical features",
                "Value": len(self.requested_features),
                "Interpretation": "Features declared by the study",
            },
            {
                "Metric": "Available numerical features",
                "Value": len(self.available_features),
                "Interpretation": "Features found in the dataset",
            },
            {
                "Metric": "Missing numerical features",
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
                    if statistics.empty
                    else int((statistics["Missing count"] > 0).sum())
                ),
                "Interpretation": "Real missing values detected by pandas",
            },
            {
                "Metric": "Features with blank values",
                "Value": (
                    0
                    if statistics.empty
                    else int((statistics["Blank count"] > 0).sum())
                ),
                "Interpretation": "Hidden text-based missingness",
            },
            {
                "Metric": "Materialized projection values",
                "Value": self.materialized_value_count,
                "Interpretation": (
                    "Values changed only in the analysis projection"
                ),
            },
            {
                "Metric": "Features with conversion issues",
                "Value": (
                    0
                    if statistics.empty
                    else int(
                        (statistics["Non-numeric count"] > 0).sum()
                    )
                ),
                "Interpretation": (
                    "Values that cannot be safely analyzed as numeric"
                ),
            },
            {
                "Metric": "Features with outlier candidates",
                "Value": len(self.features_with_outliers),
                "Interpretation": (
                    f"IQR rule with multiplier {self.iqr_multiplier:g}"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def statistics_frame(self) -> pd.DataFrame:
        """Return descriptive statistics for each available feature."""
        return self.statistics.copy(deep=True)

    def exploration_frame(
        self,
        *,
        skewness_review_threshold: float = 0.5,
        format_percentages: bool = True,
    ) -> pd.DataFrame:
        """Return a compact table for notebook numerical exploration."""
        if not isinstance(skewness_review_threshold, Real) or isinstance(
            skewness_review_threshold, bool
        ):
            raise TypeError(
                "skewness_review_threshold must be a non-negative real "
                "number."
            )

        threshold = float(skewness_review_threshold)
        if threshold < 0:
            raise ValueError(
                "skewness_review_threshold must be zero or greater."
            )

        frame = self.statistics.loc[
            :,
            [
                "Feature",
                "Minimum",
                "Q1",
                "Median",
                "Mean",
                "Q3",
                "Maximum",
                "Standard deviation",
                "IQR",
                "Skewness",
                "Outlier count",
                "Outlier percent",
            ],
        ].copy(deep=True)

        frame.insert(
            10,
            "Distribution shape",
            [
                _describe_distribution_shape(value, threshold=threshold)
                for value in frame["Skewness"]
            ],
        )

        if format_percentages:
            frame["Outlier percent"] = frame["Outlier percent"].map(
                lambda value: f"{float(value):.2%}"
            )

        return frame.loc[:, _EXPLORATION_COLUMNS]

    def outlier_summary_frame(self) -> pd.DataFrame:
        """Return IQR fences and outlier counts by feature."""
        return self.outlier_summary.copy(deep=True)

    def outliers_frame(self) -> pd.DataFrame:
        """Return sampled observations classified as candidate outliers."""
        return self.outliers.copy(deep=True)

    def conversion_issues_frame(self) -> pd.DataFrame:
        """Return values that could not be safely projected as numeric."""
        return self.conversion_issues.copy(deep=True)

    def numeric_frame(self) -> pd.DataFrame:
        """Return the non-mutating numeric analysis projection."""
        return self.numeric_projection.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_features_present: bool = True,
        require_numeric_conversion: bool = True,
        require_materialization_conditions: bool = True,
        require_no_missing_values: bool = False,
        require_no_outliers: bool = False,
    ) -> None:
        """Raise one consolidated error for selected expectations."""
        failures: list[str] = []

        if require_features_present and self.has_missing_features:
            failures.append(
                "missing numerical features: "
                + ", ".join(self.missing_features)
            )

        if require_numeric_conversion and self.has_conversion_issues:
            conversion_issues = self.conversion_issues
            non_condition = conversion_issues.loc[
                conversion_issues["Issue"]
                != "Materialization condition not satisfied"
            ]
            if not non_condition.empty:
                failures.append(
                    "numeric conversion issues found: "
                    f"{int(non_condition['Count'].sum())}"
                )

        if (
            require_materialization_conditions
            and self.has_conversion_issues
        ):
            condition_issues = self.conversion_issues.loc[
                self.conversion_issues["Issue"]
                == "Materialization condition not satisfied"
            ]
            if not condition_issues.empty:
                failures.append(
                    "materialization condition failures found: "
                    f"{int(condition_issues['Count'].sum())}"
                )

        if require_no_missing_values and self.has_missing_values:
            missing_count = int(self.statistics["Missing count"].sum())
            failures.append(f"missing numerical values found: {missing_count}")

        if require_no_outliers and self.has_outliers:
            outlier_count = int(self.outlier_summary["Outlier count"].sum())
            failures.append(f"candidate outliers found: {outlier_count}")

        if failures:
            raise NumericalAnalysisError("; ".join(failures) + ".")


def analyze_numerical_features(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    materialization_rules: Mapping[str, Mapping[str, object]] | None = None,
    iqr_multiplier: float = 1.5,
    max_outlier_samples: int = 10,
) -> NumericalFeatureReport:
    """Analyze numerical features without changing the source DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    duplicated_labels = dataframe.columns[dataframe.columns.duplicated()]
    if len(duplicated_labels) > 0:
        labels = ", ".join(str(value) for value in duplicated_labels)
        raise NumericalAnalysisError(
            "dataframe contains duplicated column labels: " + labels + "."
        )

    normalized_features = _normalize_features(features)
    normalized_rules = _normalize_materialization_rules(
        materialization_rules,
        features=normalized_features,
        dataframe=dataframe,
    )

    if not isinstance(iqr_multiplier, Real) or isinstance(
        iqr_multiplier, bool
    ):
        raise TypeError("iqr_multiplier must be a positive real number.")
    iqr_multiplier = float(iqr_multiplier)
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than zero.")

    if not isinstance(max_outlier_samples, int) or isinstance(
        max_outlier_samples, bool
    ):
        raise TypeError("max_outlier_samples must be an integer.")
    if max_outlier_samples < 0:
        raise ValueError("max_outlier_samples must be zero or greater.")

    available_features = tuple(
        feature for feature in normalized_features if feature in dataframe
    )
    missing_features = tuple(
        feature for feature in normalized_features if feature not in dataframe
    )

    projection = pd.DataFrame(index=dataframe.index)
    statistic_rows: list[dict[str, object]] = []
    outlier_summary_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []

    for feature in available_features:
        rule = normalized_rules.get(feature, {})
        materialized = _materialize_feature(
            dataframe,
            feature=feature,
            rule=rule,
        )
        numeric_series = materialized.numeric_series
        projection[feature] = numeric_series

        valid = numeric_series.dropna()
        statistics = _calculate_statistics(
            valid,
            iqr_multiplier=iqr_multiplier,
        )

        lower_mask = pd.Series(False, index=dataframe.index)
        upper_mask = pd.Series(False, index=dataframe.index)
        if statistics["Lower fence"] is not None:
            lower_mask = numeric_series < statistics["Lower fence"]
            upper_mask = numeric_series > statistics["Upper fence"]
            lower_mask = lower_mask.fillna(False)
            upper_mask = upper_mask.fillna(False)

        lower_count = int(lower_mask.sum())
        upper_count = int(upper_mask.sum())
        outlier_count = lower_count + upper_count
        valid_count = int(valid.shape[0])
        outlier_percent = (
            outlier_count / valid_count if valid_count else 0.0
        )

        non_numeric_count = int(materialized.non_numeric_mask.sum())
        condition_failure_count = int(
            materialized.condition_failure_mask.sum()
        )
        issue_count = non_numeric_count + condition_failure_count

        statistic_rows.append(
            {
                "Feature": feature,
                "Row count": len(dataframe),
                "Valid numeric count": valid_count,
                "Missing count": int(materialized.missing_mask.sum()),
                "Blank count": int(materialized.blank_mask.sum()),
                "Materialized count": int(
                    materialized.materialized_mask.sum()
                ),
                "Non-numeric count": issue_count,
                "Unique count": int(valid.nunique(dropna=True)),
                "Zero count": int((valid == 0).sum()),
                "Negative count": int((valid < 0).sum()),
                **statistics,
                "Lower outlier count": lower_count,
                "Upper outlier count": upper_count,
                "Outlier count": outlier_count,
                "Outlier percent": outlier_percent,
                "Status": (
                    "Review required" if issue_count else "Analyzed"
                ),
            }
        )

        outlier_summary_rows.append(
            {
                "Feature": feature,
                "Lower fence": statistics["Lower fence"],
                "Upper fence": statistics["Upper fence"],
                "Lower outlier count": lower_count,
                "Upper outlier count": upper_count,
                "Outlier count": outlier_count,
                "Outlier percent": outlier_percent,
                "Interpretation": (
                    "Candidate outliers require contextual review"
                    if outlier_count
                    else "No IQR outlier candidates"
                ),
            }
        )

        outlier_rows.extend(
            _build_outlier_rows(
                dataframe,
                feature=feature,
                numeric_series=numeric_series,
                lower_mask=lower_mask,
                upper_mask=upper_mask,
                lower_fence=statistics["Lower fence"],
                upper_fence=statistics["Upper fence"],
                max_samples=max_outlier_samples,
            )
        )

        issue_rows.extend(
            _build_conversion_issue_rows(
                dataframe,
                feature=feature,
                raw_series=dataframe[feature],
                non_numeric_mask=materialized.non_numeric_mask,
                condition_failure_mask=(
                    materialized.condition_failure_mask
                ),
            )
        )

    statistics_frame = pd.DataFrame(
        statistic_rows,
        columns=_STATISTICS_COLUMNS,
    )
    outlier_summary_frame = pd.DataFrame(
        outlier_summary_rows,
        columns=_OUTLIER_SUMMARY_COLUMNS,
    )
    outliers_frame = pd.DataFrame(
        outlier_rows,
        columns=_OUTLIER_COLUMNS,
    )
    conversion_issues_frame = pd.DataFrame(
        issue_rows,
        columns=_CONVERSION_ISSUE_COLUMNS,
    )

    return NumericalFeatureReport(
        requested_features=normalized_features,
        available_features=available_features,
        missing_features=missing_features,
        row_count=len(dataframe),
        iqr_multiplier=iqr_multiplier,
        statistics=statistics_frame,
        outlier_summary=outlier_summary_frame,
        outliers=outliers_frame,
        conversion_issues=conversion_issues_frame,
        numeric_projection=projection,
    )


@dataclass(frozen=True, slots=True)
class _MaterializedFeature:
    numeric_series: pd.Series
    missing_mask: pd.Series
    blank_mask: pd.Series
    materialized_mask: pd.Series
    condition_failure_mask: pd.Series
    non_numeric_mask: pd.Series


def _materialize_feature(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    rule: Mapping[str, object],
) -> _MaterializedFeature:
    raw_series = dataframe[feature]
    working = raw_series.copy(deep=True)
    missing_mask = raw_series.isna()
    blank_mask = raw_series.map(_is_blank_string).fillna(False)

    if bool(rule.get("strip_strings", False)):
        working = working.map(_strip_if_string)

    materialized_mask = pd.Series(False, index=dataframe.index)
    condition_failure_mask = pd.Series(False, index=dataframe.index)

    if "blank_replacement" in rule:
        condition_column = str(rule["condition_column"])
        condition_value = rule["condition_value"]
        condition_mask = dataframe[condition_column].eq(condition_value)
        condition_mask = condition_mask.fillna(False)
        materialized_mask = blank_mask & condition_mask
        condition_failure_mask = blank_mask & ~condition_mask

    # Remove blank strings before numeric coercion. Numeric replacements are
    # applied only to the numeric projection so pandas string extension
    # dtypes never receive floats directly.
    working = working.mask(blank_mask)
    numeric_series = pd.to_numeric(working, errors="coerce")

    if "blank_replacement" in rule:
        numeric_series = numeric_series.mask(
            materialized_mask,
            other=float(rule["blank_replacement"]),
        )

    non_numeric_mask = (
        ~missing_mask
        & ~blank_mask
        & numeric_series.isna()
    )

    return _MaterializedFeature(
        numeric_series=numeric_series.astype("float64"),
        missing_mask=missing_mask.astype(bool),
        blank_mask=blank_mask.astype(bool),
        materialized_mask=materialized_mask.astype(bool),
        condition_failure_mask=condition_failure_mask.astype(bool),
        non_numeric_mask=non_numeric_mask.astype(bool),
    )


def _calculate_statistics(
    valid: pd.Series,
    *,
    iqr_multiplier: float,
) -> dict[str, float | None]:
    if valid.empty:
        return {
            "Minimum": None,
            "Q1": None,
            "Median": None,
            "Mean": None,
            "Q3": None,
            "Maximum": None,
            "Range": None,
            "Standard deviation": None,
            "Variance": None,
            "IQR": None,
            "Skewness": None,
            "Lower fence": None,
            "Upper fence": None,
        }

    minimum = float(valid.min())
    maximum = float(valid.max())
    q1 = float(valid.quantile(0.25))
    median = float(valid.median())
    mean = float(valid.mean())
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1

    standard_deviation_value = valid.std(ddof=1)
    variance_value = valid.var(ddof=1)
    skewness_value = (
        None
        if len(valid) < 3 or valid.nunique(dropna=True) <= 1
        else valid.skew()
    )

    return {
        "Minimum": minimum,
        "Q1": q1,
        "Median": median,
        "Mean": mean,
        "Q3": q3,
        "Maximum": maximum,
        "Range": maximum - minimum,
        "Standard deviation": _optional_float(standard_deviation_value),
        "Variance": _optional_float(variance_value),
        "IQR": iqr,
        "Skewness": _optional_float(skewness_value),
        "Lower fence": q1 - iqr_multiplier * iqr,
        "Upper fence": q3 + iqr_multiplier * iqr,
    }


def _build_outlier_rows(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    numeric_series: pd.Series,
    lower_mask: pd.Series,
    upper_mask: pd.Series,
    lower_fence: float | None,
    upper_fence: float | None,
    max_samples: int,
) -> list[dict[str, object]]:
    if max_samples == 0:
        return []

    rows: list[dict[str, object]] = []
    candidate_positions = [
        position
        for position, is_outlier in enumerate(
            (lower_mask | upper_mask).tolist()
        )
        if is_outlier
    ][:max_samples]

    for position in candidate_positions:
        direction = "Lower" if bool(lower_mask.iloc[position]) else "Upper"
        rows.append(
            {
                "Feature": feature,
                "Direction": direction,
                "Value": float(numeric_series.iloc[position]),
                "Row position": position,
                "Row index": dataframe.index[position],
                "Lower fence": lower_fence,
                "Upper fence": upper_fence,
            }
        )

    return rows


def _build_conversion_issue_rows(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    raw_series: pd.Series,
    non_numeric_mask: pd.Series,
    condition_failure_mask: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    if bool(non_numeric_mask.any()):
        rows.append(
            _issue_row(
                dataframe,
                feature=feature,
                issue="Non-numeric value",
                mask=non_numeric_mask,
                raw_series=raw_series,
                potential_impact=(
                    "Prevents reliable numerical statistics and may block "
                    "later type conversion."
                ),
            )
        )

    if bool(condition_failure_mask.any()):
        rows.append(
            _issue_row(
                dataframe,
                feature=feature,
                issue="Materialization condition not satisfied",
                mask=condition_failure_mask,
                raw_series=raw_series,
                potential_impact=(
                    "A blank value cannot be safely replaced under the "
                    "declared semantic condition."
                ),
            )
        )

    return rows


def _issue_row(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    issue: str,
    mask: pd.Series,
    raw_series: pd.Series,
    potential_impact: str,
) -> dict[str, object]:
    positions = [
        position
        for position, selected in enumerate(mask.tolist())
        if selected
    ]
    sampled_positions = positions[:10]
    raw_values = tuple(
        _display_value(raw_series.iloc[position])
        for position in sampled_positions
    )
    return {
        "Feature": feature,
        "Issue": issue,
        "Count": len(positions),
        "Raw values": raw_values,
        "Row positions": tuple(sampled_positions),
        "Row indices": tuple(
            dataframe.index[position] for position in sampled_positions
        ),
        "Potential impact": potential_impact,
    }


def _normalize_features(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, str) or not isinstance(features, Sequence):
        raise TypeError("features must be a sequence of column names.")

    normalized: list[str] = []
    for feature in features:
        if not isinstance(feature, str):
            raise TypeError("feature names must be strings.")
        value = feature.strip()
        if not value:
            raise ValueError("feature names must not be blank.")
        normalized.append(value)

    if not normalized:
        raise ValueError("features must contain at least one column.")

    duplicates = _find_duplicates(normalized)
    if duplicates:
        raise ValueError(
            "features contains duplicate column names: "
            + ", ".join(duplicates)
            + "."
        )

    return tuple(normalized)


def _normalize_materialization_rules(
    rules: Mapping[str, Mapping[str, object]] | None,
    *,
    features: tuple[str, ...],
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    if rules is None:
        return {}
    if not isinstance(rules, Mapping):
        raise TypeError("materialization_rules must be a mapping.")

    normalized: dict[str, dict[str, object]] = {}
    feature_set = set(features)

    for raw_feature, raw_rule in rules.items():
        if not isinstance(raw_feature, str) or not raw_feature.strip():
            raise ValueError(
                "materialization rule feature names must be non-blank strings."
            )
        feature = raw_feature.strip()
        if feature not in feature_set:
            raise NumericalAnalysisError(
                f"materialization rule declared for non-feature {feature!r}."
            )
        if not isinstance(raw_rule, Mapping):
            raise TypeError(
                f"materialization rule for {feature!r} must be a mapping."
            )

        unknown_keys = set(raw_rule).difference(_ALLOWED_RULE_KEYS)
        if unknown_keys:
            raise NumericalAnalysisError(
                f"materialization rule for {feature!r} contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown_keys))
                + "."
            )

        rule = dict(raw_rule)
        if "strip_strings" in rule and not isinstance(
            rule["strip_strings"], bool
        ):
            raise TypeError(
                f"strip_strings for {feature!r} must be boolean."
            )

        replacement_declared = "blank_replacement" in rule
        condition_column_declared = "condition_column" in rule
        condition_value_declared = "condition_value" in rule

        if replacement_declared:
            replacement = rule["blank_replacement"]
            if not isinstance(replacement, Real) or isinstance(
                replacement, bool
            ):
                raise TypeError(
                    f"blank_replacement for {feature!r} must be numeric."
                )
            if not (
                condition_column_declared and condition_value_declared
            ):
                raise NumericalAnalysisError(
                    f"blank_replacement for {feature!r} requires both "
                    "condition_column and condition_value."
                )

        if condition_column_declared:
            condition_column = rule["condition_column"]
            if not isinstance(condition_column, str) or not condition_column.strip():
                raise ValueError(
                    f"condition_column for {feature!r} must be a non-blank string."
                )
            condition_column = condition_column.strip()
            if condition_column not in dataframe.columns:
                raise KeyError(
                    f"materialization condition column not found: "
                    f"{condition_column}."
                )
            rule["condition_column"] = condition_column

        if (
            condition_column_declared or condition_value_declared
        ) and not replacement_declared:
            raise NumericalAnalysisError(
                f"materialization condition for {feature!r} requires "
                "blank_replacement."
            )

        normalized[feature] = rule

    return normalized


def _find_duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _is_blank_string(value: object) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _strip_if_string(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    return repr(value)


def _describe_distribution_shape(
    skewness: object,
    *,
    threshold: float,
) -> str:
    """Return a descriptive label for observed skewness."""
    if skewness is None or pd.isna(skewness):
        return "Unavailable"

    skewness_value = float(skewness)
    if skewness_value >= threshold:
        return "Right-skewed"
    if skewness_value <= -threshold:
        return "Left-skewed"
    return "Approximately symmetric"


def _normalize_plot_columns(columns: int) -> int:
    if not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("columns must be an integer.")
    if columns <= 0:
        raise ValueError("columns must be greater than zero.")
    return columns


def plot_numerical_distributions(
    report: NumericalFeatureReport,
    *,
    columns: int = 4,
    bins: str | int = "auto",
    figure_width_per_column: float = 4.0,
    figure_height_per_row: float = 3.0,
):
    """Return a compact histogram overview for all numerical features."""
    from matplotlib import pyplot as plt

    if not isinstance(report, NumericalFeatureReport):
        raise TypeError("report must be a NumericalFeatureReport.")

    columns = _normalize_plot_columns(columns)
    features = report.available_features
    if not features:
        raise NumericalAnalysisError(
            "cannot plot numerical distributions without available features."
        )

    row_count = (len(features) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(
            figure_width_per_column * columns,
            figure_height_per_row * row_count,
        ),
        squeeze=False,
    )

    statistics = report.statistics.set_index("Feature")
    projection = report.numeric_projection

    for axis, feature in zip(axes.flat, features, strict=False):
        values = projection[feature].dropna()
        feature_statistics = statistics.loc[feature]

        axis.hist(values, bins=bins)

        mean = feature_statistics["Mean"]
        median = feature_statistics["Median"]

        if mean is not None and not pd.isna(mean):
            axis.axvline(
                float(mean),
                linestyle="--",
                linewidth=1,
                label="Mean",
            )
        if median is not None and not pd.isna(median):
            axis.axvline(
                float(median),
                linestyle=":",
                linewidth=1,
                label="Median",
            )

        axis.set_title(str(feature))
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize="small")

    for axis in axes.flat[len(features):]:
        axis.set_visible(False)

    figure.suptitle("Numerical Feature Distributions", y=1.01)
    figure.tight_layout()
    return figure


def plot_numerical_boxplots(
    report: NumericalFeatureReport,
    *,
    columns: int = 4,
    figure_width_per_column: float = 4.0,
    figure_height_per_row: float = 2.2,
):
    """Return a compact boxplot overview for candidate-outlier inspection."""
    from matplotlib import pyplot as plt

    if not isinstance(report, NumericalFeatureReport):
        raise TypeError("report must be a NumericalFeatureReport.")

    columns = _normalize_plot_columns(columns)
    features = report.available_features
    if not features:
        raise NumericalAnalysisError(
            "cannot plot numerical boxplots without available features."
        )

    row_count = (len(features) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(
            figure_width_per_column * columns,
            figure_height_per_row * row_count,
        ),
        squeeze=False,
    )

    projection = report.numeric_projection

    for axis, feature in zip(axes.flat, features, strict=False):
        values = projection[feature].dropna()
        if not values.empty:
            axis.boxplot(values, vert=False)
        axis.set_title(str(feature))
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.2)

    for axis in axes.flat[len(features):]:
        axis.set_visible(False)

    figure.suptitle("Numerical Feature Outlier Overview", y=1.01)
    figure.tight_layout()
    return figure
