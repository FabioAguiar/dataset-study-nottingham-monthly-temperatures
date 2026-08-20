"""Reusable, non-mutating analysis of categorical and continuous targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_DISTRIBUTION_COLUMNS: Final[list[str]] = [
    "Class",
    "Count",
    "Percentage",
    "Expected",
    "Role",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Issue",
    "Count",
    "Values",
    "Potential impact",
]


class TargetAnalysisError(ValueError):
    """Raised when target configuration or expectations are invalid."""


@dataclass(frozen=True, slots=True)
class TargetDistributionReport:
    """Summarize class counts, prevalence, imbalance, and target issues."""

    target: str
    row_count: int
    non_missing_count: int
    missing_count: int
    expected_classes: tuple[object, ...]
    positive_class: object | None
    unexpected_classes: tuple[object, ...]
    missing_expected_classes: tuple[object, ...]
    majority_classes: tuple[object, ...]
    minority_classes: tuple[object, ...]
    majority_count: int
    minority_count: int
    positive_class_count: int | None
    distribution: pd.DataFrame

    @property
    def class_count(self) -> int:
        """Return the number of classes observed at least once."""
        if self.distribution.empty:
            return 0
        return int((self.distribution["Count"] > 0).sum())

    @property
    def majority_class(self) -> object | None:
        """Return the unique majority class, or None when tied or absent."""
        if len(self.majority_classes) != 1:
            return None
        return self.majority_classes[0]

    @property
    def minority_class(self) -> object | None:
        """Return the unique minority class, or None when tied or absent."""
        if len(self.minority_classes) != 1:
            return None
        return self.minority_classes[0]

    @property
    def has_majority_tie(self) -> bool:
        """Return whether multiple classes share the largest count."""
        return len(self.majority_classes) > 1

    @property
    def has_minority_tie(self) -> bool:
        """Return whether multiple classes share the smallest positive count."""
        return len(self.minority_classes) > 1

    @property
    def majority_share(self) -> float | None:
        """Return the majority proportion among non-missing targets."""
        if self.non_missing_count == 0:
            return None
        return self.majority_count / self.non_missing_count

    @property
    def minority_share(self) -> float | None:
        """Return the minority proportion among non-missing targets."""
        if self.non_missing_count == 0:
            return None
        return self.minority_count / self.non_missing_count

    @property
    def imbalance_ratio(self) -> float | None:
        """Return majority count divided by smallest positive class count."""
        if self.minority_count <= 0:
            return None
        return self.majority_count / self.minority_count

    @property
    def positive_class_share(self) -> float | None:
        """Return positive-class prevalence among non-missing targets."""
        if (
            self.positive_class is None
            or self.positive_class_count is None
            or self.non_missing_count == 0
        ):
            return None
        return self.positive_class_count / self.non_missing_count

    @property
    def normalized_class_entropy(self) -> float | None:
        """Return normalized entropy across observed classes in [0, 1]."""
        if self.non_missing_count == 0 or self.class_count < 2:
            return None

        counts = self.distribution.loc[
            self.distribution["Count"] > 0,
            "Count",
        ]
        proportions = counts / self.non_missing_count
        entropy = -sum(
            float(value) * log(float(value))
            for value in proportions
        )
        return entropy / log(self.class_count)

    @property
    def majority_baseline_accuracy(self) -> float | None:
        """Return the accuracy of always predicting a majority class."""
        return self.majority_share

    @property
    def has_missing_values(self) -> bool:
        """Return whether the target contains missing values."""
        return self.missing_count > 0

    @property
    def has_unexpected_classes(self) -> bool:
        """Return whether observed classes were not declared as expected."""
        return bool(self.unexpected_classes)

    @property
    def has_missing_expected_classes(self) -> bool:
        """Return whether declared classes were absent from observations."""
        return bool(self.missing_expected_classes)

    @property
    def has_issues(self) -> bool:
        """Return whether missing, absent, or unexpected target values exist."""
        return (
            self.has_missing_values
            or self.has_unexpected_classes
            or self.has_missing_expected_classes
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic target-distribution metrics."""

        def percent(value: float | None) -> str:
            if value is None:
                return "Not available"
            return f"{value:.4%}"

        def class_list(values: tuple[object, ...]) -> str:
            if not values:
                return "Not available"
            return ", ".join(repr(value) for value in values)

        rows = [
            {
                "Metric": "Total rows",
                "Value": self.row_count,
                "Interpretation": "All observations",
            },
            {
                "Metric": "Non-missing target values",
                "Value": self.non_missing_count,
                "Interpretation": "Rows used for class proportions",
            },
            {
                "Metric": "Missing target values",
                "Value": self.missing_count,
                "Interpretation": (
                    "Requires review"
                    if self.has_missing_values
                    else "No missing target values"
                ),
            },
            {
                "Metric": "Observed classes",
                "Value": self.class_count,
                "Interpretation": "Classes with at least one observation",
            },
            {
                "Metric": "Majority class",
                "Value": class_list(self.majority_classes),
                "Interpretation": (
                    "Tied majority"
                    if self.has_majority_tie
                    else "Most frequent observed class"
                ),
            },
            {
                "Metric": "Minority class",
                "Value": class_list(self.minority_classes),
                "Interpretation": (
                    "Tied minority"
                    if self.has_minority_tie
                    else "Least frequent observed class"
                ),
            },
            {
                "Metric": "Majority share",
                "Value": percent(self.majority_share),
                "Interpretation": "Majority-class baseline accuracy",
            },
            {
                "Metric": "Minority share",
                "Value": percent(self.minority_share),
                "Interpretation": "Smallest observed class prevalence",
            },
            {
                "Metric": "Majority-to-minority ratio",
                "Value": (
                    "Not available"
                    if self.imbalance_ratio is None
                    else round(self.imbalance_ratio, 4)
                ),
                "Interpretation": "Descriptive imbalance indicator",
            },
            {
                "Metric": "Normalized class entropy",
                "Value": (
                    "Not available"
                    if self.normalized_class_entropy is None
                    else round(self.normalized_class_entropy, 4)
                ),
                "Interpretation": (
                    "1.0 indicates equal proportions across observed classes"
                ),
            },
        ]

        if self.positive_class is not None:
            rows.extend(
                [
                    {
                        "Metric": "Positive class",
                        "Value": repr(self.positive_class),
                        "Interpretation": "Outcome treated as positive",
                    },
                    {
                        "Metric": "Positive-class prevalence",
                        "Value": percent(self.positive_class_share),
                        "Interpretation": (
                            "Share of non-missing target values"
                        ),
                    },
                ]
            )

        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def distribution_frame(
        self,
        *,
        format_percentages: bool = False,
    ) -> pd.DataFrame:
        """Return class counts and percentages in deterministic order."""
        frame = self.distribution.copy(deep=True)
        if format_percentages:
            frame["Percentage"] = frame["Percentage"].map(
                lambda value: f"{value:.2%}"
            )
        return frame

    def issues_frame(self) -> pd.DataFrame:
        """Return target conditions requiring review."""
        rows: list[dict[str, object]] = []

        if self.has_missing_values:
            rows.append(
                {
                    "Issue": "Missing target values",
                    "Count": self.missing_count,
                    "Values": "<missing>",
                    "Potential impact": (
                        "Rows without labels cannot support supervised "
                        "training or target-based evaluation."
                    ),
                }
            )

        if self.has_missing_expected_classes:
            rows.append(
                {
                    "Issue": "Missing expected classes",
                    "Count": len(self.missing_expected_classes),
                    "Values": ", ".join(
                        repr(value)
                        for value in self.missing_expected_classes
                    ),
                    "Potential impact": (
                        "The observed dataset does not cover every declared "
                        "outcome class."
                    ),
                }
            )

        if self.has_unexpected_classes:
            unexpected_count = int(
                self.distribution.loc[
                    self.distribution["Class"].isin(
                        self.unexpected_classes
                    ),
                    "Count",
                ].sum()
            )
            rows.append(
                {
                    "Issue": "Unexpected classes",
                    "Count": unexpected_count,
                    "Values": ", ".join(
                        repr(value) for value in self.unexpected_classes
                    ),
                    "Potential impact": (
                        "Unsupported labels may invalidate target encoding "
                        "and evaluation assumptions."
                    ),
                }
            )

        return pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

    def raise_if_invalid(
        self,
        *,
        require_no_missing_target: bool = True,
        require_expected_classes_present: bool = True,
        require_no_unexpected_classes: bool = True,
    ) -> None:
        """Raise when configured target expectations are not satisfied."""
        failures: list[str] = []

        if require_no_missing_target and self.has_missing_values:
            failures.append(
                f"missing_target_values:{self.missing_count}"
            )

        if (
            require_expected_classes_present
            and self.has_missing_expected_classes
        ):
            failures.append(
                "missing_expected_classes:"
                + ",".join(
                    repr(value)
                    for value in self.missing_expected_classes
                )
            )

        if require_no_unexpected_classes and self.has_unexpected_classes:
            failures.append(
                "unexpected_classes:"
                + ",".join(
                    repr(value) for value in self.unexpected_classes
                )
            )

        if failures:
            raise TargetAnalysisError(
                "Target distribution validation failed: "
                + "; ".join(failures)
            )


def analyze_target_distribution(
    dataframe: pd.DataFrame,
    *,
    target: str,
    expected_classes: Sequence[object] | None = None,
    positive_class: object | None = None,
) -> TargetDistributionReport:
    """Analyze a categorical target without modifying the source DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_column_name(target, field="target")
    _require_unique_columns(dataframe)

    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    normalized_expected = _normalize_expected_classes(expected_classes)

    if positive_class is not None and _is_missing_scalar(positive_class):
        raise TargetAnalysisError(
            "positive_class must not be a missing value."
        )

    if (
        positive_class is not None
        and normalized_expected
        and not _contains_value(normalized_expected, positive_class)
    ):
        raise TargetAnalysisError(
            "positive_class must be included in expected_classes."
        )

    target_series = dataframe[target_name]
    missing_mask = target_series.isna()
    observed = target_series.loc[~missing_mask]

    observed_classes = tuple(pd.unique(observed))
    class_order = list(normalized_expected)
    class_order.extend(
        value
        for value in observed_classes
        if not _contains_value(class_order, value)
    )

    unexpected_classes = tuple(
        value
        for value in observed_classes
        if normalized_expected
        and not _contains_value(normalized_expected, value)
    )
    missing_expected_classes = tuple(
        value
        for value in normalized_expected
        if not _contains_value(observed_classes, value)
    )

    counts = [
        int(_value_mask(observed, class_value).sum())
        for class_value in class_order
    ]
    positive_counts = [count for count in counts if count > 0]

    if positive_counts:
        majority_count = max(positive_counts)
        minority_count = min(positive_counts)
        majority_classes = tuple(
            class_value
            for class_value, count in zip(
                class_order,
                counts,
                strict=True,
            )
            if count == majority_count
        )
        minority_classes = tuple(
            class_value
            for class_value, count in zip(
                class_order,
                counts,
                strict=True,
            )
            if count == minority_count
        )
    else:
        majority_count = 0
        minority_count = 0
        majority_classes = ()
        minority_classes = ()

    non_missing_count = int((~missing_mask).sum())
    distribution_rows: list[dict[str, object]] = []

    for class_value, count in zip(class_order, counts, strict=True):
        distribution_rows.append(
            {
                "Class": class_value,
                "Count": count,
                "Percentage": (
                    count / non_missing_count
                    if non_missing_count
                    else 0.0
                ),
                "Expected": (
                    True
                    if not normalized_expected
                    else _contains_value(
                        normalized_expected,
                        class_value,
                    )
                ),
                "Role": _class_role(
                    class_value=class_value,
                    count=count,
                    positive_class=positive_class,
                    majority_classes=majority_classes,
                    minority_classes=minority_classes,
                ),
            }
        )

    positive_class_count = None
    if positive_class is not None:
        positive_class_count = int(
            _value_mask(observed, positive_class).sum()
        )

    distribution = pd.DataFrame(
        distribution_rows,
        columns=_DISTRIBUTION_COLUMNS,
    )

    return TargetDistributionReport(
        target=target_name,
        row_count=len(dataframe),
        non_missing_count=non_missing_count,
        missing_count=int(missing_mask.sum()),
        expected_classes=normalized_expected,
        positive_class=positive_class,
        unexpected_classes=unexpected_classes,
        missing_expected_classes=missing_expected_classes,
        majority_classes=majority_classes,
        minority_classes=minority_classes,
        majority_count=majority_count,
        minority_count=minority_count,
        positive_class_count=positive_class_count,
        distribution=distribution,
    )


def plot_target_distribution(
    report: TargetDistributionReport,
    *,
    title: str,
    xlabel: str = "Class",
    ylabel: str = "Observation count",
):
    """Create a compact class-count chart with exact share labels."""
    if not isinstance(report, TargetDistributionReport):
        raise TypeError("report must be a TargetDistributionReport.")
    if not isinstance(title, str) or not title.strip():
        raise TargetAnalysisError("title must be a non-empty string.")

    from matplotlib import pyplot as plt

    distribution = report.distribution_frame()
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(
        distribution["Class"].astype(str),
        distribution["Count"],
    )
    labels = [
        f"{count:,}\n({percentage:.2%})"
        for count, percentage in zip(
            distribution["Count"],
            distribution["Percentage"],
            strict=True,
        )
    ]
    axis.bar_label(bars, labels=labels, padding=4)
    axis.set_title(title.strip())
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if not distribution.empty:
        maximum = int(distribution["Count"].max())
        if maximum > 0:
            axis.set_ylim(0, maximum * 1.18)

    figure.tight_layout()
    return figure



@dataclass(frozen=True, slots=True)
class ContinuousTargetDistributionReport:
    """Summarize a numeric target's distribution, range, and extremes."""

    target: str
    unit: str | None
    row_count: int
    non_missing_count: int
    missing_count: int
    finite_count: int
    non_finite_count: int
    unique_count: int
    minimum: float | None
    q01: float | None
    q05: float | None
    q25: float | None
    median: float | None
    mean: float | None
    q75: float | None
    q95: float | None
    q99: float | None
    maximum: float | None
    standard_deviation: float | None
    iqr: float | None
    lower_tukey_fence: float | None
    upper_tukey_fence: float | None
    lower_extreme_count: int
    upper_extreme_count: int
    finite_values: tuple[float, ...]

    @property
    def observed_range(self) -> float | None:
        """Return maximum minus minimum for finite target values."""
        if self.minimum is None or self.maximum is None:
            return None
        return self.maximum - self.minimum

    @property
    def has_missing_values(self) -> bool:
        """Return whether the target contains missing values."""
        return self.missing_count > 0

    @property
    def has_non_finite_values(self) -> bool:
        """Return whether non-missing target values include infinities."""
        return self.non_finite_count > 0

    @property
    def has_variation(self) -> bool:
        """Return whether at least two distinct finite target values exist."""
        return self.unique_count > 1

    @property
    def extreme_count(self) -> int:
        """Return observations outside the descriptive 1.5-IQR fences."""
        return self.lower_extreme_count + self.upper_extreme_count

    @property
    def extreme_share(self) -> float | None:
        """Return share of finite observations outside 1.5-IQR fences."""
        if self.finite_count == 0:
            return None
        return self.extreme_count / self.finite_count

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic summary metrics for the continuous target."""

        def metric(value: float | None) -> object:
            if value is None:
                return "Not available"
            return round(value, 6)

        rows = [
            {
                "Metric": "Total rows",
                "Value": self.row_count,
                "Interpretation": "All observations",
            },
            {
                "Metric": "Finite target values",
                "Value": self.finite_count,
                "Interpretation": "Rows used for numerical summaries",
            },
            {
                "Metric": "Missing target values",
                "Value": self.missing_count,
                "Interpretation": (
                    "Requires review"
                    if self.has_missing_values
                    else "No missing target values"
                ),
            },
            {
                "Metric": "Non-finite target values",
                "Value": self.non_finite_count,
                "Interpretation": (
                    "Requires review"
                    if self.has_non_finite_values
                    else "All non-missing target values are finite"
                ),
            },
            {
                "Metric": "Distinct finite values",
                "Value": self.unique_count,
                "Interpretation": "Observed target-value diversity",
            },
            {
                "Metric": "Minimum",
                "Value": metric(self.minimum),
                "Interpretation": _with_unit("Observed minimum", self.unit),
            },
            {
                "Metric": "Mean",
                "Value": metric(self.mean),
                "Interpretation": _with_unit("Arithmetic mean", self.unit),
            },
            {
                "Metric": "Median",
                "Value": metric(self.median),
                "Interpretation": _with_unit("50th percentile", self.unit),
            },
            {
                "Metric": "Maximum",
                "Value": metric(self.maximum),
                "Interpretation": _with_unit("Observed maximum", self.unit),
            },
            {
                "Metric": "Observed range",
                "Value": metric(self.observed_range),
                "Interpretation": _with_unit("Maximum minus minimum", self.unit),
            },
            {
                "Metric": "Standard deviation",
                "Value": metric(self.standard_deviation),
                "Interpretation": _with_unit("Sample standard deviation", self.unit),
            },
            {
                "Metric": "Interquartile range",
                "Value": metric(self.iqr),
                "Interpretation": _with_unit("Q3 minus Q1", self.unit),
            },
            {
                "Metric": "Outside 1.5-IQR fences",
                "Value": self.extreme_count,
                "Interpretation": (
                    "Descriptive extreme-value signal; not an automatic "
                    "outlier-removal rule"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def quantiles_frame(self) -> pd.DataFrame:
        """Return selected quantiles in deterministic order."""
        rows = [
            ("1%", self.q01),
            ("5%", self.q05),
            ("25%", self.q25),
            ("50%", self.median),
            ("75%", self.q75),
            ("95%", self.q95),
            ("99%", self.q99),
        ]
        return pd.DataFrame(
            [
                {
                    "Quantile": label,
                    "Value": (
                        "Not available"
                        if value is None
                        else round(value, 6)
                    ),
                    "Unit": self.unit or "Not specified",
                }
                for label, value in rows
            ],
            columns=["Quantile", "Value", "Unit"],
        )

    def extremes_frame(self) -> pd.DataFrame:
        """Return descriptive Tukey-fence evidence without removal advice."""
        rows = [
            {
                "Side": "Lower",
                "Fence": self.lower_tukey_fence,
                "Observed extreme": self.minimum,
                "Count outside fence": self.lower_extreme_count,
            },
            {
                "Side": "Upper",
                "Fence": self.upper_tukey_fence,
                "Observed extreme": self.maximum,
                "Count outside fence": self.upper_extreme_count,
            },
        ]
        frame = pd.DataFrame(rows)
        for column in ("Fence", "Observed extreme"):
            frame[column] = frame[column].map(
                lambda value: (
                    "Not available"
                    if value is None
                    else round(float(value), 6)
                )
            )
        return frame

    def issues_frame(self) -> pd.DataFrame:
        """Return target conditions that invalidate numeric interpretation."""
        rows: list[dict[str, object]] = []
        if self.has_missing_values:
            rows.append(
                {
                    "Issue": "Missing target values",
                    "Count": self.missing_count,
                    "Values": "<missing>",
                    "Potential impact": (
                        "Rows without target values cannot support supervised "
                        "regression training or target-based evaluation."
                    ),
                }
            )
        if self.has_non_finite_values:
            rows.append(
                {
                    "Issue": "Non-finite target values",
                    "Count": self.non_finite_count,
                    "Values": "<+/-inf>",
                    "Potential impact": (
                        "Infinite target values invalidate ordinary regression "
                        "losses and numerical summary statistics."
                    ),
                }
            )
        if self.finite_count > 0 and not self.has_variation:
            rows.append(
                {
                    "Issue": "Constant target",
                    "Count": self.finite_count,
                    "Values": repr(self.minimum),
                    "Potential impact": (
                        "A constant target does not define a meaningful "
                        "continuous regression prediction problem."
                    ),
                }
            )
        return pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

    def raise_if_invalid(
        self,
        *,
        require_no_missing_target: bool = True,
        require_all_values_finite: bool = True,
        require_variation: bool = True,
    ) -> None:
        """Raise when configured continuous-target expectations fail."""
        failures: list[str] = []
        if require_no_missing_target and self.has_missing_values:
            failures.append(f"missing_target_values:{self.missing_count}")
        if require_all_values_finite and self.has_non_finite_values:
            failures.append(f"non_finite_target_values:{self.non_finite_count}")
        if require_variation and self.finite_count > 0 and not self.has_variation:
            failures.append("constant_target")
        if self.finite_count == 0:
            failures.append("no_finite_target_values")
        if failures:
            raise TargetAnalysisError(
                "Continuous target validation failed: " + "; ".join(failures)
            )


def analyze_continuous_target_distribution(
    dataframe: pd.DataFrame,
    *,
    target: str,
    unit: str | None = None,
) -> ContinuousTargetDistributionReport:
    """Analyze a continuous numeric target without mutating the DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    target_name = _normalize_column_name(target, field="target")
    _require_unique_columns(dataframe)
    if target_name not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_name!r}")

    normalized_unit = None
    if unit is not None:
        normalized_unit = _normalize_column_name(unit, field="unit")

    target_series = dataframe[target_name]
    missing_mask = target_series.isna()
    observed = target_series.loc[~missing_mask]
    numeric = pd.to_numeric(observed, errors="coerce")

    conversion_failures = numeric.isna()
    if bool(conversion_failures.any()):
        raise TargetAnalysisError(
            "Continuous target contains non-numeric non-missing values: "
            f"{int(conversion_failures.sum())}"
        )

    finite_mask = numeric.map(lambda value: isfinite(float(value)))
    finite = numeric.loc[finite_mask].astype(float)
    finite_count = int(finite.shape[0])
    non_finite_count = int((~finite_mask).sum())

    if finite_count:
        quantiles = finite.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        minimum = float(finite.min())
        maximum = float(finite.max())
        mean = float(finite.mean())
        median = float(quantiles.loc[0.50])
        q01 = float(quantiles.loc[0.01])
        q05 = float(quantiles.loc[0.05])
        q25 = float(quantiles.loc[0.25])
        q75 = float(quantiles.loc[0.75])
        q95 = float(quantiles.loc[0.95])
        q99 = float(quantiles.loc[0.99])
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        lower_extreme_count = int((finite < lower_fence).sum())
        upper_extreme_count = int((finite > upper_fence).sum())
        standard_deviation_value = finite.std(ddof=1)
        standard_deviation = (
            None
            if pd.isna(standard_deviation_value)
            else float(standard_deviation_value)
        )
        finite_values = tuple(float(value) for value in finite.tolist())
    else:
        minimum = maximum = mean = median = None
        q01 = q05 = q25 = q75 = q95 = q99 = None
        iqr = lower_fence = upper_fence = None
        lower_extreme_count = upper_extreme_count = 0
        standard_deviation = None
        finite_values = ()

    return ContinuousTargetDistributionReport(
        target=target_name,
        unit=normalized_unit,
        row_count=len(dataframe),
        non_missing_count=int((~missing_mask).sum()),
        missing_count=int(missing_mask.sum()),
        finite_count=finite_count,
        non_finite_count=non_finite_count,
        unique_count=int(finite.nunique(dropna=True)),
        minimum=minimum,
        q01=q01,
        q05=q05,
        q25=q25,
        median=median,
        mean=mean,
        q75=q75,
        q95=q95,
        q99=q99,
        maximum=maximum,
        standard_deviation=standard_deviation,
        iqr=iqr,
        lower_tukey_fence=lower_fence,
        upper_tukey_fence=upper_fence,
        lower_extreme_count=lower_extreme_count,
        upper_extreme_count=upper_extreme_count,
        finite_values=finite_values,
    )


def plot_continuous_target_distribution(
    report: ContinuousTargetDistributionReport,
    *,
    title: str,
    bins: int = 30,
):
    """Create a histogram for a continuous target on its original scale."""
    if not isinstance(report, ContinuousTargetDistributionReport):
        raise TypeError("report must be a ContinuousTargetDistributionReport.")
    if not isinstance(title, str) or not title.strip():
        raise TargetAnalysisError("title must be a non-empty string.")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise TargetAnalysisError("bins must be a positive integer.")
    if not report.finite_values:
        raise TargetAnalysisError("report contains no finite target values to plot.")

    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(report.finite_values, bins=bins)
    axis.set_title(title.strip())
    xlabel = report.target
    if report.unit:
        xlabel += f" ({report.unit})"
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Observation count")
    figure.tight_layout()
    return figure


def _with_unit(label: str, unit: str | None) -> str:
    if not unit:
        return label
    return f"{label} ({unit})"


def _class_role(
    *,
    class_value: object,
    count: int,
    positive_class: object | None,
    majority_classes: tuple[object, ...],
    minority_classes: tuple[object, ...],
) -> str:
    roles: list[str] = []

    if count == 0:
        roles.append("Absent expected")
    elif _contains_value(majority_classes, class_value):
        if len(majority_classes) > 1:
            roles.append("Tied")
        else:
            roles.append("Majority")
    elif _contains_value(minority_classes, class_value):
        if len(minority_classes) > 1:
            roles.append("Tied minority")
        else:
            roles.append("Minority")
    else:
        roles.append("Intermediate")

    if positive_class is not None:
        roles.append(
            "Positive"
            if _values_equal(class_value, positive_class)
            else "Negative"
        )

    return " / ".join(roles)


def _normalize_column_name(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise TargetAnalysisError(f"{field} must not be empty.")
    return normalized


def _normalize_expected_classes(
    values: Sequence[object] | None,
) -> tuple[object, ...]:
    if values is None:
        return ()

    if isinstance(values, (str, bytes)):
        raise TypeError(
            "expected_classes must be a sequence of class values, "
            "not a string."
        )

    normalized = tuple(values)
    for value in normalized:
        if _is_missing_scalar(value):
            raise TargetAnalysisError(
                "expected_classes must not contain missing values."
            )

    for index, value in enumerate(normalized):
        if _contains_value(normalized[:index], value):
            raise TargetAnalysisError(
                f"expected_classes contains a duplicate value: {value!r}"
            )

    return normalized


def _require_unique_columns(dataframe: pd.DataFrame) -> None:
    duplicated = dataframe.columns[
        dataframe.columns.duplicated(keep=False)
    ].tolist()
    if duplicated:
        labels = ", ".join(repr(value) for value in duplicated)
        raise TargetAnalysisError(
            f"DataFrame contains duplicated column labels: {labels}"
        )


def _value_mask(series: pd.Series, value: object) -> pd.Series:
    try:
        mask = series.eq(value)
    except (TypeError, ValueError):
        mask = series.map(lambda item: _values_equal(item, value))

    return mask.fillna(False).astype(bool)


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
