"""Reusable multiclass class-profile, separation, and overlap exploration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from numbers import Integral
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_CLASS_STATISTIC_COLUMNS: Final[list[str]] = [
    "Target class",
    "Feature",
    "Row count",
    "Mean",
    "Median",
    "Q1",
    "Q3",
    "IQR",
    "Robust standardized median",
]

_PAIRWISE_COLUMNS: Final[list[str]] = [
    "Class A",
    "Class B",
    "RMS robust median gap",
    "Mean IQR overlap coefficient",
    "IQR-overlap features",
    "IQR-non-overlap features",
    "Feature count",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Item",
    "Issue",
    "Details",
    "Potential impact",
]


class ClassProfileAnalysisError(ValueError):
    """Raised when multiclass profile analysis cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class MulticlassClassProfileReport:
    """Summarize class profiles and exploratory multivariate separation."""

    requested_features: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    target_name: str
    expected_target_classes: tuple[object, ...]
    observed_target_classes: tuple[object, ...]
    unexpected_target_classes: tuple[object, ...]
    missing_expected_target_classes: tuple[object, ...]
    row_count: int
    missing_target_count: int
    class_statistics: pd.DataFrame
    pairwise_comparisons: pd.DataFrame
    pca_projection: pd.DataFrame
    pca_explained_variance_ratio: tuple[float, float]
    issues: pd.DataFrame

    @property
    def has_missing_features(self) -> bool:
        return bool(self.missing_features)

    @property
    def has_missing_target_values(self) -> bool:
        return self.missing_target_count > 0

    @property
    def has_unexpected_target_classes(self) -> bool:
        return bool(self.unexpected_target_classes)

    @property
    def has_missing_expected_target_classes(self) -> bool:
        return bool(self.missing_expected_target_classes)

    @property
    def has_non_numeric_or_missing_features(self) -> bool:
        if self.issues.empty:
            return False
        return bool(
            self.issues["Issue"].isin(
                {
                    "Non-numeric or missing feature values",
                    "Constant numerical feature",
                    "Zero global IQR",
                }
            ).any()
        )

    @property
    def is_analysis_ready(self) -> bool:
        return not (
            self.has_missing_features
            or self.has_missing_target_values
            or self.has_unexpected_target_classes
            or self.has_missing_expected_target_classes
            or self.has_non_numeric_or_missing_features
            or len(self.expected_target_classes) < 3
            or len(self.available_features) < 2
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return compact class-profile and PCA summary metrics."""
        pairwise = self.pairwise_overlap_frame()
        closest_pair = None
        closest_gap = None
        greatest_overlap_pair = None
        greatest_overlap = None

        if not pairwise.empty:
            closest = pairwise.sort_values(
                ["RMS robust median gap"],
                ascending=[True],
                kind="stable",
            ).iloc[0]
            closest_pair = f"{closest['Class A']} vs {closest['Class B']}"
            closest_gap = closest["RMS robust median gap"]

            overlap = pairwise.iloc[0]
            greatest_overlap_pair = f"{overlap['Class A']} vs {overlap['Class B']}"
            greatest_overlap = overlap["Mean IQR overlap coefficient"]

        pc1, pc2 = self.pca_explained_variance_ratio
        rows = [
            {
                "Metric": "Rows",
                "Value": self.row_count,
                "Interpretation": "Observations supplied to the analysis",
            },
            {
                "Metric": "Numerical features",
                "Value": len(self.requested_features),
                "Interpretation": "Features used to describe class profiles",
            },
            {
                "Metric": "Target classes",
                "Value": len(self.expected_target_classes),
                "Interpretation": "Classes declared by the target contract",
            },
            {
                "Metric": "Pairwise class comparisons",
                "Value": len(self.pairwise_comparisons),
                "Interpretation": "All unique unordered class pairs",
            },
            {
                "Metric": "Closest central-profile pair",
                "Value": closest_pair,
                "Interpretation": "Smallest RMS robust median gap",
            },
            {
                "Metric": "Closest central-profile gap",
                "Value": closest_gap,
                "Interpretation": "Distance measured in global-IQR units",
            },
            {
                "Metric": "Greatest central-overlap pair",
                "Value": greatest_overlap_pair,
                "Interpretation": "Largest mean class-IQR overlap coefficient",
            },
            {
                "Metric": "Greatest mean IQR overlap",
                "Value": greatest_overlap,
                "Interpretation": "0=no central overlap; 1=complete smaller-IQR overlap",
            },
            {
                "Metric": "PCA PC1 explained variance",
                "Value": pc1,
                "Interpretation": "Exploratory standardized-feature projection",
            },
            {
                "Metric": "PCA PC2 explained variance",
                "Value": pc2,
                "Interpretation": "Exploratory standardized-feature projection",
            },
            {
                "Metric": "PCA two-component explained variance",
                "Value": pc1 + pc2,
                "Interpretation": "Share represented by the 2D visualization",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def class_statistics_frame(self) -> pd.DataFrame:
        """Return per-class descriptive statistics in contract order."""
        return self.class_statistics.copy(deep=True)

    def standardized_profile_frame(self) -> pd.DataFrame:
        """Return class medians standardized by each feature's global IQR."""
        if self.class_statistics.empty:
            return pd.DataFrame(
                index=pd.Index([], name="Target class"),
                columns=list(self.requested_features),
                dtype=float,
            )

        frame = self.class_statistics.pivot(
            index="Target class",
            columns="Feature",
            values="Robust standardized median",
        )
        frame = frame.reindex(
            index=list(self.expected_target_classes),
            columns=list(self.requested_features),
        )
        frame.index.name = "Target class"
        frame.columns.name = None
        return frame.astype(float)

    def pairwise_overlap_frame(self, *, limit: int | None = None) -> pd.DataFrame:
        """Return class pairs ordered from greatest central overlap."""
        frame = self.pairwise_comparisons.sort_values(
            [
                "Mean IQR overlap coefficient",
                "RMS robust median gap",
            ],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        if limit is None:
            return frame
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ClassProfileAnalysisError("limit must be a positive integer.")
        return frame.head(int(limit)).reset_index(drop=True)

    def pca_projection_frame(self) -> pd.DataFrame:
        """Return the full exploratory two-component PCA projection."""
        return self.pca_projection.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return structural issues that limit interpretation."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_features_present: bool = True,
        require_multiclass_target: bool = True,
        require_no_missing_target: bool = True,
        require_expected_target_classes: bool = True,
        require_no_unexpected_target_classes: bool = True,
        require_numeric_complete_features: bool = True,
    ) -> None:
        """Raise when configured class-profile requirements are not met."""
        failures: list[str] = []

        if require_features_present and self.missing_features:
            failures.append("missing_features:" + ",".join(self.missing_features))
        if require_multiclass_target and len(self.expected_target_classes) < 3:
            failures.append("target_contract_is_not_multiclass")
        if require_no_missing_target and self.has_missing_target_values:
            failures.append(f"missing_target_values:{self.missing_target_count}")
        if require_expected_target_classes and self.missing_expected_target_classes:
            failures.append(
                "missing_expected_target_classes:"
                + ",".join(
                    repr(value) for value in self.missing_expected_target_classes
                )
            )
        if require_no_unexpected_target_classes and self.unexpected_target_classes:
            failures.append(
                "unexpected_target_classes:"
                + ",".join(repr(value) for value in self.unexpected_target_classes)
            )
        if require_numeric_complete_features and self.has_non_numeric_or_missing_features:
            failures.append("non_numeric_missing_or_constant_features_detected")
        if require_features_present and len(self.available_features) < 2:
            failures.append("at_least_two_features_required")

        if failures:
            raise ClassProfileAnalysisError(
                "Multiclass class-profile analysis is invalid: "
                + "; ".join(failures)
            )


def analyze_multiclass_class_profiles(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    expected_target_classes: Sequence[object],
) -> MulticlassClassProfileReport:
    """Analyze robust class profiles, pairwise overlap, and a 2D PCA view.

    Class profiles use each class median centered by the global feature median
    and scaled by the global feature IQR. Pairwise overlap compares the class
    IQRs feature by feature. PCA is exploratory only and is fitted to globally
    standardized features solely to support visualization in this notebook.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ClassProfileAnalysisError("dataframe must be a pandas DataFrame.")
    if dataframe.columns.duplicated().any():
        raise ClassProfileAnalysisError(
            "dataframe must not contain duplicated column labels."
        )

    requested_features = _normalize_feature_names(features)
    if len(requested_features) < 2:
        raise ClassProfileAnalysisError("features must contain at least two columns.")
    if not isinstance(target, str) or not target.strip():
        raise ClassProfileAnalysisError("target must be a non-empty string.")
    target_name = target.strip()
    if target_name in requested_features:
        raise ClassProfileAnalysisError("target must not be included in features.")

    expected_classes = _normalize_values(expected_target_classes)
    if len(expected_classes) < 3:
        raise ClassProfileAnalysisError(
            "expected_target_classes must contain at least three classes."
        )
    if target_name not in dataframe.columns:
        raise ClassProfileAnalysisError(
            f"Target column {target_name!r} is not present in dataframe."
        )

    source = dataframe.copy(deep=True)
    available_features = tuple(
        feature for feature in requested_features if feature in source.columns
    )
    missing_features = tuple(
        feature for feature in requested_features if feature not in source.columns
    )

    target_source = source[target_name].copy(deep=True)
    observed_classes = tuple(pd.unique(target_source.dropna()))
    unexpected_classes = tuple(
        value for value in observed_classes if value not in expected_classes
    )
    missing_expected_classes = tuple(
        value for value in expected_classes if value not in observed_classes
    )
    missing_target_count = int(target_source.isna().sum())

    issues: list[dict[str, object]] = []
    for feature in missing_features:
        issues.append(
            {
                "Scope": "Feature contract",
                "Item": feature,
                "Issue": "Missing feature",
                "Details": "Requested feature is absent from dataframe.",
                "Potential impact": "Class profiles cannot cover the declared feature set.",
            }
        )

    if missing_target_count:
        issues.append(
            {
                "Scope": "Target contract",
                "Item": target_name,
                "Issue": "Missing target values",
                "Details": f"count={missing_target_count}",
                "Potential impact": "Unlabelled rows cannot support class profiles.",
            }
        )
    if unexpected_classes:
        issues.append(
            {
                "Scope": "Target contract",
                "Item": target_name,
                "Issue": "Unexpected target classes",
                "Details": ", ".join(repr(value) for value in unexpected_classes),
                "Potential impact": "Observed outcomes do not match the declared contract.",
            }
        )
    if missing_expected_classes:
        issues.append(
            {
                "Scope": "Target contract",
                "Item": target_name,
                "Issue": "Missing expected target classes",
                "Details": ", ".join(repr(value) for value in missing_expected_classes),
                "Potential impact": "The full multiclass contract is not represented.",
            }
        )

    numeric_frame = pd.DataFrame(index=source.index)
    numeric_complete = not missing_features
    if not missing_features:
        for feature in available_features:
            converted = pd.to_numeric(source[feature], errors="coerce")
            invalid_count = int(converted.isna().sum())
            if invalid_count:
                numeric_complete = False
                issues.append(
                    {
                        "Scope": "Feature values",
                        "Item": feature,
                        "Issue": "Non-numeric or missing feature values",
                        "Details": f"count={invalid_count}",
                        "Potential impact": (
                            "Robust profiles and standardized PCA require complete numeric values."
                        ),
                    }
                )
            if converted.nunique(dropna=True) < 2:
                numeric_complete = False
                issues.append(
                    {
                        "Scope": "Feature values",
                        "Item": feature,
                        "Issue": "Constant numerical feature",
                        "Details": "Observed feature has fewer than two distinct numeric values.",
                        "Potential impact": "Robust scaling and separation evidence are undefined.",
                    }
                )
            numeric_frame[feature] = converted

    can_analyze = (
        not missing_features
        and numeric_complete
        and missing_target_count == 0
        and not unexpected_classes
        and not missing_expected_classes
    )

    class_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    pca_projection = pd.DataFrame(columns=["PC1", "PC2", target_name])
    pca_explained_variance_ratio = (0.0, 0.0)

    if can_analyze:
        global_medians = numeric_frame.median()
        global_q1 = numeric_frame.quantile(0.25)
        global_q3 = numeric_frame.quantile(0.75)
        global_iqr = global_q3 - global_q1
        zero_iqr_features = [
            feature for feature in available_features if global_iqr[feature] <= 0
        ]
        if zero_iqr_features:
            for feature in zero_iqr_features:
                issues.append(
                    {
                        "Scope": "Feature values",
                        "Item": feature,
                        "Issue": "Zero global IQR",
                        "Details": "Global feature IQR is zero despite observed variation.",
                        "Potential impact": "Robust profile standardization is undefined.",
                    }
                )
            can_analyze = False

    if can_analyze:
        class_quantiles: dict[object, dict[str, dict[str, float]]] = {}
        for target_class in expected_classes:
            class_mask = target_source.eq(target_class)
            class_numeric = numeric_frame.loc[class_mask, list(available_features)]
            class_quantiles[target_class] = {}

            for feature in available_features:
                series = class_numeric[feature]
                q1 = float(series.quantile(0.25))
                median = float(series.median())
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                robust_median = float(
                    (median - global_medians[feature]) / global_iqr[feature]
                )
                class_quantiles[target_class][feature] = {
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "robust_median": robust_median,
                }
                class_rows.append(
                    {
                        "Target class": target_class,
                        "Feature": feature,
                        "Row count": len(series),
                        "Mean": float(series.mean()),
                        "Median": median,
                        "Q1": q1,
                        "Q3": q3,
                        "IQR": iqr,
                        "Robust standardized median": robust_median,
                    }
                )

        for class_a, class_b in combinations(expected_classes, 2):
            squared_gaps: list[float] = []
            overlaps: list[float] = []
            overlap_count = 0

            for feature in available_features:
                a = class_quantiles[class_a][feature]
                b = class_quantiles[class_b][feature]
                gap = a["robust_median"] - b["robust_median"]
                squared_gaps.append(gap * gap)

                overlap = _iqr_overlap_coefficient(
                    a["q1"], a["q3"], b["q1"], b["q3"]
                )
                overlaps.append(overlap)
                if overlap > 0:
                    overlap_count += 1

            feature_count = len(available_features)
            pairwise_rows.append(
                {
                    "Class A": class_a,
                    "Class B": class_b,
                    "RMS robust median gap": sqrt(
                        sum(squared_gaps) / feature_count
                    ),
                    "Mean IQR overlap coefficient": sum(overlaps) / feature_count,
                    "IQR-overlap features": overlap_count,
                    "IQR-non-overlap features": feature_count - overlap_count,
                    "Feature count": feature_count,
                }
            )

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        standardized = StandardScaler().fit_transform(
            numeric_frame.loc[:, list(available_features)]
        )
        pca = PCA(n_components=2, svd_solver="full")
        coordinates = pca.fit_transform(standardized)
        pca_projection = pd.DataFrame(
            {
                "PC1": coordinates[:, 0],
                "PC2": coordinates[:, 1],
                target_name: target_source.to_numpy(copy=True),
            },
            index=source.index,
        )
        pca_explained_variance_ratio = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )

    return MulticlassClassProfileReport(
        requested_features=requested_features,
        available_features=available_features,
        missing_features=missing_features,
        target_name=target_name,
        expected_target_classes=expected_classes,
        observed_target_classes=observed_classes,
        unexpected_target_classes=unexpected_classes,
        missing_expected_target_classes=missing_expected_classes,
        row_count=len(source),
        missing_target_count=missing_target_count,
        class_statistics=pd.DataFrame(class_rows, columns=_CLASS_STATISTIC_COLUMNS),
        pairwise_comparisons=pd.DataFrame(pairwise_rows, columns=_PAIRWISE_COLUMNS),
        pca_projection=pca_projection,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
    )


def plot_standardized_class_profiles(
    report: MulticlassClassProfileReport,
    *,
    title: str = "Robust Standardized Class Profiles",
):
    """Plot class median profiles in global-IQR units."""
    if not isinstance(report, MulticlassClassProfileReport):
        raise ClassProfileAnalysisError(
            "report must be a MulticlassClassProfileReport."
        )
    report.raise_if_invalid()

    import matplotlib.pyplot as plt

    matrix = report.standardized_profile_frame()
    width = max(10.0, 0.7 * len(matrix.columns))
    height = max(4.5, 0.6 * len(matrix.index))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix.to_numpy(), aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Numerical feature")
    axis.set_ylabel("Target class")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=55, ha="right")
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([str(value) for value in matrix.index])
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Class median deviation (global IQR units)")
    figure.tight_layout()
    return figure


def plot_class_pca_projection(
    report: MulticlassClassProfileReport,
    *,
    max_points_per_class: int = 500,
    random_state: int = 42,
    title: str = "Exploratory PCA Class Projection",
):
    """Plot a sampled two-component PCA view of standardized features."""
    if not isinstance(report, MulticlassClassProfileReport):
        raise ClassProfileAnalysisError(
            "report must be a MulticlassClassProfileReport."
        )
    report.raise_if_invalid()
    if (
        isinstance(max_points_per_class, bool)
        or not isinstance(max_points_per_class, Integral)
        or max_points_per_class < 1
    ):
        raise ClassProfileAnalysisError(
            "max_points_per_class must be a positive integer."
        )
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise ClassProfileAnalysisError("random_state must be an integer.")

    import matplotlib.pyplot as plt

    projection = report.pca_projection_frame()
    figure, axis = plt.subplots(figsize=(9.5, 7.0))

    for class_index, target_class in enumerate(report.expected_target_classes):
        group = projection.loc[
            projection[report.target_name].eq(target_class),
            ["PC1", "PC2"],
        ]
        if len(group) > max_points_per_class:
            group = group.sample(
                n=int(max_points_per_class),
                random_state=int(random_state) + class_index,
            )
        axis.scatter(
            group["PC1"],
            group["PC2"],
            s=14,
            alpha=0.55,
            label=str(target_class),
        )

    pc1, pc2 = report.pca_explained_variance_ratio
    axis.set_title(title)
    axis.set_xlabel(f"PC1 ({pc1:.1%} explained variance)")
    axis.set_ylabel(f"PC2 ({pc2:.1%} explained variance)")
    axis.legend(title="Class", loc="best")
    figure.tight_layout()
    return figure


def _normalize_feature_names(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise ClassProfileAnalysisError("features must be a sequence of names.")
    normalized: list[str] = []
    for value in features:
        if not isinstance(value, str) or not value.strip():
            raise ClassProfileAnalysisError(
                "features must contain non-empty strings."
            )
        name = value.strip()
        if name in normalized:
            raise ClassProfileAnalysisError(
                f"features contains duplicate name {name!r}."
            )
        normalized.append(name)
    if not normalized:
        raise ClassProfileAnalysisError("features must not be empty.")
    return tuple(normalized)


def _normalize_values(values: Sequence[object]) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise ClassProfileAnalysisError(
            "expected_target_classes must be a sequence of class labels."
        )
    normalized: list[object] = []
    for value in values:
        if pd.isna(value):
            raise ClassProfileAnalysisError(
                "expected_target_classes must not contain missing values."
            )
        if value in normalized:
            raise ClassProfileAnalysisError(
                f"expected_target_classes contains duplicate value {value!r}."
            )
        normalized.append(value)
    if not normalized:
        raise ClassProfileAnalysisError(
            "expected_target_classes must not be empty."
        )
    return tuple(normalized)


def _iqr_overlap_coefficient(
    a_q1: float,
    a_q3: float,
    b_q1: float,
    b_q3: float,
) -> float:
    """Return overlap relative to the narrower class IQR."""
    a_width = max(0.0, a_q3 - a_q1)
    b_width = max(0.0, b_q3 - b_q1)
    overlap_width = max(0.0, min(a_q3, b_q3) - max(a_q1, b_q1))
    denominator = min(a_width, b_width)

    if denominator > 0:
        return min(1.0, overlap_width / denominator)

    if a_width == 0 and b_width == 0:
        return 1.0 if a_q1 == b_q1 else 0.0
    point = a_q1 if a_width == 0 else b_q1
    other_q1, other_q3 = (
        (b_q1, b_q3) if a_width == 0 else (a_q1, a_q3)
    )
    return 1.0 if other_q1 <= point <= other_q3 else 0.0
