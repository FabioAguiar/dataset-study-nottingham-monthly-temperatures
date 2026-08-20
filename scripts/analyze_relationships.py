"""Reusable, non-mutating analysis of relationships between features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import log, sqrt
from numbers import Real
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_NUMERICAL_COLUMNS: Final[list[str]] = [
    "Feature A",
    "Feature B",
    "Valid paired rows",
    "Missing paired rows",
    "Pearson correlation",
    "Rank correlation",
    "Absolute Pearson correlation",
    "Absolute rank correlation",
    "Strong association",
    "Potential redundancy",
    "Interpretation",
]

_NUMERICAL_REVIEW_COLUMNS: Final[list[str]] = [
    "Feature A",
    "Feature B",
    "Pearson correlation",
    "Rank correlation",
    "Maximum absolute association",
    "Potential redundancy",
    "Interpretation",
]

_CATEGORICAL_COLUMNS: Final[list[str]] = [
    "Feature A",
    "Feature B",
    "Valid paired rows",
    "Missing paired rows",
    "Feature A cardinality",
    "Feature B cardinality",
    "Cramer's V",
    "U(A | B)",
    "U(B | A)",
    "Maximum directional dependency",
    "Structural dependency",
    "Potential redundancy",
    "Interpretation",
]

_MIXED_COLUMNS: Final[list[str]] = [
    "Categorical feature",
    "Numerical feature",
    "Valid paired rows",
    "Missing paired rows",
    "Observed categories",
    "Eta squared",
    "Strong association",
    "Interpretation",
]

_INTERACTION_COLUMNS: Final[list[str]] = [
    "Interaction",
    "Left feature",
    "Right feature",
    "Operation",
    "Compared feature",
    "Valid rows",
    "Missing rows",
    "Pearson correlation",
    "Rank correlation",
    "Mean absolute difference",
    "Median absolute difference",
    "Mean relative difference",
    "Strong association",
    "Potential redundancy",
    "Interpretation",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Feature A",
    "Feature B",
    "Issue",
    "Details",
    "Potential impact",
]

_REQUIRED_INTERACTION_KEYS: Final[set[str]] = {
    "name",
    "left",
    "right",
    "operation",
    "compare_to",
}

_SUPPORTED_OPERATIONS: Final[set[str]] = {"product"}


class FeatureRelationshipAnalysisError(ValueError):
    """Raised when feature-relationship configuration or data are invalid."""


@dataclass(frozen=True, slots=True)
class FeatureRelationshipReport:
    """Summarize numerical, categorical, mixed, and interaction evidence."""

    requested_numerical_features: tuple[str, ...]
    available_numerical_features: tuple[str, ...]
    missing_numerical_features: tuple[str, ...]
    requested_categorical_features: tuple[str, ...]
    available_categorical_features: tuple[str, ...]
    missing_categorical_features: tuple[str, ...]
    numerical_row_count: int
    categorical_row_count: int
    indices_aligned: bool
    strong_numerical_threshold: float
    strong_categorical_threshold: float
    strong_mixed_threshold: float
    deterministic_dependency_threshold: float
    redundancy_review_threshold: float
    numerical_relationships: pd.DataFrame
    categorical_relationships: pd.DataFrame
    mixed_relationships: pd.DataFrame
    interactions: pd.DataFrame
    issues: pd.DataFrame
    numerical_pearson_matrix: pd.DataFrame
    numerical_rank_matrix: pd.DataFrame
    categorical_cramers_v_matrix: pd.DataFrame
    mixed_eta_squared_matrix: pd.DataFrame

    @property
    def has_alignment_issues(self) -> bool:
        """Return whether numerical and categorical indices differ."""
        return not self.indices_aligned

    @property
    def has_missing_features(self) -> bool:
        """Return whether any requested feature is absent."""
        return bool(
            self.missing_numerical_features
            or self.missing_categorical_features
        )

    @property
    def has_constant_features(self) -> bool:
        """Return whether a relationship could not vary due to a constant."""
        if self.issues.empty:
            return False
        return bool(
            self.issues["Issue"].isin(
                {
                    "Constant numerical feature",
                    "Constant categorical feature",
                }
            ).any()
        )

    @property
    def has_strong_numerical_relationships(self) -> bool:
        """Return whether any numerical pair meets the configured threshold."""
        if self.numerical_relationships.empty:
            return False
        return bool(
            self.numerical_relationships["Strong association"].any()
        )

    @property
    def has_strong_categorical_relationships(self) -> bool:
        """Return whether any categorical pair meets the configured threshold."""
        if self.categorical_relationships.empty:
            return False
        return bool(
            self.categorical_relationships["Cramer's V"]
            .fillna(0.0)
            .ge(self.strong_categorical_threshold)
            .any()
        )

    @property
    def has_strong_mixed_relationships(self) -> bool:
        """Return whether any mixed pair meets the configured threshold."""
        if self.mixed_relationships.empty:
            return False
        return bool(self.mixed_relationships["Strong association"].any())

    @property
    def has_structural_dependencies(self) -> bool:
        """Return whether any categorical pair is directionally deterministic."""
        if self.categorical_relationships.empty:
            return False
        return bool(
            self.categorical_relationships["Structural dependency"].any()
        )

    @property
    def has_redundancy_candidates(self) -> bool:
        """Return whether any pair or interaction merits redundancy review."""
        numerical = (
            False
            if self.numerical_relationships.empty
            else bool(
                self.numerical_relationships["Potential redundancy"].any()
            )
        )
        categorical = (
            False
            if self.categorical_relationships.empty
            else bool(
                self.categorical_relationships["Potential redundancy"].any()
            )
        )
        interactions = (
            False
            if self.interactions.empty
            else bool(self.interactions["Potential redundancy"].any())
        )
        return numerical or categorical or interactions

    @property
    def is_analysis_ready(self) -> bool:
        """Return whether alignment and requested feature contracts are valid."""
        return not self.has_alignment_issues and not self.has_missing_features

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic high-level feature-relationship metrics."""
        rows = [
            {
                "Metric": "Requested numerical features",
                "Value": len(self.requested_numerical_features),
                "Interpretation": "Numerical features declared by the study",
            },
            {
                "Metric": "Available numerical features",
                "Value": len(self.available_numerical_features),
                "Interpretation": "Numerical features found in the projection",
            },
            {
                "Metric": "Missing numerical features",
                "Value": len(self.missing_numerical_features),
                "Interpretation": (
                    "Requires review"
                    if self.missing_numerical_features
                    else "All declared numerical features are present"
                ),
            },
            {
                "Metric": "Requested categorical features",
                "Value": len(self.requested_categorical_features),
                "Interpretation": "Categorical features declared by the study",
            },
            {
                "Metric": "Available categorical features",
                "Value": len(self.available_categorical_features),
                "Interpretation": "Categorical features found in the projection",
            },
            {
                "Metric": "Missing categorical features",
                "Value": len(self.missing_categorical_features),
                "Interpretation": (
                    "Requires review"
                    if self.missing_categorical_features
                    else "All declared categorical features are present"
                ),
            },
            {
                "Metric": "Projection indices aligned",
                "Value": self.indices_aligned,
                "Interpretation": (
                    "Mixed relationships use matching observations"
                    if self.indices_aligned
                    else "Mixed relationships were not calculated"
                ),
            },
            {
                "Metric": "Numerical relationships",
                "Value": len(self.numerical_relationships),
                "Interpretation": "Unique numerical feature pairs",
            },
            {
                "Metric": "Categorical relationships",
                "Value": len(self.categorical_relationships),
                "Interpretation": "Unique categorical feature pairs",
            },
            {
                "Metric": "Categorical-numerical relationships",
                "Value": len(self.mixed_relationships),
                "Interpretation": "Cross-type feature pairs",
            },
            {
                "Metric": "Interaction candidates",
                "Value": len(self.interactions),
                "Interpretation": "Explicit derived relationships reviewed",
            },
            {
                "Metric": "Strong numerical relationships",
                "Value": (
                    0
                    if self.numerical_relationships.empty
                    else int(
                        self.numerical_relationships[
                            "Strong association"
                        ].sum()
                    )
                ),
                "Interpretation": (
                    "Absolute Pearson or rank correlation at or above "
                    f"{self.strong_numerical_threshold:g}"
                ),
            },
            {
                "Metric": "Strong categorical relationships",
                "Value": (
                    0
                    if self.categorical_relationships.empty
                    else int(
                        self.categorical_relationships["Cramer's V"]
                        .fillna(0.0)
                        .ge(self.strong_categorical_threshold)
                        .sum()
                    )
                ),
                "Interpretation": (
                    "Cramer's V at or above "
                    f"{self.strong_categorical_threshold:g}"
                ),
            },
            {
                "Metric": "Strong mixed relationships",
                "Value": (
                    0
                    if self.mixed_relationships.empty
                    else int(
                        self.mixed_relationships["Strong association"].sum()
                    )
                ),
                "Interpretation": (
                    "Eta squared at or above "
                    f"{self.strong_mixed_threshold:g}"
                ),
            },
            {
                "Metric": "Structural categorical dependencies",
                "Value": (
                    0
                    if self.categorical_relationships.empty
                    else int(
                        self.categorical_relationships[
                            "Structural dependency"
                        ].sum()
                    )
                ),
                "Interpretation": (
                    "Directional uncertainty coefficient at or above "
                    f"{self.deterministic_dependency_threshold:g}"
                ),
            },
            {
                "Metric": "Redundancy review candidates",
                "Value": _redundancy_count(self),
                "Interpretation": (
                    "Strong evidence requiring review, not automatic removal"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def numerical_summary_frame(self) -> pd.DataFrame:
        """Return a compact summary for all-numerical feature studies."""
        strong_count = (
            0
            if self.numerical_relationships.empty
            else int(self.numerical_relationships["Strong association"].sum())
        )
        redundancy_count = (
            0
            if self.numerical_relationships.empty
            else int(
                self.numerical_relationships["Potential redundancy"].sum()
            )
        )
        rows = [
            {
                "Metric": "Numerical features",
                "Value": len(self.available_numerical_features),
                "Interpretation": "Candidate features included in the analysis",
            },
            {
                "Metric": "Feature pairs",
                "Value": len(self.numerical_relationships),
                "Interpretation": "Unique numerical feature pairs assessed",
            },
            {
                "Metric": "Strong-association threshold",
                "Value": self.strong_numerical_threshold,
                "Interpretation": (
                    "Absolute Pearson or rank correlation used for review"
                ),
            },
            {
                "Metric": "Strong-association pairs",
                "Value": strong_count,
                "Interpretation": "Pairs meeting the exploratory threshold",
            },
            {
                "Metric": "Redundancy-review threshold",
                "Value": self.redundancy_review_threshold,
                "Interpretation": (
                    "Absolute Pearson correlation used to flag redundancy"
                ),
            },
            {
                "Metric": "Redundancy-review candidates",
                "Value": redundancy_count,
                "Interpretation": (
                    "Review evidence only; no feature removal is implied"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def numerical_relationships_frame(self) -> pd.DataFrame:
        """Return pairwise numerical-association evidence."""
        return self.numerical_relationships.copy(deep=True)

    def numerical_review_frame(
        self,
        *,
        minimum_absolute_association: Real | None = None,
    ) -> pd.DataFrame:
        """Return strongest numerical relationships for notebook review."""
        threshold = (
            self.strong_numerical_threshold
            if minimum_absolute_association is None
            else _validate_unit_interval_threshold(
                minimum_absolute_association,
                name="minimum_absolute_association",
            )
        )

        if self.numerical_relationships.empty:
            return pd.DataFrame(columns=_NUMERICAL_REVIEW_COLUMNS)

        frame = self.numerical_relationships.copy(deep=True)
        frame["Maximum absolute association"] = frame[
            [
                "Absolute Pearson correlation",
                "Absolute rank correlation",
            ]
        ].max(axis=1, skipna=True)
        frame = frame.loc[
            frame["Maximum absolute association"].fillna(0.0).ge(threshold)
        ]
        frame = frame.sort_values(
            by=[
                "Potential redundancy",
                "Maximum absolute association",
                "Feature A",
                "Feature B",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        )
        return frame.loc[:, _NUMERICAL_REVIEW_COLUMNS].reset_index(drop=True)

    def categorical_relationships_frame(self) -> pd.DataFrame:
        """Return pairwise categorical-association evidence."""
        return self.categorical_relationships.copy(deep=True)

    def mixed_relationships_frame(self) -> pd.DataFrame:
        """Return categorical-to-numerical association evidence."""
        return self.mixed_relationships.copy(deep=True)

    def interactions_frame(self) -> pd.DataFrame:
        """Return explicit interaction-candidate evidence."""
        return self.interactions.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return alignment, feature, and variation conditions."""
        return self.issues.copy(deep=True)

    def numerical_correlation_matrix(
        self,
        *,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Return a defensive numerical correlation matrix."""
        normalized_method = method.strip().casefold()
        if normalized_method == "pearson":
            return self.numerical_pearson_matrix.copy(deep=True)
        if normalized_method in {"rank", "spearman"}:
            return self.numerical_rank_matrix.copy(deep=True)
        raise FeatureRelationshipAnalysisError(
            "method must be 'pearson', 'rank', or 'spearman'."
        )

    def categorical_association_matrix(self) -> pd.DataFrame:
        """Return the Cramer's V association matrix."""
        return self.categorical_cramers_v_matrix.copy(deep=True)

    def mixed_association_matrix(self) -> pd.DataFrame:
        """Return the categorical-by-numerical eta-squared matrix."""
        return self.mixed_eta_squared_matrix.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_aligned_indices: bool = True,
        require_features_present: bool = True,
        require_unique_columns: bool = True,
        require_sufficient_variation: bool = False,
    ) -> None:
        """Raise when configured relationship-analysis requirements fail."""
        failures: list[str] = []

        if require_aligned_indices and self.has_alignment_issues:
            failures.append("projection_indices_not_aligned")

        if require_features_present:
            if self.missing_numerical_features:
                failures.append(
                    "missing_numerical_features:"
                    + ",".join(self.missing_numerical_features)
                )
            if self.missing_categorical_features:
                failures.append(
                    "missing_categorical_features:"
                    + ",".join(self.missing_categorical_features)
                )

        if require_unique_columns:
            duplicate_issues = self.issues.loc[
                self.issues["Issue"].isin(
                    {
                        "Duplicated numerical column labels",
                        "Duplicated categorical column labels",
                    }
                )
            ]
            if not duplicate_issues.empty:
                failures.append("duplicated_column_labels")

        if require_sufficient_variation and self.has_constant_features:
            failures.append("constant_features_detected")

        if failures:
            raise FeatureRelationshipAnalysisError(
                "Feature relationship analysis is invalid: "
                + "; ".join(failures)
            )


def analyze_feature_relationships(
    numerical_frame: pd.DataFrame,
    categorical_frame: pd.DataFrame,
    *,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    interaction_candidates: Sequence[Mapping[str, object]] = (),
    strong_numerical_threshold: Real = 0.80,
    strong_categorical_threshold: Real = 0.70,
    strong_mixed_threshold: Real = 0.25,
    deterministic_dependency_threshold: Real = 0.99,
    redundancy_review_threshold: Real = 0.90,
) -> FeatureRelationshipReport:
    """Analyze relationships between features without mutating inputs."""
    _validate_dataframe(numerical_frame, name="numerical_frame")
    _validate_dataframe(categorical_frame, name="categorical_frame")

    requested_numerical = _normalize_feature_names(
        numerical_features,
        name="numerical_features",
    )
    requested_categorical = _normalize_feature_names(
        categorical_features,
        name="categorical_features",
    )
    normalized_interactions = _normalize_interactions(interaction_candidates)

    thresholds = _validate_thresholds(
        strong_numerical_threshold=strong_numerical_threshold,
        strong_categorical_threshold=strong_categorical_threshold,
        strong_mixed_threshold=strong_mixed_threshold,
        deterministic_dependency_threshold=(
            deterministic_dependency_threshold
        ),
        redundancy_review_threshold=redundancy_review_threshold,
    )

    numerical_source = numerical_frame.copy(deep=True)
    categorical_source = categorical_frame.copy(deep=True)

    missing_numerical = tuple(
        feature
        for feature in requested_numerical
        if feature not in numerical_source.columns
    )
    available_numerical = tuple(
        feature
        for feature in requested_numerical
        if feature in numerical_source.columns
    )
    missing_categorical = tuple(
        feature
        for feature in requested_categorical
        if feature not in categorical_source.columns
    )
    available_categorical = tuple(
        feature
        for feature in requested_categorical
        if feature in categorical_source.columns
    )

    indices_aligned = numerical_source.index.equals(categorical_source.index)
    issues: list[dict[str, object]] = []

    if not indices_aligned:
        issues.append(
            {
                "Scope": "Projection alignment",
                "Feature A": None,
                "Feature B": None,
                "Issue": "Projection indices are not aligned",
                "Details": (
                    f"numerical rows={len(numerical_source)}, "
                    f"categorical rows={len(categorical_source)}"
                ),
                "Potential impact": (
                    "Mixed relationships could associate different "
                    "observations"
                ),
            }
        )

    for feature in missing_numerical:
        issues.append(
            _missing_feature_issue("Numerical", feature)
        )
    for feature in missing_categorical:
        issues.append(
            _missing_feature_issue("Categorical", feature)
        )

    numerical_rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(available_numerical, 2):
        row, row_issues = _analyze_numerical_pair(
            numerical_source,
            feature_a=feature_a,
            feature_b=feature_b,
            strong_threshold=thresholds["strong_numerical"],
            redundancy_threshold=thresholds["redundancy"],
        )
        numerical_rows.append(row)
        issues.extend(row_issues)

    categorical_rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(available_categorical, 2):
        row, row_issues = _analyze_categorical_pair(
            categorical_source,
            feature_a=feature_a,
            feature_b=feature_b,
            strong_threshold=thresholds["strong_categorical"],
            deterministic_threshold=thresholds["deterministic"],
            redundancy_threshold=thresholds["redundancy"],
        )
        categorical_rows.append(row)
        issues.extend(row_issues)

    mixed_rows: list[dict[str, object]] = []
    if indices_aligned:
        for categorical_feature in available_categorical:
            for numerical_feature in available_numerical:
                row, row_issues = _analyze_mixed_pair(
                    numerical_source,
                    categorical_source,
                    categorical_feature=categorical_feature,
                    numerical_feature=numerical_feature,
                    strong_threshold=thresholds["strong_mixed"],
                )
                mixed_rows.append(row)
                issues.extend(row_issues)

    interaction_rows: list[dict[str, object]] = []
    for candidate in normalized_interactions:
        row, row_issues = _analyze_interaction(
            numerical_source,
            candidate=candidate,
            available_features=available_numerical,
            strong_threshold=thresholds["strong_numerical"],
            redundancy_threshold=thresholds["redundancy"],
        )
        if row is not None:
            interaction_rows.append(row)
        issues.extend(row_issues)

    numerical_relationships = pd.DataFrame(
        numerical_rows,
        columns=_NUMERICAL_COLUMNS,
    )
    categorical_relationships = pd.DataFrame(
        categorical_rows,
        columns=_CATEGORICAL_COLUMNS,
    )
    mixed_relationships = pd.DataFrame(
        mixed_rows,
        columns=_MIXED_COLUMNS,
    )
    interactions = pd.DataFrame(
        interaction_rows,
        columns=_INTERACTION_COLUMNS,
    )
    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)

    pearson_matrix = _build_numerical_matrix(
        available_numerical,
        numerical_relationships,
        value_column="Pearson correlation",
    )
    rank_matrix = _build_numerical_matrix(
        available_numerical,
        numerical_relationships,
        value_column="Rank correlation",
    )
    categorical_matrix = _build_categorical_matrix(
        available_categorical,
        categorical_relationships,
    )
    mixed_matrix = _build_mixed_matrix(
        available_categorical,
        available_numerical,
        mixed_relationships,
    )

    return FeatureRelationshipReport(
        requested_numerical_features=requested_numerical,
        available_numerical_features=available_numerical,
        missing_numerical_features=missing_numerical,
        requested_categorical_features=requested_categorical,
        available_categorical_features=available_categorical,
        missing_categorical_features=missing_categorical,
        numerical_row_count=len(numerical_source),
        categorical_row_count=len(categorical_source),
        indices_aligned=indices_aligned,
        strong_numerical_threshold=thresholds["strong_numerical"],
        strong_categorical_threshold=thresholds["strong_categorical"],
        strong_mixed_threshold=thresholds["strong_mixed"],
        deterministic_dependency_threshold=thresholds["deterministic"],
        redundancy_review_threshold=thresholds["redundancy"],
        numerical_relationships=numerical_relationships,
        categorical_relationships=categorical_relationships,
        mixed_relationships=mixed_relationships,
        interactions=interactions,
        issues=issues_frame,
        numerical_pearson_matrix=pearson_matrix,
        numerical_rank_matrix=rank_matrix,
        categorical_cramers_v_matrix=categorical_matrix,
        mixed_eta_squared_matrix=mixed_matrix,
    )


def analyze_numerical_feature_relationships(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    strong_association_threshold: Real = 0.80,
    redundancy_review_threshold: Real = 0.90,
) -> FeatureRelationshipReport:
    """Analyze an all-numerical feature set with a compact public API."""
    _validate_dataframe(dataframe, name="dataframe")
    return analyze_feature_relationships(
        dataframe,
        pd.DataFrame(index=dataframe.index),
        numerical_features=features,
        categorical_features=(),
        strong_numerical_threshold=strong_association_threshold,
        redundancy_review_threshold=redundancy_review_threshold,
    )


def plot_numerical_correlation_heatmap(
    report: FeatureRelationshipReport,
    *,
    method: str = "pearson",
    title: str | None = None,
    figure_size: tuple[float, float] = (11.0, 9.0),
):
    """Return a compact heatmap for numerical feature relationships."""
    from matplotlib import pyplot as plt

    if not isinstance(report, FeatureRelationshipReport):
        raise TypeError("report must be a FeatureRelationshipReport.")

    matrix = report.numerical_correlation_matrix(method=method)
    if matrix.empty:
        raise FeatureRelationshipAnalysisError(
            "cannot plot a correlation heatmap without numerical features."
        )

    normalized_method = method.strip().casefold()
    label = "Pearson" if normalized_method == "pearson" else "Spearman rank"

    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0)

    positions = range(len(matrix.columns))
    axis.set_xticks(list(positions))
    axis.set_yticks(list(positions))
    axis.set_xticklabels(matrix.columns, rotation=60, ha="right")
    axis.set_yticklabels(matrix.index)
    axis.set_title(title or f"{label} Feature Correlation")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    return figure


def _analyze_numerical_pair(
    dataframe: pd.DataFrame,
    *,
    feature_a: str,
    feature_b: str,
    strong_threshold: float,
    redundancy_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paired = pd.DataFrame(
        {
            feature_a: pd.to_numeric(dataframe[feature_a], errors="coerce"),
            feature_b: pd.to_numeric(dataframe[feature_b], errors="coerce"),
        },
        index=dataframe.index,
    ).dropna()

    valid_rows = len(paired)
    missing_rows = len(dataframe) - valid_rows
    issues: list[dict[str, object]] = []

    constant_a = paired[feature_a].nunique(dropna=True) <= 1
    constant_b = paired[feature_b].nunique(dropna=True) <= 1
    for feature, constant in (
        (feature_a, constant_a),
        (feature_b, constant_b),
    ):
        if constant:
            issues.append(
                _constant_issue(
                    scope="Numerical relationship",
                    feature=feature,
                    other=(feature_b if feature == feature_a else feature_a),
                    categorical=False,
                )
            )

    pearson: float | None = None
    rank: float | None = None
    if valid_rows >= 2 and not constant_a and not constant_b:
        pearson = _finite_or_none(
            paired[feature_a].corr(paired[feature_b])
        )
        rank = _finite_or_none(
            paired[feature_a]
            .rank(method="average")
            .corr(paired[feature_b].rank(method="average"))
        )

    absolute_pearson = None if pearson is None else abs(pearson)
    absolute_rank = None if rank is None else abs(rank)
    maximum = max(
        value
        for value in (absolute_pearson, absolute_rank, 0.0)
        if value is not None
    )
    strong = maximum >= strong_threshold
    redundancy = (
        absolute_pearson is not None
        and absolute_pearson >= redundancy_threshold
    )

    if pearson is None:
        interpretation = "Insufficient variation for correlation"
    elif strong:
        direction = "positive" if pearson >= 0 else "negative"
        interpretation = f"Strong {direction} numerical association"
    elif maximum >= strong_threshold / 2:
        interpretation = "Moderate numerical association"
    else:
        interpretation = "Limited numerical association"

    return (
        {
            "Feature A": feature_a,
            "Feature B": feature_b,
            "Valid paired rows": valid_rows,
            "Missing paired rows": missing_rows,
            "Pearson correlation": pearson,
            "Rank correlation": rank,
            "Absolute Pearson correlation": absolute_pearson,
            "Absolute rank correlation": absolute_rank,
            "Strong association": strong,
            "Potential redundancy": redundancy,
            "Interpretation": interpretation,
        },
        issues,
    )


def _analyze_categorical_pair(
    dataframe: pd.DataFrame,
    *,
    feature_a: str,
    feature_b: str,
    strong_threshold: float,
    deterministic_threshold: float,
    redundancy_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paired = pd.DataFrame(
        {
            feature_a: _clean_categorical_series(dataframe[feature_a]),
            feature_b: _clean_categorical_series(dataframe[feature_b]),
        },
        index=dataframe.index,
    ).dropna()

    valid_rows = len(paired)
    missing_rows = len(dataframe) - valid_rows
    cardinality_a = int(paired[feature_a].nunique(dropna=True))
    cardinality_b = int(paired[feature_b].nunique(dropna=True))
    issues: list[dict[str, object]] = []

    if cardinality_a <= 1:
        issues.append(
            _constant_issue(
                scope="Categorical relationship",
                feature=feature_a,
                other=feature_b,
                categorical=True,
            )
        )
    if cardinality_b <= 1:
        issues.append(
            _constant_issue(
                scope="Categorical relationship",
                feature=feature_b,
                other=feature_a,
                categorical=True,
            )
        )

    cramers_v: float | None = None
    u_a_given_b: float | None = None
    u_b_given_a: float | None = None

    if valid_rows > 0 and cardinality_a > 1 and cardinality_b > 1:
        contingency = pd.crosstab(
            paired[feature_a],
            paired[feature_b],
            dropna=True,
        )
        cramers_v = _cramers_v(contingency)
        u_a_given_b = _uncertainty_coefficient(
            target=paired[feature_a],
            predictor=paired[feature_b],
        )
        u_b_given_a = _uncertainty_coefficient(
            target=paired[feature_b],
            predictor=paired[feature_a],
        )

    directional_values = [
        value
        for value in (u_a_given_b, u_b_given_a)
        if value is not None
    ]
    maximum_directional = (
        max(directional_values) if directional_values else None
    )
    structural = bool(
        maximum_directional is not None
        and maximum_directional >= deterministic_threshold
    )
    redundancy = bool(
        cramers_v is not None
        and cramers_v >= redundancy_threshold
        and u_a_given_b is not None
        and u_b_given_a is not None
        and min(u_a_given_b, u_b_given_a) >= deterministic_threshold
    )

    if cramers_v is None:
        interpretation = "Insufficient variation for categorical association"
    elif structural and not redundancy:
        interpretation = "Strong directional structural dependency"
    elif redundancy:
        interpretation = "Near-bidirectional categorical redundancy candidate"
    elif cramers_v >= strong_threshold:
        interpretation = "Strong categorical association"
    elif cramers_v >= strong_threshold / 2:
        interpretation = "Moderate categorical association"
    else:
        interpretation = "Limited categorical association"

    return (
        {
            "Feature A": feature_a,
            "Feature B": feature_b,
            "Valid paired rows": valid_rows,
            "Missing paired rows": missing_rows,
            "Feature A cardinality": cardinality_a,
            "Feature B cardinality": cardinality_b,
            "Cramer's V": cramers_v,
            "U(A | B)": u_a_given_b,
            "U(B | A)": u_b_given_a,
            "Maximum directional dependency": maximum_directional,
            "Structural dependency": structural,
            "Potential redundancy": redundancy,
            "Interpretation": interpretation,
        },
        issues,
    )


def _analyze_mixed_pair(
    numerical_frame: pd.DataFrame,
    categorical_frame: pd.DataFrame,
    *,
    categorical_feature: str,
    numerical_feature: str,
    strong_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paired = pd.DataFrame(
        {
            "category": _clean_categorical_series(
                categorical_frame[categorical_feature]
            ),
            "numeric": pd.to_numeric(
                numerical_frame[numerical_feature],
                errors="coerce",
            ),
        },
        index=numerical_frame.index,
    ).dropna()

    valid_rows = len(paired)
    missing_rows = len(numerical_frame) - valid_rows
    categories = int(paired["category"].nunique(dropna=True))
    numeric_unique = int(paired["numeric"].nunique(dropna=True))
    issues: list[dict[str, object]] = []

    if categories <= 1:
        issues.append(
            _constant_issue(
                scope="Mixed relationship",
                feature=categorical_feature,
                other=numerical_feature,
                categorical=True,
            )
        )
    if numeric_unique <= 1:
        issues.append(
            _constant_issue(
                scope="Mixed relationship",
                feature=numerical_feature,
                other=categorical_feature,
                categorical=False,
            )
        )

    eta_squared: float | None = None
    if valid_rows > 0 and categories > 1 and numeric_unique > 1:
        eta_squared = _eta_squared(
            categories=paired["category"],
            values=paired["numeric"],
        )

    strong = bool(
        eta_squared is not None and eta_squared >= strong_threshold
    )
    if eta_squared is None:
        interpretation = "Insufficient variation for mixed association"
    elif strong:
        interpretation = "Strong categorical-to-numerical association"
    elif eta_squared >= strong_threshold / 2:
        interpretation = "Moderate categorical-to-numerical association"
    else:
        interpretation = "Limited categorical-to-numerical association"

    return (
        {
            "Categorical feature": categorical_feature,
            "Numerical feature": numerical_feature,
            "Valid paired rows": valid_rows,
            "Missing paired rows": missing_rows,
            "Observed categories": categories,
            "Eta squared": eta_squared,
            "Strong association": strong,
            "Interpretation": interpretation,
        },
        issues,
    )


def _analyze_interaction(
    dataframe: pd.DataFrame,
    *,
    candidate: Mapping[str, str],
    available_features: Sequence[str],
    strong_threshold: float,
    redundancy_threshold: float,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    name = candidate["name"]
    left = candidate["left"]
    right = candidate["right"]
    operation = candidate["operation"]
    compare_to = candidate["compare_to"]
    required = (left, right, compare_to)
    missing = tuple(
        feature for feature in required if feature not in available_features
    )

    if missing:
        return (
            None,
            [
                {
                    "Scope": "Interaction",
                    "Feature A": left,
                    "Feature B": right,
                    "Issue": "Interaction feature missing",
                    "Details": f"{name}: {missing!r}",
                    "Potential impact": (
                        "Interaction evidence could not be calculated"
                    ),
                }
            ],
        )

    left_values = pd.to_numeric(dataframe[left], errors="coerce")
    right_values = pd.to_numeric(dataframe[right], errors="coerce")
    compared_values = pd.to_numeric(dataframe[compare_to], errors="coerce")

    if operation == "product":
        interaction_values = left_values * right_values
    else:  # configuration validation guarantees this branch is unreachable.
        raise FeatureRelationshipAnalysisError(
            f"Unsupported interaction operation: {operation!r}"
        )

    paired = pd.DataFrame(
        {
            "interaction": interaction_values,
            "compared": compared_values,
        },
        index=dataframe.index,
    ).dropna()

    valid_rows = len(paired)
    missing_rows = len(dataframe) - valid_rows
    interaction_unique = paired["interaction"].nunique(dropna=True)
    compared_unique = paired["compared"].nunique(dropna=True)
    issues: list[dict[str, object]] = []

    if interaction_unique <= 1:
        issues.append(
            _constant_issue(
                scope="Interaction",
                feature=name,
                other=compare_to,
                categorical=False,
            )
        )
    if compared_unique <= 1:
        issues.append(
            _constant_issue(
                scope="Interaction",
                feature=compare_to,
                other=name,
                categorical=False,
            )
        )

    pearson: float | None = None
    rank: float | None = None
    mean_absolute_difference: float | None = None
    median_absolute_difference: float | None = None
    mean_relative_difference: float | None = None

    if valid_rows > 0:
        absolute_difference = (
            paired["interaction"] - paired["compared"]
        ).abs()
        mean_absolute_difference = float(absolute_difference.mean())
        median_absolute_difference = float(absolute_difference.median())

        nonzero = paired["compared"].abs() > 0
        if nonzero.any():
            relative = (
                absolute_difference.loc[nonzero]
                / paired.loc[nonzero, "compared"].abs()
            )
            mean_relative_difference = float(relative.mean())

    if valid_rows >= 2 and interaction_unique > 1 and compared_unique > 1:
        pearson = _finite_or_none(
            paired["interaction"].corr(paired["compared"])
        )
        rank = _finite_or_none(
            paired["interaction"]
            .rank(method="average")
            .corr(paired["compared"].rank(method="average"))
        )

    maximum = max(
        abs(value)
        for value in (pearson, rank, 0.0)
        if value is not None
    )
    strong = maximum >= strong_threshold
    redundancy = bool(
        pearson is not None and abs(pearson) >= redundancy_threshold
    )

    if pearson is None:
        interpretation = "Insufficient variation for interaction comparison"
    elif redundancy:
        interpretation = "Strong derived-feature redundancy candidate"
    elif strong:
        interpretation = "Strong interaction relationship"
    else:
        interpretation = "Limited interaction relationship"

    return (
        {
            "Interaction": name,
            "Left feature": left,
            "Right feature": right,
            "Operation": operation,
            "Compared feature": compare_to,
            "Valid rows": valid_rows,
            "Missing rows": missing_rows,
            "Pearson correlation": pearson,
            "Rank correlation": rank,
            "Mean absolute difference": mean_absolute_difference,
            "Median absolute difference": median_absolute_difference,
            "Mean relative difference": mean_relative_difference,
            "Strong association": strong,
            "Potential redundancy": redundancy,
            "Interpretation": interpretation,
        },
        issues,
    )


def _cramers_v(contingency: pd.DataFrame) -> float | None:
    observed = contingency.astype(float)
    total = float(observed.to_numpy().sum())
    rows, columns = observed.shape
    denominator_dimension = min(rows - 1, columns - 1)
    if total <= 0 or denominator_dimension <= 0:
        return None

    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    expected = pd.DataFrame(
        [
            [
                float(row_total * column_total / total)
                for column_total in column_totals
            ]
            for row_total in row_totals
        ],
        index=observed.index,
        columns=observed.columns,
    )
    valid = expected > 0
    chi_squared = float(
        (((observed - expected) ** 2) / expected)
        .where(valid, 0.0)
        .to_numpy()
        .sum()
    )
    value = sqrt(chi_squared / (total * denominator_dimension))
    return min(max(value, 0.0), 1.0)


def _uncertainty_coefficient(
    *,
    target: pd.Series,
    predictor: pd.Series,
) -> float | None:
    target_entropy = _entropy(target)
    if target_entropy <= 0:
        return None

    joint = pd.crosstab(target, predictor, dropna=True)
    total = float(joint.to_numpy().sum())
    if total <= 0:
        return None

    conditional_entropy = 0.0
    for predictor_value in joint.columns:
        counts = joint[predictor_value]
        group_total = float(counts.sum())
        if group_total <= 0:
            continue
        probabilities = counts[counts > 0] / group_total
        group_entropy = -sum(
            float(probability) * log(float(probability), 2)
            for probability in probabilities
        )
        conditional_entropy += (group_total / total) * group_entropy

    value = (target_entropy - conditional_entropy) / target_entropy
    return min(max(float(value), 0.0), 1.0)


def _entropy(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts / total
    return -sum(
        float(probability) * log(float(probability), 2)
        for probability in probabilities
        if probability > 0
    )


def _eta_squared(
    *,
    categories: pd.Series,
    values: pd.Series,
) -> float | None:
    overall_mean = float(values.mean())
    total_sum_squares = float(((values - overall_mean) ** 2).sum())
    if total_sum_squares <= 0:
        return None

    between_sum_squares = 0.0
    grouped = values.groupby(categories, sort=False, observed=False)
    for _, group in grouped:
        if group.empty:
            continue
        between_sum_squares += len(group) * (
            float(group.mean()) - overall_mean
        ) ** 2

    value = between_sum_squares / total_sum_squares
    return min(max(float(value), 0.0), 1.0)


def _build_numerical_matrix(
    features: Sequence[str],
    relationships: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        0.0,
        index=list(features),
        columns=list(features),
        dtype=float,
    )
    for feature in features:
        matrix.loc[feature, feature] = 1.0
    for _, row in relationships.iterrows():
        feature_a = row["Feature A"]
        feature_b = row["Feature B"]
        value = row[value_column]
        if value is None or pd.isna(value):
            matrix.loc[feature_a, feature_b] = float("nan")
            matrix.loc[feature_b, feature_a] = float("nan")
        else:
            matrix.loc[feature_a, feature_b] = float(value)
            matrix.loc[feature_b, feature_a] = float(value)
    return matrix


def _build_categorical_matrix(
    features: Sequence[str],
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        0.0,
        index=list(features),
        columns=list(features),
        dtype=float,
    )
    for feature in features:
        matrix.loc[feature, feature] = 1.0
    for _, row in relationships.iterrows():
        feature_a = row["Feature A"]
        feature_b = row["Feature B"]
        value = row["Cramer's V"]
        numeric = float("nan") if value is None or pd.isna(value) else float(value)
        matrix.loc[feature_a, feature_b] = numeric
        matrix.loc[feature_b, feature_a] = numeric
    return matrix


def _build_mixed_matrix(
    categorical_features: Sequence[str],
    numerical_features: Sequence[str],
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        float("nan"),
        index=list(categorical_features),
        columns=list(numerical_features),
        dtype=float,
    )
    for _, row in relationships.iterrows():
        value = row["Eta squared"]
        matrix.loc[
            row["Categorical feature"],
            row["Numerical feature"],
        ] = (
            float("nan")
            if value is None or pd.isna(value)
            else float(value)
        )
    return matrix


def _clean_categorical_series(series: pd.Series) -> pd.Series:
    result = series.astype("object").copy(deep=True)
    blank_mask = result.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    result = result.mask(blank_mask)
    return result


def _missing_feature_issue(scope: str, feature: str) -> dict[str, object]:
    return {
        "Scope": f"{scope} feature contract",
        "Feature A": feature,
        "Feature B": None,
        "Issue": f"Missing {scope.casefold()} feature",
        "Details": feature,
        "Potential impact": "Declared relationships cannot be calculated",
    }


def _constant_issue(
    *,
    scope: str,
    feature: str,
    other: str,
    categorical: bool,
) -> dict[str, object]:
    return {
        "Scope": scope,
        "Feature A": feature,
        "Feature B": other,
        "Issue": (
            "Constant categorical feature"
            if categorical
            else "Constant numerical feature"
        ),
        "Details": f"{feature!r} has insufficient variation",
        "Potential impact": "Association metric is undefined",
    }


def _redundancy_count(report: FeatureRelationshipReport) -> int:
    count = 0
    for frame in (
        report.numerical_relationships,
        report.categorical_relationships,
        report.interactions,
    ):
        if not frame.empty and "Potential redundancy" in frame:
            count += int(frame["Potential redundancy"].sum())
    return count


def _validate_dataframe(dataframe: pd.DataFrame, *, name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated:
        raise FeatureRelationshipAnalysisError(
            f"{name} contains duplicated column labels: {duplicated!r}"
        )


def _normalize_feature_names(
    features: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of column names.")
    normalized: list[str] = []
    for feature in features:
        if not isinstance(feature, str) or not feature.strip():
            raise FeatureRelationshipAnalysisError(
                f"{name} contains an invalid column name: {feature!r}"
            )
        normalized.append(feature)
    duplicates = _find_duplicates(normalized)
    if duplicates:
        raise FeatureRelationshipAnalysisError(
            f"{name} contains duplicate column names: {duplicates!r}"
        )
    return tuple(normalized)


def _normalize_interactions(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("interaction_candidates must be a sequence of mappings.")
    normalized: list[dict[str, str]] = []
    names: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(
                "interaction_candidates entries must be mappings; "
                f"entry {index} is {type(candidate).__name__}."
            )
        keys = set(candidate)
        missing = _REQUIRED_INTERACTION_KEYS - keys
        extra = keys - _REQUIRED_INTERACTION_KEYS
        if missing or extra:
            raise FeatureRelationshipAnalysisError(
                "interaction candidate keys are invalid: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        normalized_candidate: dict[str, str] = {}
        for key in sorted(_REQUIRED_INTERACTION_KEYS):
            value = candidate[key]
            if not isinstance(value, str) or not value.strip():
                raise FeatureRelationshipAnalysisError(
                    f"interaction candidate {key!r} must be a non-empty string."
                )
            normalized_candidate[key] = value
        if normalized_candidate["operation"] not in _SUPPORTED_OPERATIONS:
            raise FeatureRelationshipAnalysisError(
                "Unsupported interaction operation: "
                f"{normalized_candidate['operation']!r}"
            )
        names.append(normalized_candidate["name"])
        normalized.append(normalized_candidate)
    duplicates = _find_duplicates(names)
    if duplicates:
        raise FeatureRelationshipAnalysisError(
            f"interaction candidate names must be unique: {duplicates!r}"
        )
    return tuple(normalized)


def _validate_unit_interval_threshold(
    value: Real,
    *,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise FeatureRelationshipAnalysisError(
            f"{name} must be between 0 and 1."
        )
    return numeric


def _validate_thresholds(**thresholds: Real) -> dict[str, float]:
    normalized: dict[str, float] = {}
    key_map = {
        "strong_numerical_threshold": "strong_numerical",
        "strong_categorical_threshold": "strong_categorical",
        "strong_mixed_threshold": "strong_mixed",
        "deterministic_dependency_threshold": "deterministic",
        "redundancy_review_threshold": "redundancy",
    }
    for source_name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{source_name} must be a real number.")
        numeric = float(value)
        if not 0 <= numeric <= 1:
            raise FeatureRelationshipAnalysisError(
                f"{source_name} must be between 0 and 1."
            )
        normalized[key_map[source_name]] = numeric
    return normalized


def _finite_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric


def _find_duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
