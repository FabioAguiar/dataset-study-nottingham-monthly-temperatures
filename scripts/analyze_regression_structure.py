"""Exploratory, non-mutating diagnostics for continuous regression structure."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_NONLINEARITY_COLUMNS: Final[list[str]] = [
    "Feature",
    "Valid rows",
    "Excluded rows",
    "Linear R squared",
    "Quadratic R squared",
    "Linear adjusted R squared",
    "Quadratic adjusted R squared",
    "Adjusted R squared gain",
    "Nonlinearity signal",
    "Interpretation",
]

_INTERACTION_COLUMNS: Final[list[str]] = [
    "Feature A",
    "Feature B",
    "Valid rows",
    "Excluded rows",
    "Quadratic main-effects adjusted R squared",
    "With interaction adjusted R squared",
    "Adjusted R squared gain",
    "Interaction signal",
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


class RegressionStructureAnalysisError(ValueError):
    """Raised when regression-structure diagnostics are not well defined."""


@dataclass(frozen=True, slots=True)
class RegressionStructureReport:
    """Summarize exploratory nonlinearity and pairwise interaction signals."""

    requested_features: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    target_name: str
    row_count: int
    missing_target_count: int
    invalid_target_count: int
    target_unique_count: int
    nonlinearity_review_threshold: float
    interaction_review_threshold: float
    nonlinearity: pd.DataFrame
    interactions: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_missing_features(self) -> bool:
        return bool(self.missing_features)

    @property
    def has_missing_target_values(self) -> bool:
        return self.missing_target_count > 0

    @property
    def has_invalid_target_values(self) -> bool:
        return self.invalid_target_count > 0

    @property
    def has_constant_target(self) -> bool:
        return self.target_unique_count < 2

    @property
    def has_feature_issues(self) -> bool:
        if self.issues.empty:
            return False
        return bool(
            self.issues["Issue"].isin(
                {
                    "Missing numerical feature",
                    "Non-numeric or non-finite feature values",
                    "Constant numerical feature",
                    "Insufficient complete rows",
                }
            ).any()
        )

    @property
    def has_nonlinearity_signals(self) -> bool:
        if self.nonlinearity.empty:
            return False
        return bool(self.nonlinearity["Nonlinearity signal"].any())

    @property
    def has_interaction_signals(self) -> bool:
        if self.interactions.empty:
            return False
        return bool(self.interactions["Interaction signal"].any())

    @property
    def is_analysis_ready(self) -> bool:
        return not (
            self.has_missing_features
            or self.has_missing_target_values
            or self.has_invalid_target_values
            or self.has_constant_target
            or self.has_feature_issues
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic high-level diagnostic metrics."""
        rows = [
            {
                "Metric": "Rows",
                "Value": self.row_count,
                "Interpretation": "Observations supplied to the diagnostics",
            },
            {
                "Metric": "Requested numerical features",
                "Value": len(self.requested_features),
                "Interpretation": "Candidate features reviewed",
            },
            {
                "Metric": "Nonlinearity review threshold",
                "Value": self.nonlinearity_review_threshold,
                "Interpretation": (
                    "Minimum adjusted-R² gain from adding a quadratic term"
                ),
            },
            {
                "Metric": "Nonlinearity signals",
                "Value": (
                    0
                    if self.nonlinearity.empty
                    else int(self.nonlinearity["Nonlinearity signal"].sum())
                ),
                "Interpretation": "Features meeting the exploratory threshold",
            },
            {
                "Metric": "Interaction review threshold",
                "Value": self.interaction_review_threshold,
                "Interpretation": (
                    "Minimum adjusted-R² gain from adding a product term "
                    "beyond quadratic main effects"
                ),
            },
            {
                "Metric": "Interaction signals",
                "Value": (
                    0
                    if self.interactions.empty
                    else int(self.interactions["Interaction signal"].sum())
                ),
                "Interpretation": "Feature pairs meeting the exploratory threshold",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def nonlinearity_frame(self) -> pd.DataFrame:
        """Return feature-level curvature evidence ordered by adjusted-R² gain."""
        if self.nonlinearity.empty:
            return self.nonlinearity.copy(deep=True)
        return self.nonlinearity.sort_values(
            ["Adjusted R squared gain", "Feature"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    def interaction_frame(self, *, limit: int | None = None) -> pd.DataFrame:
        """Return pairwise interaction evidence ordered by adjusted-R² gain."""
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise RegressionStructureAnalysisError(
                    "limit must be None or a positive integer."
                )
        frame = self.interactions.copy(deep=True)
        if frame.empty:
            return frame
        frame = frame.sort_values(
            ["Adjusted R squared gain", "Feature A", "Feature B"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        if limit is not None:
            frame = frame.head(limit).reset_index(drop=True)
        return frame

    def issues_frame(self) -> pd.DataFrame:
        """Return structural conditions limiting interpretation."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_features_present: bool = True,
        require_numeric_target: bool = True,
        require_no_missing_target: bool = True,
        require_target_variation: bool = True,
        require_complete_numeric_features: bool = True,
    ) -> None:
        """Raise one consolidated error for selected diagnostic requirements."""
        failures: list[str] = []

        if require_features_present and self.has_missing_features:
            failures.append("missing_features:" + ",".join(self.missing_features))
        if require_numeric_target and self.has_invalid_target_values:
            failures.append(f"invalid_target_values:{self.invalid_target_count}")
        if require_no_missing_target and self.has_missing_target_values:
            failures.append(f"missing_target_values:{self.missing_target_count}")
        if require_target_variation and self.has_constant_target:
            failures.append("constant_target_detected")
        if require_complete_numeric_features and self.has_feature_issues:
            failures.append("numeric_feature_issues_detected")

        if failures:
            raise RegressionStructureAnalysisError(
                "Regression-structure analysis is invalid: " + "; ".join(failures)
            )


def analyze_regression_structure(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    nonlinearity_review_threshold: Real = 0.02,
    interaction_review_threshold: Real = 0.02,
) -> RegressionStructureReport:
    """Diagnose curvature and pairwise interaction signals for regression.

    For each feature, a linear descriptive fit ``y ~ x`` is compared with
    ``y ~ x + x²``. For each feature pair, quadratic main effects
    ``y ~ x1 + x1² + x2 + x2²`` are compared with the same terms plus
    ``x1*x2``. The diagnostic signal is the gain in adjusted R².

    All predictors are standardized before polynomial/product construction.
    Fits are in-sample structural diagnostics only: they are not cross-
    validated performance estimates, model-selection results, causal effects,
    or instructions to engineer features.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise RegressionStructureAnalysisError("dataframe must be a pandas DataFrame.")
    if dataframe.columns.duplicated().any():
        raise RegressionStructureAnalysisError(
            "dataframe must not contain duplicated column labels."
        )

    requested_features = _normalize_feature_names(features)
    if len(requested_features) < 2:
        raise RegressionStructureAnalysisError(
            "features must contain at least two numerical columns."
        )
    if not isinstance(target, str) or not target.strip():
        raise RegressionStructureAnalysisError("target must be a non-empty string.")
    target_name = target.strip()
    if target_name in requested_features:
        raise RegressionStructureAnalysisError("target must not be included in features.")

    nonlinearity_threshold = _validate_threshold(
        nonlinearity_review_threshold,
        "nonlinearity_review_threshold",
    )
    interaction_threshold = _validate_threshold(
        interaction_review_threshold,
        "interaction_review_threshold",
    )

    source = dataframe.copy(deep=True)
    if target_name not in source.columns:
        raise RegressionStructureAnalysisError(
            f"Target column {target_name!r} is not present in dataframe."
        )

    available_features = tuple(
        feature for feature in requested_features if feature in source.columns
    )
    missing_features = tuple(
        feature for feature in requested_features if feature not in source.columns
    )

    target_raw = source[target_name]
    target_numeric = pd.to_numeric(target_raw, errors="coerce")
    missing_target_mask = target_raw.isna()
    invalid_target_mask = (~missing_target_mask) & (
        target_numeric.isna() | ~target_numeric.map(_is_finite_value)
    )
    valid_target_mask = (
        ~missing_target_mask
        & ~invalid_target_mask
        & target_numeric.notna()
    )
    missing_target_count = int(missing_target_mask.sum())
    invalid_target_count = int(invalid_target_mask.sum())
    target_unique_count = int(
        target_numeric.loc[valid_target_mask].nunique(dropna=True)
    )

    issues: list[dict[str, object]] = []
    for feature in missing_features:
        issues.append(
            {
                "Scope": "Feature contract",
                "Feature A": feature,
                "Feature B": None,
                "Issue": "Missing numerical feature",
                "Details": "Requested feature is absent from dataframe",
                "Potential impact": "Requested structural diagnostic is incomplete.",
            }
        )

    if missing_target_count:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature A": None,
                "Feature B": None,
                "Issue": "Missing target values",
                "Details": f"count={missing_target_count}",
                "Potential impact": "Incomplete outcomes reduce usable diagnostic rows.",
            }
        )
    if invalid_target_count:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature A": None,
                "Feature B": None,
                "Issue": "Invalid continuous target values",
                "Details": f"count={invalid_target_count}",
                "Potential impact": "Non-numeric/non-finite targets invalidate diagnostics.",
            }
        )
    if target_unique_count < 2:
        issues.append(
            {
                "Scope": "Target contract",
                "Feature A": None,
                "Feature B": None,
                "Issue": "Constant continuous target",
                "Details": f"unique finite values={target_unique_count}",
                "Potential impact": "Explained-variance diagnostics are undefined.",
            }
        )

    numeric_features: dict[str, pd.Series] = {}
    for feature in available_features:
        raw = source[feature]
        numeric = pd.to_numeric(raw, errors="coerce")
        valid = raw.notna() & numeric.notna() & numeric.map(_is_finite_value)
        invalid_count = int((~valid).sum())
        if invalid_count:
            issues.append(
                {
                    "Scope": "Feature values",
                    "Feature A": feature,
                    "Feature B": None,
                    "Issue": "Non-numeric or non-finite feature values",
                    "Details": f"count={invalid_count}",
                    "Potential impact": "Rows are excluded from affected diagnostics.",
                }
            )
        finite = numeric.loc[valid]
        if finite.nunique(dropna=True) < 2:
            issues.append(
                {
                    "Scope": "Feature variation",
                    "Feature A": feature,
                    "Feature B": None,
                    "Issue": "Constant numerical feature",
                    "Details": f"unique finite values={finite.nunique(dropna=True)}",
                    "Potential impact": "Curvature and interaction terms are undefined.",
                }
            )
        numeric_features[feature] = numeric.astype(float)

    nonlinearity_rows: list[dict[str, object]] = []
    interactions_rows: list[dict[str, object]] = []

    globally_ready = (
        not missing_features
        and missing_target_count == 0
        and invalid_target_count == 0
        and target_unique_count >= 2
    )

    if globally_ready:
        for feature in available_features:
            series = numeric_features[feature]
            mask = series.notna() & series.map(_is_finite_value) & valid_target_mask
            x = series.loc[mask].to_numpy(dtype=float)
            y = target_numeric.loc[mask].to_numpy(dtype=float)
            valid_rows = len(x)
            excluded_rows = len(source) - valid_rows

            if valid_rows < 5 or np.unique(x).size < 3:
                issues.append(
                    {
                        "Scope": "Nonlinearity diagnostic",
                        "Feature A": feature,
                        "Feature B": None,
                        "Issue": "Insufficient complete rows",
                        "Details": (
                            f"valid_rows={valid_rows}; unique_feature_values="
                            f"{np.unique(x).size}"
                        ),
                        "Potential impact": "Quadratic diagnostic was not calculated.",
                    }
                )
                metrics = (None, None, None, None, None)
            else:
                z = _standardize(x)
                linear_design = np.column_stack([np.ones(valid_rows), z])
                quadratic_design = np.column_stack(
                    [np.ones(valid_rows), z, z**2]
                )
                linear_r2, linear_adj = _fit_r_squared(
                    linear_design, y, predictor_count=1
                )
                quadratic_r2, quadratic_adj = _fit_r_squared(
                    quadratic_design, y, predictor_count=2
                )
                gain = quadratic_adj - linear_adj
                metrics = (
                    linear_r2,
                    quadratic_r2,
                    linear_adj,
                    quadratic_adj,
                    gain,
                )

            gain = metrics[4]
            signal = bool(
                gain is not None
                and isfinite(float(gain))
                and float(gain) >= nonlinearity_threshold
            )
            interpretation = (
                "Quadratic term adds notable descriptive structure"
                if signal
                else "No notable quadratic gain at the review threshold"
            )
            nonlinearity_rows.append(
                {
                    "Feature": feature,
                    "Valid rows": valid_rows,
                    "Excluded rows": excluded_rows,
                    "Linear R squared": metrics[0],
                    "Quadratic R squared": metrics[1],
                    "Linear adjusted R squared": metrics[2],
                    "Quadratic adjusted R squared": metrics[3],
                    "Adjusted R squared gain": gain,
                    "Nonlinearity signal": signal,
                    "Interpretation": interpretation,
                }
            )

        for feature_a, feature_b in combinations(available_features, 2):
            a = numeric_features[feature_a]
            b = numeric_features[feature_b]
            mask = (
                a.notna()
                & a.map(_is_finite_value)
                & b.notna()
                & b.map(_is_finite_value)
                & valid_target_mask
            )
            x1 = a.loc[mask].to_numpy(dtype=float)
            x2 = b.loc[mask].to_numpy(dtype=float)
            y = target_numeric.loc[mask].to_numpy(dtype=float)
            valid_rows = len(y)
            excluded_rows = len(source) - valid_rows

            unique_a = np.unique(x1).size
            unique_b = np.unique(x2).size
            if valid_rows < 8 or unique_a < 3 or unique_b < 3:
                issues.append(
                    {
                        "Scope": "Interaction diagnostic",
                        "Feature A": feature_a,
                        "Feature B": feature_b,
                        "Issue": "Insufficient complete rows",
                        "Details": (
                            f"valid_rows={valid_rows}; unique_a={unique_a}; "
                            f"unique_b={unique_b}"
                        ),
                        "Potential impact": "Interaction diagnostic was not calculated.",
                    }
                )
                base_adj = None
                interaction_adj = None
                gain = None
            else:
                z1 = _standardize(x1)
                z2 = _standardize(x2)
                main_design = np.column_stack(
                    [np.ones(valid_rows), z1, z1**2, z2, z2**2]
                )
                interaction_design = np.column_stack(
                    [main_design, z1 * z2]
                )
                _, base_adj = _fit_r_squared(
                    main_design, y, predictor_count=4
                )
                _, interaction_adj = _fit_r_squared(
                    interaction_design, y, predictor_count=5
                )
                gain = interaction_adj - base_adj

            signal = bool(
                gain is not None
                and isfinite(float(gain))
                and float(gain) >= interaction_threshold
            )
            interpretation = (
                "Product term adds structure beyond quadratic main effects"
                if signal
                else "No notable interaction gain at the review threshold"
            )
            interactions_rows.append(
                {
                    "Feature A": feature_a,
                    "Feature B": feature_b,
                    "Valid rows": valid_rows,
                    "Excluded rows": excluded_rows,
                    "Quadratic main-effects adjusted R squared": base_adj,
                    "With interaction adjusted R squared": interaction_adj,
                    "Adjusted R squared gain": gain,
                    "Interaction signal": signal,
                    "Interpretation": interpretation,
                }
            )

    return RegressionStructureReport(
        requested_features=requested_features,
        available_features=available_features,
        missing_features=missing_features,
        target_name=target_name,
        row_count=len(source),
        missing_target_count=missing_target_count,
        invalid_target_count=invalid_target_count,
        target_unique_count=target_unique_count,
        nonlinearity_review_threshold=nonlinearity_threshold,
        interaction_review_threshold=interaction_threshold,
        nonlinearity=pd.DataFrame(
            nonlinearity_rows,
            columns=_NONLINEARITY_COLUMNS,
        ),
        interactions=pd.DataFrame(
            interactions_rows,
            columns=_INTERACTION_COLUMNS,
        ),
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
    )


def plot_nonlinearity_signals(
    report: RegressionStructureReport,
    *,
    title: str = "Regression Nonlinearity Signals",
):
    """Plot adjusted-R² gain from adding one quadratic term per feature."""
    _validate_report(report)
    frame = report.nonlinearity_frame()
    if frame.empty:
        raise RegressionStructureAnalysisError(
            "No nonlinearity evidence is available to plot."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RegressionStructureAnalysisError(
            "matplotlib is required to plot regression diagnostics."
        ) from exc

    frame = frame.sort_values(
        ["Adjusted R squared gain", "Feature"],
        ascending=[True, True],
        na_position="first",
    ).reset_index(drop=True)
    figure_height = max(4.5, 0.38 * len(frame) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        frame["Feature"],
        frame["Adjusted R squared gain"].fillna(0.0),
    )
    axis.axvline(
        report.nonlinearity_review_threshold,
        linestyle="--",
        linewidth=1,
        label=(
            "Review threshold "
            f"({report.nonlinearity_review_threshold:.3f})"
        ),
    )
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel("Adjusted R² gain: quadratic vs linear")
    axis.set_ylabel("Feature")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_interaction_signals(
    report: RegressionStructureReport,
    *,
    limit: int = 15,
    title: str = "Regression Pairwise Interaction Signals",
):
    """Plot strongest product-term gains beyond quadratic main effects."""
    _validate_report(report)
    frame = report.interaction_frame(limit=limit)
    if frame.empty:
        raise RegressionStructureAnalysisError(
            "No interaction evidence is available to plot."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RegressionStructureAnalysisError(
            "matplotlib is required to plot regression diagnostics."
        ) from exc

    frame = frame.copy(deep=True)
    frame["Pair"] = frame["Feature A"] + " × " + frame["Feature B"]
    frame = frame.sort_values(
        ["Adjusted R squared gain", "Pair"],
        ascending=[True, True],
        na_position="first",
    ).reset_index(drop=True)
    figure_height = max(5.0, 0.36 * len(frame) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        frame["Pair"],
        frame["Adjusted R squared gain"].fillna(0.0),
    )
    axis.axvline(
        report.interaction_review_threshold,
        linestyle="--",
        linewidth=1,
        label=(
            "Review threshold "
            f"({report.interaction_review_threshold:.3f})"
        ),
    )
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel(
        "Adjusted R² gain: product term beyond quadratic main effects"
    )
    axis.set_ylabel("Feature pair")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def _normalize_feature_names(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise RegressionStructureAnalysisError(
            "features must be a sequence of column names."
        )
    normalized: list[str] = []
    for feature in features:
        if not isinstance(feature, str) or not feature.strip():
            raise RegressionStructureAnalysisError(
                "features must contain non-empty strings."
            )
        name = feature.strip()
        if name in normalized:
            raise RegressionStructureAnalysisError(
                f"features contains duplicate column {name!r}."
            )
        normalized.append(name)
    return tuple(normalized)


def _validate_threshold(value: Real, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise RegressionStructureAnalysisError(
            f"{name} must be a non-negative real number."
        )
    threshold = float(value)
    if not isfinite(threshold) or threshold < 0:
        raise RegressionStructureAnalysisError(
            f"{name} must be a finite non-negative real number."
        )
    return threshold


def _is_finite_value(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not isfinite(scale) or scale <= 0:
        raise RegressionStructureAnalysisError(
            "Cannot standardize a feature without finite variation."
        )
    return (values - mean) / scale


def _fit_r_squared(
    design: np.ndarray,
    target: np.ndarray,
    *,
    predictor_count: int,
) -> tuple[float, float]:
    n_rows = int(len(target))
    if n_rows <= predictor_count + 1:
        raise RegressionStructureAnalysisError(
            "Insufficient rows for adjusted R squared."
        )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((target - fitted) ** 2))
    centered = target - float(np.mean(target))
    total_sum = float(np.sum(centered**2))
    if total_sum <= 0:
        raise RegressionStructureAnalysisError(
            "Target must vary to calculate R squared."
        )
    r_squared = 1.0 - residual_sum / total_sum
    adjusted = 1.0 - (
        (1.0 - r_squared) * (n_rows - 1) / (n_rows - predictor_count - 1)
    )
    return float(r_squared), float(adjusted)


def _validate_report(report: RegressionStructureReport) -> None:
    if not isinstance(report, RegressionStructureReport):
        raise RegressionStructureAnalysisError(
            "report must be a RegressionStructureReport."
        )
