"""Tests for the portable Notebook-01 exploration handoff."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.build_exploration_handoff import (
    ExplorationHandoffError,
    build_static_continuous_regression_exploration_handoff,
    build_static_multiclass_exploration_handoff,
    load_and_validate_exploration_handoff,
)


FEATURES = ("Area", "Perimeter", "Compactness")
CLASSES = ("A", "B", "C")


class TargetReport:
    has_issues = False
    class_count = 3
    imbalance_ratio = 2.0
    normalized_class_entropy = 0.9
    majority_classes = ("A",)
    minority_classes = ("C",)

    def distribution_frame(self, *, format_percentages: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Class": CLASSES,
                "Count": [6, 4, 3],
                "Proportion": [6 / 13, 4 / 13, 3 / 13],
                "Role": ["Majority", "Intermediate", "Minority"],
            }
        )


class DuplicateReport:
    has_source_identifiers = False
    exact_duplicate_group_count = 1
    exact_duplicate_row_count = 2
    target_conflict_group_count = 0


class LeakageReport:
    is_structurally_valid = True
    has_direct_target_leakage = False
    confirmed_derived_dependency_count = 1

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Feature"])

    def dependency_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Derived feature": "Compactness",
                    "Dependency status": "Confirmed from retained columns",
                    "Target-derived": False,
                },
                {
                    "Derived feature": "Perimeter",
                    "Dependency status": "Declared dependency not confirmed",
                    "Target-derived": False,
                },
            ]
        )


class QualityReport:
    is_structurally_valid = True

    def findings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Finding ID": "DQ-001",
                    "Title": "Review exact matches",
                    "Disposition": "Review",
                }
            ]
        )

    def validated_non_issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Non-issue ID": "NI-001",
                    "Title": "No missing values",
                    "Disposition": "No action",
                }
            ]
        )


class InsightsReport:
    is_structurally_valid = True

    def key_insights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Insight ID": "INS-001",
                    "Theme": "Class separation",
                    "Title": "Overlap remains",
                    "Relevance": "High",
                    "Status": "Observed",
                    "Summary": "Some class profiles overlap.",
                    "Modeling implication": "Inspect confusion.",
                    "Interpretation boundary": "EDA only.",
                }
            ]
        )

    def hypotheses_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Hypothesis ID": "HYP-001",
                    "Title": "Overlap predicts confusion",
                    "Hypothesis": "Overlapping classes may be confused.",
                    "Required validation": "Inspect validation confusion matrices.",
                }
            ]
        )

    def limitations_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Limitation ID": "LIM-001",
                    "Title": "EDA is not performance",
                    "Limitation type": "Modeling",
                }
            ]
        )


class PreparationReport:
    is_structurally_valid = True
    is_ready_for_deterministic_preparation = True
    is_ready_for_split_execution = True
    is_ready_for_modeling = False

    def decisions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Decision ID": "PREP-001",
                    "Domain": "Cleaning",
                    "Title": "Preserve source",
                    "Status": "Approved",
                }
            ]
        )

    def execution_plan_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Step ID": "STEP-001",
                    "Sequence": 1,
                    "Action": "Create prepared copy",
                    "Status": "Planned",
                }
            ]
        )

    def guardrails_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Guardrail ID": "GRD-001",
                    "Title": "Protect target",
                    "Status": "Active",
                }
            ]
        )

    def split_policy_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Policy item": "Train fraction", "Value": 0.70},
                {"Policy item": "Validation fraction", "Value": 0.15},
                {"Policy item": "Test fraction", "Value": 0.15},
                {"Policy item": "Stratification field", "Value": "Class"},
                {"Policy item": "Random seed", "Value": 42},
                {"Policy item": "Final test holdout", "Value": True},
                {"Policy item": "Disjoint partitions", "Value": True},
                {"Policy item": "Identifier grouping", "Value": ()},
                {
                    "Policy item": "Temporal policy status",
                    "Value": "Resolved snapshot fallback",
                },
            ]
        )


class RelationshipReport:
    numerical_relationships = pd.DataFrame(
        [
            {
                "Feature A": "Area",
                "Feature B": "Perimeter",
                "Potential redundancy": True,
            }
        ]
    )


class ContinuousTargetReport:
    target = "Strength"
    unit = "MPa"
    row_count = 4
    finite_count = 4
    missing_count = 0
    non_finite_count = 0
    unique_count = 4
    minimum = 10.0
    mean = 25.0
    median = 25.0
    maximum = 40.0
    standard_deviation = 12.909944
    iqr = 15.0
    extreme_count = 1
    extreme_share = 0.25
    has_variation = True

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Issue", "Count", "Values", "Potential impact"])

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": "Mean",
                    "Value": self.mean,
                    "Interpretation": "Arithmetic mean in MPa",
                },
                {
                    "Metric": "Outside 1.5-IQR fences",
                    "Value": self.extreme_count,
                    "Interpretation": "Descriptive only",
                },
            ]
        )

    def quantiles_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Quantile": "25%", "Value": 17.5, "Unit": "MPa"},
                {"Quantile": "50%", "Value": 25.0, "Unit": "MPa"},
                {"Quantile": "75%", "Value": 32.5, "Unit": "MPa"},
            ]
        )

    def extremes_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Side": "Upper",
                    "Fence": 55.0,
                    "Observed extreme": 40.0,
                    "Count outside fence": 1,
                }
            ]
        )


class ContinuousLeakageReport:
    is_structurally_valid = True
    has_direct_target_leakage = False
    confirmed_derived_dependency_count = 0

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Feature"])

    def dependency_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Derived feature",
                "Dependency status",
                "Target-derived",
            ]
        )


class ContinuousInsightsReport:
    is_structurally_valid = True

    def key_insights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Insight ID": "INS-001",
                    "Theme": "Regression structure",
                    "Title": "Nonlinearity requires validation",
                    "Relevance": "High",
                    "Status": "Observed",
                    "Summary": "Exploration shows nonlinear structure.",
                    "Modeling implication": "Compare flexible candidates.",
                    "Interpretation boundary": "EDA only.",
                }
            ]
        )

    def hypotheses_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Hypothesis ID": "HYP-001",
                    "Title": "Flexible regression may help",
                    "Hypothesis": (
                        "Nonlinear candidates may outperform an additive baseline."
                    ),
                    "Required validation": (
                        "Compare candidates with leakage-safe validation."
                    ),
                }
            ]
        )

    def limitations_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Limitation ID": "LIM-001",
                    "Title": "EDA is not performance",
                    "Limitation type": "Modeling",
                }
            ]
        )


class ContinuousPreparationReport(PreparationReport):
    def split_policy_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Policy item": "Train fraction", "Value": 0.70},
                {"Policy item": "Validation fraction", "Value": 0.15},
                {"Policy item": "Test fraction", "Value": 0.15},
                {"Policy item": "Stratification field", "Value": None},
                {"Policy item": "Random seed", "Value": 42},
                {"Policy item": "Shuffle random split", "Value": True},
                {"Policy item": "Final test holdout", "Value": True},
                {"Policy item": "Disjoint partitions", "Value": True},
                {"Policy item": "Identifier grouping", "Value": ()},
                {
                    "Policy item": "Temporal policy status",
                    "Value": "Resolved snapshot fallback",
                },
                {
                    "Policy item": "Random-split fallback",
                    "Value": "Approved static snapshot fallback",
                },
            ]
        )


def source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": [1, 2, 3],
            "Perimeter": [2.0, 3.0, 4.0],
            "Compactness": [0.7, 0.8, 0.9],
            "Class": ["A", "B", "C"],
        }
    )


def continuous_source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Cement": [100.0, 200.0, 300.0, 400.0],
            "Age": [7, 28, 56, 90],
            "Strength": [10.0, 20.0, 30.0, 40.0],
        }
    )


def build_continuous_report(tmp_path: Path, **overrides):
    source = tmp_path / "dataset.csv"
    source.write_text(
        "Cement,Age,Strength\n100,7,10\n",
        encoding="utf-8",
    )
    params = {
        "dataset_slug": "concrete-compressive-strength",
        "source_repository": "UCI Machine Learning Repository",
        "source_dataset_id": 165,
        "source_file": source,
        "project_root": tmp_path,
        "source_dataframe": continuous_source_dataframe(),
        "target_contract": SimpleNamespace(
            target="Strength",
            problem_type="continuous_regression",
            target_semantics="Continuous / quantitative",
            expected_unit="MPa",
            source_unit="MPa",
            prediction_output=(
                "Continuous numeric value on the original target scale"
            ),
        ),
        "feature_columns": ("Cement", "Age"),
        "numerical_features": ("Cement", "Age"),
        "identifier_columns": (),
        "target_report": ContinuousTargetReport(),
        "duplicate_report": DuplicateReport(),
        "feature_relationship_report": RelationshipReport(),
        "leakage_report": ContinuousLeakageReport(),
        "quality_report": QualityReport(),
        "insights_report": ContinuousInsightsReport(),
        "preparation_report": ContinuousPreparationReport(),
    }
    params.update(overrides)
    return build_static_continuous_regression_exploration_handoff(**params)


def build_report(tmp_path: Path, **overrides):
    source = tmp_path / "dataset.csv"
    source.write_text(
        "Area,Perimeter,Compactness,Class\n1,2,0.7,A\n",
        encoding="utf-8",
    )
    params = {
        "dataset_slug": "dry-bean",
        "source_repository": "UCI Machine Learning Repository",
        "source_dataset_id": 602,
        "source_file": source,
        "project_root": tmp_path,
        "source_dataframe": source_dataframe(),
        "target_contract": SimpleNamespace(
            target="Class",
            expected_classes=CLASSES,
            problem_type="multiclass_classification",
            class_semantics="Nominal / unordered",
        ),
        "feature_columns": FEATURES,
        "numerical_features": FEATURES,
        "identifier_columns": (),
        "target_report": TargetReport(),
        "duplicate_report": DuplicateReport(),
        "feature_relationship_report": RelationshipReport(),
        "leakage_report": LeakageReport(),
        "quality_report": QualityReport(),
        "insights_report": InsightsReport(),
        "preparation_report": PreparationReport(),
    }
    params.update(overrides)
    return build_static_multiclass_exploration_handoff(**params)


def test_builds_ready_portable_handoff(tmp_path: Path) -> None:
    report = build_report(tmp_path)

    assert report.is_structurally_valid
    assert report.is_handoff_ready
    assert report.payload["schema_version"] == "exploration-handoff.v1"
    assert report.payload["source"]["dataset_id"] == 602
    assert report.payload["source"]["path"] == "dataset.csv"
    assert report.payload["prediction_contract"]["positive_class"] is None
    assert report.payload["feature_contract"]["feature_columns"] == list(FEATURES)
    assert report.payload["preparation_contract"]["split_policy"]["random_seed"] == 42
    assert report.payload["readiness"]["model_selection_ready"] is False


def test_open_reviews_preserve_nonblocking_evidence(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    reviews = report.open_reviews_frame()

    assert not reviews.empty
    assert reviews["blocking"].eq(False).all()
    assert "Duplicate identity" in set(reviews["theme"])
    assert "Derived-feature dependency" in set(reviews["theme"])
    assert "Feature redundancy" in set(reviews["theme"])
    assert "Class support" in set(reviews["theme"])
    assert "Exploratory hypothesis" in set(reviews["theme"])


def test_next_steps_keep_notebook_03_waiting(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    steps = report.next_steps_frame()

    notebook_02 = steps.loc[steps["Notebook"].eq("02_data_preparation.ipynb")]
    notebook_03 = steps.loc[
        steps["Notebook"].eq("03_model_selection_and_evaluation.ipynb")
    ]

    assert notebook_02["Status"].eq("Ready").all()
    assert notebook_03["Status"].eq("Waiting on Notebook 02").all()


def test_write_is_atomic_and_reloadable(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "artifacts/exploration/dry-bean/exploration-handoff.json"

    persisted = report.write(destination)

    assert persisted.path == destination.resolve()
    assert persisted.size_bytes > 0
    assert len(persisted.sha256) == 64

    payload = load_and_validate_exploration_handoff(
        destination,
        expected_dataset_slug="dry-bean",
        expected_source_dataset_id=602,
    )
    assert payload["readiness"]["split_execution_ready"] is True
    assert payload["source"]["sha256"]


def test_write_is_deterministic_for_same_payload(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"

    first = report.write(destination)
    second = report.write(destination)

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes


def test_target_cannot_be_predictor(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        feature_columns=(*FEATURES, "Class"),
        numerical_features=(*FEATURES, "Class"),
    )

    assert not report.is_structurally_valid
    with pytest.raises(ExplorationHandoffError, match="not ready"):
        report.raise_if_invalid()


def test_static_dry_bean_handoff_requires_all_numeric_features(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        numerical_features=("Area", "Perimeter"),
    )

    assert not report.is_structurally_valid
    issues = report.issues_frame()
    assert issues["Issue"].str.contains("entirely numerical").any()


def test_upstream_leakage_failure_blocks_handoff(tmp_path: Path) -> None:
    leakage = LeakageReport()
    leakage.has_direct_target_leakage = True  # type: ignore[attr-defined]

    report = build_report(tmp_path, leakage_report=leakage)

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains("Leakage audit").any()


def test_split_readiness_is_required(tmp_path: Path) -> None:
    preparation = PreparationReport()
    preparation.is_ready_for_split_execution = False  # type: ignore[attr-defined]

    report = build_report(tmp_path, preparation_report=preparation)

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains("Split execution").any()


def test_loader_rejects_tampered_contract(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["prediction_contract"]["target_classes"] = ["A", "B"]
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExplorationHandoffError,
        match="multiclass target contract",
    ):
        load_and_validate_exploration_handoff(destination)


def test_loader_rejects_wrong_source_dataset_id(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    with pytest.raises(
        ExplorationHandoffError,
        match="source dataset ID mismatch",
    ):
        load_and_validate_exploration_handoff(
            destination,
            expected_source_dataset_id=999,
        )

def test_builds_ready_continuous_regression_handoff(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)

    assert report.is_structurally_valid
    assert report.is_handoff_ready
    contract = report.payload["prediction_contract"]
    assert contract["problem_type"] == "continuous_regression"
    assert contract["target_classes"] == []
    assert contract["class_semantics"] is None
    assert contract["target_semantics"] == "Continuous / quantitative"
    assert contract["target_unit"] == "MPa"
    assert report.payload["target_distribution"]["finite_count"] == 4
    assert (
        report.payload["preparation_contract"]["split_policy"][
            "stratification_field"
        ]
        is None
    )
    assert report.payload["readiness"]["model_selection_ready"] is False


def test_continuous_handoff_carries_target_extreme_review(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)
    reviews = report.open_reviews_frame()

    assert "Target extremes" in set(reviews["theme"])
    assert reviews["blocking"].eq(False).all()


def test_continuous_next_steps_do_not_require_class_stratification(
    tmp_path: Path,
) -> None:
    report = build_continuous_report(tmp_path)
    steps = report.next_steps_frame()
    text = " ".join(
        steps.loc[
            steps["Notebook"].eq("02_data_preparation.ipynb"),
            ["Action", "Acceptance criterion"],
        ]
        .astype(str)
        .to_numpy()
        .ravel()
    ).lower()

    assert "non-stratified" in text
    assert "target classes" not in text


def test_continuous_handoff_write_is_reloadable(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)
    destination = (
        tmp_path
        / "artifacts/exploration/concrete-compressive-strength/"
        / "exploration-handoff.json"
    )

    persisted = report.write(destination)
    payload = load_and_validate_exploration_handoff(
        destination,
        expected_dataset_slug="concrete-compressive-strength",
        expected_source_dataset_id=165,
    )

    assert len(persisted.sha256) == 64
    assert payload["prediction_contract"]["problem_type"] == "continuous_regression"
    assert payload["prediction_contract"]["target_unit"] == "MPa"


def test_continuous_handoff_rejects_stratification(tmp_path: Path) -> None:
    preparation = ContinuousPreparationReport()
    original = preparation.split_policy_frame

    def stratified_split_policy() -> pd.DataFrame:
        frame = original()
        frame.loc[
            frame["Policy item"].eq("Stratification field"),
            "Value",
        ] = "StrengthBin"
        return frame

    preparation.split_policy_frame = stratified_split_policy  # type: ignore[method-assign]
    report = build_continuous_report(
        tmp_path,
        preparation_report=preparation,
    )

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains(
        "non-stratified"
    ).any()


def test_continuous_loader_rejects_target_classes(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["prediction_contract"]["target_classes"] = ["low", "high"]
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExplorationHandoffError,
        match="must not declare target classes",
    ):
        load_and_validate_exploration_handoff(destination)


def test_continuous_loader_rejects_missing_semantics(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["prediction_contract"]["target_semantics"] = ""
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExplorationHandoffError,
        match="missing target semantics",
    ):
        load_and_validate_exploration_handoff(destination)


def test_continuous_summary_uses_non_stratified_gate(tmp_path: Path) -> None:
    report = build_continuous_report(tmp_path)
    summary = report.summary_frame()

    split_row = summary.loc[summary["Gate"].eq("Snapshot split may execute")]
    assert len(split_row) == 1
    assert split_row["Ready"].eq(True).all()
    assert split_row["Interpretation"].str.contains(
        "non-stratified"
    ).all()

