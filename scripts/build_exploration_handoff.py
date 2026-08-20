"""Versioned Notebook-01 exploration handoff for static supervised studies.

The module serializes validated exploratory contracts and preparation decisions
without executing preparation, splitting, preprocessing, feature selection, or
model fitting. The resulting JSON is intended to be reloadable from a fresh
kernel by the next notebook.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


HANDOFF_SCHEMA_VERSION = "exploration-handoff.v1"
HANDOFF_ARTIFACT_TYPE = "exploration_handoff"

_ISSUE_COLUMNS = ["Scope", "Issue", "Details"]
_SUMMARY_COLUMNS = ["Gate", "Ready", "Interpretation"]
_NEXT_STEP_COLUMNS = [
    "Notebook",
    "Sequence",
    "Action",
    "Status",
    "Acceptance criterion",
]
_EXPECTED_OUTPUT_COLUMNS = ["Notebook", "Expected output", "Purpose"]
_ARTIFACT_COLUMNS = ["Artifact", "Value"]


class ExplorationHandoffError(RuntimeError):
    """Raised when a Notebook-01 handoff cannot be built or validated."""


def _text(value: object) -> str:
    return str(value).strip()


def _unique_text_tuple(values: Sequence[object] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if not item:
            continue
        if item in seen:
            raise ExplorationHandoffError(
                f"Duplicate contract value is not allowed: {item!r}."
            )
        seen.add(item)
        result.append(item)
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _frame_records(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise ExplorationHandoffError("Expected a pandas DataFrame report.")
    selected = frame
    if columns is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ExplorationHandoffError(
                f"Report is missing required columns: {missing!r}."
            )
        selected = frame.loc[:, list(columns)]
    return _json_safe(selected.to_dict(orient="records"))


def _split_value(split_frame: pd.DataFrame, label: str) -> object:
    matches = split_frame.loc[split_frame["Policy item"].eq(label), "Value"]
    if len(matches) != 1:
        raise ExplorationHandoffError(
            f"Expected exactly one split-policy item {label!r}; found {len(matches)}."
        )
    return deepcopy(matches.iloc[0])


def _optional_split_value(
    split_frame: pd.DataFrame,
    label: str,
    *,
    default: object = None,
) -> object:
    matches = split_frame.loc[split_frame["Policy item"].eq(label), "Value"]
    if len(matches) > 1:
        raise ExplorationHandoffError(
            f"Expected at most one split-policy item {label!r}; found {len(matches)}."
        )
    if matches.empty:
        return deepcopy(default)
    return deepcopy(matches.iloc[0])


def _dependency_review(leakage_report: object) -> tuple[str, ...]:
    frame = leakage_report.dependency_frame()
    if frame.empty:
        return ()
    mask = frame["Dependency status"].eq("Declared dependency not confirmed")
    return tuple(str(value) for value in frame.loc[mask, "Derived feature"])


def _redundancy_count(feature_relationship_report: object) -> int:
    frame = getattr(
        feature_relationship_report,
        "numerical_relationships",
        pd.DataFrame(),
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0
    if "Potential redundancy" not in frame.columns:
        return 0
    return int(frame["Potential redundancy"].fillna(False).sum())


def _open_reviews(
    *,
    duplicate_report: object,
    leakage_report: object,
    feature_relationship_report: object,
    target_report: object,
    insights_report: object,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []

    exact_groups = int(
        getattr(duplicate_report, "exact_duplicate_group_count", 0)
    )
    exact_rows = int(getattr(duplicate_report, "exact_duplicate_row_count", 0))
    has_ids = bool(getattr(duplicate_report, "has_source_identifiers", False))
    if exact_groups and not has_ids:
        reviews.append(
            {
                "review_id": "REV-001",
                "theme": "Duplicate identity",
                "blocking": False,
                "summary": (
                    f"{exact_groups} exact-row group(s) covering {exact_rows} rows "
                    "exist without an independent source identifier."
                ),
                "continuation": (
                    "Preserve rows. During partition validation, report any "
                    "cross-partition identical feature profiles as sensitivity "
                    "evidence; do not silently deduplicate or regroup them."
                ),
            }
        )

    unconfirmed = _dependency_review(leakage_report)
    if unconfirmed:
        reviews.append(
            {
                "review_id": "REV-002",
                "theme": "Derived-feature dependency",
                "blocking": False,
                "summary": (
                    "Declared mathematical dependency not numerically confirmed "
                    f"for: {', '.join(unconfirmed)}."
                ),
                "continuation": (
                    "Retain the affected feature(s) in the baseline and test any "
                    "ablation only inside leakage-safe model selection."
                ),
            }
        )

    redundancy_count = _redundancy_count(feature_relationship_report)
    if redundancy_count:
        reviews.append(
            {
                "review_id": "REV-003",
                "theme": "Feature redundancy",
                "blocking": False,
                "summary": (
                    f"{redundancy_count} feature pair(s) meet the declared "
                    "redundancy-review threshold."
                ),
                "continuation": (
                    "Use the complete feature set as baseline; compare "
                    "regularization or ablation only on training/validation data."
                ),
            }
        )

    imbalance_ratio = getattr(target_report, "imbalance_ratio", None)
    entropy = getattr(target_report, "normalized_class_entropy", None)
    if imbalance_ratio is not None:
        reviews.append(
            {
                "review_id": "REV-004",
                "theme": "Class support",
                "blocking": False,
                "summary": (
                    "Multiclass support is unequal; majority/minority ratio="
                    f"{float(imbalance_ratio):.6g}, normalized entropy="
                    f"{float(entropy):.6g}" if entropy is not None else
                    f"{float(imbalance_ratio):.6g}."
                ),
                "continuation": (
                    "Preserve all classes with stratification. Evaluate class "
                    "weighting or training-only resampling only if validation "
                    "metrics justify it."
                ),
            }
        )

    hypotheses = insights_report.hypotheses_frame()
    for index, row in hypotheses.iterrows():
        reviews.append(
            {
                "review_id": f"REV-HYP-{index + 1:03d}",
                "theme": "Exploratory hypothesis",
                "blocking": False,
                "summary": str(row.get("Hypothesis", row.get("Title", ""))),
                "continuation": str(
                    row.get(
                        "Required validation",
                        "Validate in later modeling stages.",
                    )
                ),
            }
        )

    return reviews


def _next_steps() -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 1,
            "Action": "Reload the UCI source from a fresh kernel and verify identity",
            "Status": "Ready",
            "Acceptance criterion": (
                "Source dataset ID, file hash, row count, column order, target, "
                "and feature roles match the persisted exploration handoff."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 2,
            "Action": "Reconstruct and validate the source-backed contracts",
            "Status": "Ready",
            "Acceptance criterion": (
                "Notebook 02 does not depend on live variables from Notebook 01 "
                "and independently validates all required fields and classes."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 3,
            "Action": "Create an unchanged defensive prepared projection",
            "Status": "Ready",
            "Acceptance criterion": (
                "All source rows and validated values are preserved; no equality-"
                "only deduplication or generic outlier treatment is performed."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 4,
            "Action": "Separate baseline predictors and readable multiclass target",
            "Status": "Ready",
            "Acceptance criterion": (
                "X contains exactly the 16 declared candidate features and y "
                "contains exactly the seven nominal target classes."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 5,
            "Action": "Execute the approved stratified 70/15/15 snapshot split",
            "Status": "Ready",
            "Acceptance criterion": (
                "Train, validation, and test partitions are reproducible, disjoint, "
                "class-aware, and generated with the declared random seed."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 6,
            "Action": "Validate partition integrity and repeated-profile evidence",
            "Status": "Ready",
            "Acceptance criterion": (
                "All source rows occur in exactly one partition, every class is "
                "represented, and cross-partition identical profiles are reported "
                "without being treated as proven duplicate identities."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 7,
            "Action": "Persist the preparation handoff and seal the final test set",
            "Status": "Ready",
            "Acceptance criterion": (
                "Prepared data, feature manifest, split manifest, quality evidence, "
                "and partition artifacts are validated and reloadable by Notebook 03."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 1,
            "Action": "Load and validate the frozen preparation handoff",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Model selection never reconstructs partitions differently or "
                "fits preprocessing on validation/test data."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 2,
            "Action": "Establish multiclass baselines and compare candidate models",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "All candidates use the same partitions and model-appropriate "
                "train-fitted pipelines."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 3,
            "Action": "Evaluate imbalance and redundancy alternatives",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Class weighting, training-only resampling, regularization, and "
                "feature ablation are adopted only from held-out validation evidence."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 4,
            "Action": "Evaluate multiclass errors and class-pair overlap hypotheses",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Macro and per-class metrics plus confusion evidence test the "
                "exploratory overlap hypotheses without using the final test set."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 5,
            "Action": "Freeze the candidate before one-time final-test evaluation",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Model family, preprocessing, features, hyperparameters, and all "
                "selection choices are frozen before final test access."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=_NEXT_STEP_COLUMNS)


def _expected_outputs(dataset_slug: str) -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": f"artifacts/preparation/{dataset_slug}/preparation-manifest.json",
            "Purpose": "Record source identity, preparation policy, and readiness.",
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": f"artifacts/preparation/{dataset_slug}/feature-manifest.json",
            "Purpose": "Freeze feature order, target classes, and predictor roles.",
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": f"artifacts/preparation/{dataset_slug}/split-manifest.json",
            "Purpose": "Prove reproducible partition policy and isolation.",
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": f"artifacts/preparation/{dataset_slug}/quality-evidence.json",
            "Purpose": "Prove post-preparation integrity and source preservation.",
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Expected output": f"artifacts/model-selection/{dataset_slug}/model-selection-handoff.json",
            "Purpose": "Freeze the selected candidate and downstream evaluation contract.",
        },
    ]
    return pd.DataFrame(rows, columns=_EXPECTED_OUTPUT_COLUMNS)


def _continuous_open_reviews(
    *,
    duplicate_report: object,
    leakage_report: object,
    feature_relationship_report: object,
    target_report: object,
    insights_report: object,
) -> list[dict[str, Any]]:
    """Carry forward non-blocking continuous-regression review evidence."""
    reviews = _open_reviews(
        duplicate_report=duplicate_report,
        leakage_report=leakage_report,
        feature_relationship_report=feature_relationship_report,
        target_report=target_report,
        insights_report=insights_report,
    )

    extreme_count = int(getattr(target_report, "extreme_count", 0))
    finite_count = int(getattr(target_report, "finite_count", 0))
    if extreme_count:
        extreme_share = (
            None if finite_count <= 0 else extreme_count / finite_count
        )
        reviews.insert(
            min(3, len(reviews)),
            {
                "review_id": "REV-004",
                "theme": "Target extremes",
                "blocking": False,
                "summary": (
                    f"{extreme_count} target observation(s) fall outside the "
                    "descriptive 1.5-IQR fences"
                    + (
                        "."
                        if extreme_share is None
                        else f" ({extreme_share:.2%} of finite target values)."
                    )
                ),
                "continuation": (
                    "Preserve these observations. Later regression evaluation "
                    "should report residual/error sensitivity without using "
                    "target extremes as an automatic deletion rule."
                ),
            },
        )

    # Keep review identifiers deterministic after inserting a continuous-only item.
    for index, review in enumerate(reviews, start=1):
        if not str(review.get("review_id", "")).startswith("REV-HYP-"):
            review["review_id"] = f"REV-{index:03d}"

    return reviews


def _continuous_next_steps() -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 1,
            "Action": "Reload the UCI source from a fresh kernel and verify identity",
            "Status": "Ready",
            "Acceptance criterion": (
                "Source dataset ID, file hash, row count, column order, target, "
                "and feature roles match the persisted exploration handoff."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 2,
            "Action": "Reconstruct and validate the continuous-regression contracts",
            "Status": "Ready",
            "Acceptance criterion": (
                "Notebook 02 does not depend on live variables from Notebook 01 "
                "and independently validates target semantics, units, feature roles, "
                "and declared source-type resolutions."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 3,
            "Action": "Create an unchanged defensive prepared projection",
            "Status": "Ready",
            "Acceptance criterion": (
                "All released rows and validated numerical values are preserved; "
                "source-type metadata conflicts are resolved without rounding, "
                "truncation, generic outlier treatment, or equality-only deduplication."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 4,
            "Action": "Separate validated predictors and the continuous target",
            "Status": "Ready",
            "Acceptance criterion": (
                "X contains exactly the declared candidate features and y remains "
                "numeric, continuous, and on the original target scale."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 5,
            "Action": "Execute the approved reproducible 70/15/15 snapshot split",
            "Status": "Ready",
            "Acceptance criterion": (
                "Train, validation, and test partitions are shuffled, reproducible, "
                "disjoint, non-stratified, and generated with the declared random seed."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 6,
            "Action": "Validate partition integrity and continuous-target coverage",
            "Status": "Ready",
            "Acceptance criterion": (
                "Every source row occurs in exactly one partition; target range and "
                "quantiles plus repeated-profile evidence are reported diagnostically "
                "without seed-shopping or target-derived regrouping."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Sequence": 7,
            "Action": "Persist the preparation handoff and seal the final test set",
            "Status": "Ready",
            "Acceptance criterion": (
                "Prepared data, feature manifest, split manifest, quality evidence, "
                "and partition artifacts are validated and reloadable by Notebook 03."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 1,
            "Action": "Load and validate the frozen preparation handoff",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Model selection never reconstructs partitions differently or fits "
                "preprocessing on validation/test data."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 2,
            "Action": "Establish regression baselines and compare candidate models",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "All candidates use the same frozen partitions, regression metrics, "
                "and model-appropriate train-fitted pipelines."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 3,
            "Action": "Evaluate scaling, redundancy, nonlinearity, and interactions",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Transformations, feature ablations, and flexible model structure "
                "are adopted only from leakage-safe training/validation evidence."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 4,
            "Action": "Evaluate regression errors and exploratory sensitivity hypotheses",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Held-out regression metrics and residual diagnostics test target-"
                "extreme, repeated-profile, and structural hypotheses without using "
                "the final test set for selection."
            ),
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Sequence": 5,
            "Action": "Freeze the candidate before one-time final-test evaluation",
            "Status": "Waiting on Notebook 02",
            "Acceptance criterion": (
                "Model family, preprocessing, features, hyperparameters, and all "
                "selection choices are frozen before final test access."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=_NEXT_STEP_COLUMNS)


def _continuous_expected_outputs(dataset_slug: str) -> pd.DataFrame:
    rows = [
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": (
                f"artifacts/preparation/{dataset_slug}/preparation-manifest.json"
            ),
            "Purpose": "Record source identity, preparation policy, and readiness.",
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": (
                f"artifacts/preparation/{dataset_slug}/feature-manifest.json"
            ),
            "Purpose": (
                "Freeze feature order, continuous-target semantics/unit, and "
                "predictor roles."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": (
                f"artifacts/preparation/{dataset_slug}/split-manifest.json"
            ),
            "Purpose": (
                "Prove the reproducible non-stratified partition policy and isolation."
            ),
        },
        {
            "Notebook": "02_data_preparation.ipynb",
            "Expected output": (
                f"artifacts/preparation/{dataset_slug}/quality-evidence.json"
            ),
            "Purpose": "Prove post-preparation integrity and source preservation.",
        },
        {
            "Notebook": "03_model_selection_and_evaluation.ipynb",
            "Expected output": (
                f"artifacts/model-selection/{dataset_slug}/model-selection-handoff.json"
            ),
            "Purpose": (
                "Freeze the selected regression candidate and downstream "
                "evaluation contract."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=_EXPECTED_OUTPUT_COLUMNS)


@dataclass(frozen=True, slots=True)
class PersistedExplorationHandoff:
    """Metadata for one materialized exploration handoff."""

    path: Path
    sha256: str
    size_bytes: int

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {"Artifact": "Path", "Value": self.path.as_posix()},
            {"Artifact": "SHA-256", "Value": self.sha256},
            {"Artifact": "Bytes", "Value": self.size_bytes},
        ]
        return pd.DataFrame(rows, columns=_ARTIFACT_COLUMNS)


@dataclass(frozen=True, slots=True)
class ExplorationHandoffReport:
    """Validated, serializable Notebook-01 transition contract."""

    payload: dict[str, Any]
    issues: pd.DataFrame
    next_steps: pd.DataFrame
    expected_outputs: pd.DataFrame

    @property
    def is_structurally_valid(self) -> bool:
        return self.issues.empty

    @property
    def is_handoff_ready(self) -> bool:
        readiness = self.payload.get("readiness", {})
        return bool(
            self.is_structurally_valid
            and readiness.get("notebook_01_complete") is True
            and readiness.get("deterministic_preparation_ready") is True
            and readiness.get("split_execution_ready") is True
        )

    def summary_frame(self) -> pd.DataFrame:
        readiness = self.payload["readiness"]
        problem_type = self.payload.get("prediction_contract", {}).get(
            "problem_type"
        )
        if problem_type == "continuous_regression":
            split_gate = "Snapshot split may execute"
            split_interpretation = (
                "The 70/15/15 shuffled non-stratified snapshot policy is resolved."
            )
        else:
            split_gate = "Stratified snapshot split may execute"
            split_interpretation = (
                "The 70/15/15 class-stratified snapshot policy is resolved."
            )
        rows = [
            {
                "Gate": "Notebook 01 analysis complete",
                "Ready": readiness["notebook_01_complete"],
                "Interpretation": "Exploration, quality, leakage, insight, and decision contracts are complete.",
            },
            {
                "Gate": "Handoff artifact structurally valid",
                "Ready": self.is_structurally_valid,
                "Interpretation": "Serialized source, role, evidence, and continuation contracts are coherent.",
            },
            {
                "Gate": "Deterministic preparation may begin",
                "Ready": readiness["deterministic_preparation_ready"],
                "Interpretation": "Notebook 02 may independently reconstruct and preserve the validated source.",
            },
            {
                "Gate": split_gate,
                "Ready": readiness["split_execution_ready"],
                "Interpretation": split_interpretation,
            },
            {
                "Gate": "Model selection may begin",
                "Ready": readiness["model_selection_ready"],
                "Interpretation": "Expected to remain false until Notebook 02 publishes validated partitions.",
            },
            {
                "Gate": "Notebook 01 handoff ready",
                "Ready": self.is_handoff_ready,
                "Interpretation": "Notebook 02 can continue from a fresh kernel using the persisted contract.",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def next_steps_frame(self) -> pd.DataFrame:
        return self.next_steps.copy(deep=True)

    def expected_outputs_frame(self) -> pd.DataFrame:
        return self.expected_outputs.copy(deep=True)

    def open_reviews_frame(self) -> pd.DataFrame:
        return pd.DataFrame(deepcopy(self.payload["open_reviews"]))

    def issues_frame(self) -> pd.DataFrame:
        return self.issues.copy(deep=True)

    def raise_if_invalid(self) -> None:
        if self.is_structurally_valid and self.is_handoff_ready:
            return
        if not self.issues.empty:
            details = "; ".join(
                f"{row['Scope']}: {row['Issue']} ({row['Details']})"
                for _, row in self.issues.iterrows()
            )
        else:
            details = "handoff readiness gates are not satisfied"
        raise ExplorationHandoffError(
            f"Notebook-01 exploration handoff is not ready: {details}."
        )

    def write(self, destination: str | Path) -> PersistedExplorationHandoff:
        """Atomically persist the validated JSON handoff."""
        self.raise_if_invalid()
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            _json_safe(self.payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        return PersistedExplorationHandoff(
            path=path,
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )


def build_static_multiclass_exploration_handoff(
    *,
    dataset_slug: str,
    source_repository: str,
    source_dataset_id: int,
    source_file: str | Path,
    project_root: str | Path | None,
    source_dataframe: pd.DataFrame,
    target_contract: object,
    feature_columns: Sequence[object],
    numerical_features: Sequence[object],
    identifier_columns: Sequence[object] | None,
    target_report: object,
    duplicate_report: object,
    feature_relationship_report: object,
    leakage_report: object,
    quality_report: object,
    insights_report: object,
    preparation_report: object,
) -> ExplorationHandoffReport:
    """Build the final Notebook-01 handoff without modifying analytical data."""
    issues: list[dict[str, str]] = []

    slug = _text(dataset_slug)
    repository = _text(source_repository)
    source = Path(source_file).expanduser().resolve()
    project = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )
    features = _unique_text_tuple(feature_columns)
    numericals = _unique_text_tuple(numerical_features)
    identifiers = _unique_text_tuple(identifier_columns)
    target = _text(getattr(target_contract, "target", ""))
    classes = tuple(deepcopy(getattr(target_contract, "expected_classes", ())))
    problem_type = _text(getattr(target_contract, "problem_type", ""))

    if not slug:
        issues.append({"Scope": "Dataset", "Issue": "Missing dataset slug", "Details": "dataset_slug is empty"})
    if not repository:
        issues.append({"Scope": "Source", "Issue": "Missing repository name", "Details": "source_repository is empty"})
    if not source.is_file():
        issues.append({"Scope": "Source", "Issue": "Source file unavailable", "Details": source.as_posix()})
    if not isinstance(source_dataframe, pd.DataFrame) or source_dataframe.empty:
        issues.append({"Scope": "Source", "Issue": "Invalid source dataframe", "Details": "A non-empty pandas DataFrame is required"})
    if problem_type != "multiclass_classification":
        issues.append({"Scope": "Target", "Issue": "Unexpected problem type", "Details": repr(problem_type)})
    if len(classes) < 3:
        issues.append({"Scope": "Target", "Issue": "Multiclass contract incomplete", "Details": f"Declared classes={len(classes)}"})
    if not features:
        issues.append({"Scope": "Features", "Issue": "No candidate features", "Details": "feature_columns is empty"})
    if target in set(features):
        issues.append({"Scope": "Features", "Issue": "Target included in predictors", "Details": target})
    if set(numericals) != set(features):
        issues.append({
            "Scope": "Features",
            "Issue": "Dry Bean handoff expects an entirely numerical baseline",
            "Details": f"features={len(features)}, numerical={len(numericals)}",
        })
    source_columns = tuple(str(value) for value in source_dataframe.columns)
    unknown_roles = [
        value
        for value in (*features, *identifiers, target)
        if value and value not in set(source_columns)
    ]
    if unknown_roles:
        issues.append({"Scope": "Roles", "Issue": "Declared fields missing from source", "Details": repr(tuple(unknown_roles))})

    contract_checks = (
        ("Target distribution", not bool(getattr(target_report, "has_issues", True))),
        ("Leakage audit", bool(getattr(leakage_report, "is_structurally_valid", False)) and not bool(getattr(leakage_report, "has_direct_target_leakage", True))),
        ("Initial data quality", bool(getattr(quality_report, "is_structurally_valid", False))),
        ("Exploratory insights", bool(getattr(insights_report, "is_structurally_valid", False))),
        ("Preparation decisions", bool(getattr(preparation_report, "is_structurally_valid", False))),
    )
    for name, valid in contract_checks:
        if not valid:
            issues.append({"Scope": "Upstream contract", "Issue": f"{name} is not handoff-safe", "Details": "Required validation/readiness condition is false"})

    prep_ready = bool(
        getattr(
            preparation_report,
            "is_ready_for_deterministic_preparation",
            False,
        )
    )
    split_ready = bool(
        getattr(preparation_report, "is_ready_for_split_execution", False)
    )
    if not prep_ready:
        issues.append({"Scope": "Readiness", "Issue": "Deterministic preparation not ready", "Details": "Stage 18 did not clear deterministic preparation"})
    if not split_ready:
        issues.append({"Scope": "Readiness", "Issue": "Split execution not ready", "Details": "Stage 18 did not clear the split policy"})

    split_frame = preparation_report.split_policy_frame()
    decisions_frame = preparation_report.decisions_frame()
    execution_frame = preparation_report.execution_plan_frame()
    guardrails_frame = preparation_report.guardrails_frame()

    train_fraction = _split_value(split_frame, "Train fraction")
    validation_fraction = _split_value(split_frame, "Validation fraction")
    test_fraction = _split_value(split_frame, "Test fraction")
    stratification_field = _split_value(split_frame, "Stratification field")
    random_seed = _split_value(split_frame, "Random seed")

    distribution = target_report.distribution_frame(format_percentages=False)
    quality_findings = quality_report.findings_frame()
    non_issues = quality_report.validated_non_issues_frame()
    key_insights = insights_report.key_insights_frame()
    hypotheses = insights_report.hypotheses_frame()
    limitations = insights_report.limitations_frame()
    dependencies = leakage_report.dependency_frame()
    open_reviews = _open_reviews(
        duplicate_report=duplicate_report,
        leakage_report=leakage_report,
        feature_relationship_report=feature_relationship_report,
        target_report=target_report,
        insights_report=insights_report,
    )
    next_steps = _next_steps()
    expected_outputs = _expected_outputs(slug)

    source_sha256 = _sha256_file(source) if source.is_file() else None

    payload: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "artifact_type": HANDOFF_ARTIFACT_TYPE,
        "dataset_slug": slug,
        "source": {
            "repository": repository,
            "dataset_id": int(source_dataset_id),
            "path": _relative_posix(source, project),
            "sha256": source_sha256,
            "row_count": int(len(source_dataframe)),
            "column_count": int(len(source_dataframe.columns)),
            "column_order": list(source_columns),
        },
        "prediction_contract": {
            "problem_type": problem_type,
            "target_column": target,
            "target_classes": _json_safe(classes),
            "class_semantics": _text(
                getattr(target_contract, "class_semantics", "Nominal / unordered")
            ),
            "positive_class": None,
        },
        "feature_contract": {
            "identifier_columns": list(identifiers),
            "feature_columns": list(features),
            "numerical_features": list(numericals),
            "categorical_features": [],
            "baseline_feature_count": len(features),
        },
        "target_distribution": {
            "class_count": int(getattr(target_report, "class_count", 0)),
            "imbalance_ratio": _json_safe(
                getattr(target_report, "imbalance_ratio", None)
            ),
            "normalized_class_entropy": _json_safe(
                getattr(target_report, "normalized_class_entropy", None)
            ),
            "majority_classes": _json_safe(
                getattr(target_report, "majority_classes", ())
            ),
            "minority_classes": _json_safe(
                getattr(target_report, "minority_classes", ())
            ),
            "distribution": _frame_records(distribution),
        },
        "data_quality": {
            "structurally_valid": bool(
                getattr(quality_report, "is_structurally_valid", False)
            ),
            "findings": _frame_records(quality_findings),
            "validated_non_issues": _frame_records(non_issues),
            "duplicate_identity": {
                "source_identifiers_available": bool(
                    getattr(duplicate_report, "has_source_identifiers", False)
                ),
                "exact_duplicate_group_count": int(
                    getattr(duplicate_report, "exact_duplicate_group_count", 0)
                ),
                "exact_duplicate_row_count": int(
                    getattr(duplicate_report, "exact_duplicate_row_count", 0)
                ),
                "target_conflict_group_count": int(
                    getattr(duplicate_report, "target_conflict_group_count", 0)
                ),
            },
        },
        "leakage_and_dependencies": {
            "direct_target_leakage": bool(
                getattr(leakage_report, "has_direct_target_leakage", True)
            ),
            "target_proxy_candidate_count": len(
                leakage_report.target_proxy_candidates_frame()
            ),
            "confirmed_derived_dependency_count": int(
                getattr(leakage_report, "confirmed_derived_dependency_count", 0)
            ),
            "dependencies": _frame_records(dependencies),
        },
        "exploratory_synthesis": {
            "key_insights": _frame_records(key_insights),
            "hypotheses": _frame_records(hypotheses),
            "limitations": _frame_records(limitations),
        },
        "preparation_contract": {
            "decisions": _frame_records(decisions_frame),
            "execution_plan": _frame_records(execution_frame),
            "guardrails": _frame_records(guardrails_frame),
            "split_policy": {
                "train_fraction": _json_safe(train_fraction),
                "validation_fraction": _json_safe(validation_fraction),
                "test_fraction": _json_safe(test_fraction),
                "stratification_field": _json_safe(stratification_field),
                "random_seed": _json_safe(random_seed),
                "test_holdout_untouched": bool(
                    _split_value(split_frame, "Final test holdout")
                ),
                "disjoint_partitions_required": bool(
                    _split_value(split_frame, "Disjoint partitions")
                ),
                "identifier_grouping": _json_safe(
                    _split_value(split_frame, "Identifier grouping")
                ),
                "temporal_policy_status": _json_safe(
                    _split_value(split_frame, "Temporal policy status")
                ),
            },
        },
        "open_reviews": open_reviews,
        "continuation": {
            "next_steps": _frame_records(next_steps),
            "expected_outputs": _frame_records(expected_outputs),
            "fresh_kernel_required": True,
            "notebook_02_must_revalidate_source": True,
            "notebook_03_must_consume_frozen_preparation_handoff": True,
        },
        "readiness": {
            "notebook_01_complete": len(issues) == 0,
            "deterministic_preparation_ready": prep_ready,
            "split_execution_ready": split_ready,
            "model_selection_ready": False,
            "model_selection_waits_for_notebook_02": True,
        },
    }

    return ExplorationHandoffReport(
        payload=_json_safe(payload),
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
        next_steps=next_steps,
        expected_outputs=expected_outputs,
    )


def build_static_continuous_regression_exploration_handoff(
    *,
    dataset_slug: str,
    source_repository: str,
    source_dataset_id: int,
    source_file: str | Path,
    project_root: str | Path | None,
    source_dataframe: pd.DataFrame,
    target_contract: object,
    feature_columns: Sequence[object],
    numerical_features: Sequence[object],
    identifier_columns: Sequence[object] | None,
    target_report: object,
    duplicate_report: object,
    feature_relationship_report: object,
    leakage_report: object,
    quality_report: object,
    insights_report: object,
    preparation_report: object,
) -> ExplorationHandoffReport:
    """Build the final Notebook-01 handoff for static continuous regression."""
    issues: list[dict[str, str]] = []

    slug = _text(dataset_slug)
    repository = _text(source_repository)
    source = Path(source_file).expanduser().resolve()
    project = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )
    features = _unique_text_tuple(feature_columns)
    numericals = _unique_text_tuple(numerical_features)
    identifiers = _unique_text_tuple(identifier_columns)
    target = _text(getattr(target_contract, "target", ""))
    problem_type = _text(getattr(target_contract, "problem_type", ""))
    target_semantics = _text(
        getattr(target_contract, "target_semantics", "")
    )
    target_unit = (
        getattr(target_contract, "expected_unit", None)
        or getattr(target_contract, "source_unit", None)
    )
    target_unit = None if target_unit is None else _text(target_unit)
    prediction_output = _text(
        getattr(
            target_contract,
            "prediction_output",
            "Continuous numeric value on the original target scale",
        )
    )

    if not slug:
        issues.append({
            "Scope": "Dataset",
            "Issue": "Missing dataset slug",
            "Details": "dataset_slug is empty",
        })
    if not repository:
        issues.append({
            "Scope": "Source",
            "Issue": "Missing repository name",
            "Details": "source_repository is empty",
        })
    if not source.is_file():
        issues.append({
            "Scope": "Source",
            "Issue": "Source file unavailable",
            "Details": source.as_posix(),
        })
    if not isinstance(source_dataframe, pd.DataFrame) or source_dataframe.empty:
        issues.append({
            "Scope": "Source",
            "Issue": "Invalid source dataframe",
            "Details": "A non-empty pandas DataFrame is required",
        })
    if problem_type != "continuous_regression":
        issues.append({
            "Scope": "Target",
            "Issue": "Unexpected problem type",
            "Details": repr(problem_type),
        })
    if not target_semantics:
        issues.append({
            "Scope": "Target",
            "Issue": "Continuous target semantics missing",
            "Details": "target_semantics is empty",
        })
    if not prediction_output:
        issues.append({
            "Scope": "Target",
            "Issue": "Prediction output contract missing",
            "Details": "prediction_output is empty",
        })
    if not features:
        issues.append({
            "Scope": "Features",
            "Issue": "No candidate features",
            "Details": "feature_columns is empty",
        })
    if target in set(features):
        issues.append({
            "Scope": "Features",
            "Issue": "Target included in predictors",
            "Details": target,
        })
    if set(numericals) != set(features):
        issues.append({
            "Scope": "Features",
            "Issue": (
                "Static continuous-regression handoff expects an entirely "
                "numerical baseline"
            ),
            "Details": f"features={len(features)}, numerical={len(numericals)}",
        })

    source_columns = tuple(str(value) for value in source_dataframe.columns)
    unknown_roles = [
        value
        for value in (*features, *identifiers, target)
        if value and value not in set(source_columns)
    ]
    if unknown_roles:
        issues.append({
            "Scope": "Roles",
            "Issue": "Declared fields missing from source",
            "Details": repr(tuple(unknown_roles)),
        })

    target_report_target = _text(getattr(target_report, "target", ""))
    if target_report_target and target_report_target != target:
        issues.append({
            "Scope": "Target",
            "Issue": "Target report does not match prediction contract",
            "Details": f"{target_report_target!r} != {target!r}",
        })

    target_issues_method = getattr(target_report, "issues_frame", None)
    if callable(target_issues_method):
        target_issues = target_issues_method()
        target_safe = isinstance(target_issues, pd.DataFrame) and target_issues.empty
    else:
        target_safe = bool(
            int(getattr(target_report, "missing_count", 1)) == 0
            and int(getattr(target_report, "non_finite_count", 1)) == 0
            and bool(getattr(target_report, "has_variation", False))
        )

    contract_checks = (
        ("Target distribution", target_safe),
        (
            "Leakage audit",
            bool(getattr(leakage_report, "is_structurally_valid", False))
            and not bool(
                getattr(leakage_report, "has_direct_target_leakage", True)
            ),
        ),
        (
            "Initial data quality",
            bool(getattr(quality_report, "is_structurally_valid", False)),
        ),
        (
            "Exploratory insights",
            bool(getattr(insights_report, "is_structurally_valid", False)),
        ),
        (
            "Preparation decisions",
            bool(getattr(preparation_report, "is_structurally_valid", False)),
        ),
    )
    for name, valid in contract_checks:
        if not valid:
            issues.append({
                "Scope": "Upstream contract",
                "Issue": f"{name} is not handoff-safe",
                "Details": "Required validation/readiness condition is false",
            })

    prep_ready = bool(
        getattr(
            preparation_report,
            "is_ready_for_deterministic_preparation",
            False,
        )
    )
    split_ready = bool(
        getattr(preparation_report, "is_ready_for_split_execution", False)
    )
    if not prep_ready:
        issues.append({
            "Scope": "Readiness",
            "Issue": "Deterministic preparation not ready",
            "Details": "Stage 18 did not clear deterministic preparation",
        })
    if not split_ready:
        issues.append({
            "Scope": "Readiness",
            "Issue": "Split execution not ready",
            "Details": "Stage 18 did not clear the split policy",
        })

    split_frame = preparation_report.split_policy_frame()
    decisions_frame = preparation_report.decisions_frame()
    execution_frame = preparation_report.execution_plan_frame()
    guardrails_frame = preparation_report.guardrails_frame()

    train_fraction = _split_value(split_frame, "Train fraction")
    validation_fraction = _split_value(split_frame, "Validation fraction")
    test_fraction = _split_value(split_frame, "Test fraction")
    stratification_field = _split_value(split_frame, "Stratification field")
    random_seed = _split_value(split_frame, "Random seed")

    if stratification_field not in (None, "", ()):
        issues.append({
            "Scope": "Split policy",
            "Issue": "Continuous snapshot split must remain non-stratified",
            "Details": repr(stratification_field),
        })

    quality_findings = quality_report.findings_frame()
    non_issues = quality_report.validated_non_issues_frame()
    key_insights = insights_report.key_insights_frame()
    hypotheses = insights_report.hypotheses_frame()
    limitations = insights_report.limitations_frame()
    dependencies = leakage_report.dependency_frame()
    open_reviews = _continuous_open_reviews(
        duplicate_report=duplicate_report,
        leakage_report=leakage_report,
        feature_relationship_report=feature_relationship_report,
        target_report=target_report,
        insights_report=insights_report,
    )
    next_steps = _continuous_next_steps()
    expected_outputs = _continuous_expected_outputs(slug)

    source_sha256 = _sha256_file(source) if source.is_file() else None

    target_summary = target_report.summary_frame()
    target_quantiles = target_report.quantiles_frame()
    target_extremes = target_report.extremes_frame()

    payload: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "artifact_type": HANDOFF_ARTIFACT_TYPE,
        "dataset_slug": slug,
        "source": {
            "repository": repository,
            "dataset_id": int(source_dataset_id),
            "path": _relative_posix(source, project),
            "sha256": source_sha256,
            "row_count": int(len(source_dataframe)),
            "column_count": int(len(source_dataframe.columns)),
            "column_order": list(source_columns),
        },
        "prediction_contract": {
            "problem_type": problem_type,
            "target_column": target,
            "target_classes": [],
            "class_semantics": None,
            "positive_class": None,
            "target_semantics": target_semantics,
            "target_unit": target_unit,
            "prediction_output": prediction_output,
        },
        "feature_contract": {
            "identifier_columns": list(identifiers),
            "feature_columns": list(features),
            "numerical_features": list(numericals),
            "categorical_features": [],
            "baseline_feature_count": len(features),
        },
        "target_distribution": {
            "unit": target_unit or getattr(target_report, "unit", None),
            "row_count": int(getattr(target_report, "row_count", 0)),
            "finite_count": int(getattr(target_report, "finite_count", 0)),
            "missing_count": int(getattr(target_report, "missing_count", 0)),
            "non_finite_count": int(
                getattr(target_report, "non_finite_count", 0)
            ),
            "unique_count": int(getattr(target_report, "unique_count", 0)),
            "minimum": _json_safe(getattr(target_report, "minimum", None)),
            "mean": _json_safe(getattr(target_report, "mean", None)),
            "median": _json_safe(getattr(target_report, "median", None)),
            "maximum": _json_safe(getattr(target_report, "maximum", None)),
            "standard_deviation": _json_safe(
                getattr(target_report, "standard_deviation", None)
            ),
            "iqr": _json_safe(getattr(target_report, "iqr", None)),
            "extreme_count": int(getattr(target_report, "extreme_count", 0)),
            "extreme_share": _json_safe(
                getattr(target_report, "extreme_share", None)
            ),
            "summary": _frame_records(target_summary),
            "quantiles": _frame_records(target_quantiles),
            "extremes": _frame_records(target_extremes),
        },
        "data_quality": {
            "structurally_valid": bool(
                getattr(quality_report, "is_structurally_valid", False)
            ),
            "findings": _frame_records(quality_findings),
            "validated_non_issues": _frame_records(non_issues),
            "duplicate_identity": {
                "source_identifiers_available": bool(
                    getattr(duplicate_report, "has_source_identifiers", False)
                ),
                "exact_duplicate_group_count": int(
                    getattr(duplicate_report, "exact_duplicate_group_count", 0)
                ),
                "exact_duplicate_row_count": int(
                    getattr(duplicate_report, "exact_duplicate_row_count", 0)
                ),
                "target_conflict_group_count": int(
                    getattr(duplicate_report, "target_conflict_group_count", 0)
                ),
            },
        },
        "leakage_and_dependencies": {
            "direct_target_leakage": bool(
                getattr(leakage_report, "has_direct_target_leakage", True)
            ),
            "target_proxy_candidate_count": len(
                leakage_report.target_proxy_candidates_frame()
            ),
            "confirmed_derived_dependency_count": int(
                getattr(leakage_report, "confirmed_derived_dependency_count", 0)
            ),
            "dependencies": _frame_records(dependencies),
        },
        "exploratory_synthesis": {
            "key_insights": _frame_records(key_insights),
            "hypotheses": _frame_records(hypotheses),
            "limitations": _frame_records(limitations),
        },
        "preparation_contract": {
            "decisions": _frame_records(decisions_frame),
            "execution_plan": _frame_records(execution_frame),
            "guardrails": _frame_records(guardrails_frame),
            "split_policy": {
                "train_fraction": _json_safe(train_fraction),
                "validation_fraction": _json_safe(validation_fraction),
                "test_fraction": _json_safe(test_fraction),
                "stratification_field": None,
                "random_seed": _json_safe(random_seed),
                "shuffle_random_split": bool(
                    _optional_split_value(
                        split_frame,
                        "Shuffle random split",
                        default=True,
                    )
                ),
                "test_holdout_untouched": bool(
                    _split_value(split_frame, "Final test holdout")
                ),
                "disjoint_partitions_required": bool(
                    _split_value(split_frame, "Disjoint partitions")
                ),
                "identifier_grouping": _json_safe(
                    _split_value(split_frame, "Identifier grouping")
                ),
                "temporal_policy_status": _json_safe(
                    _split_value(split_frame, "Temporal policy status")
                ),
                "random_split_fallback": _json_safe(
                    _optional_split_value(
                        split_frame,
                        "Random-split fallback",
                    )
                ),
            },
        },
        "open_reviews": open_reviews,
        "continuation": {
            "next_steps": _frame_records(next_steps),
            "expected_outputs": _frame_records(expected_outputs),
            "fresh_kernel_required": True,
            "notebook_02_must_revalidate_source": True,
            "notebook_03_must_consume_frozen_preparation_handoff": True,
        },
        "readiness": {
            "notebook_01_complete": len(issues) == 0,
            "deterministic_preparation_ready": prep_ready,
            "split_execution_ready": split_ready,
            "model_selection_ready": False,
            "model_selection_waits_for_notebook_02": True,
        },
    }

    return ExplorationHandoffReport(
        payload=_json_safe(payload),
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
        next_steps=next_steps,
        expected_outputs=expected_outputs,
    )


def load_and_validate_exploration_handoff(
    path: str | Path,
    *,
    expected_dataset_slug: str | None = None,
    expected_source_dataset_id: int | None = None,
) -> dict[str, Any]:
    """Load a persisted handoff and validate its portable top-level contract."""
    handoff_path = Path(path).expanduser().resolve()
    if not handoff_path.is_file():
        raise ExplorationHandoffError(
            f"Exploration handoff artifact is missing: {handoff_path.as_posix()}."
        )
    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExplorationHandoffError(
            f"Exploration handoff JSON is invalid: {exc}."
        ) from exc

    required = {
        "schema_version",
        "artifact_type",
        "dataset_slug",
        "source",
        "prediction_contract",
        "feature_contract",
        "preparation_contract",
        "continuation",
        "readiness",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ExplorationHandoffError(
            f"Exploration handoff is missing required fields: {missing!r}."
        )
    if payload["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ExplorationHandoffError(
            "Unexpected exploration handoff schema version: "
            f"{payload['schema_version']!r}."
        )
    if payload["artifact_type"] != HANDOFF_ARTIFACT_TYPE:
        raise ExplorationHandoffError(
            f"Unexpected exploration handoff artifact type: {payload['artifact_type']!r}."
        )
    if expected_dataset_slug is not None and payload["dataset_slug"] != expected_dataset_slug:
        raise ExplorationHandoffError(
            "Exploration handoff dataset slug mismatch: "
            f"{payload['dataset_slug']!r} != {expected_dataset_slug!r}."
        )
    if (
        expected_source_dataset_id is not None
        and int(payload["source"].get("dataset_id", -1))
        != int(expected_source_dataset_id)
    ):
        raise ExplorationHandoffError(
            "Exploration handoff source dataset ID mismatch."
        )
    readiness = payload["readiness"]
    if readiness.get("notebook_01_complete") is not True:
        raise ExplorationHandoffError(
            "Exploration handoff does not declare Notebook 01 complete."
        )
    if readiness.get("split_execution_ready") is not True:
        raise ExplorationHandoffError(
            "Exploration handoff does not authorize split execution."
        )

    prediction_contract = payload["prediction_contract"]
    target = prediction_contract.get("target_column")
    features = payload["feature_contract"].get("feature_columns", [])
    classes = prediction_contract.get("target_classes", [])
    problem_type = prediction_contract.get("problem_type")

    if target in features:
        raise ExplorationHandoffError(
            "Persisted handoff includes the target among predictor features."
        )

    if problem_type == "multiclass_classification":
        if len(classes) < 3:
            raise ExplorationHandoffError(
                "Persisted handoff does not contain a multiclass target contract."
            )
    elif problem_type == "continuous_regression":
        if classes:
            raise ExplorationHandoffError(
                "Persisted continuous regression handoff must not declare "
                "target classes."
            )
        if not _text(prediction_contract.get("target_semantics", "")):
            raise ExplorationHandoffError(
                "Persisted continuous regression handoff is missing target semantics."
            )
        if not _text(prediction_contract.get("prediction_output", "")):
            raise ExplorationHandoffError(
                "Persisted continuous regression handoff is missing the "
                "prediction-output contract."
            )
        split_policy = payload["preparation_contract"].get("split_policy", {})
        if split_policy.get("stratification_field") not in (None, "", []):
            raise ExplorationHandoffError(
                "Persisted continuous regression handoff must use the approved "
                "non-stratified snapshot policy."
            )
    else:
        raise ExplorationHandoffError(
            "Persisted exploration handoff has unsupported problem type: "
            f"{problem_type!r}."
        )

    return deepcopy(payload)
