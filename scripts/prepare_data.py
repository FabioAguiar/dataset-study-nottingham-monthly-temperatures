"""Deterministic and reusable tabular data-preparation utilities.

The module contains dataset-agnostic validation, conditional numeric
materialization, classification splitting, manifest construction, atomic
artifact persistence, and preparation-handoff validation. Dataset-specific
contracts are supplied by callers.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from pandas.api import types as pandas_types
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split


CONTRACT_VERSION = "tabular-preparation.v1"
CSV_ENCODING = "utf-8"
CSV_LINE_TERMINATOR = "\n"
_VOLATILE_SEMANTIC_KEYS = frozenset(
    {
        "created_at",
        "created_at_utc",
        "generated_at",
        "generated_at_utc",
        "timestamp",
        "updated_at",
        "updated_at_utc",
    }
)


class PreparationError(RuntimeError):
    """Base exception for preparation failures."""


class DatasetValidationError(PreparationError):
    """Raised when a raw or prepared dataset violates its contract."""


class ConditionalMaterializationError(PreparationError):
    """Raised when a conditional numeric rule cannot be applied safely."""


class SplitPolicyError(PreparationError):
    """Raised when a classification split policy is invalid."""


class PartitionValidationError(PreparationError):
    """Raised when generated partitions violate isolation or coverage rules."""


class ArtifactConflictError(PreparationError):
    """Raised when an existing artifact is semantically divergent."""


class HandoffValidationError(PreparationError):
    """Raised when a persisted preparation handoff is invalid."""


@dataclass(frozen=True, slots=True)
class SourceIdentityReport:
    """Validated identity evidence for an independently acquired source."""

    dataset_slug: str
    source_repository: str
    source_dataset_id: int
    source_path: str
    source_sha256: str
    row_count: int
    column_count: int
    column_order: tuple[str, ...]
    target_column: str
    target_classes: tuple[Any, ...]
    feature_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    problem_type: str
    checks: tuple[tuple[str, bool], ...]

    @property
    def is_valid(self) -> bool:
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_slug": self.dataset_slug,
            "source_repository": self.source_repository,
            "source_dataset_id": self.source_dataset_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_order": list(self.column_order),
            "target_column": self.target_column,
            "target_classes": list(self.target_classes),
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "identifier_columns": list(self.identifier_columns),
            "identifier_count": len(self.identifier_columns),
            "problem_type": self.problem_type,
            "checks": dict(self.checks),
            "valid": self.is_valid,
        }


def _copy_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.copy(deep=True)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _tuple_mapping(mapping: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), copy.deepcopy(value)) for key, value in mapping.items())


def _mapping_from_tuple(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in items}


@dataclass(frozen=True, slots=True)
class ConditionalNumericRule:
    """Declare one deterministic conditional numeric materialization rule."""

    column: str
    condition_column: str
    condition_value: Any
    blank_replacement: float | int
    strip_strings: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""
        return {
            "column": self.column,
            "condition_column": self.condition_column,
            "condition_value": copy.deepcopy(self.condition_value),
            "blank_replacement": self.blank_replacement,
            "strip_strings": self.strip_strings,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Immutable summary of a dataset-contract validation."""

    stage: str
    row_count: int
    column_count: int
    column_order: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    target_counts: tuple[tuple[str, int], ...]
    observed_categories: tuple[tuple[str, tuple[Any, ...]], ...]
    checks: tuple[tuple[str, bool], ...]

    @property
    def is_valid(self) -> bool:
        """Return whether every recorded check passed."""
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""
        return {
            "stage": self.stage,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_order": list(self.column_order),
            "dtypes": dict(self.dtypes),
            "target_counts": dict(self.target_counts),
            "observed_categories": {
                column: list(values)
                for column, values in self.observed_categories
            },
            "checks": dict(self.checks),
            "valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Prepared DataFrame plus deterministic materialization evidence."""

    _dataframe: pd.DataFrame
    materialized_counts: tuple[tuple[str, int], ...]
    invalid_conversion_counts: tuple[tuple[str, int], ...]
    rules: tuple[ConditionalNumericRule, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_dataframe", _copy_frame(self._dataframe))

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the prepared DataFrame."""
        return _copy_frame(self._dataframe)

    def materialization_count(self, column: str) -> int:
        """Return the number of materialized values for one column."""
        return dict(self.materialized_counts).get(column, 0)

    def as_dict(self) -> dict[str, Any]:
        """Return materialization evidence without exposing mutable frames."""
        return {
            "materialized_counts": dict(self.materialized_counts),
            "invalid_conversion_counts": dict(self.invalid_conversion_counts),
            "rules": [rule.as_dict() for rule in self.rules],
        }


@dataclass(frozen=True, slots=True)
class DatasetRoles:
    """Defensive lineage, predictor, and target projections."""

    _lineage: pd.DataFrame
    _features: pd.DataFrame
    _target: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lineage", _copy_frame(self._lineage))
        object.__setattr__(self, "_features", _copy_frame(self._features))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def lineage(self) -> pd.DataFrame:
        return _copy_frame(self._lineage)

    @property
    def features(self) -> pd.DataFrame:
        return _copy_frame(self._features)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)


@dataclass(frozen=True, slots=True)
class ClassificationSplitPolicy:
    """Declarative split policy for an educational classification snapshot."""

    evaluation_mode: str
    purpose: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    stratify_by: str
    random_seed: int
    shuffle: bool
    educational_justification: str
    operational_validity: str = "unconfirmed"
    temporal_contract_status: str = "unresolved"
    feature_inference_availability: str = "unconfirmed"

    @property
    def second_stage_seed(self) -> int:
        """Return a deterministic distinct seed for the temporary split."""
        return self.random_seed + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_mode": self.evaluation_mode,
            "purpose": self.purpose,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "stratify_by": self.stratify_by,
            "random_seed": self.random_seed,
            "shuffle": self.shuffle,
            "educational_justification": self.educational_justification,
            "operational_validity": self.operational_validity,
            "temporal_contract_status": self.temporal_contract_status,
            "feature_inference_availability": self.feature_inference_availability,
            "stage_seeds": {
                "train_vs_temporary": self.random_seed,
                "validation_vs_test": self.second_stage_seed,
            },
        }


@dataclass(frozen=True, slots=True)
class ContinuousRegressionSplitPolicy:
    """Declarative non-stratified policy for a continuous-regression snapshot."""

    evaluation_mode: str
    purpose: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    random_seed: int
    shuffle: bool
    stratify_by: None = None
    educational_justification: str = ""
    operational_validity: str = "unconfirmed"
    temporal_contract_status: str = "resolved_static_snapshot"
    feature_inference_availability: str = "unconfirmed"

    @property
    def second_stage_seed(self) -> int:
        return self.random_seed + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_type": "continuous_regression",
            "evaluation_mode": self.evaluation_mode,
            "purpose": self.purpose,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "stratify_by": None,
            "random_seed": self.random_seed,
            "shuffle": self.shuffle,
            "educational_justification": self.educational_justification,
            "operational_validity": self.operational_validity,
            "temporal_contract_status": self.temporal_contract_status,
            "feature_inference_availability": self.feature_inference_availability,
            "stage_seeds": {
                "train_vs_temporary": self.random_seed,
                "validation_vs_test": self.second_stage_seed,
            },
        }


@dataclass(frozen=True, slots=True)
class DatasetPartitions:
    """Defensive train, validation, and test DataFrame partitions."""

    _train: pd.DataFrame
    _validation: pd.DataFrame
    _test: pd.DataFrame
    split_method: str
    rounding_method: str
    _membership: tuple[tuple[str, tuple[str, ...]], ...] = ()
    membership_kind: str = "unspecified"
    membership_semantics: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_train", _copy_frame(self._train))
        object.__setattr__(self, "_validation", _copy_frame(self._validation))
        object.__setattr__(self, "_test", _copy_frame(self._test))
        object.__setattr__(
            self,
            "_membership",
            tuple((str(name), tuple(values)) for name, values in self._membership),
        )

    @property
    def train(self) -> pd.DataFrame:
        return _copy_frame(self._train)

    @property
    def validation(self) -> pd.DataFrame:
        return _copy_frame(self._validation)

    @property
    def test(self) -> pd.DataFrame:
        return _copy_frame(self._test)

    def as_mapping(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

    def membership_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return persisted technical or source-identifier membership."""
        return {name: tuple(values) for name, values in self._membership}


@dataclass(frozen=True, slots=True)
class PartitionValidationReport:
    """Immutable partition-integrity evidence."""

    row_counts: tuple[tuple[str, int], ...]
    class_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    class_prevalence: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    membership: tuple[tuple[str, tuple[str, ...]], ...]
    checks: tuple[tuple[str, bool], ...]
    prevalence_tolerance: float
    membership_kind: str = "source_identifier"
    membership_semantics: str = "source identifier values"
    entity_disjointness_status: str = "validated_from_source_identifiers"

    @property
    def is_valid(self) -> bool:
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_counts": dict(self.row_counts),
            "class_counts": {
                name: dict(values) for name, values in self.class_counts
            },
            "class_prevalence": {
                name: dict(values) for name, values in self.class_prevalence
            },
            "membership": {
                name: list(values) for name, values in self.membership
            },
            "membership_kind": self.membership_kind,
            "membership_semantics": self.membership_semantics,
            "entity_disjointness_status": self.entity_disjointness_status,
            "checks": dict(self.checks),
            "prevalence_tolerance": self.prevalence_tolerance,
            "valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class RegressionPartitionValidationReport:
    """Partition integrity and descriptive continuous-target evidence."""

    row_counts: tuple[tuple[str, int], ...]
    target_diagnostics: tuple[tuple[str, Mapping[str, Any]], ...]
    membership: tuple[tuple[str, tuple[str, ...]], ...]
    checks: tuple[tuple[str, bool], ...]
    membership_kind: str
    membership_semantics: str
    entity_disjointness_status: str

    @property
    def is_valid(self) -> bool:
        return all(value for _, value in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_counts": dict(self.row_counts),
            "target_diagnostics": {
                name: _copy_mapping(values) for name, values in self.target_diagnostics
            },
            "membership": {name: list(values) for name, values in self.membership},
            "membership_kind": self.membership_kind,
            "membership_semantics": self.membership_semantics,
            "entity_disjointness_status": self.entity_disjointness_status,
            "checks": dict(self.checks),
            "valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Result of one atomic preparation-artifact transaction."""

    statuses: tuple[tuple[str, str], ...]
    sha256: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "statuses": dict(self.statuses),
            "sha256": dict(self.sha256),
        }


@dataclass(frozen=True, slots=True)
class PreparationHandoff:
    """Validated persisted data and manifests for the next notebook."""

    _prepared: pd.DataFrame
    _train: pd.DataFrame
    _validation: pd.DataFrame
    _test: pd.DataFrame
    _manifests: tuple[tuple[str, Mapping[str, Any]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_prepared", _copy_frame(self._prepared))
        object.__setattr__(self, "_train", _copy_frame(self._train))
        object.__setattr__(self, "_validation", _copy_frame(self._validation))
        object.__setattr__(self, "_test", _copy_frame(self._test))
        object.__setattr__(
            self,
            "_manifests",
            tuple((name, _copy_mapping(value)) for name, value in self._manifests),
        )

    @property
    def prepared(self) -> pd.DataFrame:
        return _copy_frame(self._prepared)

    @property
    def train(self) -> pd.DataFrame:
        return _copy_frame(self._train)

    @property
    def validation(self) -> pd.DataFrame:
        return _copy_frame(self._validation)

    @property
    def test(self) -> pd.DataFrame:
        return _copy_frame(self._test)

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        return {name: _copy_mapping(value) for name, value in self._manifests}


@dataclass(frozen=True, slots=True)
class ModelSelectionPreparationHandoff:
    """Preparation view that cannot materialize prepared or test tabular data."""

    _train: pd.DataFrame
    _validation: pd.DataFrame
    _manifests: tuple[tuple[str, Mapping[str, Any]], ...]
    _prepared_integrity_reference: tuple[tuple[str, Any], ...]
    _test_integrity_reference: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_train", _copy_frame(self._train))
        object.__setattr__(self, "_validation", _copy_frame(self._validation))
        object.__setattr__(self, "_manifests", tuple(
            (name, _copy_mapping(value)) for name, value in self._manifests
        ))

    @property
    def train(self) -> pd.DataFrame:
        return _copy_frame(self._train)

    @property
    def validation(self) -> pd.DataFrame:
        return _copy_frame(self._validation)

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        return {name: _copy_mapping(value) for name, value in self._manifests}

    @property
    def prepared_integrity_reference(self) -> dict[str, Any]:
        return _mapping_from_tuple(self._prepared_integrity_reference)

    @property
    def sealed_test_integrity_reference(self) -> dict[str, Any]:
        return _mapping_from_tuple(self._test_integrity_reference)


def fingerprint_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of the bytes stored in ``path``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path.name}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_scalar(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def fingerprint_dataframe(dataframe: pd.DataFrame) -> str:
    """Return a stable logical SHA-256 fingerprint for a DataFrame.

    The canonical payload includes column order, dtype strings, index names,
    index values, and row values. It is independent of object identity.
    """
    frame = _copy_frame(dataframe)
    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [
            None if name is None else str(name) for name in frame.index.names
        ],
        "index": [
            [_normalize_scalar(item) for item in value]
            if isinstance(value, tuple)
            else _normalize_scalar(value)
            for value in frame.index.tolist()
        ],
        "records": [
            [_normalize_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to deterministic UTF-8 CSV bytes."""
    buffer = io.StringIO(newline="")
    dataframe.to_csv(
        buffer,
        index=False,
        encoding=CSV_ENCODING,
        lineterminator=CSV_LINE_TERMINATOR,
        float_format="%.15g",
    )
    return buffer.getvalue().encode(CSV_ENCODING)


def fingerprint_dataframe_csv(dataframe: pd.DataFrame) -> str:
    """Return the SHA-256 digest of deterministic CSV bytes."""
    return hashlib.sha256(dataframe_csv_bytes(dataframe)).hexdigest()


def json_artifact_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a JSON artifact exactly as the atomic writer will persist it."""
    return (
        json.dumps(
            _copy_mapping(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def fingerprint_json_artifact(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of deterministic JSON artifact bytes."""
    return hashlib.sha256(json_artifact_bytes(payload)).hexdigest()


def validate_source_against_exploration_handoff(
    dataframe: pd.DataFrame,
    *,
    handoff: Mapping[str, Any],
    source_file: str | Path,
    project_root: str | Path,
    dataset_slug: str,
    source_repository: str,
    source_dataset_id: int,
    metadata_file: str | Path | None = None,
    variables_file: str | Path | None = None,
) -> SourceIdentityReport:
    """Fail closed when an independent acquisition differs from Notebook 01.

    The gate validates source bytes and logical source identity as separate
    claims. UCI metadata and variable roles are checked when their materialized
    files are supplied.
    """
    frame = _copy_frame(dataframe)
    root = Path(project_root).expanduser().resolve()
    source_path = Path(source_file).expanduser().resolve()
    try:
        logical_path = source_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DatasetValidationError(
            "Source identity gate requires a source file inside project_root."
        ) from exc

    source_contract = handoff.get("source")
    prediction_contract = handoff.get("prediction_contract")
    feature_contract = handoff.get("feature_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (source_contract, prediction_contract, feature_contract)
    ):
        raise DatasetValidationError(
            "Exploration handoff source, prediction, or feature contract is invalid."
        )

    expected_slug = str(handoff.get("dataset_slug", ""))
    expected_repository = str(source_contract.get("repository", ""))
    expected_dataset_id = source_contract.get("dataset_id")
    expected_path = str(source_contract.get("path", ""))
    expected_sha = str(source_contract.get("sha256", ""))
    expected_rows = source_contract.get("row_count")
    expected_columns = source_contract.get("column_count")
    expected_order = tuple(str(value) for value in source_contract.get("column_order", ()))
    target_column = str(prediction_contract.get("target_column", ""))
    target_classes = tuple(prediction_contract.get("target_classes", ()))
    problem_type = str(prediction_contract.get("problem_type", ""))
    feature_columns = tuple(
        str(value) for value in feature_contract.get("feature_columns", ())
    )
    identifier_columns = tuple(
        str(value) for value in feature_contract.get("identifier_columns", ())
    )

    comparisons = {
        "dataset_slug": (dataset_slug, expected_slug),
        "source_repository": (source_repository, expected_repository),
        "source_dataset_id": (source_dataset_id, expected_dataset_id),
        "source_path": (logical_path, expected_path),
        "source_sha256": (fingerprint_file(source_path), expected_sha),
        "row_count": (len(frame), expected_rows),
        "column_count": (len(frame.columns), expected_columns),
        "column_order": (tuple(str(value) for value in frame.columns), expected_order),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise DatasetValidationError(
                f"Source identity mismatch for {label}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    if target_column not in frame.columns:
        raise DatasetValidationError(
            f"Source identity mismatch: target column {target_column!r} is absent."
        )
    observed_classes = tuple(pd.unique(frame[target_column]).tolist())
    if problem_type == "multiclass_classification":
        if set(observed_classes) != set(target_classes):
            raise DatasetValidationError(
                "Source identity mismatch for target classes: "
                f"observed={observed_classes!r}, expected={target_classes!r}."
            )
    elif problem_type == "continuous_regression":
        if target_classes:
            raise DatasetValidationError(
                "Continuous-regression handoff must declare empty target_classes."
            )
        if prediction_contract.get("positive_class") is not None:
            raise DatasetValidationError(
                "Continuous-regression handoff must not declare a positive class."
            )
        if prediction_contract.get("class_semantics") is not None:
            raise DatasetValidationError(
                "Continuous-regression handoff must not declare class semantics."
            )
        target_semantics = prediction_contract.get("target_semantics")
        if not isinstance(target_semantics, str) or not target_semantics.strip():
            raise DatasetValidationError("Continuous target semantics are inconsistent.")
        target_unit = prediction_contract.get("target_unit")
        if target_unit is not None and (
            not isinstance(target_unit, str) or not target_unit.strip()
        ):
            raise DatasetValidationError("Continuous target unit is inconsistent.")
        numeric_target = pd.to_numeric(frame[target_column], errors="coerce")
        if not pandas_types.is_numeric_dtype(frame[target_column]):
            raise DatasetValidationError("Continuous target must have a numeric dtype.")
        if numeric_target.isna().any() or not numeric_target.map(math.isfinite).all():
            raise DatasetValidationError("Continuous target must be complete and finite.")
    else:
        raise DatasetValidationError(
            f"Unexpected exploration problem type: {problem_type!r}."
        )
    if len(feature_columns) != int(feature_contract.get("baseline_feature_count", -1)):
        raise DatasetValidationError("Exploration handoff feature count is inconsistent.")
    if any(column not in frame.columns for column in (*feature_columns, *identifier_columns)):
        raise DatasetValidationError(
            "Source identity mismatch: declared feature or identifier columns are absent."
        )
    if set((*identifier_columns, *feature_columns, target_column)) != set(frame.columns):
        raise DatasetValidationError(
            "Source identity mismatch: feature, identifier, and target roles do not "
            "cover the acquired schema."
        )
    if problem_type == "multiclass_classification" and prediction_contract.get("positive_class") is not None:
        raise DatasetValidationError(
            "Multiclass exploration handoff must not declare a positive class."
        )

    checks: list[tuple[str, bool]] = [
        ("dataset_slug_matches", True),
        ("source_repository_matches", True),
        ("uci_dataset_id_matches", True),
        ("logical_source_path_matches", True),
        ("source_sha256_matches", True),
        ("source_shape_matches", True),
        ("column_order_matches", True),
        ("target_contract_matches", True),
        ("feature_order_matches", True),
        ("identifier_contract_matches", True),
        ("problem_type_matches", True),
    ]

    if metadata_file is not None:
        metadata_path = Path(metadata_file).expanduser().resolve()
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetValidationError("UCI metadata.json is missing or invalid.") from exc
        if not isinstance(metadata, Mapping):
            raise DatasetValidationError("UCI metadata.json must contain an object.")
        metadata_id = metadata.get("uci_id", metadata.get("id"))
        if metadata_id is None or int(metadata_id) != int(source_dataset_id):
            raise DatasetValidationError(
                "Source identity mismatch for the UCI metadata dataset ID."
            )
        checks.append(("uci_metadata_dataset_id_matches", True))

    if variables_file is not None:
        variables_path = Path(variables_file).expanduser().resolve()
        try:
            variables = pd.read_csv(variables_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DatasetValidationError("UCI variables.csv is missing or invalid.") from exc
        normalized_columns = {str(column).strip().lower(): column for column in variables.columns}
        if "name" not in normalized_columns or "role" not in normalized_columns:
            raise DatasetValidationError(
                "UCI variables.csv must contain name and role columns."
            )
        names = variables[normalized_columns["name"]].astype(str).str.strip()
        roles = variables[normalized_columns["role"]].astype(str).str.strip().str.lower()
        variable_features = tuple(names.loc[roles.eq("feature")].tolist())
        variable_targets = tuple(names.loc[roles.eq("target")].tolist())
        if variable_features != feature_columns:
            raise DatasetValidationError(
                "Source identity mismatch for UCI feature-variable order."
            )
        if variable_targets != (target_column,):
            raise DatasetValidationError(
                "Source identity mismatch for the UCI target-variable contract."
            )
        if problem_type == "continuous_regression":
            type_column = normalized_columns.get("type")
            if type_column is None:
                raise DatasetValidationError(
                    "UCI variables.csv must declare source variable types."
                )
            decisions = handoff.get("preparation_contract", {}).get("decisions", [])
            for column in feature_columns:
                source_rows = variables.loc[names.eq(column)]
                if len(source_rows) != 1:
                    raise DatasetValidationError(
                        f"Source type provenance is missing for {column!r}."
                    )
                declared_type = str(source_rows.iloc[0][type_column]).strip().casefold()
                series = frame[column]
                requires_resolution = (
                    declared_type == "integer"
                    and pandas_types.is_numeric_dtype(series.dtype)
                    and bool(((pd.to_numeric(series) % 1).abs() > 0).any())
                )
                if not requires_resolution:
                    continue
                matching = []
                for item in decisions:
                    if not isinstance(item, Mapping) or item.get("Status") != "Approved":
                        continue
                    affected = item.get("Affected fields", [])
                    operation = str(item.get("Operation", "")).casefold()
                    if (
                        isinstance(affected, list)
                        and column in affected
                        and "preserve" in operation
                        and "round" in operation
                        and "truncate" in operation
                        and ("coerce" in operation or "integer" in operation)
                    ):
                        matching.append(item)
                if len(matching) != 1:
                    raise DatasetValidationError(
                        f"Required source-type resolution is absent or inconsistent for {column!r}."
                    )
        checks.append(("uci_variable_roles_match", True))

    return SourceIdentityReport(
        dataset_slug=dataset_slug,
        source_repository=source_repository,
        source_dataset_id=int(source_dataset_id),
        source_path=logical_path,
        source_sha256=fingerprint_file(source_path),
        row_count=len(frame),
        column_count=len(frame.columns),
        column_order=tuple(str(value) for value in frame.columns),
        target_column=target_column,
        target_classes=target_classes,
        feature_columns=feature_columns,
        identifier_columns=identifier_columns,
        problem_type=problem_type,
        checks=tuple(checks),
    )


def _is_string_series(series: pd.Series) -> bool:
    if pandas_types.is_string_dtype(series.dtype):
        return True
    non_missing = series.dropna()
    return non_missing.map(lambda value: isinstance(value, str)).all()


def _matches_expected_type(
    series: pd.Series,
    expectation: str,
    *,
    allow_numeric_text: bool,
) -> bool:
    normalized = expectation.strip().lower()
    if normalized == "string":
        return _is_string_series(series)
    if normalized == "integer":
        return pandas_types.is_integer_dtype(series.dtype)
    if normalized == "numeric":
        if pandas_types.is_numeric_dtype(series.dtype):
            return True
        if not allow_numeric_text:
            return False
        normalized_values = series.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        non_blank = normalized_values.loc[
            ~normalized_values.map(
                lambda value: isinstance(value, str) and value == ""
            )
        ]
        converted = pd.to_numeric(non_blank, errors="coerce")
        return converted.notna().all()
    if normalized == "boolean":
        return pandas_types.is_bool_dtype(series.dtype)
    raise ValueError(f"Unsupported expected type: {expectation!r}")


def _validate_contract_configuration(
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    columns = tuple(column_order)
    identifiers = tuple(identifier_columns)
    features = tuple(feature_columns)
    if not columns:
        raise ValueError("column_order cannot be empty.")
    if not target_column:
        raise ValueError("target_column cannot be empty.")
    if len(set(columns)) != len(columns):
        raise ValueError("column_order contains duplicate names.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("identifier_columns contains duplicate names.")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns contains duplicate names.")
    overlap = (set(identifiers) & set(features)) | ({target_column} & set(features))
    if overlap:
        raise ValueError(f"Dataset roles overlap: {sorted(overlap)}")
    declared = set(identifiers) | set(features) | {target_column}
    if declared != set(columns):
        missing = sorted(set(columns) - declared)
        extra = sorted(declared - set(columns))
        raise ValueError(
            "Role declarations must cover column_order exactly; "
            f"unassigned={missing}, outside_order={extra}."
        )


def validate_raw_dataset(
    dataframe: pd.DataFrame,
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    expected_types: Mapping[str, str],
    numeric_text_columns: Sequence[str] = (),
    allow_unexpected_columns: bool = False,
    require_all_expected_categories: bool = True,
    problem_type: str = "classification",
) -> DatasetValidationReport:
    """Validate a raw dataset without mutating it."""
    _validate_contract_configuration(
        column_order=column_order,
        identifier_columns=identifier_columns,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    frame = _copy_frame(dataframe)
    expected_order = tuple(column_order)
    observed_order = tuple(str(column) for column in frame.columns)
    missing = [column for column in expected_order if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in expected_order]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")
    if unexpected and not allow_unexpected_columns:
        raise DatasetValidationError(f"Unexpected columns: {unexpected}")
    if not allow_unexpected_columns and observed_order != expected_order:
        raise DatasetValidationError(
            "Column order does not match the declared raw schema."
        )

    undeclared_types = [column for column in expected_order if column not in expected_types]
    missing_type_columns = [column for column in expected_types if column not in frame.columns]
    if undeclared_types or missing_type_columns:
        raise DatasetValidationError(
            "Expected type declarations must match the schema exactly; "
            f"undeclared={undeclared_types}, missing_columns={missing_type_columns}."
        )

    for column in identifier_columns:
        series = frame[column]
        if series.isna().any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains missing values."
            )
        blank_mask = series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if blank_mask.any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains blank values."
            )
        if series.duplicated(keep=False).any():
            raise DatasetValidationError(
                f"Identifier column '{column}' contains duplicate values."
            )

    expected_target = tuple(target_classes)
    target = frame[target_column]
    if target.isna().any():
        raise DatasetValidationError(
            f"Target column '{target_column}' contains missing values."
        )
    observed_target = tuple(pd.unique(target).tolist())
    if problem_type == "continuous_regression":
        if expected_target:
            raise ValueError("Continuous regression cannot declare target classes.")
        if not pandas_types.is_numeric_dtype(target):
            raise DatasetValidationError("Continuous target must have a numeric dtype.")
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.isna().any() or not numeric_target.map(math.isfinite).all():
            raise DatasetValidationError("Continuous target must be complete and finite.")
    else:
        if not expected_target:
            raise ValueError("target_classes cannot be empty.")
        unexpected_target = [value for value in observed_target if value not in expected_target]
        absent_target = [value for value in expected_target if value not in observed_target]
        if unexpected_target:
            raise DatasetValidationError(
                f"Target column '{target_column}' contains unexpected classes: "
                f"{unexpected_target}."
            )
        if absent_target:
            raise DatasetValidationError(
                f"Target column '{target_column}' is missing expected classes: "
                f"{absent_target}."
            )

    observed_categories: list[tuple[str, tuple[Any, ...]]] = []
    for column, expected_values in categorical_expected_values.items():
        if column not in frame.columns:
            raise DatasetValidationError(
                f"Categorical contract references missing column '{column}'."
            )
        series = frame[column]
        if series.isna().any():
            raise DatasetValidationError(
                f"Categorical column '{column}' contains missing values."
            )
        expected = tuple(expected_values)
        observed = tuple(pd.unique(series).tolist())
        unexpected_values = [value for value in observed if value not in expected]
        absent_values = [value for value in expected if value not in observed]
        if unexpected_values:
            raise DatasetValidationError(
                f"Categorical column '{column}' contains unexpected values: "
                f"{unexpected_values}."
            )
        if require_all_expected_categories and absent_values:
            raise DatasetValidationError(
                f"Categorical column '{column}' is missing expected values: "
                f"{absent_values}."
            )
        observed_categories.append((column, observed))

    numeric_text = set(numeric_text_columns)
    for column, expectation in expected_types.items():
        if not _matches_expected_type(
            frame[column],
            expectation,
            allow_numeric_text=column in numeric_text,
        ):
            raise DatasetValidationError(
                f"Column '{column}' does not match expected type "
                f"'{expectation}'; observed dtype is '{frame[column].dtype}'."
            )

    target_counts = tuple(
        (str(value), int((target == value).sum())) for value in expected_target
    )
    identifier_checks = (
        (("identifiers_complete", True), ("identifiers_unique", True))
        if identifier_columns
        else (("source_identifier_absence_respected", True),)
    )
    checks = (
        ("required_columns_present", True),
        ("column_order_valid", observed_order == expected_order or allow_unexpected_columns),
        *identifier_checks,
        ("target_valid", True),
        ("categories_valid", True),
        ("types_valid", True),
    )
    return DatasetValidationReport(
        stage="raw",
        row_count=len(frame),
        column_count=len(frame.columns),
        column_order=observed_order,
        dtypes=tuple((str(column), str(frame[column].dtype)) for column in frame.columns),
        target_counts=target_counts,
        observed_categories=tuple(observed_categories),
        checks=checks,
    )


def materialize_conditional_numeric_values(
    dataframe: pd.DataFrame,
    rule: ConditionalNumericRule,
) -> tuple[pd.DataFrame, int, int]:
    """Apply one conditional blank-to-number rule to a defensive copy."""
    frame = _copy_frame(dataframe)
    for column in (rule.column, rule.condition_column):
        if column not in frame.columns:
            raise ConditionalMaterializationError(
                f"Conditional materialization references missing column '{column}'."
            )

    source = frame[rule.column]
    if source.isna().any():
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains null values; only explicit blank "
            "strings may be materialized by this rule."
        )

    normalized = source.map(
        lambda value: value.strip()
        if rule.strip_strings and isinstance(value, str)
        else value
    )
    blank_mask = normalized.map(
        lambda value: isinstance(value, str) and value == ""
    )
    condition_mask = frame[rule.condition_column].eq(rule.condition_value)
    invalid_blank_mask = blank_mask & ~condition_mask
    if invalid_blank_mask.any():
        count = int(invalid_blank_mask.sum())
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains {count} blank value(s) where "
            f"'{rule.condition_column}' != {rule.condition_value!r}."
        )

    materialization_mask = blank_mask & condition_mask

    # Convert textual values before assigning the numeric replacement. Pandas 3
    # infers text columns as a strict string dtype, which rejects direct
    # assignment of numbers into the string-backed Series. Replacing authorized
    # blanks with a missing sentinel preserves the textual validation boundary;
    # the numeric replacement is applied only after conversion.
    numeric_candidate = normalized.mask(materialization_mask, pd.NA)
    converted = pd.to_numeric(numeric_candidate, errors="coerce")

    invalid_mask = converted.isna() & ~materialization_mask
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        samples = tuple(
            str(value)
            for value in normalized.loc[invalid_mask].head(5).tolist()
        )
        raise ConditionalMaterializationError(
            f"Column '{rule.column}' contains {invalid_count} non-convertible "
            f"value(s); samples={samples}."
        )

    converted = converted.astype("float64")
    converted.loc[materialization_mask] = float(rule.blank_replacement)
    frame[rule.column] = converted
    return frame, int(materialization_mask.sum()), invalid_count


def prepare_tabular_dataset(
    dataframe: pd.DataFrame,
    *,
    conditional_numeric_rules: Sequence[ConditionalNumericRule] = (),
) -> PreparedDataset:
    """Create a defensive prepared projection using only explicit rules."""
    prepared = _copy_frame(dataframe)
    original_columns = tuple(prepared.columns)
    original_index = prepared.index.copy(deep=True)
    materialized_counts: list[tuple[str, int]] = []
    invalid_counts: list[tuple[str, int]] = []
    rules = tuple(copy.deepcopy(tuple(conditional_numeric_rules)))

    for rule in rules:
        prepared, materialized_count, invalid_count = (
            materialize_conditional_numeric_values(prepared, rule)
        )
        materialized_counts.append((rule.column, materialized_count))
        invalid_counts.append((rule.column, invalid_count))

    if tuple(prepared.columns) != original_columns:
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the column order."
        )
    if not prepared.index.equals(original_index):
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the DataFrame index."
        )
    if len(prepared) != len(dataframe):
        raise ConditionalMaterializationError(
            "Preparation unexpectedly changed the row count."
        )

    return PreparedDataset(
        _dataframe=prepared,
        materialized_counts=tuple(materialized_counts),
        invalid_conversion_counts=tuple(invalid_counts),
        rules=rules,
    )


def validate_prepared_dataset(
    raw_dataframe: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    *,
    column_order: Sequence[str],
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    expected_types: Mapping[str, str],
    authorized_changed_columns: Sequence[str],
    expected_row_count: int | None = None,
    expected_materialized_counts: Mapping[str, int] | None = None,
    observed_materialized_counts: Mapping[str, int] | None = None,
    problem_type: str = "classification",
) -> DatasetValidationReport:
    """Validate prepared data and prove preservation of the raw projection."""
    raw = _copy_frame(raw_dataframe)
    prepared = _copy_frame(prepared_dataframe)
    if expected_row_count is not None and len(prepared) != expected_row_count:
        raise DatasetValidationError(
            f"Prepared row count is {len(prepared)}, expected {expected_row_count}."
        )
    if raw.shape != prepared.shape:
        raise DatasetValidationError(
            f"Prepared shape {prepared.shape} differs from raw shape {raw.shape}."
        )
    if tuple(raw.columns) != tuple(prepared.columns):
        raise DatasetValidationError("Prepared column order differs from raw data.")
    if not raw.index.equals(prepared.index):
        raise DatasetValidationError("Prepared index differs from raw data.")

    authorized = set(authorized_changed_columns)
    for column in raw.columns:
        if column in authorized:
            continue
        try:
            pd.testing.assert_series_equal(
                raw[column], prepared[column], check_dtype=True, check_names=True
            )
        except AssertionError as exc:
            raise DatasetValidationError(
                f"Unauthorized values changed in column '{column}'."
            ) from exc

    if expected_materialized_counts is not None:
        observed = dict(observed_materialized_counts or {})
        for column, expected_count in expected_materialized_counts.items():
            if observed.get(column) != expected_count:
                raise DatasetValidationError(
                    f"Materialized count for '{column}' is {observed.get(column)}, "
                    f"expected {expected_count}."
                )

    report = validate_raw_dataset(
        prepared,
        column_order=column_order,
        identifier_columns=identifier_columns,
        feature_columns=feature_columns,
        target_column=target_column,
        target_classes=target_classes,
        categorical_expected_values=categorical_expected_values,
        expected_types=expected_types,
        numeric_text_columns=(),
        allow_unexpected_columns=False,
        require_all_expected_categories=True,
        problem_type=problem_type,
    )
    return DatasetValidationReport(
        stage="prepared",
        row_count=report.row_count,
        column_count=report.column_count,
        column_order=report.column_order,
        dtypes=report.dtypes,
        target_counts=report.target_counts,
        observed_categories=report.observed_categories,
        checks=report.checks
        + (
            ("raw_shape_preserved", True),
            ("raw_index_preserved", True),
            ("unauthorized_columns_unchanged", True),
            ("materialization_counts_valid", True),
        ),
    )


def separate_dataset_roles(
    dataframe: pd.DataFrame,
    *,
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> DatasetRoles:
    """Return defensive lineage, X, and y projections in declared order."""
    frame = _copy_frame(dataframe)
    required = tuple(identifier_columns) + tuple(feature_columns) + (target_column,)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetValidationError(f"Role separation is missing columns: {missing}")
    if set(identifier_columns) & set(feature_columns):
        raise DatasetValidationError("Identifier columns cannot be predictors.")
    if target_column in feature_columns:
        raise DatasetValidationError("Target column cannot be a predictor.")
    return DatasetRoles(
        _lineage=frame.loc[:, list(identifier_columns)],
        _features=frame.loc[:, list(feature_columns)],
        _target=frame.loc[:, target_column],
    )


def validate_split_policy(
    policy: ClassificationSplitPolicy,
    *,
    known_columns: Iterable[str],
) -> dict[str, bool]:
    """Validate an educational stratified-random snapshot policy."""
    if policy.evaluation_mode != "stratified_random_snapshot":
        raise SplitPolicyError(
            "evaluation_mode must be 'stratified_random_snapshot'."
        )
    if policy.purpose != "educational_benchmark":
        raise SplitPolicyError("purpose must be 'educational_benchmark'.")
    if not policy.educational_justification.strip():
        raise SplitPolicyError("An educational justification is required.")
    fractions = (
        policy.train_fraction,
        policy.validation_fraction,
        policy.test_fraction,
    )
    if any(not isinstance(value, (int, float)) for value in fractions):
        raise SplitPolicyError("Split fractions must be numeric.")
    if any(value <= 0 for value in fractions):
        raise SplitPolicyError("Split fractions must be greater than zero.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SplitPolicyError("Split fractions must sum to 1.0.")
    if policy.stratify_by not in set(known_columns):
        raise SplitPolicyError(
            f"Unknown stratification target: {policy.stratify_by!r}."
        )
    if isinstance(policy.random_seed, bool) or not isinstance(policy.random_seed, int):
        raise SplitPolicyError("random_seed must be an integer.")
    if not policy.shuffle:
        raise SplitPolicyError(
            "shuffle must be true for a stratified random snapshot."
        )
    if policy.operational_validity != "unconfirmed":
        raise SplitPolicyError(
            "operational_validity must remain 'unconfirmed' for this benchmark."
        )
    allowed_temporal_statuses = {"unresolved", "resolved_static_snapshot"}
    if policy.temporal_contract_status not in allowed_temporal_statuses:
        raise SplitPolicyError(
            "temporal_contract_status must be 'unresolved' or "
            "'resolved_static_snapshot'."
        )
    if policy.feature_inference_availability != "unconfirmed":
        raise SplitPolicyError(
            "feature_inference_availability must remain 'unconfirmed'."
        )
    temporal_check = (
        {"temporal_contract_unresolved": True}
        if policy.temporal_contract_status == "unresolved"
        else {"static_snapshot_contract_resolved": True}
    )
    return {
        "mode_valid": True,
        "purpose_valid": True,
        "fractions_valid": True,
        "stratification_target_known": True,
        "seed_valid": True,
        "shuffle_valid": True,
        "educational_boundary_valid": True,
        "operational_validity_unconfirmed": True,
        **temporal_check,
        "feature_inference_availability_unconfirmed": True,
    }


def _canonical_row_scalar(value: Any) -> Any:
    normalized = _normalize_scalar(value)
    if isinstance(normalized, bool):
        return {"type": "bool", "value": normalized}
    if isinstance(normalized, float):
        if not math.isfinite(normalized):
            raise PartitionValidationError(
                "Partition membership cannot fingerprint non-finite values."
            )
        return {"type": "number", "value": format(normalized, ".15g")}
    if isinstance(normalized, int):
        return {"type": "number", "value": str(normalized)}
    if normalized is None:
        return {"type": "null", "value": None}
    return {"type": "string", "value": str(normalized)}


def _row_content_fingerprints(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    selected_columns = tuple(columns) if columns is not None else tuple(frame.columns)
    missing = [column for column in selected_columns if column not in frame.columns]
    if missing:
        raise PartitionValidationError(
            f"Row fingerprint columns are missing: {missing}."
        )
    fingerprints: list[str] = []
    for row in frame.loc[:, list(selected_columns)].itertuples(index=False, name=None):
        payload = [
            _canonical_row_scalar(value)
            for value in row
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fingerprints.append(hashlib.sha256(encoded).hexdigest())
    return tuple(fingerprints)


def _source_identifier_membership_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
    if not columns:
        raise ValueError("Source-identifier membership requires identifier columns.")
    key_frame = frame.loc[:, list(columns)].copy(deep=True)
    return tuple(
        key_frame.apply(
            lambda row: json.dumps(
                [_normalize_scalar(value) for value in row.tolist()],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            axis=1,
        ).tolist()
    )


def _technical_occurrence_membership_keys(frame: pd.DataFrame) -> tuple[str, ...]:
    """Create source-order-bound occurrence tokens without inventing an ID.

    Equal rows receive the same content hash and distinct occurrence ordinals.
    The token is technical persistence evidence only; it is neither a predictor
    nor evidence that two equal rows represent the same real-world entity.
    """
    occurrences: Counter[str] = Counter()
    keys: list[str] = []
    for row_hash in _row_content_fingerprints(frame):
        occurrence = occurrences[row_hash]
        occurrences[row_hash] += 1
        keys.append(f"row-occurrence-v1:{row_hash}:{occurrence:08d}")
    return tuple(keys)


def _membership_contract(
    frame: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> tuple[tuple[str, ...], str, str]:
    if identifier_columns:
        return (
            _source_identifier_membership_keys(frame, identifier_columns),
            "source_identifier",
            "declared source identifier tuple",
        )
    return (
        _technical_occurrence_membership_keys(frame),
        "technical_row_occurrence",
        (
            "source-order-bound full-row hash plus occurrence ordinal; technical "
            "partition evidence only, not source or entity identity"
        ),
    )


def _row_hash_from_occurrence_key(value: str) -> str:
    prefix = "row-occurrence-v1:"
    if not value.startswith(prefix):
        raise PartitionValidationError(
            "Invalid technical row-occurrence membership token."
        )
    remainder = value[len(prefix):]
    row_hash, separator, ordinal = remainder.rpartition(":")
    if (
        not separator
        or len(row_hash) != 64
        or any(character not in "0123456789abcdef" for character in row_hash)
        or len(ordinal) != 8
        or not ordinal.isdigit()
    ):
        raise PartitionValidationError(
            "Invalid technical row-occurrence membership token."
        )
    return row_hash


def split_classification_dataset(
    dataframe: pd.DataFrame,
    *,
    policy: ClassificationSplitPolicy,
    identifier_columns: Sequence[str],
    target_classes: Sequence[Any],
) -> DatasetPartitions:
    """Create deterministic stratified partitions with stable membership.

    Rows are ordered by declared source identifiers when present. Without an
    identifier, a full-row hash plus source-order occurrence ordinal provides
    technical, non-semantic membership evidence. Returned rows are restored to
    source position within each partition.
    """
    frame = _copy_frame(dataframe)
    validate_split_policy(policy, known_columns=frame.columns)
    for column in identifier_columns:
        if column not in frame.columns:
            raise SplitPolicyError(f"Identifier column not found: {column}")
    if frame.empty:
        raise SplitPolicyError("Cannot split an empty dataset.")
    target = frame[policy.stratify_by]
    absent = [value for value in target_classes if value not in set(target)]
    if absent:
        raise SplitPolicyError(
            f"Cannot stratify because expected target classes are absent: {absent}."
        )

    working = frame.copy(deep=True)
    working["__source_position__"] = range(len(working))
    source_membership, membership_kind, membership_semantics = _membership_contract(
        frame, identifier_columns
    )
    working["__membership_key__"] = list(source_membership)
    if working["__membership_key__"].duplicated().any():
        raise SplitPolicyError("Membership keys must be unique before splitting.")
    canonical = working.sort_values(
        "__membership_key__", kind="stable"
    ).reset_index(drop=True)
    canonical_positions = list(range(len(canonical)))
    temporary_fraction = policy.validation_fraction + policy.test_fraction

    train_positions, temporary_positions = train_test_split(
        canonical_positions,
        test_size=temporary_fraction,
        random_state=policy.random_seed,
        shuffle=policy.shuffle,
        stratify=canonical[policy.stratify_by],
    )
    temporary = canonical.iloc[temporary_positions]
    relative_test_fraction = policy.test_fraction / temporary_fraction
    validation_positions, test_positions = train_test_split(
        temporary_positions,
        test_size=relative_test_fraction,
        random_state=policy.second_stage_seed,
        shuffle=policy.shuffle,
        stratify=temporary[policy.stratify_by],
    )

    def project(positions: Sequence[int]) -> tuple[pd.DataFrame, tuple[str, ...]]:
        source_positions = sorted(
            int(value)
            for value in canonical.iloc[list(positions)]["__source_position__"].tolist()
        )
        projected = frame.iloc[source_positions].copy(deep=True)
        projected_membership = tuple(source_membership[position] for position in source_positions)
        return projected, projected_membership

    train, train_membership = project(train_positions)
    validation, validation_membership = project(validation_positions)
    test, test_membership = project(test_positions)

    return DatasetPartitions(
        _train=train,
        _validation=validation,
        _test=test,
        split_method=(
            "two_stage_sklearn_train_test_split_with_stratification_and_"
            + (
                "identifier_sorted_membership"
                if identifier_columns
                else "technical_row_occurrence_sorted_membership"
            )
        ),
        rounding_method=(
            "scikit-learn float test_size semantics: each held-out size is "
            "rounded up with ceil; the remainder is assigned to the first set"
        ),
        _membership=(
            ("train", train_membership),
            ("validation", validation_membership),
            ("test", test_membership),
        ),
        membership_kind=membership_kind,
        membership_semantics=membership_semantics,
    )


def validate_dataset_partitions(
    source_dataframe: pd.DataFrame,
    partitions: DatasetPartitions,
    *,
    identifier_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    prevalence_tolerance: float = 0.02,
) -> PartitionValidationReport:
    """Validate class preservation, isolation, coverage, and stable order."""
    if prevalence_tolerance < 0:
        raise ValueError("prevalence_tolerance cannot be negative.")
    source = _copy_frame(source_dataframe)
    partition_map = partitions.as_mapping()
    source_membership, inferred_kind, inferred_semantics = _membership_contract(
        source, identifier_columns
    )
    membership_kind = (
        partitions.membership_kind
        if partitions.membership_kind != "unspecified"
        else inferred_kind
    )
    membership_semantics = (
        partitions.membership_semantics
        if partitions.membership_semantics != "unspecified"
        else inferred_semantics
    )
    if membership_kind != inferred_kind:
        raise PartitionValidationError(
            "Partition membership kind conflicts with the source identifier contract."
        )
    source_membership_set = set(source_membership)
    if len(source_membership_set) != len(source):
        raise PartitionValidationError("Source membership identifiers are not unique.")

    source_counts = source[target_column].value_counts(dropna=False)
    source_prevalence = source[target_column].value_counts(normalize=True, dropna=False)
    row_counts: list[tuple[str, int]] = []
    class_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    class_prevalence: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    membership: list[tuple[str, tuple[str, ...]]] = []
    membership_sets: dict[str, set[str]] = {}
    source_position = {value: position for position, value in enumerate(source_membership)}
    persisted_membership = partitions.membership_mapping()
    if not persisted_membership and not identifier_columns:
        raise PartitionValidationError(
            "Technical row-occurrence membership evidence is required when the "
            "source has no identifier columns."
        )

    for name in ("train", "validation", "test"):
        frame = partition_map[name]
        if frame.empty:
            raise PartitionValidationError(f"Partition '{name}' is empty.")
        if tuple(frame.columns) != tuple(source.columns):
            raise PartitionValidationError(
                f"Partition '{name}' does not preserve source columns."
            )
        observed = tuple(pd.unique(frame[target_column]).tolist())
        absent = [value for value in target_classes if value not in observed]
        if absent:
            raise PartitionValidationError(
                f"Partition '{name}' is missing expected classes: {absent}."
            )
        if persisted_membership:
            values = tuple(persisted_membership.get(name, ()))
        else:
            values = _source_identifier_membership_keys(frame, identifier_columns)
        if len(values) != len(frame):
            raise PartitionValidationError(
                f"Partition '{name}' membership count differs from its row count."
            )
        values_set = set(values)
        if len(values_set) != len(frame):
            raise PartitionValidationError(
                f"Partition '{name}' contains duplicate membership identifiers."
            )
        unknown_values = [value for value in values if value not in source_position]
        if unknown_values:
            raise PartitionValidationError(
                f"Partition '{name}' contains membership outside the source."
            )
        if identifier_columns:
            observed_identifier_membership = _source_identifier_membership_keys(
                frame, identifier_columns
            )
            if observed_identifier_membership != values:
                raise PartitionValidationError(
                    f"Partition '{name}' identifier membership does not match its rows."
                )
        else:
            expected_hashes = Counter(_row_hash_from_occurrence_key(value) for value in values)
            observed_hashes = Counter(_row_content_fingerprints(frame))
            if observed_hashes != expected_hashes:
                raise PartitionValidationError(
                    f"Partition '{name}' row multiplicities do not match technical "
                    "membership evidence."
                )
        positions = [source_position[value] for value in values]
        if positions != sorted(positions):
            raise PartitionValidationError(
                f"Partition '{name}' is not in stable source-row order."
            )
        counts = tuple(
            (str(value), int((frame[target_column] == value).sum()))
            for value in target_classes
        )
        prevalences = tuple(
            (
                str(value),
                float((frame[target_column] == value).mean()),
            )
            for value in target_classes
        )
        for value in target_classes:
            observed_prevalence = float((frame[target_column] == value).mean())
            expected_prevalence = float(source_prevalence.get(value, 0.0))
            if abs(observed_prevalence - expected_prevalence) > prevalence_tolerance:
                raise PartitionValidationError(
                    f"Partition '{name}' prevalence for {value!r} differs from "
                    f"source by more than {prevalence_tolerance:.4f}."
                )
        row_counts.append((name, len(frame)))
        class_counts.append((name, counts))
        class_prevalence.append((name, prevalences))
        membership.append((name, values))
        membership_sets[name] = values_set

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}_{right}": membership_sets[left] & membership_sets[right]
        for left, right in pairs
    }
    if any(overlaps.values()):
        raise PartitionValidationError(
            "Partition membership overlaps were detected."
        )
    union = set().union(*membership_sets.values())
    if union != source_membership_set:
        missing = len(source_membership_set - union)
        extra = len(union - source_membership_set)
        raise PartitionValidationError(
            f"Partition coverage mismatch: missing={missing}, extra={extra}."
        )
    if sum(dict(row_counts).values()) != len(source):
        raise PartitionValidationError("Partition row counts do not cover the source.")

    membership_checks = (
        (
            ("source_identifier_membership_disjoint", True),
            ("entity_disjointness_validated", True),
        )
        if identifier_columns
        else (
            ("technical_occurrence_membership_disjoint", True),
            ("row_multiplicity_preserved", True),
            ("entity_disjointness_not_claimed_without_source_identifiers", True),
        )
    )
    checks = (
        ("all_partitions_present", True),
        ("all_classes_present", True),
        ("class_prevalence_within_tolerance", True),
        *membership_checks,
        ("partition_membership_isolated", True),
        ("full_coverage", True),
        ("row_count_preserved", True),
        ("stable_source_order", True),
        ("test_holdout_isolated", True),
    )
    return PartitionValidationReport(
        row_counts=tuple(row_counts),
        class_counts=tuple(class_counts),
        class_prevalence=tuple(class_prevalence),
        membership=tuple(membership),
        checks=checks,
        prevalence_tolerance=prevalence_tolerance,
        membership_kind=membership_kind,
        membership_semantics=membership_semantics,
        entity_disjointness_status=(
            "validated_from_source_identifiers"
            if identifier_columns
            else "not_claimed_without_source_identifiers"
        ),
    )


def validate_regression_split_policy(
    policy: ContinuousRegressionSplitPolicy,
) -> None:
    """Validate a shuffled, reproducible policy that never uses the target."""
    if policy.evaluation_mode != "shuffled_random_snapshot":
        raise SplitPolicyError("Regression evaluation_mode must be shuffled_random_snapshot.")
    if type(policy.random_seed) is not int:
        raise SplitPolicyError("random_seed must be an integer.")
    if policy.shuffle is not True or policy.stratify_by is not None:
        raise SplitPolicyError("Regression snapshot must shuffle with stratify_by=None.")
    fractions = (policy.train_fraction, policy.validation_fraction, policy.test_fraction)
    if any(not isinstance(value, (int, float)) or value <= 0 or value >= 1 for value in fractions):
        raise SplitPolicyError("Split fractions must be numeric values strictly between 0 and 1.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SplitPolicyError("Split fractions must sum to 1.0.")
    if policy.operational_validity != "unconfirmed":
        raise SplitPolicyError("Operational validity must remain unconfirmed.")


def split_continuous_regression_dataset(
    dataframe: pd.DataFrame,
    *,
    policy: ContinuousRegressionSplitPolicy,
    identifier_columns: Sequence[str] = (),
) -> DatasetPartitions:
    """Split source positions in two stages without consulting target values."""
    frame = _copy_frame(dataframe)
    validate_regression_split_policy(policy)
    if frame.empty:
        raise SplitPolicyError("Cannot split an empty dataset.")
    for column in identifier_columns:
        if column not in frame:
            raise SplitPolicyError(f"Identifier column not found: {column}")
    source_membership, membership_kind, membership_semantics = _membership_contract(
        frame, identifier_columns
    )
    positions = list(range(len(frame)))
    temporary_fraction = policy.validation_fraction + policy.test_fraction
    train_positions, temporary_positions = train_test_split(
        positions,
        test_size=temporary_fraction,
        random_state=policy.random_seed,
        shuffle=True,
        stratify=None,
    )
    relative_test_fraction = policy.test_fraction / temporary_fraction
    validation_positions, test_positions = train_test_split(
        temporary_positions,
        test_size=relative_test_fraction,
        random_state=policy.second_stage_seed,
        shuffle=True,
        stratify=None,
    )

    def project(selected: Sequence[int]) -> tuple[pd.DataFrame, tuple[str, ...]]:
        stable = sorted(int(value) for value in selected)
        return (
            frame.iloc[stable].copy(deep=True),
            tuple(source_membership[position] for position in stable),
        )

    train, train_membership = project(train_positions)
    validation, validation_membership = project(validation_positions)
    test, test_membership = project(test_positions)
    return DatasetPartitions(
        _train=train,
        _validation=validation,
        _test=test,
        split_method="two_stage_sklearn_train_test_split_shuffled_non_stratified_source_positions",
        rounding_method=(
            "scikit-learn float test_size semantics: each held-out size is rounded "
            "up with ceil; the remainder is assigned to the first set"
        ),
        _membership=(("train", train_membership), ("validation", validation_membership), ("test", test_membership)),
        membership_kind=membership_kind,
        membership_semantics=membership_semantics,
    )


def describe_continuous_target(series: pd.Series) -> dict[str, Any]:
    """Return diagnostics only; callers must not use them to choose a split."""
    if not pandas_types.is_numeric_dtype(series):
        raise PartitionValidationError("Continuous target must have a numeric dtype.")
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise PartitionValidationError("Continuous target must be complete and finite.")
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(values.count()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std(ddof=1)),
        "quantiles": {f"{int(q * 100)}%": float(value) for q, value in quantiles.items()},
        "diagnostic_only": True,
        "used_for_assignment_or_seed_selection": False,
    }


def validate_regression_partitions(
    source_dataframe: pd.DataFrame,
    partitions: DatasetPartitions,
    *,
    identifier_columns: Sequence[str],
    target_column: str,
) -> RegressionPartitionValidationReport:
    """Prove schema, occurrence coverage, isolation, and finite continuous y."""
    source = _copy_frame(source_dataframe)
    if target_column not in source:
        raise PartitionValidationError("Continuous target column is absent.")
    describe_continuous_target(source[target_column])
    source_membership, inferred_kind, inferred_semantics = _membership_contract(source, identifier_columns)
    if partitions.membership_kind != inferred_kind:
        raise PartitionValidationError("Partition membership kind conflicts with source.")
    persisted = partitions.membership_mapping()
    source_positions = {value: position for position, value in enumerate(source_membership)}
    membership_sets: dict[str, set[str]] = {}
    rows: list[tuple[str, int]] = []
    diagnostics: list[tuple[str, Mapping[str, Any]]] = []
    membership: list[tuple[str, tuple[str, ...]]] = []
    for name, frame in partitions.as_mapping().items():
        if frame.empty or tuple(frame.columns) != tuple(source.columns):
            raise PartitionValidationError(f"Partition '{name}' is empty or has a schema mismatch.")
        values = tuple(persisted.get(name, ()))
        if len(values) != len(frame) or len(set(values)) != len(values):
            raise PartitionValidationError(f"Partition '{name}' membership is invalid.")
        if any(value not in source_positions for value in values):
            raise PartitionValidationError(f"Partition '{name}' membership is outside source.")
        if not identifier_columns:
            expected_hashes = Counter(_row_hash_from_occurrence_key(value) for value in values)
            if Counter(_row_content_fingerprints(frame)) != expected_hashes:
                raise PartitionValidationError(f"Partition '{name}' row multiplicity mismatch.")
        elif _source_identifier_membership_keys(frame, identifier_columns) != values:
            raise PartitionValidationError(f"Partition '{name}' identifier membership mismatch.")
        positions = [source_positions[value] for value in values]
        if positions != sorted(positions):
            raise PartitionValidationError(f"Partition '{name}' is not in stable source order.")
        membership_sets[name] = set(values)
        rows.append((name, len(frame)))
        diagnostics.append((name, describe_continuous_target(frame[target_column])))
        membership.append((name, values))
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(membership_sets[a] & membership_sets[b] for a, b in pairs):
        raise PartitionValidationError("Partition membership overlaps were detected.")
    if set().union(*membership_sets.values()) != set(source_membership):
        raise PartitionValidationError("Partition coverage mismatch.")
    if sum(dict(rows).values()) != len(source):
        raise PartitionValidationError("Partition row counts do not cover source.")
    checks = (
        ("all_partitions_present", True), ("all_partitions_non_empty", True),
        ("schema_and_column_order_preserved", True), ("continuous_target_complete_finite", True),
        ("technical_occurrence_membership_disjoint", True), ("row_multiplicity_preserved", True),
        ("partition_membership_isolated", True), ("full_coverage", True),
        ("row_count_preserved", True), ("stable_source_order", True),
        ("non_stratified_assignment", True), ("test_holdout_isolated", True),
        ("entity_disjointness_not_claimed_without_source_identifiers", not bool(identifier_columns)),
    )
    return RegressionPartitionValidationReport(
        row_counts=tuple(rows), target_diagnostics=tuple(diagnostics),
        membership=tuple(membership), checks=checks,
        membership_kind=inferred_kind, membership_semantics=inferred_semantics,
        entity_disjointness_status=("validated_from_source_identifiers" if identifier_columns else "not_claimed_without_source_identifiers"),
    )


def analyze_repeated_profiles_across_partitions(
    source_dataframe: pd.DataFrame,
    partitions: DatasetPartitions,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    identifier_columns: Sequence[str] = (),
    max_samples: int = 20,
) -> dict[str, Any]:
    """Describe equality evidence without inferring duplicate entity identity."""
    if max_samples < 0:
        raise ValueError("max_samples cannot be negative.")
    source = _copy_frame(source_dataframe)
    features = tuple(feature_columns)
    missing = [column for column in (*features, target_column) if column not in source]
    if missing:
        raise DatasetValidationError(
            f"Repeated-profile analysis is missing columns: {missing}."
        )

    source_exact = Counter(_row_content_fingerprints(source))
    source_profiles = Counter(
        _row_content_fingerprints(source, columns=features)
    )
    partition_exact: dict[str, Counter[str]] = {}
    partition_profiles: dict[str, Counter[str]] = {}
    for name, frame in partitions.as_mapping().items():
        partition_exact[name] = Counter(_row_content_fingerprints(frame))
        partition_profiles[name] = Counter(
            _row_content_fingerprints(frame, columns=features)
        )

    combined_exact: Counter[str] = Counter()
    combined_profiles: Counter[str] = Counter()
    for counter in partition_exact.values():
        combined_exact.update(counter)
    for counter in partition_profiles.values():
        combined_profiles.update(counter)
    if combined_exact != source_exact or combined_profiles != source_profiles:
        raise PartitionValidationError(
            "Repeated-profile multiplicities changed across partitions."
        )

    cross_partition_exact = {
        row_hash: {
            name: int(counts[row_hash])
            for name, counts in partition_exact.items()
            if counts[row_hash]
        }
        for row_hash, total in source_exact.items()
        if total > 1
        and sum(counts[row_hash] > 0 for counts in partition_exact.values()) > 1
    }
    cross_partition_profiles = {
        profile_hash: {
            name: int(counts[profile_hash])
            for name, counts in partition_profiles.items()
            if counts[profile_hash]
        }
        for profile_hash, total in source_profiles.items()
        if total > 1
        and sum(counts[profile_hash] > 0 for counts in partition_profiles.values()) > 1
    }

    profile_targets: defaultdict[str, set[str]] = defaultdict(set)
    profile_hashes = _row_content_fingerprints(source, columns=features)
    for profile_hash, target_value in zip(
        profile_hashes,
        source[target_column].tolist(),
        strict=True,
    ):
        profile_targets[profile_hash].add(str(target_value))
    target_conflict_profiles = {
        key: sorted(values)
        for key, values in profile_targets.items()
        if len(values) > 1
    }

    def sample(mapping: Mapping[str, Mapping[str, int]]) -> list[dict[str, Any]]:
        return [
            {"profile_sha256": key, "partition_counts": dict(value)}
            for key, value in sorted(mapping.items())[:max_samples]
        ]

    return {
        "evidence_type": "repeated_profile_partition_review",
        "source_identifier_available": bool(identifier_columns),
        "identity_interpretation": (
            "source_identifier_supported"
            if identifier_columns
            else (
                "row equality is observational evidence only; duplicate entity "
                "identity and same-grain leakage are not proven"
            )
        ),
        "proven_duplicate_identity": False if not identifier_columns else None,
        "source_exact_row_equality_group_count": int(
            sum(count > 1 for count in source_exact.values())
        ),
        "source_exact_row_equality_row_count": int(
            sum(count for count in source_exact.values() if count > 1)
        ),
        "source_repeated_feature_profile_group_count": int(
            sum(count > 1 for count in source_profiles.values())
        ),
        "cross_partition_exact_row_equality_group_count": len(cross_partition_exact),
        "cross_partition_repeated_feature_profile_group_count": len(
            cross_partition_profiles
        ),
        "target_conflicting_feature_profile_group_count": len(target_conflict_profiles),
        "exact_row_multiplicity_preserved": combined_exact == source_exact,
        "feature_profile_multiplicity_preserved": combined_profiles == source_profiles,
        "cross_partition_exact_row_samples": sample(cross_partition_exact),
        "cross_partition_feature_profile_samples": sample(cross_partition_profiles),
    }


def _relative_posix(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"Artifact path must be project-relative: {candidate.name}")
    normalized = Path(os.path.normpath(str(candidate)))
    if normalized == Path(".") or ".." in normalized.parts:
        raise ValueError(f"Invalid project-relative artifact path: {candidate}")
    return normalized.as_posix()


def runtime_versions() -> dict[str, str]:
    """Return versions relevant to deterministic preparation."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
    }


def build_preparation_manifest(
    *,
    dataset_slug: str,
    source_path: str | Path,
    source_sha256: str,
    prepared_path: str | Path,
    prepared_sha256: str,
    raw_report: DatasetValidationReport,
    prepared_report: DatasetValidationReport,
    preparation: PreparedDataset,
    raw_fingerprint_before: str,
    raw_fingerprint_after: str,
    source_sha256_after: str,
    deterministic_rules: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    source_identity: Mapping[str, Any] | None = None,
    contract_version: str = CONTRACT_VERSION,
    upstream_exploration: Mapping[str, Any] | None = None,
    source_type_resolutions: Sequence[Mapping[str, Any]] = (),
    runtime_version_evidence: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a versioned preparation manifest."""
    payload = {
        "schema_version": "preparation-manifest.v1",
        "artifact_type": "preparation_manifest",
        "dataset_slug": dataset_slug,
        "contract_version": contract_version,
        "source_path": _relative_posix(source_path),
        "source_filename": Path(source_path).name,
        "source_sha256": source_sha256,
        "source_sha256_after": source_sha256_after,
        "prepared_path": _relative_posix(prepared_path),
        "prepared_sha256": prepared_sha256,
        "source_row_count": raw_report.row_count,
        "prepared_row_count": prepared_report.row_count,
        "source_column_count": raw_report.column_count,
        "prepared_column_count": prepared_report.column_count,
        "column_order": list(prepared_report.column_order),
        "deterministic_rules": copy.deepcopy(list(deterministic_rules)),
        "materialized_value_counts": dict(preparation.materialized_counts),
        "invalid_conversion_counts": dict(preparation.invalid_conversion_counts),
        "row_preservation_checks": {
            "row_count_preserved": raw_report.row_count == prepared_report.row_count,
            "column_count_preserved": raw_report.column_count == prepared_report.column_count,
            "column_order_preserved": raw_report.column_order == prepared_report.column_order,
        },
        "raw_immutability_checks": {
            "logical_fingerprint_before": raw_fingerprint_before,
            "logical_fingerprint_after": raw_fingerprint_after,
            "logical_fingerprint_preserved": raw_fingerprint_before == raw_fingerprint_after,
            "source_sha256_preserved": source_sha256 == source_sha256_after,
        },
        "runtime_versions": _copy_mapping(
            runtime_version_evidence
            if runtime_version_evidence is not None
            else runtime_versions()
        ),
        "readiness": _copy_mapping(readiness),
        "source_identity_gate": _copy_mapping(source_identity or {}),
    }
    if upstream_exploration is not None:
        payload["upstream_exploration"] = _copy_mapping(upstream_exploration)
    if source_type_resolutions:
        payload["source_type_resolutions"] = copy.deepcopy(list(source_type_resolutions))
    return payload


def build_feature_manifest(
    *,
    dataset_slug: str,
    identifier_columns: Sequence[str],
    feature_columns: Sequence[str],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    categorical_expected_values: Mapping[str, Sequence[Any]],
    target_column: str,
    target_classes: Sequence[Any],
    expected_dtypes: Mapping[str, Any],
    preprocessing_contract: Mapping[str, Any],
    prohibited_predictors: Sequence[str],
    positive_target_class: Any = None,
    target_encoding: Mapping[Any, int] | None = None,
    problem_type: str | None = None,
    target_semantics: str | None = None,
    target_unit: str | None = None,
    prediction_output: str | None = None,
) -> dict[str, Any]:
    """Build an ordered feature and future-preprocessing contract."""
    features = tuple(feature_columns)
    numerical = tuple(numerical_features)
    categorical = tuple(categorical_features)
    identifiers = tuple(identifier_columns)
    classes = tuple(target_classes)
    is_regression = problem_type == "continuous_regression"
    if not classes and not is_regression:
        raise ValueError("target_classes cannot be empty.")
    if is_regression and classes:
        raise ValueError("Continuous regression cannot declare target classes.")
    if set(numerical) & set(categorical):
        raise ValueError("Numerical and categorical feature roles overlap.")
    if set((*numerical, *categorical)) != set(features):
        raise ValueError(
            "Numerical and categorical roles must cover feature_columns exactly."
        )
    resolved_problem_type = problem_type or (
        "binary_classification" if len(classes) == 2 else "multiclass_classification"
    )
    if resolved_problem_type == "multiclass_classification" and positive_target_class is not None:
        raise ValueError("Multiclass targets cannot declare a positive target class.")
    if resolved_problem_type == "binary_classification" and len(classes) != 2:
        raise ValueError("Binary classification requires exactly two target classes.")
    encoding = {} if is_regression else (
        dict(target_encoding)
        if target_encoding is not None
        else {value: index for index, value in enumerate(classes)}
    )
    if not is_regression and (set(encoding) != set(classes) or set(encoding.values()) != set(range(len(classes)))):
        raise ValueError(
            "target_encoding must map every target class bijectively to 0..n-1."
        )
    legacy_binary = (
        problem_type is None
        and target_semantics is None
        and len(classes) == 2
        and positive_target_class is not None
    )
    if is_regression:
        if positive_target_class is not None or target_encoding:
            raise ValueError("Continuous regression cannot define classes or target encoding.")
        if not target_semantics or not target_unit or not prediction_output:
            raise ValueError("Continuous target semantics, unit, and prediction output are required.")
    payload = {
        "schema_version": (
            "feature-manifest.v1" if legacy_binary else ("feature-manifest.v3" if is_regression else "feature-manifest.v2")
        ),
        "artifact_type": "feature_manifest",
        "dataset_slug": dataset_slug,
        "identifier_columns": list(identifiers),
        "feature_columns": list(features),
        "numerical_features": list(numerical),
        "categorical_features": list(categorical),
        "categorical_expected_values": {
            column: list(values)
            for column, values in categorical_expected_values.items()
        },
        "target_column": target_column,
        "target_classes": list(classes),
        "positive_target_class": positive_target_class,
        "expected_dtypes": _copy_mapping(expected_dtypes),
        "preprocessing_contract": _copy_mapping(preprocessing_contract),
        "prohibited_predictors": list(prohibited_predictors),
    }
    if not is_regression:
        payload["target_encoding_contract"] = {str(key): value for key, value in encoding.items()}
    else:
        payload["problem_type"] = "continuous_regression"
        payload["target_contract"] = {
            "semantics": target_semantics,
            "unit": target_unit,
            "prediction_output": prediction_output,
            "persisted_target": "continuous_numeric_original_scale",
            "target_encoding": "not_applicable",
            "target_transformed": False,
        }
    if not legacy_binary and not is_regression:
        payload["problem_type"] = resolved_problem_type
        payload["target_contract"] = {
            "semantics": target_semantics or "nominal_unordered",
            "ordered_class_contract": list(classes),
            "class_order_purpose": "deterministic technical contract, not ordinal rank",
            "positive_class": positive_target_class,
            "persisted_labels_remain_readable": True,
            "encoding_required_for_persisted_target": False,
        }
    return payload


def build_split_manifest(
    *,
    dataset_slug: str,
    policy: ClassificationSplitPolicy | ContinuousRegressionSplitPolicy,
    partitions: DatasetPartitions,
    validation: PartitionValidationReport | RegressionPartitionValidationReport,
    partition_paths: Mapping[str, str | Path],
    partition_sha256: Mapping[str, str],
    repeated_profile_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned split manifest with explicit membership."""
    paths = {name: _relative_posix(path) for name, path in partition_paths.items()}
    required = {"train", "validation", "test"}
    if set(paths) != required or set(partition_sha256) != required:
        raise ValueError("Partition paths and fingerprints must cover train/validation/test.")
    report = validation.as_dict()
    is_regression = isinstance(validation, RegressionPartitionValidationReport)
    schema_version = ("split-manifest.v3" if is_regression else (
        "split-manifest.v2"
        if report["membership_kind"] == "technical_row_occurrence"
        else "split-manifest.v1"
    ))
    payload = {
        "schema_version": schema_version,
        "artifact_type": "split_manifest",
        "dataset_slug": dataset_slug,
        **policy.as_dict(),
        "split_method": partitions.split_method,
        "rounding_method": partitions.rounding_method,
        "partition_paths": paths,
        "partition_sha256": dict(partition_sha256),
        "row_counts": report["row_counts"],
        "membership": report["membership"],
        "membership_kind": report["membership_kind"],
        "membership_semantics": report["membership_semantics"],
        "entity_disjointness_status": report["entity_disjointness_status"],
        "isolation_checks": report["checks"],
        "test_holdout_policy": (
            "The test partition is isolated and must not be used for feature, "
            "preprocessing, model, hyperparameter, threshold, or metric selection."
        ),
        "operational_modeling_ready": False,
        "educational_model_selection_ready": True,
    }
    if is_regression:
        payload["target_diagnostics"] = report["target_diagnostics"]
        payload["stratification"] = None
        payload["target_bins_created"] = False
        payload["seed_shopping_performed"] = False
    else:
        payload["class_counts"] = report["class_counts"]
        payload["class_prevalence"] = report["class_prevalence"]
        payload["prevalence_tolerance"] = report["prevalence_tolerance"]
    if repeated_profile_evidence is not None:
        payload["repeated_profile_evidence"] = _copy_mapping(
            repeated_profile_evidence
        )
    return payload


def build_quality_evidence(
    *,
    dataset_slug: str,
    raw_report: DatasetValidationReport,
    prepared_report: DatasetValidationReport,
    partition_report: PartitionValidationReport | RegressionPartitionValidationReport,
    preparation: PreparedDataset,
    fingerprints: Mapping[str, Any],
    readiness: Mapping[str, Any],
    preservation_checks: Mapping[str, Any],
    repeated_profile_evidence: Mapping[str, Any] | None = None,
    source_type_resolutions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build consolidated preparation quality evidence."""
    payload = {
        "schema_version": "quality-evidence.v1",
        "artifact_type": "preparation_quality_evidence",
        "dataset_slug": dataset_slug,
        "raw_schema_validation": raw_report.as_dict(),
        "prepared_dataset_validation": prepared_report.as_dict(),
        "partition_validation": partition_report.as_dict(),
        "materializations": preparation.as_dict(),
        "fingerprint_checks": _copy_mapping(fingerprints),
        "preservation_checks": _copy_mapping(preservation_checks),
        "readiness": _copy_mapping(readiness),
        "operational_block": {
            "operational_modeling_ready": False,
            "operational_validity": readiness.get(
                "operational_validity", "unconfirmed"
            ),
            "temporal_contract_status": readiness.get(
                "temporal_contract_status", "unresolved"
            ),
            "feature_inference_availability": readiness.get(
                "feature_inference_availability", "unconfirmed"
            ),
        },
    }
    if repeated_profile_evidence is not None:
        payload["repeated_profile_evidence"] = _copy_mapping(
            repeated_profile_evidence
        )
    if source_type_resolutions:
        payload["source_type_resolutions"] = copy.deepcopy(list(source_type_resolutions))
    return payload


def build_preparation_handoff_manifest(
    *,
    dataset_slug: str,
    component_paths: Mapping[str, str | Path],
    component_payloads: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, Any],
    upstream_exploration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-circular integrity index for preparation JSON components."""
    required = {
        "preparation_manifest",
        "feature_manifest",
        "split_manifest",
        "quality_evidence",
    }
    if set(component_paths) != required or set(component_payloads) != required:
        raise ValueError(
            "Preparation handoff components must cover all four required manifests."
        )
    references = {
        name: {
            "path": _relative_posix(component_paths[name]),
            "sha256": fingerprint_json_artifact(component_payloads[name]),
            "schema_version": component_payloads[name].get("schema_version"),
        }
        for name in sorted(required)
    }
    payload = {
        "schema_version": "preparation-handoff.v1",
        "artifact_type": "preparation_handoff",
        "dataset_slug": dataset_slug,
        "components": references,
        "readiness": _copy_mapping(readiness),
        "consumer_contract": {
            "prepared_and_partitions_are_frozen": True,
            "model_selection_must_not_resplit": True,
            "test_partition_sealed": True,
            "test_partition_evaluated": False,
        },
    }
    if upstream_exploration is not None:
        payload["upstream_exploration"] = _copy_mapping(upstream_exploration)
    return payload


def _semantic_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_SEMANTIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_normalize(item) for item in value]
    return _normalize_scalar(value)


def semantically_equivalent(left: Any, right: Any) -> bool:
    """Compare values while ignoring explicitly volatile timestamp fields."""
    return _semantic_normalize(left) == _semantic_normalize(right)


def _resolve_project_artifact_path(root: Path, relative_path: str | Path) -> Path:
    relative = Path(_relative_posix(relative_path))
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes project root: {relative.as_posix()}") from exc
    return candidate


def _validate_staged_csv(path: Path, expected: pd.DataFrame) -> None:
    reloaded = pd.read_csv(path)
    if tuple(reloaded.columns) != tuple(expected.columns):
        raise PreparationError(
            f"Staged CSV column validation failed for {path.name}."
        )
    if len(reloaded) != len(expected):
        raise PreparationError(
            f"Staged CSV row validation failed for {path.name}."
        )


def write_preparation_artifacts(
    *,
    project_root: str | Path,
    csv_artifacts: Mapping[str | Path, pd.DataFrame],
    json_artifacts: Mapping[str | Path, Mapping[str, Any]],
    overwrite: bool = False,
) -> ArtifactWriteResult:
    """Persist a complete artifact set with staging, conflict checks, and rollback.

    Existing byte-equivalent CSVs and semantically equivalent JSON documents
    are accepted as idempotent. Divergent files fail before promotion unless
    ``overwrite=True`` was supplied deliberately.
    """
    root = Path(project_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not csv_artifacts and not json_artifacts:
        raise ValueError("At least one artifact must be supplied.")

    csv_inputs = {
        _relative_posix(path): _copy_frame(frame)
        for path, frame in csv_artifacts.items()
    }
    json_inputs = {
        _relative_posix(path): _copy_mapping(payload)
        for path, payload in json_artifacts.items()
    }
    overlap = set(csv_inputs) & set(json_inputs)
    if overlap:
        raise ValueError(f"Artifact paths are duplicated across formats: {sorted(overlap)}")

    transaction_id = uuid.uuid4().hex
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".preparation-staging-{transaction_id}-", dir=root)
    )
    backup_root = Path(
        tempfile.mkdtemp(prefix=f".preparation-backup-{transaction_id}-", dir=root)
    )
    all_inputs: dict[str, tuple[str, Any]] = {
        **{path: ("csv", value) for path, value in csv_inputs.items()},
        **{path: ("json", value) for path, value in json_inputs.items()},
    }
    staged_paths: dict[str, Path] = {}
    statuses: dict[str, str] = {}
    digests: dict[str, str] = {}
    promoted: list[str] = []
    backed_up: list[str] = []

    try:
        for relative, (kind, payload) in all_inputs.items():
            staged = staging_root / Path(relative)
            staged.parent.mkdir(parents=True, exist_ok=True)
            if kind == "csv":
                staged.write_bytes(dataframe_csv_bytes(payload))
                _validate_staged_csv(staged, payload)
            else:
                staged.write_bytes(json_artifact_bytes(payload))
                json.loads(staged.read_text(encoding="utf-8"))
            staged_paths[relative] = staged
            digests[relative] = fingerprint_file(staged)

        # Detect every conflict before mutating any destination.
        for relative, (kind, payload) in all_inputs.items():
            destination = _resolve_project_artifact_path(root, relative)
            if not destination.exists():
                statuses[relative] = "created"
                continue
            if not destination.is_file():
                raise ArtifactConflictError(
                    f"Artifact destination is not a file: {relative}"
                )
            if kind == "csv":
                equivalent = destination.read_bytes() == staged_paths[relative].read_bytes()
            else:
                try:
                    existing_payload = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ArtifactConflictError(
                        f"Existing JSON artifact is unreadable: {relative}"
                    ) from exc
                equivalent = semantically_equivalent(existing_payload, payload)
            if equivalent:
                statuses[relative] = "reused_equivalent"
            elif overwrite:
                statuses[relative] = "overwritten"
            else:
                raise ArtifactConflictError(
                    f"Existing artifact is semantically divergent: {relative}. "
                    "No overwrite was performed."
                )

        for relative in sorted(all_inputs):
            status = statuses[relative]
            if status == "reused_equivalent":
                digests[relative] = fingerprint_file(
                    _resolve_project_artifact_path(root, relative)
                )
                continue
            destination = _resolve_project_artifact_path(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = backup_root / Path(relative)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(relative)
            os.replace(staged_paths[relative], destination)
            promoted.append(relative)
            digests[relative] = fingerprint_file(destination)

    except Exception:
        for relative in reversed(promoted):
            destination = _resolve_project_artifact_path(root, relative)
            destination.unlink(missing_ok=True)
        for relative in reversed(backed_up):
            backup = backup_root / Path(relative)
            destination = _resolve_project_artifact_path(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)

    return ArtifactWriteResult(
        statuses=tuple(sorted(statuses.items())),
        sha256=tuple(sorted(digests.items())),
    )


def _load_json_artifact(root: Path, relative_path: str | Path) -> dict[str, Any]:
    path = _resolve_project_artifact_path(root, relative_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HandoffValidationError(
            f"Required handoff artifact is missing: {_relative_posix(relative_path)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HandoffValidationError(
            f"Invalid JSON handoff artifact: {_relative_posix(relative_path)}"
        ) from exc
    if not isinstance(payload, dict):
        raise HandoffValidationError(
            f"Handoff artifact must contain a JSON object: {_relative_posix(relative_path)}"
        )
    return payload


def load_and_validate_preparation_handoff(
    *,
    project_root: str | Path,
    preparation_handoff_path: str | Path | None = None,
    preparation_manifest_path: str | Path | None = None,
    feature_manifest_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    quality_evidence_path: str | Path | None = None,
    _model_selection_safe: bool = False,
) -> PreparationHandoff:
    """Load persisted artifacts, verify fingerprints, and return the handoff."""
    root = Path(project_root).expanduser().resolve()
    handoff_manifest: dict[str, Any] | None = None
    supplied_paths = {
        "preparation_manifest": preparation_manifest_path,
        "feature_manifest": feature_manifest_path,
        "split_manifest": split_manifest_path,
        "quality_evidence": quality_evidence_path,
    }
    if preparation_handoff_path is not None:
        handoff_manifest = _load_json_artifact(root, preparation_handoff_path)
        if handoff_manifest.get("schema_version") != "preparation-handoff.v1":
            raise HandoffValidationError(
                "Unexpected preparation handoff schema_version."
            )
        if handoff_manifest.get("artifact_type") != "preparation_handoff":
            raise HandoffValidationError(
                "Unexpected preparation handoff artifact_type."
            )
        components = handoff_manifest.get("components")
        if not isinstance(components, Mapping) or set(components) != set(supplied_paths):
            raise HandoffValidationError(
                "Preparation handoff component references are incomplete."
            )
        resolved_paths: dict[str, str] = {}
        for name, reference in components.items():
            if not isinstance(reference, Mapping):
                raise HandoffValidationError(
                    f"Preparation handoff component '{name}' is invalid."
                )
            relative = reference.get("path")
            expected_sha = reference.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_sha, str):
                raise HandoffValidationError(
                    f"Preparation handoff component '{name}' lacks path or SHA-256."
                )
            component_file = _resolve_project_artifact_path(root, relative)
            try:
                observed_component_sha = fingerprint_file(component_file)
            except FileNotFoundError as exc:
                raise HandoffValidationError(
                    f"Preparation handoff component is missing: {name}."
                ) from exc
            if observed_component_sha != expected_sha:
                raise HandoffValidationError(
                    f"Preparation handoff component fingerprint mismatch: {name}."
                )
            explicit = supplied_paths[name]
            if explicit is not None and _relative_posix(explicit) != relative:
                raise HandoffValidationError(
                    f"Explicit path conflicts with preparation handoff: {name}."
                )
            resolved_paths[name] = relative
        preparation_manifest_path = resolved_paths["preparation_manifest"]
        feature_manifest_path = resolved_paths["feature_manifest"]
        split_manifest_path = resolved_paths["split_manifest"]
        quality_evidence_path = resolved_paths["quality_evidence"]
    elif any(value is None for value in supplied_paths.values()):
        raise ValueError(
            "Supply preparation_handoff_path or all four legacy component paths."
        )

    assert preparation_manifest_path is not None
    assert feature_manifest_path is not None
    assert split_manifest_path is not None
    assert quality_evidence_path is not None
    preparation_manifest = _load_json_artifact(root, preparation_manifest_path)
    feature_manifest = _load_json_artifact(root, feature_manifest_path)
    split_manifest = _load_json_artifact(root, split_manifest_path)
    quality_evidence = _load_json_artifact(root, quality_evidence_path)

    expected_schemas: dict[str, tuple[dict[str, Any], set[str]]] = {
        "preparation": (preparation_manifest, {"preparation-manifest.v1"}),
        "feature": (
            feature_manifest,
            {"feature-manifest.v1", "feature-manifest.v2", "feature-manifest.v3"},
        ),
        "split": (split_manifest, {"split-manifest.v1", "split-manifest.v2", "split-manifest.v3"}),
        "quality": (quality_evidence, {"quality-evidence.v1"}),
    }
    for name, (payload, expected) in expected_schemas.items():
        if payload.get("schema_version") not in expected:
            raise HandoffValidationError(
                f"Unexpected {name} schema_version: {payload.get('schema_version')!r}."
            )
    slugs = {
        payload.get("dataset_slug")
        for payload, _ in expected_schemas.values()
    }
    if handoff_manifest is not None:
        slugs.add(handoff_manifest.get("dataset_slug"))
    if len(slugs) != 1 or None in slugs:
        raise HandoffValidationError("Handoff dataset_slug values are inconsistent.")
    if split_manifest.get("operational_validity") != "unconfirmed":
        raise HandoffValidationError(
            "Handoff operational_validity must remain unconfirmed."
        )
    if split_manifest.get("operational_modeling_ready") is not False:
        raise HandoffValidationError(
            "Handoff operational_modeling_ready must remain false."
        )
    if split_manifest.get("educational_model_selection_ready") is not True:
        raise HandoffValidationError(
            "Handoff educational_model_selection_ready must be true."
        )
    if feature_manifest.get("schema_version") == "feature-manifest.v2":
        problem_type = feature_manifest.get("problem_type")
        target_contract = feature_manifest.get("target_contract")
        if problem_type not in {"binary_classification", "multiclass_classification"}:
            raise HandoffValidationError("Feature manifest problem_type is invalid.")
        if not isinstance(target_contract, Mapping):
            raise HandoffValidationError("Feature manifest target_contract is invalid.")
        if problem_type == "multiclass_classification":
            if feature_manifest.get("positive_target_class") is not None:
                raise HandoffValidationError(
                    "Multiclass feature manifest cannot define a positive class."
                )
            if target_contract.get("semantics") != "nominal_unordered":
                raise HandoffValidationError(
                    "Multiclass target semantics must remain nominal and unordered."
                )
            if target_contract.get("ordered_class_contract") != feature_manifest.get(
                "target_classes"
            ):
                raise HandoffValidationError(
                    "Multiclass ordered class contract is inconsistent."
                )
    elif feature_manifest.get("schema_version") == "feature-manifest.v3":
        target_contract = feature_manifest.get("target_contract")
        if feature_manifest.get("problem_type") != "continuous_regression":
            raise HandoffValidationError("Continuous feature manifest problem_type is invalid.")
        if feature_manifest.get("target_classes") != []:
            raise HandoffValidationError("Continuous feature manifest cannot define classes.")
        if feature_manifest.get("positive_target_class") is not None:
            raise HandoffValidationError("Continuous feature manifest cannot define a positive class.")
        if "target_encoding_contract" in feature_manifest:
            raise HandoffValidationError("Continuous feature manifest cannot define target encoding.")
        if not isinstance(target_contract, Mapping):
            raise HandoffValidationError("Continuous target contract is invalid.")

    if split_manifest.get("schema_version") == "split-manifest.v3" and split_manifest.get(
        "problem_type"
    ) != feature_manifest.get("problem_type"):
        raise HandoffValidationError("Split manifest problem_type is inconsistent.")

    if handoff_manifest is not None and "upstream_exploration" in handoff_manifest:
        upstream = handoff_manifest["upstream_exploration"]
        if not isinstance(upstream, Mapping):
            raise HandoffValidationError("Upstream exploration lineage is invalid.")
        upstream_path = upstream.get("path")
        upstream_sha = upstream.get("sha256")
        if not isinstance(upstream_path, str) or not isinstance(upstream_sha, str):
            raise HandoffValidationError("Upstream exploration lineage lacks path or SHA-256.")
        if fingerprint_file(_resolve_project_artifact_path(root, upstream_path)) != upstream_sha:
            raise HandoffValidationError("Upstream exploration fingerprint mismatch.")
        if upstream.get("dataset_slug") != feature_manifest.get("dataset_slug"):
            raise HandoffValidationError("Upstream exploration dataset_slug mismatch.")
        upstream_payload = _load_json_artifact(root, upstream_path)
        if upstream_payload.get("schema_version") != upstream.get("schema_version"):
            raise HandoffValidationError("Upstream exploration schema_version mismatch.")
        if upstream_payload.get("dataset_slug") != upstream.get("dataset_slug"):
            raise HandoffValidationError("Upstream exploration payload dataset_slug mismatch.")
        source_contract = upstream_payload.get("source")
        prediction_contract = upstream_payload.get("prediction_contract")
        feature_contract = upstream_payload.get("feature_contract")
        if not all(
            isinstance(value, Mapping)
            for value in (source_contract, prediction_contract, feature_contract)
        ):
            raise HandoffValidationError("Upstream exploration contracts are invalid.")
        if source_contract.get("dataset_id") != upstream.get("source_dataset_id"):
            raise HandoffValidationError("Upstream source dataset ID mismatch.")
        if source_contract.get("sha256") != upstream.get("source_dataset_sha256"):
            raise HandoffValidationError("Upstream source dataset SHA-256 mismatch.")
        if prediction_contract.get("problem_type") != feature_manifest.get("problem_type"):
            raise HandoffValidationError("Upstream prediction problem_type mismatch.")
        if prediction_contract.get("target_column") != feature_manifest.get("target_column"):
            raise HandoffValidationError("Upstream target column mismatch.")
        if prediction_contract.get("target_classes") != feature_manifest.get("target_classes"):
            raise HandoffValidationError("Upstream target classes mismatch.")
        if feature_contract.get("feature_columns") != feature_manifest.get("feature_columns"):
            raise HandoffValidationError("Upstream feature order mismatch.")
        if feature_contract.get("identifier_columns") != feature_manifest.get("identifier_columns"):
            raise HandoffValidationError("Upstream identifier contract mismatch.")
        if feature_manifest.get("problem_type") == "continuous_regression":
            if target_contract.get("semantics") != prediction_contract.get("target_semantics"):
                raise HandoffValidationError("Continuous target semantics mismatch.")
            if target_contract.get("unit") != prediction_contract.get("target_unit"):
                raise HandoffValidationError("Continuous target unit mismatch.")

    prepared_relative = preparation_manifest.get("prepared_path")
    partition_paths = split_manifest.get("partition_paths")
    if not isinstance(prepared_relative, str) or not isinstance(partition_paths, dict):
        raise HandoffValidationError("Handoff data paths are missing or invalid.")

    prepared_path = _resolve_project_artifact_path(root, prepared_relative)
    expected_prepared_sha = preparation_manifest.get("prepared_sha256")
    if fingerprint_file(prepared_path) != expected_prepared_sha:
        raise HandoffValidationError("Prepared CSV fingerprint mismatch.")
    expected_columns = feature_manifest.get("identifier_columns", []) + feature_manifest.get(
        "feature_columns", []
    ) + [feature_manifest.get("target_column")]
    # Feature order is not necessarily contiguous with identifiers/target in the
    # source schema, so the preparation manifest remains authoritative.
    authoritative_columns = preparation_manifest.get("column_order")
    if set(expected_columns) != set(authoritative_columns):
        raise HandoffValidationError("Feature roles do not cover prepared columns.")

    prepared = None if _model_selection_safe else pd.read_csv(prepared_path)
    if prepared is not None:
        if list(prepared.columns) != authoritative_columns:
            raise HandoffValidationError("Prepared CSV column order mismatch.")
        if len(prepared) != preparation_manifest.get("prepared_row_count"):
            raise HandoffValidationError("Prepared CSV row count mismatch.")

    loaded_partitions: dict[str, pd.DataFrame] = {}
    partition_sha = split_manifest.get("partition_sha256", {})
    for name in ("train", "validation", "test"):
        relative = partition_paths.get(name)
        if not isinstance(relative, str):
            raise HandoffValidationError(f"Missing path for partition '{name}'.")
        path = _resolve_project_artifact_path(root, relative)
        if fingerprint_file(path) != partition_sha.get(name):
            raise HandoffValidationError(
                f"Partition CSV fingerprint mismatch: {name}."
            )
        if _model_selection_safe and name == "test":
            continue
        frame = pd.read_csv(path)
        if list(frame.columns) != authoritative_columns:
            raise HandoffValidationError(
                f"Partition '{name}' column order mismatch."
            )
        if len(frame) != split_manifest.get("row_counts", {}).get(name):
            raise HandoffValidationError(
                f"Partition '{name}' row count mismatch."
            )
        loaded_partitions[name] = frame

    partition_set = None if _model_selection_safe else DatasetPartitions(
        _train=loaded_partitions["train"],
        _validation=loaded_partitions["validation"],
        _test=loaded_partitions["test"],
        split_method=str(split_manifest.get("split_method")),
        rounding_method=str(split_manifest.get("rounding_method")),
        _membership=tuple(
            (
                name,
                tuple(split_manifest.get("membership", {}).get(name, ())),
            )
            for name in ("train", "validation", "test")
        ),
        membership_kind=str(
            split_manifest.get("membership_kind", "source_identifier")
        ),
        membership_semantics=str(
            split_manifest.get(
                "membership_semantics", "declared source identifier tuple"
            )
        ),
    )
    if _model_selection_safe:
        target_column = feature_manifest.get("target_column")
        row_counts = split_manifest.get("row_counts")
        membership = split_manifest.get("membership")
        if not isinstance(row_counts, Mapping) or not isinstance(membership, Mapping):
            raise HandoffValidationError("Split membership contract is missing or invalid.")
        membership_sets: dict[str, set[Any]] = {}
        for name in ("train", "validation", "test"):
            values = membership.get(name)
            expected_count = row_counts.get(name)
            if not isinstance(values, list) or len(values) != expected_count:
                raise HandoffValidationError(
                    f"Partition '{name}' membership length is inconsistent."
                )
            membership_sets[name] = set(values)
            if len(membership_sets[name]) != len(values):
                raise HandoffValidationError(
                    f"Partition '{name}' membership contains duplicate tokens."
                )
        if any(
            membership_sets[left].intersection(membership_sets[right])
            for left, right in (("train", "validation"), ("train", "test"),
                                ("validation", "test"))
        ):
            raise HandoffValidationError("Split membership partitions are not disjoint.")
        if sum(int(row_counts[name]) for name in ("train", "validation", "test")) != preparation_manifest.get(
            "prepared_row_count"
        ):
            raise HandoffValidationError("Split membership does not cover the prepared row count.")
        for name in ("train", "validation"):
            frame = loaded_partitions[name]
            if list(frame.columns) != authoritative_columns:
                raise HandoffValidationError(f"Partition '{name}' column order mismatch.")
            target = frame[target_column]
            if (not pandas_types.is_numeric_dtype(target) or target.isna().any()
                    or not all(math.isfinite(float(value)) for value in target)):
                raise HandoffValidationError(
                    f"Partition '{name}' target must be complete, numeric, and finite."
                )
    elif feature_manifest.get("problem_type") == "continuous_regression":
        validate_regression_partitions(
            prepared, partition_set,
            identifier_columns=feature_manifest["identifier_columns"],
            target_column=feature_manifest["target_column"],
        )
    else:
        validate_dataset_partitions(
            prepared,
            partition_set,
            identifier_columns=feature_manifest["identifier_columns"],
            target_column=feature_manifest["target_column"],
            target_classes=feature_manifest["target_classes"],
            prevalence_tolerance=float(split_manifest.get("prevalence_tolerance", 0.02)),
        )

    fingerprints = quality_evidence.get("fingerprint_checks", {})
    if isinstance(fingerprints, Mapping):
        quality_prepared_sha = fingerprints.get("prepared_sha256")
        if quality_prepared_sha is not None and quality_prepared_sha != expected_prepared_sha:
            raise HandoffValidationError(
                "Prepared SHA-256 differs between preparation and quality artifacts."
            )
        quality_partition_sha = fingerprints.get("partition_sha256")
        if quality_partition_sha is not None and quality_partition_sha != partition_sha:
            raise HandoffValidationError(
                "Partition SHA-256 values differ between split and quality artifacts."
            )
    quality_readiness = quality_evidence.get("readiness", {})
    preparation_readiness = preparation_manifest.get("readiness", {})
    readiness_contracts = [
        (quality_readiness, False),
        (preparation_readiness, False),
    ]
    if handoff_manifest is not None:
        readiness_contracts.append(
            (
                handoff_manifest.get("readiness", {}),
                feature_manifest.get("problem_type") == "continuous_regression",
            )
        )
        consumer_contract = handoff_manifest.get("consumer_contract", {})
        if (
            not isinstance(consumer_contract, Mapping)
            or (
                "test_partition_sealed" in consumer_contract
                and consumer_contract.get("test_partition_sealed") is not True
            )
            or (
                "test_partition_evaluated" in consumer_contract
                and consumer_contract.get("test_partition_evaluated") is not False
            )
            or (
                "model_selection_must_not_resplit" in consumer_contract
                and consumer_contract.get("model_selection_must_not_resplit") is not True
            )
        ):
            raise HandoffValidationError(
                "Preparation consumer contract does not preserve the sealed test partition."
            )
    for readiness, require_sealing in readiness_contracts:
        if not isinstance(readiness, Mapping):
            raise HandoffValidationError("Preparation readiness contract is invalid.")
        if readiness.get("educational_model_selection_ready") is not True:
            raise HandoffValidationError(
                "Preparation readiness does not enable educational model selection."
            )
        if require_sealing and readiness.get("test_partition_sealed") is not True:
            raise HandoffValidationError(
                "Preparation readiness must keep the test partition sealed."
            )
        if (
            "test_partition_evaluated" in readiness
            and readiness.get("test_partition_evaluated") is not False
        ):
            raise HandoffValidationError(
                "Preparation handoff must declare the test partition unevaluated."
            )
        if readiness.get("model_selected", False) is not False:
            raise HandoffValidationError(
                "Preparation handoff cannot declare a selected model."
            )
        if readiness.get("final_model_trained", False) is not False:
            raise HandoffValidationError(
                "Preparation handoff cannot declare a trained final model."
            )
        if (
            "operational_modeling_ready" in readiness
            and readiness.get("operational_modeling_ready") is not False
        ):
            raise HandoffValidationError(
                "Preparation handoff cannot claim operational modeling readiness."
            )

    manifest_items: list[tuple[str, Mapping[str, Any]]] = [
        ("preparation_manifest", preparation_manifest),
        ("feature_manifest", feature_manifest),
        ("split_manifest", split_manifest),
        ("quality_evidence", quality_evidence),
    ]
    if handoff_manifest is not None:
        manifest_items.append(("preparation_handoff", handoff_manifest))
    manifests = tuple(manifest_items)
    if _model_selection_safe:
        test_relative = partition_paths["test"]
        return ModelSelectionPreparationHandoff(
            _train=loaded_partitions["train"],
            _validation=loaded_partitions["validation"],
            _manifests=manifests,
            _prepared_integrity_reference=_tuple_mapping({
                "path": prepared_relative,
                "sha256": expected_prepared_sha,
                "row_count": preparation_manifest.get("prepared_row_count"),
            }),
            _test_integrity_reference=_tuple_mapping({
                "path": test_relative,
                "sha256": partition_sha["test"],
                "row_count": split_manifest.get("row_counts", {}).get("test"),
                "sealed": True,
                "evaluated": False,
            }),
        )
    return PreparationHandoff(
        _prepared=prepared,
        _train=loaded_partitions["train"],
        _validation=loaded_partitions["validation"],
        _test=loaded_partitions["test"],
        _manifests=manifests,
    )


def load_and_validate_preparation_for_model_selection(**kwargs: Any) -> ModelSelectionPreparationHandoff:
    """Authenticate preparation while parsing only train and validation CSVs."""
    result = load_and_validate_preparation_handoff(
        **kwargs, _model_selection_safe=True
    )
    assert isinstance(result, ModelSelectionPreparationHandoff)
    return result
