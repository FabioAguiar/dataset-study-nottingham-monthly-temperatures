"""Generic, deterministic final-model training and bundle materialization.

The module is dataset-agnostic. Dataset roles, paths, selected estimator,
hyperparameters, thresholds, and upstream contracts are supplied by callers.
It enforces a train+validation final fit, a frozen decision contract before test
access, one probability evaluation of test, complete-pipeline serialization,
and atomic/idempotent artifact persistence.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scripts.prepare_data import load_and_validate_preparation_handoff
from scripts.select_models import (
    build_candidate_pipeline,
    compute_multiclass_metrics,
    load_and_validate_model_selection_handoff,
)


FINAL_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "final-pipeline.joblib",
    "final-model-manifest.json",
    "final-test-evidence.json",
    "inference-bundle.json",
    "final-model-handoff.json",
)

SCHEMAS: Mapping[str, tuple[str, str]] = {
    "final-model-manifest.json": (
        "final-model-manifest.v1",
        "final_model_manifest",
    ),
    "final-test-evidence.json": (
        "final-test-evidence.v1",
        "final_test_evidence",
    ),
    "inference-bundle.json": ("inference-bundle.v1", "inference_bundle"),
    "final-model-handoff.json": (
        "final-model-handoff.v1",
        "final_model_handoff",
    ),
}

_VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "generated_at",
        "timestamp",
        "fit_duration_seconds",
        "duration_seconds",
        "byte_sha256",
    }
)
_REPEATED_PROFILE_INTERPRETATION_ALIASES: Mapping[str, str] = {
    "repeated profiles do not prove duplicate identity or leakage": (
        "Repeated-profile evidence does not prove duplicate identity or leakage"
    ),
    "repeated profiles do not prove duplicate identity or leakage.": (
        "Repeated-profile evidence does not prove duplicate identity or leakage."
    ),
    "Repeated-profile analysis is sensitivity evidence and does not prove duplicate identity or leakage.": (
        "Repeated-profile evidence does not prove duplicate identity or leakage."
    ),
}


class FinalizationError(RuntimeError):
    """Base error for final-model operations."""


class FinalizationContractError(FinalizationError, ValueError):
    """Raised when a supplied finalization contract is invalid."""


class UpstreamHandoffError(FinalizationError):
    """Raised when an upstream handoff is invalid or inconsistent."""


class TestAccessError(FinalizationError):
    """Raised when test is accessed before the frozen/fitted gate."""


class DuplicateTestEvaluationError(FinalizationError):
    """Raised when test probability evaluation is attempted twice."""


class SerializationValidationError(FinalizationError):
    """Raised when a serialized pipeline fails integrity validation."""


class ArtifactConflictError(FinalizationError):
    """Raised for partial or semantically divergent final artifact sets."""


class UntrustedArtifactError(FinalizationError):
    """Raised when binary artifact trust/integrity cannot be established."""


@dataclass(frozen=True, slots=True)
class FrozenFinalizationContract:
    """Immutable model, feature, target, partition, and threshold decisions."""

    dataset_slug: str
    model_id: str
    model_family: str
    hyperparameters: tuple[tuple[str, Any], ...]
    random_state: int | None
    feature_columns: tuple[str, ...]
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    target_column: str
    target_classes: tuple[Any, ...]
    target_encoding: tuple[tuple[Any, int], ...]
    positive_class: Any
    educational_threshold: float
    threshold_scenario_id: str
    threshold_selection_partition: str
    preprocessing_contract: tuple[tuple[str, Any], ...]
    training_partitions: tuple[str, ...] = ("train", "validation")
    evaluation_partition: str = "test"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_slug": self.dataset_slug,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "hyperparameters": dict(self.hyperparameters),
            "random_state": self.random_state,
            "feature_columns": list(self.feature_columns),
            "numerical_features": list(self.numerical_features),
            "categorical_features": list(self.categorical_features),
            "identifier_columns": list(self.identifier_columns),
            "target_column": self.target_column,
            "target_classes": list(self.target_classes),
            "target_encoding": dict(self.target_encoding),
            "positive_class": self.positive_class,
            "positive_encoded_label": dict(self.target_encoding)[self.positive_class],
            "educational_threshold": self.educational_threshold,
            "threshold_scenario_id": self.threshold_scenario_id,
            "threshold_selection_partition": self.threshold_selection_partition,
            "preprocessing_contract": dict(self.preprocessing_contract),
            "training_partitions": list(self.training_partitions),
            "evaluation_partition": self.evaluation_partition,
            "operational_threshold": "unresolved",
            "operational_validity": "unconfirmed",
        }


@dataclass(frozen=True, slots=True)
class FinalTrainingData:
    """Defensive final train+validation features and encoded target."""

    _features: pd.DataFrame
    _target: pd.Series
    class_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)

    @property
    def row_count(self) -> int:
        return int(len(self._features))


@dataclass(frozen=True, slots=True)
class TestPartitionData:
    """Defensive test features and encoded target loaded after the access gate."""

    _features: pd.DataFrame
    _target: pd.Series
    row_count: int
    class_counts: tuple[tuple[str, int], ...]
    partition_path: str
    partition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)


@dataclass(slots=True)
class EvaluationGuard:
    """Per-execution guard preventing a second test probability evaluation."""

    evaluated: bool = False
    probability_call_count: int = 0


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    """Aggregate-only final test results derived from one probability vector."""

    probability_metrics: Mapping[str, float]
    threshold_default: Mapping[str, Any]
    threshold_educational: Mapping[str, Any]
    precision_recall_curve: Mapping[str, list[float]]
    roc_curve: Mapping[str, list[float]]
    calibration_curve: Mapping[str, list[float]]
    unknown_categories_report: Mapping[str, list[Any]]
    generalization_deltas: Mapping[str, float]
    probability_sha256: str
    test_probability_evaluation_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return _deepcopy(
            {
                "probability_metrics": self.probability_metrics,
                "threshold_default": self.threshold_default,
                "threshold_educational": self.threshold_educational,
                "precision_recall_curve": self.precision_recall_curve,
                "roc_curve": self.roc_curve,
                "calibration_curve": self.calibration_curve,
                "unknown_categories_report": self.unknown_categories_report,
                "generalization_deltas": self.generalization_deltas,
                "probability_sha256": self.probability_sha256,
                "test_probability_evaluation_count": self.test_probability_evaluation_count,
            }
        )


@dataclass(frozen=True, slots=True)
class SerializedPipeline:
    """Validated staging serialization metadata."""

    path: Path
    byte_sha256: str
    state_fingerprint: str
    descriptor: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Outcome of an atomic final-artifact transaction."""

    output_directory: Path
    created: tuple[str, ...]
    replaced: tuple[str, ...]
    idempotent: bool
    byte_sha256: Mapping[str, str]
    semantic_sha256: Mapping[str, str]


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise FinalizationContractError("NaN is not allowed in canonical artifacts.")
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in _REPEATED_PROFILE_INTERPRETATION_ALIASES.items():
            result = result.replace(old, new)
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return _jsonable(value)


def semantic_fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_strip_volatile(value)))


def _strip_volatile_without_text_aliases(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile_without_text_aliases(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile_without_text_aliases(item) for item in value]
    return _jsonable(value)


def _replace_text_alias(value: Any, *, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, Mapping):
        return {
            key: _replace_text_alias(item, old=old, new=new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_text_alias(item, old=old, new=new) for item in value]
    return value


def _semantic_fingerprint_matches_declared(
    payload: Mapping[str, Any],
    declared: str | None,
) -> bool:
    if not isinstance(declared, str):
        return False
    if semantic_fingerprint(payload) == declared:
        return True
    for old, new in _REPEATED_PROFILE_INTERPRETATION_ALIASES.items():
        legacy_payload = _replace_text_alias(payload, old=new, new=old)
        legacy_fingerprint = sha256_bytes(
            canonical_json_bytes(_strip_volatile_without_text_aliases(legacy_payload))
        )
        if legacy_fingerprint == declared:
            return True
    return False


def _require_relative_path(value: str | Path, *, field: str) -> str:
    rendered = Path(value).as_posix() if isinstance(value, Path) else str(value)
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or ".." in pure.parts:
        raise FinalizationContractError(f"{field} must be project-relative: {rendered}")
    if len(rendered) >= 3 and rendered[1:3] in {":/", ":\\"}:
        raise FinalizationContractError(f"{field} must not be absolute: {rendered}")
    return pure.as_posix()


def _validate_paths_recursively(value: Any, *, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if (str(key) == "path" or str(key).endswith("_path")) and item is not None:
                _require_relative_path(item, field=field)
            elif str(key).endswith("_paths") and isinstance(item, Sequence) and not isinstance(item, str):
                for index, path in enumerate(item):
                    _require_relative_path(path, field=f"{field}[{index}]")
            else:
                _validate_paths_recursively(item, prefix=field)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_paths_recursively(item, prefix=f"{prefix}[{index}]")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


# ---------------------------------------------------------------------------
# Upstream gates and frozen contract
# ---------------------------------------------------------------------------


def validate_upstream_handoff_contracts(
    *,
    project_root: str | Path,
    preparation_paths: Mapping[str, str | Path],
    model_selection_handoff_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load and fully validate the preparation and model-selection handoffs."""

    required = {
        "preparation_manifest_path",
        "feature_manifest_path",
        "split_manifest_path",
        "quality_evidence_path",
    }
    if set(preparation_paths) != required:
        raise UpstreamHandoffError(
            f"Preparation path keys mismatch: expected={sorted(required)}"
        )
    preparation = load_and_validate_preparation_handoff(
        project_root=project_root,
        **{key: preparation_paths[key] for key in sorted(required)},
    )
    selection = load_and_validate_model_selection_handoff(
        project_root=project_root,
        handoff_path=model_selection_handoff_path,
    )
    validate_finalization_contract(selection)
    manifests = preparation.manifests
    feature_manifest = manifests["feature_manifest"]
    if list(selection["feature_columns"]) != list(feature_manifest["feature_columns"]):
        raise UpstreamHandoffError("Feature order differs between upstream handoffs.")
    if dict(selection["target_encoding"]) != dict(feature_manifest["target_encoding_contract"]):
        raise UpstreamHandoffError("Target encoding differs between upstream handoffs.")
    if selection["positive_class"] != feature_manifest["positive_target_class"]:
        raise UpstreamHandoffError("Positive class differs between upstream handoffs.")
    return preparation, _deepcopy(selection)


def validate_finalization_contract(model_selection_handoff: Mapping[str, Any]) -> None:
    """Validate readiness and absolute non-operational constraints from notebook 03."""

    if model_selection_handoff.get("schema_version") != "model-selection-handoff.v1":
        raise FinalizationContractError("Invalid model-selection handoff schema.")
    if model_selection_handoff.get("artifact_type") != "model_selection_handoff":
        raise FinalizationContractError("Invalid model-selection handoff artifact type.")
    if model_selection_handoff.get("test_partition_sealed") is not True:
        raise FinalizationContractError("Test must be sealed at finalization input.")
    if model_selection_handoff.get("test_partition_evaluated") is not False:
        raise FinalizationContractError("Test must not have been evaluated upstream.")
    readiness = model_selection_handoff.get("readiness", {})
    if readiness.get("final_model_training_ready") is not True:
        raise FinalizationContractError("final_model_training_ready must be true.")
    if model_selection_handoff.get("final_model_trained") is not False:
        raise FinalizationContractError("Final model must not be trained upstream.")
    if model_selection_handoff.get("model_artifact") is not None:
        raise FinalizationContractError("Upstream model artifact must be absent.")
    if model_selection_handoff.get("model_artifact_materialized") is not False:
        raise FinalizationContractError("Upstream model artifact must be unmaterialized.")
    if model_selection_handoff.get("model_bundle_materialized") is not False:
        raise FinalizationContractError("Upstream bundle must be unmaterialized.")
    if model_selection_handoff.get("operational_modeling_ready") is not False:
        raise FinalizationContractError("Operational modeling readiness must remain false.")
    if model_selection_handoff.get("operational_validity") != "unconfirmed":
        raise FinalizationContractError("Operational validity must remain unconfirmed.")
    if model_selection_handoff.get("operational_threshold") != "unresolved":
        raise FinalizationContractError("Operational threshold must remain unresolved.")
    _validate_paths_recursively(model_selection_handoff)


def validate_frozen_model_contract(
    contract: FrozenFinalizationContract,
    *,
    expected_model_id: str | None = None,
) -> None:
    """Validate an immutable finalization decision set."""

    if not contract.dataset_slug or not contract.model_id or not contract.model_family:
        raise FinalizationContractError("Dataset and model identity are required.")
    if expected_model_id is not None and contract.model_id != expected_model_id:
        raise FinalizationContractError("Frozen model differs from selected model.")
    features = list(contract.feature_columns)
    if not features or len(features) != len(set(features)):
        raise FinalizationContractError("Feature columns must be unique and non-empty.")
    if set(contract.numerical_features) & set(contract.categorical_features):
        raise FinalizationContractError("Numerical and categorical roles overlap.")
    if set(contract.numerical_features) | set(contract.categorical_features) != set(features):
        raise FinalizationContractError("Feature roles do not cover the frozen feature set.")
    prohibited = set(contract.identifier_columns) | {contract.target_column}
    if prohibited & set(features):
        raise FinalizationContractError("Identifiers/target cannot enter feature columns.")
    encoding = dict(contract.target_encoding)
    if set(encoding) != set(contract.target_classes) or sorted(encoding.values()) != [0, 1]:
        raise FinalizationContractError("Target encoding must map exactly two classes to 0/1.")
    if contract.positive_class not in encoding or encoding[contract.positive_class] != 1:
        raise FinalizationContractError("Positive class must be encoded as 1.")
    if not 0.0 <= float(contract.educational_threshold) <= 1.0:
        raise FinalizationContractError("Educational threshold must be in [0, 1].")
    if contract.threshold_selection_partition != "validation":
        raise FinalizationContractError("Educational threshold origin must be validation.")


def freeze_finalization_decisions(
    *,
    dataset_slug: str,
    model_selection_handoff: Mapping[str, Any],
    identifier_columns: Sequence[str],
    target_column: str,
    target_classes: Sequence[Any],
    estimator_random_state: int | None,
) -> FrozenFinalizationContract:
    """Freeze all decisions before fit and any evaluative test access."""

    validate_finalization_contract(model_selection_handoff)
    if model_selection_handoff.get("dataset_slug") != dataset_slug:
        raise FinalizationContractError("Dataset slug differs from model-selection handoff.")
    threshold = model_selection_handoff["selected_educational_threshold"]
    contract = FrozenFinalizationContract(
        dataset_slug=str(dataset_slug),
        model_id=str(model_selection_handoff["selected_model_id"]),
        model_family=str(model_selection_handoff["selected_model_family"]),
        hyperparameters=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_hyperparameters"]).items())
        ),
        random_state=estimator_random_state,
        feature_columns=tuple(model_selection_handoff["feature_columns"]),
        numerical_features=tuple(model_selection_handoff["numerical_features"]),
        categorical_features=tuple(model_selection_handoff["categorical_features"]),
        identifier_columns=tuple(identifier_columns),
        target_column=str(target_column),
        target_classes=tuple(target_classes),
        target_encoding=tuple(
            (key, int(value))
            for key, value in model_selection_handoff["target_encoding"].items()
        ),
        positive_class=model_selection_handoff["positive_class"],
        educational_threshold=float(threshold["threshold"]),
        threshold_scenario_id=str(threshold["scenario_id"]),
        threshold_selection_partition="validation",
        preprocessing_contract=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_preprocessing_contract"]).items())
        ),
    )
    validate_frozen_model_contract(contract, expected_model_id=model_selection_handoff["selected_model_id"])
    return contract


# ---------------------------------------------------------------------------
# Partition roles and pipeline
# ---------------------------------------------------------------------------


def validate_final_partition_roles(
    frame: pd.DataFrame,
    *,
    contract: FrozenFinalizationContract,
    partition_name: str,
) -> None:
    """Validate exact columns, target classes, and feature order for one partition."""

    expected = [*contract.identifier_columns, *contract.feature_columns, contract.target_column]
    if list(frame.columns) != expected:
        raise FinalizationContractError(
            f"{partition_name} column order mismatch. expected={expected}, observed={list(frame.columns)}"
        )
    observed = set(frame[contract.target_column].dropna().unique().tolist())
    if observed != set(contract.target_classes):
        raise FinalizationContractError(
            f"{partition_name} target classes mismatch: {sorted(map(str, observed))}"
        )


def assemble_final_training_data(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    contract: FrozenFinalizationContract,
) -> FinalTrainingData:
    """Concatenate train+validation defensively; test is intentionally not accepted."""

    train_copy = train.copy(deep=True)
    validation_copy = validation.copy(deep=True)
    validate_final_partition_roles(train_copy, contract=contract, partition_name="train")
    validate_final_partition_roles(validation_copy, contract=contract, partition_name="validation")
    combined = pd.concat([train_copy, validation_copy], axis=0, ignore_index=True)
    features = combined.loc[:, list(contract.feature_columns)].copy(deep=True)
    encoded = combined[contract.target_column].map(dict(contract.target_encoding))
    if encoded.isna().any():
        raise FinalizationContractError("Target encoding produced missing values.")
    target = encoded.astype("int64")
    counts = tuple(
        (str(key), int(value))
        for key, value in combined[contract.target_column].value_counts().sort_index().items()
    )
    if set(target.unique().tolist()) != {0, 1}:
        raise FinalizationContractError("Final training target must contain encoded classes 0 and 1.")
    return FinalTrainingData(features, target, counts)


def reconstruct_selected_pipeline(
    *,
    estimator: BaseEstimator,
    contract: FrozenFinalizationContract,
) -> Pipeline:
    """Reconstruct the selected unfitted pipeline from a supplied estimator instance."""

    model = clone(estimator)
    if model.__class__.__name__ != contract.model_family:
        raise FinalizationContractError(
            f"Estimator family differs from frozen contract: expected={contract.model_family}, observed={model.__class__.__name__}"
        )
    params = dict(contract.hyperparameters)
    accepted = model.get_params(deep=True)
    model_params: dict[str, Any] = {}
    for key, value in params.items():
        if not key.startswith("model__"):
            raise FinalizationContractError(f"Selected hyperparameter lacks model__ prefix: {key}")
        model_key = key.removeprefix("model__")
        if model_key not in accepted:
            raise FinalizationContractError(f"Estimator does not accept selected parameter: {model_key}")
        model_params[model_key] = value
    if contract.random_state is not None and "random_state" in accepted:
        model_params["random_state"] = contract.random_state
    model.set_params(**model_params)
    pipeline = build_candidate_pipeline(
        estimator=model,
        numerical_features=contract.numerical_features,
        categorical_features=contract.categorical_features,
        scale_numerical=False,
    )
    verify_pipeline_contract(pipeline, contract=contract, require_fitted=False)
    return pipeline


def _is_fitted(pipeline: Pipeline) -> bool:
    try:
        check_is_fitted(pipeline)
        check_is_fitted(pipeline.named_steps["preprocess"])
        check_is_fitted(pipeline.named_steps["model"])
        return True
    except Exception:
        return False


def verify_pipeline_contract(
    pipeline: Pipeline,
    *,
    contract: FrozenFinalizationContract,
    require_fitted: bool,
) -> None:
    """Verify pipeline structure, preprocessing semantics, parameters, and fitted state."""

    if not isinstance(pipeline, Pipeline):
        raise FinalizationContractError("Final model artifact must be an sklearn Pipeline.")
    if list(pipeline.named_steps) != ["preprocess", "model"]:
        raise FinalizationContractError("Pipeline steps must be preprocess then model.")
    preprocess = pipeline.named_steps["preprocess"]
    if not isinstance(preprocess, ColumnTransformer):
        raise FinalizationContractError("Preprocess step must be a ColumnTransformer.")
    if preprocess.remainder != "drop" or float(preprocess.sparse_threshold) != 0.0:
        raise FinalizationContractError("ColumnTransformer must drop remainder and force dense output.")
    transformers = {name: (transformer, list(columns)) for name, transformer, columns in preprocess.transformers}
    if transformers.get("numerical", (None, []))[0] != "passthrough":
        raise FinalizationContractError("Selected family must use numerical passthrough.")
    if transformers.get("numerical", (None, []))[1] != list(contract.numerical_features):
        raise FinalizationContractError("Numerical feature order differs from frozen contract.")
    categorical_transformer, categorical_columns = transformers.get("categorical", (None, []))
    if categorical_columns != list(contract.categorical_features):
        raise FinalizationContractError("Categorical feature order differs from frozen contract.")
    if categorical_transformer is None or categorical_transformer.__class__.__name__ != "OneHotEncoder":
        raise FinalizationContractError("Categorical transformer must be OneHotEncoder.")
    if categorical_transformer.handle_unknown != "ignore" or categorical_transformer.drop is not None:
        raise FinalizationContractError("OneHotEncoder policy differs from frozen contract.")
    if hasattr(categorical_transformer, "sparse_output") and categorical_transformer.sparse_output is not False:
        raise FinalizationContractError("OneHotEncoder output must be dense.")
    if hasattr(categorical_transformer, "sparse") and not hasattr(categorical_transformer, "sparse_output") and categorical_transformer.sparse is not False:
        raise FinalizationContractError("OneHotEncoder output must be dense.")
    model_params = pipeline.named_steps["model"].get_params(deep=False)
    for key, expected in contract.hyperparameters:
        name = key.removeprefix("model__")
        if model_params.get(name) != expected:
            raise FinalizationContractError(f"Model parameter mismatch: {name}")
    if contract.random_state is not None and "random_state" in model_params:
        if model_params["random_state"] != contract.random_state:
            raise FinalizationContractError("Estimator random_state mismatch.")
    fitted = _is_fitted(pipeline)
    if require_fitted and not fitted:
        raise FinalizationContractError("Pipeline must be fitted.")
    if not require_fitted and fitted:
        raise FinalizationContractError("Pipeline must be unfitted before final fit.")


def validate_test_access_gate(
    *,
    contract: FrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    test_path: str | Path,
    expected_sha256: str,
    project_root: str | Path,
) -> Path:
    """Authorize test loading only after decisions are frozen and final fit completed."""

    validate_frozen_model_contract(contract)
    verify_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    relative = _require_relative_path(test_path, field="test_path")
    root = Path(project_root).resolve()
    absolute = (root / relative).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise TestAccessError("Test path escapes project root.") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"Test partition is missing: {relative}")
    observed = sha256_file(absolute)
    if observed != expected_sha256:
        raise TestAccessError(
            f"Test SHA-256 mismatch: expected={expected_sha256}, observed={observed}"
        )
    return absolute


def load_test_partition_after_freeze(
    *,
    project_root: str | Path,
    test_path: str | Path,
    expected_sha256: str,
    fitted_pipeline: Pipeline,
    contract: FrozenFinalizationContract,
) -> TestPartitionData:
    """Load test directly from its validated path after the final fit gate."""

    absolute = validate_test_access_gate(
        contract=contract,
        fitted_pipeline=fitted_pipeline,
        test_path=test_path,
        expected_sha256=expected_sha256,
        project_root=project_root,
    )
    frame = pd.read_csv(absolute)
    validate_final_partition_roles(frame, contract=contract, partition_name="test")
    features = frame.loc[:, list(contract.feature_columns)].copy(deep=True)
    encoded = frame[contract.target_column].map(dict(contract.target_encoding))
    if encoded.isna().any():
        raise TestAccessError("Test target encoding produced missing values.")
    counts = tuple(
        (str(key), int(value))
        for key, value in frame[contract.target_column].value_counts().sort_index().items()
    )
    return TestPartitionData(
        features,
        encoded.astype("int64"),
        int(len(frame)),
        counts,
        _require_relative_path(test_path, field="test_path"),
        expected_sha256,
    )


# ---------------------------------------------------------------------------
# One-time evaluation and aggregate diagnostics
# ---------------------------------------------------------------------------


def compute_fbeta(y_true: Sequence[int], y_pred: Sequence[int], *, beta: float) -> float:
    return float(fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0))


def evaluate_fixed_threshold(
    *, y_true: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Evaluate one fixed threshold without selecting or modifying it."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise FinalizationContractError("Threshold must be in [0, 1].")
    probability_series = pd.Series(probabilities, dtype="float64").copy(deep=True)
    labels = (probability_series.to_numpy(copy=True) >= float(threshold)).astype(int)
    true = pd.Series(y_true, dtype="int64").to_numpy(copy=True)
    tn, fp, fn, tp = confusion_matrix(true, labels, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(true, labels, pos_label=1, zero_division=0)),
        "recall": float(recall_score(true, labels, pos_label=1, zero_division=0)),
        "f1": float(f1_score(true, labels, pos_label=1, zero_division=0)),
        "f2": compute_fbeta(true, labels, beta=2.0),
        "balanced_accuracy": float(balanced_accuracy_score(true, labels)),
        "accuracy_contextual": float(accuracy_score(true, labels)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positive_count": int(labels.sum()),
        "predicted_positive_rate": float(labels.mean()),
    }


def compute_probability_metrics(
    *, y_true: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    true = pd.Series(y_true, dtype="int64").to_numpy(copy=True)
    prob = pd.Series(probabilities, dtype="float64").to_numpy(copy=True)
    return {
        "average_precision": float(average_precision_score(true, prob)),
        "roc_auc": float(roc_auc_score(true, prob)),
        "log_loss": float(log_loss(true, prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(true, prob)),
    }


def compute_generalization_deltas(
    *,
    validation_metrics: Mapping[str, Any],
    test_probability_metrics: Mapping[str, float],
    test_default_threshold: Mapping[str, Any],
    test_educational_threshold: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
) -> dict[str, float]:
    """Compute descriptive test-minus-validation deltas; never drive decisions."""

    return {
        "average_precision_test_minus_validation": float(test_probability_metrics["average_precision"] - validation_metrics["average_precision"]),
        "roc_auc_test_minus_validation": float(test_probability_metrics["roc_auc"] - validation_metrics["roc_auc"]),
        "brier_score_test_minus_validation": float(test_probability_metrics["brier_score"] - validation_metrics["brier_score"]),
        "log_loss_test_minus_validation": float(test_probability_metrics["log_loss"] - validation_metrics["log_loss"]),
        "default_precision_test_minus_validation": float(test_default_threshold["precision"] - validation_metrics["precision"]),
        "default_recall_test_minus_validation": float(test_default_threshold["recall"] - validation_metrics["recall"]),
        "educational_precision_test_minus_validation": float(test_educational_threshold["precision"] - validation_educational_threshold["precision"]),
        "educational_recall_test_minus_validation": float(test_educational_threshold["recall"] - validation_educational_threshold["recall"]),
    }


def report_unknown_categories(
    *, fitted_pipeline: Pipeline, features: pd.DataFrame, categorical_features: Sequence[str]
) -> dict[str, list[Any]]:
    """Report categories in an evaluation frame absent from fitted vocabularies."""

    preprocess = fitted_pipeline.named_steps["preprocess"]
    categorical = preprocess.named_transformers_["categorical"]
    report: dict[str, list[Any]] = {}
    for column, fitted_values in zip(categorical_features, categorical.categories_, strict=True):
        known = set(_jsonable(list(fitted_values)))
        observed = set(_jsonable(features[column].dropna().unique().tolist()))
        unknown = sorted(observed - known, key=lambda value: str(value))
        if unknown:
            report[str(column)] = unknown
    return report


def _positive_probability_index(pipeline: Pipeline) -> int:
    classes = list(_jsonable(pipeline.named_steps["model"].classes_))
    if 1 not in classes:
        raise FinalizationContractError("Fitted estimator does not expose encoded positive class 1.")
    return classes.index(1)


def evaluate_final_model_once(
    *,
    fitted_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    educational_threshold: float,
    educational_recall_target: float,
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
    categorical_features: Sequence[str],
    guard: EvaluationGuard,
) -> FinalEvaluation:
    """Call predict_proba exactly once and derive all final aggregate results."""

    if guard.evaluated:
        raise DuplicateTestEvaluationError("Test probability evaluation already occurred.")
    if not _is_fitted(fitted_pipeline):
        raise TestAccessError("Final pipeline must be fitted before test evaluation.")
    features = x_test.copy(deep=True)
    target = y_test.copy(deep=True)
    matrix = fitted_pipeline.predict_proba(features)  # exactly one test call
    guard.probability_call_count += 1
    if guard.probability_call_count != 1:
        raise DuplicateTestEvaluationError("predict_proba call count exceeded one.")
    probabilities = matrix[:, _positive_probability_index(fitted_pipeline)].copy()
    guard.evaluated = True

    probability_metrics = compute_probability_metrics(y_true=target, probabilities=probabilities)
    default = evaluate_fixed_threshold(y_true=target, probabilities=probabilities, threshold=0.5)
    educational = evaluate_fixed_threshold(
        y_true=target, probabilities=probabilities, threshold=educational_threshold
    )
    educational["educational_recall_target"] = float(educational_recall_target)
    educational["educational_recall_target_satisfied"] = bool(
        educational["recall"] >= educational_recall_target
    )

    precision, recall, pr_thresholds = precision_recall_curve(target, probabilities, pos_label=1)
    fpr, tpr, roc_thresholds = roc_curve(target, probabilities, pos_label=1)
    fraction_positive, mean_predicted = calibration_curve(
        target, probabilities, n_bins=10, strategy="quantile"
    )
    deltas = compute_generalization_deltas(
        validation_metrics=validation_metrics,
        test_probability_metrics=probability_metrics,
        test_default_threshold=default,
        test_educational_threshold=educational,
        validation_educational_threshold=validation_educational_threshold,
    )
    return FinalEvaluation(
        probability_metrics=probability_metrics,
        threshold_default=default,
        threshold_educational=educational,
        precision_recall_curve={
            "precision": _jsonable(precision),
            "recall": _jsonable(recall),
            "thresholds": _jsonable(pr_thresholds),
        },
        roc_curve={
            "false_positive_rate": _jsonable(fpr),
            "true_positive_rate": _jsonable(tpr),
            "thresholds": _jsonable(roc_thresholds),
        },
        calibration_curve={
            "mean_predicted_probability": _jsonable(mean_predicted),
            "fraction_of_positives": _jsonable(fraction_positive),
            "n_bins": 10,
            "strategy": "quantile",
        },
        unknown_categories_report=report_unknown_categories(
            fitted_pipeline=fitted_pipeline,
            features=features,
            categorical_features=categorical_features,
        ),
        generalization_deltas=deltas,
        probability_sha256=sha256_bytes(pd.Series(probabilities).to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# Fitted-state descriptor and serialization
# ---------------------------------------------------------------------------


def describe_fitted_pipeline(
    *,
    pipeline: Pipeline,
    contract: FrozenFinalizationContract,
    training_data: FinalTrainingData,
    train_sha256: str,
    validation_sha256: str,
    sample_size: int = 32,
) -> dict[str, Any]:
    """Build a canonical JSON descriptor of fitted pipeline state."""

    verify_pipeline_contract(pipeline, contract=contract, require_fitted=True)
    preprocess = pipeline.named_steps["preprocess"]
    categorical = preprocess.named_transformers_["categorical"]
    names = preprocess.get_feature_names_out().tolist()
    sample = training_data.features.iloc[: min(sample_size, training_data.row_count)].copy(deep=True)
    probabilities = pipeline.predict_proba(sample)[:, _positive_probability_index(pipeline)].copy()
    descriptor = {
        "pipeline_class": f"{pipeline.__class__.__module__}.{pipeline.__class__.__name__}",
        "steps": list(pipeline.named_steps),
        "model_class": f"{pipeline.named_steps['model'].__class__.__module__}.{pipeline.named_steps['model'].__class__.__name__}",
        "model_parameters": {
            key: pipeline.named_steps["model"].get_params(deep=False).get(key.removeprefix("model__"))
            for key, _ in contract.hyperparameters
        },
        "random_state": contract.random_state,
        "feature_order": list(contract.feature_columns),
        "categorical_vocabularies": {
            column: _jsonable(values)
            for column, values in zip(contract.categorical_features, categorical.categories_, strict=True)
        },
        "transformed_feature_names": names,
        "estimator_classes": _jsonable(pipeline.named_steps["model"].classes_),
        "transformed_feature_count": len(names),
        "training_partition_sha256": {
            "train": train_sha256,
            "validation": validation_sha256,
        },
        "training_row_count": training_data.row_count,
        "training_target_class_counts": dict(training_data.class_counts),
        "sample_probability_sha256": sha256_bytes(
            pd.Series(probabilities).to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
        ),
        "sample_size": int(len(sample)),
        "runtime_major_minor": {
            key: ".".join(value.split(".")[:2])
            for key, value in runtime_versions().items()
        },
    }
    return _jsonable(descriptor)


def compute_fitted_model_fingerprint(descriptor: Mapping[str, Any]) -> str:
    return semantic_fingerprint(descriptor)


def serialize_pipeline_to_staging(
    *, pipeline: Pipeline, staging_path: str | Path
) -> str:
    """Serialize a complete fitted Pipeline and return its byte SHA-256."""

    if not isinstance(pipeline, Pipeline) or not _is_fitted(pipeline):
        raise SerializationValidationError("Only a fitted sklearn Pipeline can be serialized.")
    path = Path(staging_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return sha256_file(path)


def validate_serialized_pipeline(
    *,
    staging_path: str | Path,
    expected_sha256: str,
    contract: FrozenFinalizationContract,
    reference_pipeline: Pipeline,
    validation_sample: pd.DataFrame,
) -> Pipeline:
    """Reload and verify a trusted staging joblib without using test data."""

    path = Path(staging_path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise SerializationValidationError("Serialized pipeline SHA-256 mismatch.")
    loaded = joblib.load(path)
    verify_pipeline_contract(loaded, contract=contract, require_fitted=True)
    classes = list(_jsonable(loaded.named_steps["model"].classes_))
    if classes != [0, 1]:
        raise SerializationValidationError(f"Unexpected estimator classes: {classes}")
    sample = validation_sample.copy(deep=True)
    expected = reference_pipeline.predict_proba(sample)
    observed_probabilities = loaded.predict_proba(sample)
    if expected.shape != observed_probabilities.shape or not pd.DataFrame(expected).equals(pd.DataFrame(observed_probabilities)):
        # Exact equality is expected for same-runtime joblib round-trip.
        import numpy as np

        if not np.allclose(expected, observed_probabilities, rtol=0.0, atol=0.0):
            raise SerializationValidationError("Round-trip probabilities differ.")
    return loaded


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------


def build_final_test_evidence(
    *,
    contract: FrozenFinalizationContract,
    test_partition: TestPartitionData,
    evaluation: FinalEvaluation,
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "final-test-evidence.v1",
        "artifact_type": "final_test_evidence",
        "dataset_slug": contract.dataset_slug,
        "partition": "test",
        "partition_path": test_partition.partition_path,
        "partition_sha256": test_partition.partition_sha256,
        "row_count": test_partition.row_count,
        "class_counts": dict(test_partition.class_counts),
        "test_loaded_only_after_final_fit": True,
        "test_probability_evaluation_count": 1,
        "positive_class": contract.positive_class,
        "encoded_positive_label": 1,
        "probability_metrics": _deepcopy(evaluation.probability_metrics),
        "threshold_0_50": _deepcopy(evaluation.threshold_default),
        "educational_threshold": _deepcopy(evaluation.threshold_educational),
        "precision_recall_curve": _deepcopy(evaluation.precision_recall_curve),
        "roc_curve": _deepcopy(evaluation.roc_curve),
        "calibration_curve": _deepcopy(evaluation.calibration_curve),
        "unknown_categories_report": _deepcopy(evaluation.unknown_categories_report),
        "selected_validation_metrics": _deepcopy(validation_metrics),
        "selected_validation_educational_threshold": _deepcopy(validation_educational_threshold),
        "generalization_deltas": _deepcopy(evaluation.generalization_deltas),
        "frozen_model_contract": {
            "dataset_slug": contract.dataset_slug,
            "model_id": contract.model_id,
            "model_family": contract.model_family,
            "hyperparameters": dict(contract.hyperparameters),
            "feature_columns": list(contract.feature_columns),
            "numerical_features": list(contract.numerical_features),
            "categorical_features": list(contract.categorical_features),
            "target_column_metadata_only": contract.target_column,
            "target_encoding": dict(contract.target_encoding),
            "positive_class": contract.positive_class,
            "operational_validity": "unconfirmed",
            "operational_threshold": "unresolved",
        },
        "frozen_threshold_contract": {
            "threshold": contract.educational_threshold,
            "scenario_id": contract.threshold_scenario_id,
            "origin": "validation",
            "purpose": "educational",
        },
        "probability_vector_sha256_aggregate_only": evaluation.probability_sha256,
        "no_post_test_adjustment": True,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "test_used_for_threshold_selection": False,
        "test_used_for_feature_selection": False,
        "test_used_for_preprocessing_selection": False,
        "individual_rows_persisted": False,
        "operational_validity": "unconfirmed",
    }


def _fitted_vocabularies(pipeline: Pipeline, contract: FrozenFinalizationContract) -> dict[str, list[Any]]:
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    return {
        column: _jsonable(values)
        for column, values in zip(contract.categorical_features, encoder.categories_, strict=True)
    }


def build_inference_bundle(
    *,
    contract: FrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
) -> dict[str, Any]:
    verify_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    preprocess = fitted_pipeline.named_steps["preprocess"]
    negative_class = next(value for value in contract.target_classes if value != contract.positive_class)
    return {
        "schema_version": "inference-bundle.v1",
        "artifact_type": "inference_bundle",
        "dataset_slug": contract.dataset_slug,
        "bundle_version": "1.0.0",
        "model_artifact_path": _require_relative_path(model_artifact_path, field="model_artifact_path"),
        "model_artifact_format": "joblib",
        "model_artifact_sha256": model_artifact_sha256,
        "model_state_fingerprint": model_state_fingerprint,
        "model_id": contract.model_id,
        "model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "pipeline_class": f"{fitted_pipeline.__class__.__module__}.{fitted_pipeline.__class__.__name__}",
        "preprocessing_embedded": True,
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns_excluded_from_model": list(contract.identifier_columns),
        "target_column_metadata_only": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "positive_encoded_label": 1,
        "negative_class": negative_class,
        "categorical_strategy": "one_hot",
        "unknown_category_policy": "ignore_and_report",
        "drop_category": None,
        "numerical_scaling": "none",
        "fitted_categorical_vocabularies": _fitted_vocabularies(fitted_pipeline, contract),
        "transformed_feature_names": preprocess.get_feature_names_out().tolist(),
        "expected_input_dtypes": _deepcopy(expected_input_dtypes),
        "required_input_columns": list(contract.feature_columns),
        "prohibited_input_columns": [*contract.identifier_columns, contract.target_column],
        "missing_value_policy": _deepcopy(missing_value_policy),
        "educational_decision_threshold": contract.educational_threshold,
        "threshold_scenario": contract.threshold_scenario_id,
        "threshold_selection_partition": "validation",
        "operational_threshold": "unresolved",
        "output_contract": {
            "positive_class_probability": "float in [0,1]",
            "educational_prediction_encoded": "integer 0 or 1",
            "educational_prediction_label": list(contract.target_classes),
            "educational_threshold": contract.educational_threshold,
            "operational_prediction_available": False,
        },
        "runtime_version_requirements": runtime_versions(),
        "security_note": "Load joblib only from a trusted source after verifying its SHA-256.",
        "limitations": [
            "Educational snapshot evaluation only; operational validity is unconfirmed.",
            "The educational threshold is not a production decision policy.",
            "Temporal contract and feature inference availability remain unresolved.",
        ],
        "readiness": {
            "educational_inference_demo_ready": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "operational_modeling_ready": False,
        },
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


def build_final_model_manifest(
    *,
    contract: FrozenFinalizationContract,
    upstream_references: Mapping[str, Any],
    training_data: FinalTrainingData,
    test_partition: TestPartitionData,
    fit_duration_seconds: float,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    fitted_state_descriptor: Mapping[str, Any],
    final_artifact_paths: Mapping[str, str],
    final_artifact_fingerprints: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-manifest.v1",
        "artifact_type": "final_model_manifest",
        "dataset_slug": contract.dataset_slug,
        "upstream_references": _deepcopy(upstream_references),
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "preprocessing_contract": dict(contract.preprocessing_contract),
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns": list(contract.identifier_columns),
        "target_column": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "training_partitions": list(contract.training_partitions),
        "final_training_row_count": training_data.row_count,
        "final_training_class_counts": dict(training_data.class_counts),
        "test_row_count": test_partition.row_count,
        "test_class_counts": dict(test_partition.class_counts),
        "educational_threshold": contract.educational_threshold,
        "threshold_scenario_id": contract.threshold_scenario_id,
        "threshold_origin": "validation",
        "threshold_purpose": "educational",
        "test_access_policy": {
            "loaded_after_final_fit": True,
            "test_evaluation_count": 1,
            "used_for_adjustment": False,
        },
        "model_artifact_path": _require_relative_path(model_artifact_path, field="model_artifact_path"),
        "artifact_format": "joblib",
        "model_artifact_byte_sha256": model_artifact_sha256,
        "fitted_state_semantic_fingerprint": model_state_fingerprint,
        "fitted_state_descriptor": _deepcopy(fitted_state_descriptor),
        "fit_duration_seconds": float(fit_duration_seconds),
        "runtime_versions": runtime_versions(),
        "final_artifact_paths": _deepcopy(final_artifact_paths),
        "final_artifact_fingerprints": _deepcopy(final_artifact_fingerprints),
        "readiness": {
            "educational_final_model_completed": True,
            "final_model_trained": True,
            "final_test_evaluation_completed": True,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "final_model_handoff_ready": True,
            "educational_inference_demo_ready": True,
            "operational_modeling_ready": False,
        },
        "limitations": [
            "Educational snapshot validation only.",
            "Operational validity is unconfirmed.",
            "Operational threshold remains unresolved.",
        ],
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


def build_final_model_handoff(
    *,
    contract: FrozenFinalizationContract,
    preparation_handoff_references: Mapping[str, Any],
    model_selection_handoff_references: Mapping[str, Any],
    final_references: Mapping[str, Mapping[str, str]],
    evaluation: FinalEvaluation,
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-handoff.v1",
        "artifact_type": "final_model_handoff",
        "dataset_slug": contract.dataset_slug,
        "preparation_handoff_references": _deepcopy(preparation_handoff_references),
        "model_selection_handoff_references": _deepcopy(model_selection_handoff_references),
        "final_references": _deepcopy(final_references),
        "model_state_fingerprint": final_references["model_artifact"]["semantic_sha256"],
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "preprocessing": dict(contract.preprocessing_contract),
        "feature_order": list(contract.feature_columns),
        "target_encoding": dict(contract.target_encoding),
        "positive_class": contract.positive_class,
        "educational_threshold": contract.educational_threshold,
        "educational_threshold_scenario": contract.threshold_scenario_id,
        "final_training_partitions": list(contract.training_partitions),
        "final_evaluation_partition": "test",
        "final_test_metrics": evaluation.as_dict(),
        "notebook_05_instructions": [
            "Validate this handoff and the inference bundle.",
            "Verify the model artifact SHA-256 before trusted loading.",
            "Load only the complete fitted pipeline; do not refit it.",
            "Do not use the test partition for the demonstration.",
            "Use independent inputs in the declared feature order.",
            "Apply the embedded preprocessing through the pipeline.",
            "Generate positive-class probabilities.",
            "Apply the educational threshold only as a demonstration.",
            "State that the educational threshold is not operational.",
        ],
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_test_evaluation_completed": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_sealed_at_input": True,
        "test_partition_evaluated": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "test_partition_used_for_model_selection": False,
        "test_partition_used_for_threshold_selection": False,
        "api_implemented": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }


# ---------------------------------------------------------------------------
# Validation, persistence, idempotence, and trusted loading
# ---------------------------------------------------------------------------


def _validate_json_artifact(filename: str, payload: Mapping[str, Any]) -> None:
    expected_schema, expected_type = SCHEMAS[filename]
    if payload.get("schema_version") != expected_schema:
        raise FinalizationContractError(f"Invalid schema for {filename}.")
    if payload.get("artifact_type") != expected_type:
        raise FinalizationContractError(f"Invalid artifact type for {filename}.")
    if payload.get("operational_validity") == "confirmed":
        raise FinalizationContractError("Operational validity cannot be confirmed here.")
    if "operational_threshold" in payload and payload.get("operational_threshold") != "unresolved":
        raise FinalizationContractError("Operational threshold must remain unresolved.")
    _validate_paths_recursively(payload)
    rendered = canonical_json_bytes(payload)
    for prohibited in (b"individual_predictions", b"row_predictions", b"individual_probabilities"):
        if prohibited in rendered and filename == "final-test-evidence.json":
            raise FinalizationContractError("Final test evidence contains row-level prediction data.")


def inspect_final_artifact_set(output_directory: str | Path) -> str:
    output = Path(output_directory)
    present = [(output / filename).is_file() for filename in FINAL_ARTIFACT_FILENAMES]
    if not any(present):
        return "absent"
    if all(present):
        return "complete"
    return "partial"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalizationContractError(f"JSON artifact must be an object: {path.name}")
    return payload


def _validate_complete_set(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate schemas, hashes, cross-references, and model identity for a complete set."""

    state = inspect_final_artifact_set(output)
    if state != "complete":
        raise ArtifactConflictError(f"Final artifact set is {state}, not complete.")
    payloads: dict[str, dict[str, Any]] = {}
    for filename in SCHEMAS:
        payload = _load_json(output / filename)
        _validate_json_artifact(filename, payload)
        payloads[filename] = payload

    handoff = payloads["final-model-handoff.json"]
    bundle = payloads["inference-bundle.json"]
    manifest = payloads["final-model-manifest.json"]
    evidence = payloads["final-test-evidence.json"]
    slugs = {
        handoff.get("dataset_slug"),
        bundle.get("dataset_slug"),
        manifest.get("dataset_slug"),
        evidence.get("dataset_slug"),
    }
    if len(slugs) != 1:
        raise ArtifactConflictError("Dataset slug differs within the final artifact set.")

    model_relative = PurePosixPath(
        _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    )
    expected_model_path = (output / "final-pipeline.joblib").resolve()
    inferred_root = expected_model_path
    for _ in model_relative.parts:
        inferred_root = inferred_root.parent
    if (inferred_root / model_relative).resolve() != expected_model_path:
        raise ArtifactConflictError("Bundle model path does not identify the complete-set joblib.")
    model_hash = sha256_file(expected_model_path)
    if model_hash != bundle.get("model_artifact_sha256"):
        raise ArtifactConflictError("Existing joblib hash differs from the inference bundle.")

    references = handoff.get("final_references", {})
    for name, reference in references.items():
        relative = PurePosixPath(
            _require_relative_path(reference["path"], field=f"final_references.{name}.path")
        )
        absolute = (inferred_root / relative).resolve()
        try:
            absolute.relative_to(inferred_root)
        except ValueError as exc:
            raise ArtifactConflictError("Existing final reference escapes project root.") from exc
        if not absolute.is_file():
            raise ArtifactConflictError(f"Existing final reference is missing: {relative}")
        if sha256_file(absolute) != reference.get("byte_sha256"):
            raise ArtifactConflictError(f"Existing final reference hash mismatch: {relative}")

    manifest_fingerprints = manifest.get("final_artifact_fingerprints", {})
    expected_fingerprint_inputs = {
        "final-pipeline.joblib": (model_hash, bundle.get("model_state_fingerprint")),
        "final-test-evidence.json": (
            sha256_file(output / "final-test-evidence.json"),
            semantic_fingerprint(evidence),
        ),
        "inference-bundle.json": (
            sha256_file(output / "inference-bundle.json"),
            semantic_fingerprint(bundle),
        ),
    }
    for filename, (byte_hash, semantic_hash) in expected_fingerprint_inputs.items():
        declared = manifest_fingerprints.get(filename)
        if not isinstance(declared, Mapping):
            raise ArtifactConflictError(f"Manifest fingerprint missing: {filename}")
        if declared.get("byte_sha256") != byte_hash or declared.get("semantic_sha256") != semantic_hash:
            raise ArtifactConflictError(f"Manifest fingerprint mismatch: {filename}")
    if bundle.get("model_state_fingerprint") != handoff.get("model_state_fingerprint"):
        raise ArtifactConflictError("Existing model-state fingerprints are inconsistent.")
    return handoff, bundle, manifest, evidence

def validate_existing_finalization_equivalence(
    *, output_directory: str | Path, contract: FrozenFinalizationContract
) -> bool:
    """Validate complete existing artifacts against the frozen upstream contract."""

    output = Path(output_directory)
    handoff, bundle, _, _ = _validate_complete_set(output)
    checks = {
        "dataset_slug": contract.dataset_slug,
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "educational_threshold": contract.educational_threshold,
    }
    for key, expected in checks.items():
        observed = handoff.get(key)
        if observed != expected:
            raise ArtifactConflictError(
                f"Existing final artifact differs at {key}: expected={expected!r}, observed={observed!r}"
            )
    if handoff.get("feature_order") != list(contract.feature_columns):
        raise ArtifactConflictError("Existing final artifact feature order is divergent.")
    if handoff.get("selected_hyperparameters") != dict(contract.hyperparameters):
        raise ArtifactConflictError("Existing final hyperparameters are divergent.")
    if bundle.get("model_state_fingerprint") != handoff.get("model_state_fingerprint"):
        raise ArtifactConflictError("Existing model-state fingerprints are inconsistent.")
    if handoff.get("test_partition_evaluation_count") != 1:
        raise ArtifactConflictError("Existing test evaluation count must remain one.")
    return True


def write_final_model_artifacts(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    pipeline: Pipeline,
    contract: FrozenFinalizationContract,
    training_data: FinalTrainingData,
    train_sha256: str,
    validation_sha256: str,
    test_partition: TestPartitionData,
    evaluation: FinalEvaluation,
    fit_duration_seconds: float,
    upstream_references: Mapping[str, Any],
    preparation_handoff_references: Mapping[str, Any],
    model_selection_handoff_references: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    validation_educational_threshold: Mapping[str, Any],
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
    overwrite: bool = False,
) -> ArtifactWriteResult:
    """Stage, validate, and atomically promote the five final artifacts."""

    root = Path(project_root).resolve()
    relative_output = _require_relative_path(output_directory, field="output_directory")
    output = (root / relative_output).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise FinalizationContractError("Output directory escapes project root.") from exc
    state = inspect_final_artifact_set(output)
    if state == "partial":
        raise ArtifactConflictError("Partial final artifact set detected; refusing repair.")
    if state == "complete" and not overwrite:
        validate_existing_finalization_equivalence(output_directory=output, contract=contract)
        byte_hashes = {name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES}
        semantic_hashes = {
            name: semantic_fingerprint(_load_json(output / name))
            for name in SCHEMAS
        }
        semantic_hashes["final-pipeline.joblib"] = _load_json(
            output / "inference-bundle.json"
        )["model_state_fingerprint"]
        return ArtifactWriteResult(output, (), (), True, byte_hashes, semantic_hashes)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".final-model-staging-", dir=output.parent))
    backup_root = Path(tempfile.mkdtemp(prefix=".final-model-backup-", dir=output.parent))
    staging = staging_root / output.name
    staging.mkdir(parents=True, exist_ok=True)
    promoted: list[str] = []
    backed_up: list[str] = []
    existing = {name: (output / name).is_file() for name in FINAL_ARTIFACT_FILENAMES}
    try:
        descriptor = describe_fitted_pipeline(
            pipeline=pipeline,
            contract=contract,
            training_data=training_data,
            train_sha256=train_sha256,
            validation_sha256=validation_sha256,
        )
        state_fp = compute_fitted_model_fingerprint(descriptor)
        model_staging = staging / "final-pipeline.joblib"
        model_hash = serialize_pipeline_to_staging(pipeline=pipeline, staging_path=model_staging)
        sample = training_data.features.iloc[: min(32, training_data.row_count)]
        validate_serialized_pipeline(
            staging_path=model_staging,
            expected_sha256=model_hash,
            contract=contract,
            reference_pipeline=pipeline,
            validation_sample=sample,
        )
        model_rel = f"{relative_output}/final-pipeline.joblib"
        evidence = build_final_test_evidence(
            contract=contract,
            test_partition=test_partition,
            evaluation=evaluation,
            validation_metrics=validation_metrics,
            validation_educational_threshold=validation_educational_threshold,
        )
        bundle = build_inference_bundle(
            contract=contract,
            fitted_pipeline=pipeline,
            model_artifact_path=model_rel,
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fp,
            expected_input_dtypes=expected_input_dtypes,
            missing_value_policy=missing_value_policy,
        )
        preliminary = {
            "final-test-evidence.json": evidence,
            "inference-bundle.json": bundle,
        }
        fingerprints: dict[str, dict[str, str]] = {
            "final-pipeline.joblib": {
                "byte_sha256": model_hash,
                "semantic_sha256": state_fp,
            }
        }
        for filename, payload in preliminary.items():
            _validate_json_artifact(filename, payload)
            content = canonical_json_text(payload).encode("utf-8")
            (staging / filename).write_bytes(content)
            fingerprints[filename] = {
                "byte_sha256": sha256_bytes(content),
                "semantic_sha256": semantic_fingerprint(payload),
            }
        paths = {name: f"{relative_output}/{name}" for name in FINAL_ARTIFACT_FILENAMES}
        manifest = build_final_model_manifest(
            contract=contract,
            upstream_references=upstream_references,
            training_data=training_data,
            test_partition=test_partition,
            fit_duration_seconds=fit_duration_seconds,
            model_artifact_path=model_rel,
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fp,
            fitted_state_descriptor=descriptor,
            final_artifact_paths=paths,
            final_artifact_fingerprints=fingerprints,
        )
        _validate_json_artifact("final-model-manifest.json", manifest)
        manifest_content = canonical_json_text(manifest).encode("utf-8")
        (staging / "final-model-manifest.json").write_bytes(manifest_content)
        fingerprints["final-model-manifest.json"] = {
            "byte_sha256": sha256_bytes(manifest_content),
            "semantic_sha256": semantic_fingerprint(manifest),
        }
        final_refs = {
            "model_artifact": {
                "path": model_rel,
                **fingerprints["final-pipeline.joblib"],
            },
            "final_model_manifest": {
                "path": paths["final-model-manifest.json"],
                **fingerprints["final-model-manifest.json"],
            },
            "final_test_evidence": {
                "path": paths["final-test-evidence.json"],
                **fingerprints["final-test-evidence.json"],
            },
            "inference_bundle": {
                "path": paths["inference-bundle.json"],
                **fingerprints["inference-bundle.json"],
            },
        }
        handoff = build_final_model_handoff(
            contract=contract,
            preparation_handoff_references=preparation_handoff_references,
            model_selection_handoff_references=model_selection_handoff_references,
            final_references=final_refs,
            evaluation=evaluation,
        )
        _validate_json_artifact("final-model-handoff.json", handoff)
        handoff_content = canonical_json_text(handoff).encode("utf-8")
        (staging / "final-model-handoff.json").write_bytes(handoff_content)
        fingerprints["final-model-handoff.json"] = {
            "byte_sha256": sha256_bytes(handoff_content),
            "semantic_sha256": semantic_fingerprint(handoff),
        }
        # Validate every staged artifact and cross-reference before promotion.
        for filename in SCHEMAS:
            _validate_json_artifact(filename, _load_json(staging / filename))
        if sha256_file(model_staging) != bundle["model_artifact_sha256"]:
            raise SerializationValidationError("Bundle/model staging hash mismatch.")

        if state == "complete" and overwrite:
            # Divergence was explicitly authorized; preserve rollback copies.
            pass
        output.mkdir(parents=True, exist_ok=True)
        for filename in FINAL_ARTIFACT_FILENAMES:
            destination = output / filename
            if destination.exists():
                backup = backup_root / filename
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(filename)
            os.replace(staging / filename, destination)
            promoted.append(filename)

        # Complete-set validation after promotion.
        for filename in SCHEMAS:
            _validate_json_artifact(filename, _load_json(output / filename))
        loaded_bundle = load_and_validate_inference_bundle(
            project_root=root,
            bundle_path=paths["inference-bundle.json"],
        )
        load_trusted_pipeline_from_bundle(project_root=root, bundle=loaded_bundle)
        load_and_validate_final_model_handoff(
            project_root=root,
            handoff_path=paths["final-model-handoff.json"],
        )
        byte_hashes = {name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES}
        semantic_hashes = {**{name: values["semantic_sha256"] for name, values in fingerprints.items()}}
        return ArtifactWriteResult(
            output,
            tuple(name for name in promoted if not existing[name]),
            tuple(name for name in promoted if existing[name]),
            False,
            byte_hashes,
            semantic_hashes,
        )
    except Exception:
        for filename in reversed(promoted):
            destination = output / filename
            if destination.exists():
                destination.unlink()
        for filename in reversed(backed_up):
            backup = backup_root / filename
            destination = output / filename
            if backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def load_and_validate_inference_bundle(
    *, project_root: str | Path, bundle_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle_path, field="bundle_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FinalizationContractError("Bundle path escapes project root.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Inference bundle not found: {relative}")
    bundle = _load_json(path)
    _validate_json_artifact("inference-bundle.json", bundle)
    model_relative = _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    model_path = (root / model_relative).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_relative}")
    if sha256_file(model_path) != bundle["model_artifact_sha256"]:
        raise SerializationValidationError("Model artifact hash differs from inference bundle.")
    return _deepcopy(bundle)


def load_and_validate_final_model_handoff(
    *, project_root: str | Path, handoff_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(handoff_path, field="handoff_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Final model handoff not found: {relative}")
    handoff = _load_json(path)
    _validate_json_artifact("final-model-handoff.json", handoff)
    required_true = {
        "educational_final_model_completed",
        "final_model_trained",
        "model_artifact_materialized",
        "model_bundle_materialized",
        "final_test_evaluation_completed",
        "final_model_handoff_ready",
        "educational_inference_demo_ready",
        "test_partition_evaluated",
    }
    if any(handoff.get(key) is not True for key in required_true):
        raise FinalizationContractError("Final handoff readiness is incomplete.")
    if handoff.get("test_partition_evaluation_count") != 1:
        raise FinalizationContractError("Final test evaluation count must equal one.")
    if handoff.get("test_partition_used_for_adjustment") is not False:
        raise FinalizationContractError("Test must not be used for adjustment.")
    for reference in handoff.get("final_references", {}).values():
        ref_path = _require_relative_path(reference["path"], field="final_reference.path")
        absolute = (root / ref_path).resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"Referenced final artifact missing: {ref_path}")
        if sha256_file(absolute) != reference["byte_sha256"]:
            raise FinalizationContractError(f"Referenced final artifact hash mismatch: {ref_path}")
    return _deepcopy(handoff)


def load_trusted_pipeline_from_bundle(
    *, project_root: str | Path, bundle: Mapping[str, Any]
) -> Pipeline:
    """Load a joblib only after bundle validation and exact SHA-256 verification."""

    _validate_json_artifact("inference-bundle.json", bundle)
    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {relative}")
    observed = sha256_file(path)
    if observed != bundle["model_artifact_sha256"]:
        raise UntrustedArtifactError("Refusing to load joblib with a divergent SHA-256.")
    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline) or not _is_fitted(loaded):
        raise SerializationValidationError("Trusted artifact is not a fitted sklearn Pipeline.")
    return loaded


# ---------------------------------------------------------------------------
# Continuous-regression finalization (v3)
# ---------------------------------------------------------------------------

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scripts.prepare_data import load_and_validate_preparation_for_model_selection
from scripts.select_models import compute_regression_metrics


REGRESSION_FINAL_SCHEMAS: Mapping[str, tuple[str, str]] = {
    "final-model-manifest.json": ("final-model-manifest.v3", "final_model_manifest"),
    "final-test-evidence.json": ("final-test-evidence.v3", "final_test_evidence"),
    "inference-bundle.json": ("inference-bundle.v3", "inference_bundle"),
    "final-model-handoff.json": ("final-model-handoff.v3", "final_model_handoff"),
}


@dataclass(frozen=True, slots=True)
class RegressionFrozenFinalizationContract:
    payload: tuple[tuple[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return _deepcopy(dict(self.payload))

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(self.as_dict())


def freeze_regression_finalization_decisions(*, handoff: Mapping[str, Any],
        handoff_path: str, handoff_sha256: str, preparation: Any) -> RegressionFrozenFinalizationContract:
    if handoff.get("schema_version") != "model-selection-handoff.v3":
        raise FinalizationContractError("Continuous finalization requires model-selection-handoff.v3.")
    split = preparation.manifests["split_manifest"]
    policy = handoff.get("final_training_instructions", {})
    required_policy = {"fit_partitions": ["train", "validation"], "final_evaluation_partition": "test",
        "access_test_only_after_contract_freeze_and_final_fit": True, "evaluate_test_once": True,
        "do_not_retune": True, "do_not_change_feature_policy": True,
        "do_not_change_hyperparameters": True, "do_not_change_preprocessing": True}
    if any(policy.get(k) != v for k, v in required_policy.items()):
        raise FinalizationContractError("Final-training policy is incomplete or divergent.")
    required = ("dataset_slug", "selected_model_id", "selected_model_family",
                "selected_hyperparameters", "selected_estimator_fixed_constructor_parameters",
                "selected_feature_policy", "selected_feature_columns",
                "selected_preprocessing_contract", "target_contract")
    if any(key not in handoff or handoff[key] in (None, "", [], {}) for key in required):
        raise FinalizationContractError("Model-selection winner contract is incomplete.")
    if handoff.get("problem_type") != "continuous_regression":
        raise FinalizationContractError("Continuous finalization requires continuous_regression.")
    readiness = handoff.get("readiness", {})
    if (handoff.get("test_partition_sealed") is not True
            or handoff.get("test_partition_evaluated") is not False
            or readiness.get("final_model_training_ready") is not True
            or readiness.get("final_model_trained") is not False
            or readiness.get("model_artifact_materialized") is not False
            or readiness.get("model_bundle_materialized") is not False):
        raise FinalizationContractError("Upstream readiness does not permit finalization.")
    data = {
        "dataset_slug": handoff["dataset_slug"], "problem_type": "continuous_regression",
        "model_selection_handoff": {"path": handoff_path, "schema_version": handoff["schema_version"],
                                    "sha256": handoff_sha256},
        "preparation_lineage": {"handoff": handoff["preparation_handoff_reference"],
                                "artifact_hashes": handoff["preparation_artifact_hashes"]},
        "selected_model_id": handoff["selected_model_id"], "selected_model_family": handoff["selected_model_family"],
        "selected_hyperparameters": handoff["selected_hyperparameters"],
        "selected_estimator_fixed_constructor_parameters": handoff["selected_estimator_fixed_constructor_parameters"],
        "random_state": handoff.get("random_seeds", {}).get("estimators"),
        "feature_policy": handoff["selected_feature_policy"], "feature_order": handoff["selected_feature_columns"],
        "preprocessing": handoff["selected_preprocessing_contract"], "target_contract": handoff["target_contract"],
        "training_partitions": ["train", "validation"],
        "training_row_counts": {"train": split["row_counts"]["train"],
                                "validation": split["row_counts"]["validation"]},
        "training_row_count": split["row_counts"]["train"] + split["row_counts"]["validation"],
        "evaluation_partition": "test", "test_reference": preparation.sealed_test_integrity_reference,
        "test_access_rule": "only_after_final_fit_and_verified_checkpoint", "evaluate_once": True,
        "do_not_retune": True, "do_not_change_feature_policy": True,
        "do_not_change_hyperparameters": True, "do_not_change_preprocessing": True,
    }
    return RegressionFrozenFinalizationContract(tuple((k, _deepcopy(v)) for k, v in data.items()))


def reconstruct_regression_selected_pipeline(contract: RegressionFrozenFinalizationContract) -> Pipeline:
    data = contract.as_dict(); family = data["selected_model_family"]
    allowed = {"Ridge": Ridge, "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor}
    if family not in allowed: raise FinalizationContractError(f"Unsupported regression family: {family}")
    estimator = allowed[family]()
    accepted = estimator.get_params(deep=False)
    fixed = data["selected_estimator_fixed_constructor_parameters"]
    if any(k not in accepted for k in fixed): raise FinalizationContractError("Unsupported fixed constructor parameter.")
    estimator.set_params(**fixed)
    selected = {}
    for key, value in data["selected_hyperparameters"].items():
        if not key.startswith("model__"): raise FinalizationContractError("Selected parameter lacks model__ prefix.")
        plain = key.removeprefix("model__")
        if plain not in accepted: raise FinalizationContractError(f"Unsupported selected parameter: {plain}")
        selected[plain] = value
    estimator.set_params(**selected)
    prep = data["preprocessing"]; features = data["feature_order"]
    if prep.get("type") != "Pipeline" or prep.get("categorical_features") not in ([], None):
        raise FinalizationContractError("Unsupported regression preprocessing contract.")
    numerical = prep.get("numerical_features", features)
    if numerical != features:
        raise FinalizationContractError("Preprocessing numerical feature order differs.")
    if prep.get("scale_numerical") is True and prep.get("scaler") not in ("StandardScaler", None):
        raise FinalizationContractError("Unsupported numerical scaler.")
    transformer = StandardScaler() if prep.get("scale_numerical") else "passthrough"
    preprocess = ColumnTransformer([("numerical", transformer, numerical)], remainder="drop", sparse_threshold=0.0)
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    if _is_fitted(pipeline): raise FinalizationContractError("Reconstructed regression pipeline is already fitted.")
    return pipeline


def assemble_regression_final_training_data(*, train: pd.DataFrame, validation: pd.DataFrame,
        feature_columns: Sequence[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    expected = list(feature_columns) + [target_column]
    for name, frame in (("train", train), ("validation", validation)):
        if list(frame.columns) != expected:
            raise FinalizationContractError(f"{name} schema or feature order differs.")
    combined = pd.concat([train.copy(deep=True), validation.copy(deep=True)], ignore_index=True)
    y = combined[target_column]
    if not pd.api.types.is_numeric_dtype(y) or y.isna().any() or not np.isfinite(y.to_numpy(float)).all():
        raise FinalizationContractError("Final regression target must be complete, numeric, and finite.")
    return combined.loc[:, list(feature_columns)].copy(deep=True), y.copy(deep=True)


def describe_regression_fitted_pipeline(*, pipeline: Pipeline,
        contract: RegressionFrozenFinalizationContract | Mapping[str, Any]) -> dict[str, Any]:
    """Build a byte-independent, recomputable fitted-state descriptor."""
    data = contract.as_dict() if isinstance(contract, RegressionFrozenFinalizationContract) else _deepcopy(contract)
    if not isinstance(pipeline, Pipeline) or not _is_fitted(pipeline):
        raise SerializationValidationError("Regression pipeline is not fitted.")
    preprocess = pipeline.named_steps.get("preprocess"); model = pipeline.named_steps.get("model")
    try:
        transformed = [str(value) for value in preprocess.get_feature_names_out()]
    except Exception as exc:
        raise SerializationValidationError("Cannot derive transformed regression feature names.") from exc
    indicators: dict[str, Any] = {}
    for name in ("n_features_in_", "feature_names_in_", "n_iter_", "n_trees_per_iteration_",
                 "train_score_", "validation_score_", "coef_", "intercept_", "tree_",
                 "estimators_", "is_categorical_", "n_features_in_"):
        if hasattr(model, name):
            value = getattr(model, name)
            if name in ("estimators_", "tree_"):
                value = len(value) if hasattr(value, "__len__") else value.__class__.__name__
            indicators[name] = _jsonable(value)
    return {
        "pipeline_class": pipeline.__class__.__name__,
        "step_names": list(pipeline.named_steps),
        "preprocessor_class": preprocess.__class__.__name__,
        "model_class": model.__class__.__name__,
        "selected_hyperparameters": data["selected_hyperparameters"],
        "fixed_constructor_parameters": data["selected_estimator_fixed_constructor_parameters"],
        "feature_order": data["feature_order"],
        "preprocessing_contract": data["preprocessing"],
        "transformed_feature_names": transformed,
        "fitted_state_indicators": indicators,
    }


def compute_regression_model_state_fingerprint(descriptor: Mapping[str, Any]) -> str:
    return semantic_fingerprint(descriptor)


def _regression_model_params(bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(bundle["model_contract"]["fixed_constructor_parameters"])
    result.update({k.removeprefix("model__"): v for k, v in bundle["model_contract"]["selected_hyperparameters"].items()})
    return result


def _validate_regression_complete_set(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not all((directory / n).is_file() for n in FINAL_ARTIFACT_FILENAMES):
        raise ArtifactConflictError("Regression final artifact set is partial.")
    payloads = {n: _load_json(directory/n) for n in REGRESSION_FINAL_SCHEMAS}
    for name, (schema, kind) in REGRESSION_FINAL_SCHEMAS.items():
        if payloads[name].get("schema_version") != schema or payloads[name].get("artifact_type") != kind:
            raise FinalizationContractError(f"Invalid continuous v3 artifact: {name}")
    manifest=payloads["final-model-manifest.json"]; evidence=payloads["final-test-evidence.json"]
    bundle=payloads["inference-bundle.json"]; handoff=payloads["final-model-handoff.json"]
    model_path=directory/"final-pipeline.joblib"; model_sha=sha256_file(model_path)
    common_fields = ("dataset_slug", "problem_type", "selected_model_id", "selected_model_family",
                     "feature_order", "target_contract", "preprocessing_contract",
                     "frozen_finalization_contract_fingerprint")
    for field in common_fields:
        expected = manifest.get(field)
        for payload in (evidence, bundle, handoff):
            if payload.get(field) != expected:
                raise FinalizationContractError(f"Continuous final artifacts disagree on {field}.")
    if manifest.get("problem_type") != "continuous_regression":
        raise FinalizationContractError("Final artifacts are not continuous regression.")
    contract = manifest.get("frozen_finalization_contract", {})
    if semantic_fingerprint(contract) != manifest.get("frozen_finalization_contract_fingerprint"):
        raise FinalizationContractError("Frozen finalization fingerprint mismatch.")
    selection_ref = manifest.get("model_selection_handoff_reference")
    if any(payload.get("model_selection_handoff_reference") != selection_ref for payload in (evidence,bundle,handoff)):
        raise FinalizationContractError("Model-selection lineage differs across final artifacts.")
    for field in ("selected_hyperparameters", "selected_estimator_fixed_constructor_parameters"):
        if manifest.get(field) != bundle.get("model_contract", {}).get(
                "selected_hyperparameters" if field == "selected_hyperparameters" else "fixed_constructor_parameters"):
            raise FinalizationContractError(f"Model contract differs for {field}.")
    if (model_sha != bundle.get("model_artifact_sha256")
            or model_sha != manifest.get("model_artifact",{}).get("byte_sha256")
            or model_sha != handoff.get("model_artifact_reference", {}).get("sha256")):
        raise UntrustedArtifactError("Continuous final model SHA mismatch.")
    state = manifest.get("model_artifact", {}).get("state_fingerprint")
    descriptor = manifest.get("model_artifact", {}).get("state_descriptor")
    if (not descriptor or compute_regression_model_state_fingerprint(descriptor) != state
            or bundle.get("model_state_fingerprint") != state
            or bundle.get("model_state_descriptor") != descriptor
            or handoff.get("model_artifact_reference", {}).get("state_fingerprint") != state):
        raise FinalizationContractError("Continuous model-state fingerprint mismatch.")
    refs=handoff.get("sibling_references",{})
    for name in ("final-model-manifest.json","final-test-evidence.json","inference-bundle.json"):
        if refs.get(name,{}).get("sha256") != sha256_file(directory/name):
            raise FinalizationContractError(f"Final handoff sibling hash mismatch: {name}")
    if evidence.get("test_prediction_call_count") != 1 or evidence.get("test_partition_evaluation_count") != 1 or evidence.get("no_post_test_adjustment") is not True:
        raise FinalizationContractError("One-time test evidence is invalid.")
    if (manifest.get("training_row_count") != handoff.get("training_row_count")
            or evidence.get("partition_reference") != manifest.get("test_reference")
            or evidence.get("metric_contract") != handoff.get("metric_contract")
            or evidence.get("metrics") != handoff.get("final_test_metrics")
            or evidence.get("validation_to_test_deltas") != handoff.get("validation_to_test_deltas")
            or handoff.get("test_partition_evaluation_count") != 1):
        raise FinalizationContractError("Training/test evidence differs across final artifacts.")
    readiness = handoff.get("readiness", {})
    required_true = ("model_selection_handoff_validated", "selected_candidate_reconstructed",
                     "frozen_finalization_contract_validated", "final_training_completed",
                     "final_model_trained", "test_partition_opened_after_final_fit",
                     "final_test_evaluation_completed", "test_partition_evaluated",
                     "no_model_selection_decision_changed_after_test",
                     "final_model_artifact_materialized", "inference_bundle_materialized",
                     "inference_demo_ready")
    if (any(readiness.get(k) is not True for k in required_true)
            or readiness.get("operational_modeling_ready") is not False
            or readiness.get("operational_validity") != "unconfirmed"):
        raise FinalizationContractError("Final readiness contract is invalid.")
    return handoff,bundle,manifest,evidence


def _write_regression_final_artifacts(*, output: Path, staged_model: Path,
        payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    staging=Path(tempfile.mkdtemp(prefix="regression-final-")); promoted=[]
    try:
        shutil.copyfile(staged_model, staging/"final-pipeline.joblib")
        for name,payload in payloads.items(): (staging/name).write_bytes(canonical_json_bytes(payload)+b"\n")
        output.mkdir(parents=True,exist_ok=True)
        if any((output/n).exists() for n in FINAL_ARTIFACT_FILENAMES):
            raise ArtifactConflictError("Final artifact state changed during promotion.")
        for name in FINAL_ARTIFACT_FILENAMES:
            os.replace(staging/name,output/name); promoted.append(name)
        _validate_regression_complete_set(output)
        return {"status":"created","sha256":{n:sha256_file(output/n) for n in FINAL_ARTIFACT_FILENAMES}}
    except Exception:
        for name in reversed(promoted): (output/name).unlink(missing_ok=True)
        raise
    finally: shutil.rmtree(staging,ignore_errors=True)


def run_regression_finalization(*, project_root: str|Path, model_selection_handoff_path: str|Path,
        output_directory: str|Path) -> dict[str, Any]:
    root=Path(project_root).resolve(); output_rel=str(_require_relative_path(output_directory,field="output_directory"))
    out=(root/output_rel).resolve(); receipt=out/".final-evaluation-state.json"
    present=[(out/n).exists() for n in FINAL_ARTIFACT_FILENAMES]
    if any(present):
        if not all(present): raise ArtifactConflictError("Regression final artifact set is partial.")
        handoff,bundle,manifest,evidence=_validate_regression_complete_set(out)
        requested=str(_require_relative_path(model_selection_handoff_path,field="model_selection_handoff_path"))
        upstream=manifest.get("model_selection_handoff_reference", {})
        if (upstream.get("path") != requested or not (root/requested).is_file()
                or sha256_file(root/requested) != upstream.get("sha256")):
            raise ArtifactConflictError("Existing final set is divergent from the requested model-selection handoff.")
        load_and_validate_model_selection_handoff(project_root=root,handoff_path=requested)
        load_trusted_pipeline_from_bundle(project_root=root,bundle=bundle)
        return {"status":"reused_equivalent","final_fit_count":0,"test_parse_count":0,"test_predict_count":0,
            "test_evaluation_count":evidence["test_partition_evaluation_count"],"metrics":evidence["metrics"],
            "contract_fingerprint":manifest["frozen_finalization_contract_fingerprint"]}
    if receipt.is_file():
        state=_load_json(receipt)
        if state.get("state") == "evaluation_started":
            raise DuplicateTestEvaluationError("A prior test evaluation started without a complete artifact set.")
        raise ArtifactConflictError("Finalization receipt exists without complete final artifacts.")
    selection=load_and_validate_model_selection_handoff(project_root=root,handoff_path=model_selection_handoff_path)
    selection_rel=str(_require_relative_path(model_selection_handoff_path,field="model_selection_handoff_path"))
    prep_ref=selection["preparation_handoff_reference"]
    preparation=load_and_validate_preparation_for_model_selection(project_root=root,preparation_handoff_path=prep_ref["path"])
    contract=freeze_regression_finalization_decisions(handoff=selection,handoff_path=selection_rel,
        handoff_sha256=sha256_file(root/selection_rel),preparation=preparation)
    data=contract.as_dict(); features=data["feature_order"]; target=data["target_contract"]["column"]
    x,y=assemble_regression_final_training_data(train=preparation.train,validation=preparation.validation,
        feature_columns=features,target_column=target)
    if len(x) != data["training_row_count"]:
        raise FinalizationContractError("Final training row count differs from frozen contract.")
    pipeline=reconstruct_regression_selected_pipeline(contract); pipeline.fit(x,y)
    if not _is_fitted(pipeline): raise SerializationValidationError("Regression pipeline did not become fitted.")
    checkpoint_dir=Path(tempfile.mkdtemp(prefix="regression-checkpoint-")); checkpoint=checkpoint_dir/"pipeline.joblib"
    try:
        # This is the sole serialization; these exact bytes are later promoted.
        joblib.dump(pipeline,checkpoint); checkpoint_sha=sha256_file(checkpoint)
        trusted=joblib.load(checkpoint)
        if not _is_fitted(trusted): raise SerializationValidationError("Staging reload is not fitted.")
        descriptor=describe_regression_fitted_pipeline(pipeline=trusted,contract=contract)
        model_state=compute_regression_model_state_fingerprint(descriptor)
        reference_smoke=np.asarray(pipeline.predict(x.iloc[:5].copy()),dtype=float)
        reload_smoke=np.asarray(trusted.predict(x.iloc[:5].copy()),dtype=float)
        if not np.allclose(reference_smoke,reload_smoke,rtol=1e-12,atol=1e-12):
            raise SerializationValidationError("Training-sample serialization roundtrip differs.")
        test_ref=data["test_reference"]; test_path=root/_require_relative_path(test_ref["path"],field="test.path")
        if sha256_file(test_path) != test_ref["sha256"]: raise TestAccessError("Sealed test SHA mismatch.")
        out.mkdir(parents=True,exist_ok=True)
        receipt.write_bytes(canonical_json_bytes({"schema_version":"final-evaluation-state.v1",
            "state":"evaluation_started","contract_fingerprint":contract.fingerprint,
            "test_reference":test_ref})+b"\n")
        test=pd.read_csv(test_path)
        if (sha256_file(test_path) != test_ref["sha256"] or len(test) != test_ref["row_count"]
                or list(test.columns) != features+[target]):
            raise TestAccessError("Test partition integrity or schema differs after opening.")
        test_target=test[target]
        if (not pd.api.types.is_numeric_dtype(test_target) or test_target.isna().any()
                or not np.isfinite(test_target.to_numpy(float)).all()):
            raise TestAccessError("Test target must be complete, numeric, and finite.")
        predictions=np.asarray(trusted.predict(test.loc[:,features].copy()),dtype=float)
        metrics=compute_regression_metrics(test_target.to_numpy(float),predictions)
        validation=selection["selected_validation_evidence"]
        deltas={f"test_{m}_minus_validation_{m}":metrics[m]-validation[m] for m in ("mae","rmse","r2","medae")}
        selection_ref={"path":selection_rel,"schema_version":selection["schema_version"],
                       "sha256":sha256_file(root/selection_rel)}
        model_rel=f"{output_rel}/final-pipeline.joblib"; metric_contract={"primary":"mae","direction":"lower_is_better","unit":selection["target_contract"]["unit"],"metrics":["mae","rmse","r2","medae"]}
        common={"dataset_slug":selection["dataset_slug"],"problem_type":"continuous_regression",
            "model_selection_handoff_reference":selection_ref,
            "selected_model_id":selection["selected_model_id"],"selected_model_family":selection["selected_model_family"],
            "feature_order":features,"target_contract":selection["target_contract"],
            "preprocessing_contract":data["preprocessing"],
            "frozen_finalization_contract_fingerprint":contract.fingerprint}
        manifest={**common,"schema_version":"final-model-manifest.v3","artifact_type":"final_model_manifest",
            "preparation_lineage":data["preparation_lineage"],"frozen_finalization_contract":data,
            "selected_hyperparameters":data["selected_hyperparameters"],
            "selected_estimator_fixed_constructor_parameters":data["selected_estimator_fixed_constructor_parameters"],
            "feature_policy":data["feature_policy"],"training_partitions":data["training_partitions"],
            "training_row_count":len(x),"evaluation_partition":"test","test_reference":test_ref,
            "final_fit_count":1,"test_evaluation_count":1,
            "model_artifact":{"path":model_rel,"byte_sha256":checkpoint_sha,"state_fingerprint":model_state,"state_descriptor":descriptor},
            "runtime_versions":{"python":platform.python_version(),"scikit_learn":sklearn.__version__,"pandas":pd.__version__,"joblib":joblib.__version__},
            "operational_validity":"unconfirmed","operational_modeling_ready":False}
        evidence={**common,"schema_version":"final-test-evidence.v3","artifact_type":"final_test_evidence",
            "partition":"test","partition_reference":test_ref,"row_count":len(test),"metric_contract":metric_contract,"metrics":metrics,
            "validation_to_test_deltas":deltas,"test_loaded_only_after_final_fit":True,
            "test_prediction_call_count":1,"test_partition_evaluation_count":1,
            "test_used_for_model_selection":False,"test_used_for_hyperparameter_selection":False,
            "test_used_for_feature_selection":False,"test_used_for_preprocessing_selection":False,"no_post_test_adjustment":True}
        bundle={**common,"schema_version":"inference-bundle.v3","artifact_type":"inference_bundle",
            "model_artifact_path":model_rel,"model_artifact_sha256":checkpoint_sha,
            "model_state_fingerprint":model_state,"model_state_descriptor":descriptor,
            "input_feature_dtypes":{c:{"dtype":str(x[c].dtype),"role":"numerical_feature"} for c in features},
            "model_contract":{"family":selection["selected_model_family"],"fixed_constructor_parameters":data["selected_estimator_fixed_constructor_parameters"],"selected_hyperparameters":data["selected_hyperparameters"]},
            "prediction_contract":{"type":"continuous_numeric","scale":"original_target_scale","unit":selection["target_contract"]["unit"]},
            "lineage":{"model_selection":selection_ref,"preparation":data["preparation_lineage"]},
            "runtime_compatibility":{"python":platform.python_version(),"scikit_learn":sklearn.__version__},
            "security_note":"Load joblib only after exact SHA-256 verification.",
            "readiness":{"inference_demo_ready":True,"operational_modeling_ready":False,"operational_validity":"unconfirmed"}}
        payloads={"final-model-manifest.json":manifest,"final-test-evidence.json":evidence,"inference-bundle.json":bundle}
        siblings={n:{"path":f"{output_rel}/{n}","sha256":sha256_bytes(canonical_json_bytes(p)+b"\n")} for n,p in payloads.items()}
        readiness={k:True for k in ("model_selection_handoff_validated","selected_candidate_reconstructed",
            "frozen_finalization_contract_validated","final_training_completed","final_model_trained",
            "test_partition_opened_after_final_fit","final_test_evaluation_completed","test_partition_evaluated",
            "no_model_selection_decision_changed_after_test","final_model_artifact_materialized",
            "inference_bundle_materialized","inference_demo_ready")}
        readiness.update({"final_fit_count":1,"test_partition_evaluation_count":1,"test_prediction_call_count":1,
            "test_partition_used_for_adjustment":False,"operational_modeling_ready":False,"operational_validity":"unconfirmed"})
        payloads["final-model-handoff.json"]={**common,"schema_version":"final-model-handoff.v3","artifact_type":"final_model_handoff",
            "selected_hyperparameters":data["selected_hyperparameters"],
            "selected_estimator_fixed_constructor_parameters":data["selected_estimator_fixed_constructor_parameters"],
            "model_artifact_reference":{"path":model_rel,"sha256":checkpoint_sha,"state_fingerprint":model_state},
            "bundle_reference":siblings["inference-bundle.json"],"manifest_reference":siblings["final-model-manifest.json"],
            "test_evidence_reference":siblings["final-test-evidence.json"],"sibling_references":siblings,
            "prediction_contract":bundle["prediction_contract"],"metric_contract":metric_contract,
            "final_test_metrics":metrics,"validation_to_test_deltas":deltas,
            "training_partitions":data["training_partitions"],"training_row_count":len(x),"evaluation_partition":"test",
            "test_partition_evaluation_count":1,"readiness":readiness}
        result=_write_regression_final_artifacts(output=out,staged_model=checkpoint,payloads=payloads)
        receipt.write_bytes(canonical_json_bytes({"schema_version":"final-evaluation-state.v1","state":"complete",
            "contract_fingerprint":contract.fingerprint,"test_partition_evaluation_count":1})+b"\n")
        return {**result,"final_fit_count":1,"test_parse_count":1,"test_predict_count":1,"test_evaluation_count":1,
            "metrics":metrics,"validation_to_test_deltas":deltas,"contract_fingerprint":contract.fingerprint,
            "model_state_fingerprint":model_state,"model_state_descriptor":descriptor,
            "staged_joblib_sha256":checkpoint_sha,"roundtrip_verified":True,"test_row_count":len(test)}
    finally:
        shutil.rmtree(checkpoint_dir,ignore_errors=True)


_load_bundle_v1_v2 = load_and_validate_inference_bundle
_load_handoff_v1_v2 = load_and_validate_final_model_handoff
_load_manifest_v1_v2 = globals().get("load_and_validate_final_model_manifest")
_load_evidence_v1_v2 = globals().get("load_and_validate_final_test_evidence")
_load_pipeline_v1_v2 = load_trusted_pipeline_from_bundle


def load_and_validate_inference_bundle(*, project_root: str|Path, bundle_path: str|Path) -> dict[str,Any]:
    root=Path(project_root).resolve(); rel=_require_relative_path(bundle_path,field="bundle_path"); preview=_load_json(root/rel)
    if preview.get("schema_version") != "inference-bundle.v3": return _load_bundle_v1_v2(project_root=root,bundle_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[1])


def load_and_validate_final_model_handoff(*, project_root: str|Path, handoff_path: str|Path) -> dict[str,Any]:
    root=Path(project_root).resolve(); rel=_require_relative_path(handoff_path,field="handoff_path"); preview=_load_json(root/rel)
    if preview.get("schema_version") != "final-model-handoff.v3": return _load_handoff_v1_v2(project_root=root,handoff_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[0])


def load_and_validate_final_model_manifest(*, project_root: str|Path, manifest_path: str|Path) -> dict[str,Any]:
    root=Path(project_root).resolve(); rel=_require_relative_path(manifest_path,field="manifest_path"); preview=_load_json(root/rel)
    if preview.get("schema_version") != "final-model-manifest.v3": return _load_manifest_v1_v2(project_root=root,manifest_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[2])


def load_and_validate_final_test_evidence(*, project_root: str|Path, evidence_path: str|Path) -> dict[str,Any]:
    root=Path(project_root).resolve(); rel=_require_relative_path(evidence_path,field="evidence_path"); preview=_load_json(root/rel)
    if preview.get("schema_version") != "final-test-evidence.v3": return _load_evidence_v1_v2(project_root=root,evidence_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[3])


def load_trusted_pipeline_from_bundle(*, project_root: str|Path, bundle: Mapping[str,Any]) -> Pipeline:
    if bundle.get("schema_version") != "inference-bundle.v3": return _load_pipeline_v1_v2(project_root=project_root,bundle=bundle)
    root=Path(project_root).resolve(); rel=_require_relative_path(bundle["model_artifact_path"],field="model_artifact_path"); path=root/rel
    if sha256_file(path) != bundle.get("model_artifact_sha256"): raise UntrustedArtifactError("Refusing to load regression joblib with divergent SHA-256.")
    loaded=joblib.load(path)
    if not _is_fitted(loaded) or loaded.named_steps["model"].__class__.__name__ != bundle["model_contract"]["family"]:
        raise SerializationValidationError("Trusted regression pipeline contract mismatch.")
    params=loaded.named_steps["model"].get_params(deep=False)
    if any(params.get(k)!=v for k,v in _regression_model_params(bundle).items()): raise SerializationValidationError("Regression model parameters differ.")
    descriptor=describe_regression_fitted_pipeline(pipeline=loaded,contract={
        "selected_hyperparameters":bundle["model_contract"]["selected_hyperparameters"],
        "selected_estimator_fixed_constructor_parameters":bundle["model_contract"]["fixed_constructor_parameters"],
        "feature_order":bundle["feature_order"],"preprocessing":bundle["preprocessing_contract"]})
    if (descriptor != bundle.get("model_state_descriptor")
            or compute_regression_model_state_fingerprint(descriptor) != bundle.get("model_state_fingerprint")):
        raise SerializationValidationError("Regression fitted-state descriptor or fingerprint differs.")
    return loaded


# Explicit aliases retained for a readable notebook API.
validate_finalization_contract = validate_finalization_contract


# ---------------------------------------------------------------------------
# Multiclass finalization (v2)
# ---------------------------------------------------------------------------


MULTICLASS_FINAL_SCHEMAS: Mapping[str, tuple[str, str]] = {
    "final-model-manifest.json": (
        "final-model-manifest.v2",
        "final_model_manifest",
    ),
    "final-test-evidence.json": (
        "final-test-evidence.v2",
        "final_test_evidence",
    ),
    "inference-bundle.json": ("inference-bundle.v2", "inference_bundle"),
    "final-model-handoff.json": (
        "final-model-handoff.v2",
        "final_model_handoff",
    ),
}


@dataclass(frozen=True, slots=True)
class MulticlassFrozenFinalizationContract:
    """Immutable seven-or-more-class decisions frozen before final fit."""

    dataset_slug: str
    problem_type: str
    model_selection_handoff_path: str
    model_selection_handoff_sha256: str
    model_id: str
    model_family: str
    hyperparameters: tuple[tuple[str, Any], ...]
    random_state: int | None
    feature_policy: str
    feature_columns: tuple[str, ...]
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    target_column: str
    target_classes: tuple[Any, ...]
    target_encoding: tuple[tuple[Any, int], ...]
    target_semantics: str
    preprocessing_contract: tuple[tuple[str, Any], ...]
    imbalance_policy: tuple[tuple[str, Any], ...]
    decision_rule: str
    training_partitions: tuple[str, ...]
    evaluation_partition: str
    test_partition_path: str
    test_partition_sha256: str
    test_partition_row_count: int
    access_test_only_after_contract_freeze_and_final_fit: bool = True
    evaluate_test_once: bool = True

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "dataset_slug": self.dataset_slug,
                "problem_type": self.problem_type,
                "model_selection_handoff_reference": {
                    "path": self.model_selection_handoff_path,
                    "byte_sha256": self.model_selection_handoff_sha256,
                },
                "model_id": self.model_id,
                "model_family": self.model_family,
                "hyperparameters": dict(self.hyperparameters),
                "random_state": self.random_state,
                "feature_policy": self.feature_policy,
                "feature_columns": list(self.feature_columns),
                "numerical_features": list(self.numerical_features),
                "categorical_features": list(self.categorical_features),
                "identifier_columns": list(self.identifier_columns),
                "target_column": self.target_column,
                "target_classes": list(self.target_classes),
                "target_encoding": dict(self.target_encoding),
                "target_semantics": self.target_semantics,
                "positive_class": "not_applicable",
                "binary_threshold": "not_applicable",
                "operational_threshold": "not_applicable",
                "preprocessing_contract": dict(self.preprocessing_contract),
                "imbalance_policy": dict(self.imbalance_policy),
                "decision_rule": self.decision_rule,
                "training_partitions": list(self.training_partitions),
                "evaluation_partition": self.evaluation_partition,
                "test_access_policy": {
                    "access_only_after_contract_freeze_and_final_fit": self.access_test_only_after_contract_freeze_and_final_fit,
                    "evaluate_once": self.evaluate_test_once,
                },
                "test_partition_reference": {
                    "path": self.test_partition_path,
                    "byte_sha256": self.test_partition_sha256,
                    "row_count": self.test_partition_row_count,
                },
            }
        )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(self.as_dict())


@dataclass(frozen=True, slots=True)
class MulticlassFinalTrainingData:
    """Defensive train+validation data with readable nominal targets."""

    _features: pd.DataFrame
    _target: pd.Series
    class_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)

    @property
    def row_count(self) -> int:
        return int(len(self._features))


@dataclass(frozen=True, slots=True)
class MulticlassTestPartitionData:
    """Test data created only after the frozen/fitted access gate."""

    _features: pd.DataFrame
    _target: pd.Series
    row_count: int
    class_counts: tuple[tuple[str, int], ...]
    partition_path: str
    partition_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_features", self._features.copy(deep=True))
        object.__setattr__(self, "_target", self._target.copy(deep=True))

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)

    @property
    def target(self) -> pd.Series:
        return self._target.copy(deep=True)


@dataclass(frozen=True, slots=True)
class MulticlassFinalEvaluation:
    """Aggregate-only official test evidence from one probability call."""

    metrics: Mapping[str, Any]
    per_class: Sequence[Mapping[str, Any]]
    confusion_matrix: Mapping[str, Any]
    estimator_class_order: tuple[Any, ...]
    output_class_order: tuple[Any, ...]
    validation_to_test: Mapping[str, Any]
    confusion_pair_comparison: Mapping[str, Any]
    repeated_profile_sensitivity: Mapping[str, Any]
    probability_sha256: str
    test_probability_evaluation_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return _deepcopy(
            {
                "metrics": self.metrics,
                "per_class": list(self.per_class),
                "confusion_matrix": self.confusion_matrix,
                "estimator_class_order": list(self.estimator_class_order),
                "output_class_order": list(self.output_class_order),
                "validation_to_test": self.validation_to_test,
                "confusion_pair_comparison": self.confusion_pair_comparison,
                "repeated_profile_sensitivity": self.repeated_profile_sensitivity,
                "probability_sha256_aggregate_only": self.probability_sha256,
                "test_probability_evaluation_count": self.test_probability_evaluation_count,
            }
        )


def validate_multiclass_finalization_contract(
    model_selection_handoff: Mapping[str, Any],
) -> None:
    """Validate Notebook-03 v2 readiness without introducing binary semantics."""

    if model_selection_handoff.get("schema_version") != "model-selection-handoff.v2":
        raise FinalizationContractError("Invalid multiclass model-selection handoff schema.")
    if model_selection_handoff.get("artifact_type") != "model_selection_handoff":
        raise FinalizationContractError("Invalid multiclass model-selection artifact type.")
    if model_selection_handoff.get("problem_type") != "multiclass_classification":
        raise FinalizationContractError("Finalization input is not multiclass classification.")
    classes = model_selection_handoff.get("target_classes")
    if not isinstance(classes, list) or len(classes) < 3 or len(set(classes)) != len(classes):
        raise FinalizationContractError("Multiclass target classes are invalid.")
    if model_selection_handoff.get("target_semantics") != "nominal_unordered":
        raise FinalizationContractError("Multiclass target semantics must be nominal_unordered.")
    if model_selection_handoff.get("positive_class") is not None:
        raise FinalizationContractError("Multiclass finalization cannot define a positive class.")
    for field in ("binary_threshold", "operational_threshold"):
        value = model_selection_handoff.get(field)
        if not isinstance(value, Mapping) or value.get("status") != "not_applicable" or value.get("value") is not None:
            raise FinalizationContractError(f"{field} must be explicitly not_applicable.")
    if model_selection_handoff.get("decision_rule") != "argmax_class_score_or_probability":
        raise FinalizationContractError("Multiclass decision rule is invalid.")
    if model_selection_handoff.get("test_partition_sealed") is not True:
        raise FinalizationContractError("Test must be sealed at finalization input.")
    if model_selection_handoff.get("test_partition_evaluated") is not False:
        raise FinalizationContractError("Test must be unevaluated at finalization input.")
    readiness = model_selection_handoff.get("readiness", {})
    required_true = {
        "preparation_handoff_validated",
        "selected_candidate_frozen",
        "imbalance_policy_frozen",
        "multiclass_decision_rule_frozen",
        "final_model_training_ready",
        "test_partition_sealed",
    }
    if any(readiness.get(key) is not True for key in required_true):
        raise FinalizationContractError("Multiclass final-training readiness is incomplete.")
    if model_selection_handoff.get("final_model_trained") is not False:
        raise FinalizationContractError("Final model must not be trained upstream.")
    if model_selection_handoff.get("model_artifact") is not None:
        raise FinalizationContractError("Upstream model artifact must be absent.")
    if model_selection_handoff.get("model_artifact_materialized") is not False:
        raise FinalizationContractError("Upstream model artifact must be unmaterialized.")
    if model_selection_handoff.get("model_bundle_materialized") is not False:
        raise FinalizationContractError("Upstream model bundle must be unmaterialized.")
    if model_selection_handoff.get("operational_modeling_ready") is not False:
        raise FinalizationContractError("Operational readiness must remain false.")
    if model_selection_handoff.get("operational_validity") != "unconfirmed":
        raise FinalizationContractError("Operational validity must remain unconfirmed.")
    preprocessing = model_selection_handoff.get("selected_preprocessing_contract", {})
    expected_preprocessing = {
        "pipeline": "sklearn.pipeline.Pipeline",
        "numerical_scaling": "none",
        "categorical_processing": "not_applicable",
        "feature_projection": model_selection_handoff.get("selected_feature_columns"),
        "learned_preprocessing_in_notebook_02": False,
        "scaling_fit_scope": "inside_training_fold_or_final_training_only",
    }
    if preprocessing != expected_preprocessing:
        raise FinalizationContractError("Selected multiclass preprocessing contract changed.")
    if model_selection_handoff.get("selected_imbalance_policy") != {
        "strategy": "none",
        "class_weight": None,
        "resampling": "none",
    }:
        raise FinalizationContractError("Selected multiclass imbalance policy changed.")
    instructions = model_selection_handoff.get("final_training_instructions", {})
    required_instructions = {
        "reconstruct_pipeline_from_contract": True,
        "fit_partitions": ["train", "validation"],
        "final_evaluation_partition": "test",
        "access_test_only_after_contract_freeze_and_final_fit": True,
        "evaluate_test_once": True,
        "do_not_retune": True,
        "do_not_change_feature_policy": True,
        "do_not_change_imbalance_policy": True,
        "do_not_change_hyperparameters": True,
        "decision_rule": "argmax_class_score_or_probability",
    }
    if any(instructions.get(key) != value for key, value in required_instructions.items()):
        raise FinalizationContractError("Frozen final-training instructions changed.")
    _validate_paths_recursively(model_selection_handoff)


def freeze_multiclass_finalization_decisions(
    *,
    dataset_slug: str,
    model_selection_handoff: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    model_selection_handoff_path: str | Path,
    model_selection_handoff_sha256: str,
) -> MulticlassFrozenFinalizationContract:
    """Freeze every model/data decision before the final fit and test access."""

    validate_multiclass_finalization_contract(model_selection_handoff)
    if model_selection_handoff.get("dataset_slug") != dataset_slug:
        raise FinalizationContractError("Dataset slug differs from model-selection handoff.")
    features = list(model_selection_handoff["selected_feature_columns"])
    if features != list(feature_manifest.get("feature_columns", ())):
        raise FinalizationContractError("Selected feature order differs from preparation.")
    if model_selection_handoff["target_classes"] != feature_manifest.get("target_classes"):
        raise FinalizationContractError("Target class order differs from preparation.")
    if model_selection_handoff["target_encoding"] != feature_manifest.get("target_encoding_contract"):
        raise FinalizationContractError("Target encoding differs from preparation.")
    test_path = split_manifest.get("partition_paths", {}).get("test")
    test_sha = split_manifest.get("partition_sha256", {}).get("test")
    test_rows = split_manifest.get("row_counts", {}).get("test")
    if not isinstance(test_path, str) or not isinstance(test_sha, str) or not isinstance(test_rows, int):
        raise FinalizationContractError("Frozen test integrity reference is incomplete.")
    instructions = model_selection_handoff["final_training_instructions"]
    contract = MulticlassFrozenFinalizationContract(
        dataset_slug=dataset_slug,
        problem_type="multiclass_classification",
        model_selection_handoff_path=_require_relative_path(
            model_selection_handoff_path, field="model_selection_handoff_path"
        ),
        model_selection_handoff_sha256=str(model_selection_handoff_sha256),
        model_id=str(model_selection_handoff["selected_model_id"]),
        model_family=str(model_selection_handoff["selected_model_family"]),
        hyperparameters=tuple(sorted(_deepcopy(model_selection_handoff["selected_hyperparameters"]).items())),
        random_state=int(model_selection_handoff["random_seeds"]["estimators"]),
        feature_policy=str(model_selection_handoff["selected_feature_policy"]),
        feature_columns=tuple(features),
        numerical_features=tuple(feature_manifest["numerical_features"]),
        categorical_features=tuple(feature_manifest["categorical_features"]),
        identifier_columns=tuple(feature_manifest["identifier_columns"]),
        target_column=str(feature_manifest["target_column"]),
        target_classes=tuple(feature_manifest["target_classes"]),
        target_encoding=tuple(
            (label, int(feature_manifest["target_encoding_contract"][label]))
            for label in feature_manifest["target_classes"]
        ),
        target_semantics=str(model_selection_handoff["target_semantics"]),
        preprocessing_contract=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_preprocessing_contract"]).items())
        ),
        imbalance_policy=tuple(
            sorted(_deepcopy(model_selection_handoff["selected_imbalance_policy"]).items())
        ),
        decision_rule=str(model_selection_handoff["decision_rule"]),
        training_partitions=tuple(instructions["fit_partitions"]),
        evaluation_partition=str(instructions["final_evaluation_partition"]),
        test_partition_path=test_path,
        test_partition_sha256=test_sha,
        test_partition_row_count=int(test_rows),
    )
    validate_multiclass_frozen_model_contract(contract)
    return contract


def validate_multiclass_frozen_model_contract(
    contract: MulticlassFrozenFinalizationContract,
) -> None:
    if contract.problem_type != "multiclass_classification":
        raise FinalizationContractError("Frozen problem type is not multiclass.")
    if contract.model_family != "HistGradientBoostingClassifier":
        raise FinalizationContractError("Unsupported frozen multiclass model family.")
    features = list(contract.feature_columns)
    if not features or len(features) != len(set(features)):
        raise FinalizationContractError("Frozen feature columns must be unique and non-empty.")
    if tuple(contract.numerical_features) != tuple(contract.feature_columns):
        raise FinalizationContractError("Dry Bean features must remain numerical-only.")
    if contract.categorical_features:
        raise FinalizationContractError("Dry Bean cannot acquire categorical features.")
    if (set(contract.identifier_columns) | {contract.target_column}) & set(features):
        raise FinalizationContractError("Target/identifiers cannot enter final features.")
    if len(contract.target_classes) < 3 or len(set(contract.target_classes)) != len(contract.target_classes):
        raise FinalizationContractError("Frozen multiclass target contract is invalid.")
    encoding = dict(contract.target_encoding)
    if list(encoding) != list(contract.target_classes):
        raise FinalizationContractError("Target encoding order must follow target contract order.")
    if sorted(encoding.values()) != list(range(len(contract.target_classes))):
        raise FinalizationContractError("Target encoding must be deterministic 0..K-1 metadata.")
    if contract.target_semantics != "nominal_unordered":
        raise FinalizationContractError("Frozen target semantics changed.")
    if contract.feature_policy != "all_features":
        raise FinalizationContractError("Frozen feature policy must remain all_features.")
    if contract.decision_rule != "argmax_class_score_or_probability":
        raise FinalizationContractError("Frozen decision rule changed.")
    if contract.training_partitions != ("train", "validation") or contract.evaluation_partition != "test":
        raise FinalizationContractError("Final partition roles changed.")
    if not contract.access_test_only_after_contract_freeze_and_final_fit or not contract.evaluate_test_once:
        raise FinalizationContractError("Test access policy changed.")
    if dict(contract.imbalance_policy) != {"class_weight": None, "resampling": "none", "strategy": "none"}:
        raise FinalizationContractError("Frozen imbalance policy changed.")
    preprocessing = dict(contract.preprocessing_contract)
    if preprocessing.get("numerical_scaling") != "none" or preprocessing.get("categorical_processing") != "not_applicable":
        raise FinalizationContractError("Frozen preprocessing changed.")
    if preprocessing.get("feature_projection") != list(contract.feature_columns):
        raise FinalizationContractError("Frozen preprocessing feature projection changed.")
    if contract.test_partition_row_count <= 0 or len(contract.test_partition_sha256) != 64:
        raise FinalizationContractError("Frozen test integrity reference is invalid.")


def validate_multiclass_final_partition_roles(
    frame: pd.DataFrame,
    *,
    contract: MulticlassFrozenFinalizationContract,
    partition_name: str,
) -> None:
    expected = [*contract.identifier_columns, *contract.feature_columns, contract.target_column]
    if list(frame.columns) != expected:
        raise FinalizationContractError(
            f"{partition_name} column order mismatch. expected={expected}, observed={list(frame.columns)}"
        )
    if frame[list(contract.feature_columns)].isna().any().any():
        raise FinalizationContractError(f"{partition_name} contains missing predictor values.")
    observed = set(frame[contract.target_column].dropna().tolist())
    if observed != set(contract.target_classes):
        raise FinalizationContractError(f"{partition_name} target classes differ from contract.")


def assemble_multiclass_final_training_data(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    contract: MulticlassFrozenFinalizationContract,
) -> MulticlassFinalTrainingData:
    """Assemble train+validation in persisted order; no test argument exists."""

    train_copy = train.copy(deep=True)
    validation_copy = validation.copy(deep=True)
    validate_multiclass_final_partition_roles(train_copy, contract=contract, partition_name="train")
    validate_multiclass_final_partition_roles(validation_copy, contract=contract, partition_name="validation")
    combined = pd.concat([train_copy, validation_copy], axis=0, ignore_index=True)
    features = combined.loc[:, list(contract.feature_columns)].copy(deep=True)
    target = combined[contract.target_column].astype("string").copy(deep=True)
    counts = tuple(
        (str(label), int((target == label).sum())) for label in contract.target_classes
    )
    if set(target.unique().tolist()) != set(contract.target_classes):
        raise FinalizationContractError("Final training target does not contain every class.")
    return MulticlassFinalTrainingData(features, target, counts)


def reconstruct_multiclass_selected_pipeline(
    *, contract: MulticlassFrozenFinalizationContract
) -> Pipeline:
    """Reconstruct exactly the Notebook-03 HGB pipeline and parameters."""

    validate_multiclass_frozen_model_contract(contract)
    estimator = HistGradientBoostingClassifier(random_state=contract.random_state)
    accepted = estimator.get_params(deep=True)
    parameters: dict[str, Any] = {}
    for key, value in contract.hyperparameters:
        if not key.startswith("model__"):
            raise FinalizationContractError(f"Selected hyperparameter lacks model__ prefix: {key}")
        name = key.removeprefix("model__")
        if name not in accepted:
            raise FinalizationContractError(f"Estimator does not accept selected parameter: {name}")
        parameters[name] = value
    parameters["random_state"] = contract.random_state
    estimator.set_params(**parameters)
    pipeline = build_candidate_pipeline(
        estimator=estimator,
        numerical_features=contract.feature_columns,
        categorical_features=(),
        scale_numerical=False,
    )
    verify_multiclass_pipeline_contract(pipeline, contract=contract, require_fitted=False)
    return pipeline


def verify_multiclass_pipeline_contract(
    pipeline: Pipeline,
    *,
    contract: MulticlassFrozenFinalizationContract,
    require_fitted: bool,
) -> None:
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["preprocess", "model"]:
        raise FinalizationContractError("Final multiclass artifact must be preprocess+model Pipeline.")
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not isinstance(preprocess, ColumnTransformer):
        raise FinalizationContractError("Multiclass preprocess step must be ColumnTransformer.")
    if model.__class__.__name__ != contract.model_family:
        raise FinalizationContractError("Fitted estimator family differs from contract.")
    if preprocess.remainder != "drop" or float(preprocess.sparse_threshold) != 0.0:
        raise FinalizationContractError("Multiclass ColumnTransformer policy changed.")
    transformers = {name: (transformer, list(columns)) for name, transformer, columns in preprocess.transformers}
    if set(transformers) != {"numerical"}:
        raise FinalizationContractError("Multiclass pipeline acquired an unexpected transformer.")
    transformer, columns = transformers["numerical"]
    if transformer != "passthrough" or columns != list(contract.feature_columns):
        raise FinalizationContractError("Numerical passthrough feature order changed.")
    params = model.get_params(deep=False)
    for key, expected in contract.hyperparameters:
        if params.get(key.removeprefix("model__")) != expected:
            raise FinalizationContractError(f"Model parameter mismatch: {key}")
    if params.get("random_state") != contract.random_state:
        raise FinalizationContractError("Estimator random_state mismatch.")
    fitted = _is_fitted(pipeline)
    if require_fitted and not fitted:
        raise FinalizationContractError("Multiclass pipeline must be fitted.")
    if not require_fitted and fitted:
        raise FinalizationContractError("Multiclass pipeline must be unfitted before final fit.")
    if require_fitted:
        classes = list(_jsonable(model.classes_))
        if set(classes) != set(contract.target_classes):
            raise FinalizationContractError("Estimator fitted classes differ from target contract.")


def validate_multiclass_test_access_gate(
    *,
    contract: MulticlassFrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    project_root: str | Path,
) -> Path:
    """Hash the sealed test bytes before any CSV parsing is allowed."""

    validate_multiclass_frozen_model_contract(contract)
    verify_multiclass_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    root = Path(project_root).resolve()
    relative = _require_relative_path(contract.test_partition_path, field="test_partition_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TestAccessError("Test path escapes project root.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Test partition is missing: {relative}")
    observed = sha256_file(path)
    if observed != contract.test_partition_sha256:
        raise TestAccessError(
            f"Test SHA-256 mismatch: expected={contract.test_partition_sha256}, observed={observed}"
        )
    return path


def load_multiclass_test_partition_after_fit(
    *,
    project_root: str | Path,
    fitted_pipeline: Pipeline,
    contract: MulticlassFrozenFinalizationContract,
) -> MulticlassTestPartitionData:
    path = validate_multiclass_test_access_gate(
        contract=contract, fitted_pipeline=fitted_pipeline, project_root=project_root
    )
    frame = pd.read_csv(path)
    validate_multiclass_final_partition_roles(frame, contract=contract, partition_name="test")
    if len(frame) != contract.test_partition_row_count:
        raise TestAccessError("Test row count differs from the sealed contract.")
    target = frame[contract.target_column].astype("string")
    counts = tuple(
        (str(label), int((target == label).sum())) for label in contract.target_classes
    )
    return MulticlassTestPartitionData(
        frame.loc[:, list(contract.feature_columns)],
        target,
        int(len(frame)),
        counts,
        contract.test_partition_path,
        contract.test_partition_sha256,
    )


def _rank_multiclass_confusion_pairs(
    confusion: Mapping[str, Any], target_classes: Sequence[Any]
) -> list[dict[str, Any]]:
    classes = list(target_classes)
    if list(confusion.get("class_order", ())) != classes:
        raise FinalizationContractError("Confusion matrix class order differs from contract.")
    counts = np.asarray(confusion["counts"], dtype=int)
    normalized = np.asarray(confusion["row_normalized"], dtype=float)
    if counts.shape != (len(classes), len(classes)):
        raise FinalizationContractError("Confusion matrix shape differs from contract.")
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(classes):
        for right_index in range(left_index + 1, len(classes)):
            right = classes[right_index]
            pairs.append(
                {
                    "class_pair": [left, right],
                    "left_to_right_count": int(counts[left_index, right_index]),
                    "right_to_left_count": int(counts[right_index, left_index]),
                    "mutual_confusion_count": int(
                        counts[left_index, right_index] + counts[right_index, left_index]
                    ),
                    "mutual_row_normalized_rate": float(
                        normalized[left_index, right_index]
                        + normalized[right_index, left_index]
                    ),
                }
            )
    pairs.sort(
        key=lambda row: (
            -row["mutual_confusion_count"],
            -row["mutual_row_normalized_rate"],
            tuple(str(value) for value in row["class_pair"]),
        )
    )
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
    return pairs


def compare_multiclass_confusion_pairs(
    *,
    validation_confusion: Mapping[str, Any],
    test_confusion: Mapping[str, Any],
    target_classes: Sequence[Any],
    focal_pairs: Sequence[Sequence[Any]] = (("DERMASON", "SIRA"), ("BARBUNYA", "CALI")),
) -> dict[str, Any]:
    validation_pairs = _rank_multiclass_confusion_pairs(validation_confusion, target_classes)
    test_pairs = _rank_multiclass_confusion_pairs(test_confusion, target_classes)

    def locate(rows: Sequence[Mapping[str, Any]], pair: Sequence[Any]) -> Mapping[str, Any]:
        wanted = set(pair)
        return next(row for row in rows if set(row["class_pair"]) == wanted)

    comparisons = []
    for pair in focal_pairs:
        validation = locate(validation_pairs, pair)
        test = locate(test_pairs, pair)
        delta = int(test["mutual_confusion_count"] - validation["mutual_confusion_count"])
        comparisons.append(
            {
                "class_pair": list(pair),
                "validation": _deepcopy(validation),
                "test": _deepcopy(test),
                "mutual_confusion_count_delta_test_minus_validation": delta,
                "pattern_direction": (
                    "strengthened" if delta > 0 else "weakened" if delta < 0 else "persisted"
                ),
                "ranking_changed": test["rank"] != validation["rank"],
            }
        )
    return {
        "ranked_test_pairs": test_pairs,
        "focal_pair_comparisons": comparisons,
        "interpretation_boundary": (
            "Descriptive generalization evidence only; confusion changes do not reopen model selection."
        ),
    }


def compute_multiclass_generalization_review(
    *,
    validation_evidence: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    validation_metrics = validation_evidence["metrics"]
    test_metrics = test_evidence["metrics"]
    metrics = (
        "macro_f1",
        "balanced_accuracy",
        "macro_recall",
        "weighted_f1",
        "accuracy",
        "log_loss",
    )
    deltas = {
        f"{metric}_test_minus_validation": float(test_metrics[metric] - validation_metrics[metric])
        for metric in metrics
    }
    validation_by_class = {
        str(row["class"]): row for row in validation_evidence["per_class"]
    }
    test_by_class = {str(row["class"]): row for row in test_evidence["per_class"]}
    per_class = [
        {
            "class": label,
            "validation_recall": float(validation_by_class[str(label)]["recall"]),
            "test_recall": float(test_by_class[str(label)]["recall"]),
            "recall_delta_test_minus_validation": float(
                test_by_class[str(label)]["recall"]
                - validation_by_class[str(label)]["recall"]
            ),
        }
        for label in test_evidence["confusion_matrix"]["class_order"]
    ]
    validation_worst = min(validation_evidence["per_class"], key=lambda row: row["recall"])
    test_worst = min(test_evidence["per_class"], key=lambda row: row["recall"])
    return {
        "aggregate_deltas": deltas,
        "per_class_recall": per_class,
        "validation_worst_recall_class": validation_worst["class"],
        "validation_worst_recall": float(validation_worst["recall"]),
        "test_worst_recall_class": test_worst["class"],
        "test_worst_recall": float(test_worst["recall"]),
        "selection_reopened": False,
        "interpretation": "Descriptive comparison only; no pass/fail gate or post-test adjustment.",
    }


def compute_multiclass_repeated_profile_sensitivity(
    *,
    final_training_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_test: pd.Series,
    predictions: Sequence[Any],
    probabilities: np.ndarray,
    target_classes: Sequence[Any],
    official_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if list(final_training_features.columns) != list(test_features.columns):
        raise FinalizationContractError("Repeated-profile feature order differs.")
    train_profiles = pd.MultiIndex.from_frame(final_training_features.reset_index(drop=True))
    test_profiles = pd.MultiIndex.from_frame(test_features.reset_index(drop=True))
    repeated_mask = test_profiles.isin(train_profiles)
    keep_mask = ~repeated_mask
    filtered: Mapping[str, Any] | None = None
    if int(keep_mask.sum()) > 0:
        filtered = compute_multiclass_metrics(
            y_true=y_test.reset_index(drop=True)[keep_mask],
            y_pred=pd.Series(predictions)[keep_mask],
            target_classes=target_classes,
            probabilities=np.asarray(probabilities)[keep_mask],
            probability_class_order=target_classes,
        )["metrics"]
    return {
        "analysis_type": "non_destructive_repeated_feature_profile_sensitivity",
        "matching_feature_columns": list(final_training_features.columns),
        "official_full_test_row_count": int(len(test_features)),
        "repeated_profile_test_row_count": int(repeated_mask.sum()),
        "sensitivity_row_count": int(keep_mask.sum()),
        "official_full_test_metrics": {
            key: _deepcopy(official_metrics[key])
            for key in (
                "macro_f1",
                "balanced_accuracy",
                "minimum_per_class_recall",
                "row_count",
            )
        },
        "excluding_repeated_profile_metrics": (
            None
            if filtered is None
            else {
                key: _deepcopy(filtered[key])
                for key in (
                    "macro_f1",
                    "balanced_accuracy",
                    "minimum_per_class_recall",
                    "row_count",
                )
            }
        ),
        "macro_f1_delta_excluding_minus_full": (
            None if filtered is None else float(filtered["macro_f1"] - official_metrics["macro_f1"])
        ),
        "interpretation": (
            "Sensitivity only: the official test artifact remains complete; Repeated-profile evidence does not prove duplicate identity or leakage."
        ),
    }


def evaluate_multiclass_final_model_once(
    *,
    fitted_pipeline: Pipeline,
    test_partition: MulticlassTestPartitionData,
    final_training_features: pd.DataFrame,
    contract: MulticlassFrozenFinalizationContract,
    validation_evidence: Mapping[str, Any],
    guard: EvaluationGuard,
) -> MulticlassFinalEvaluation:
    """Evaluate test exactly once; no search space or alternative candidate is accepted."""

    if guard.evaluated or guard.probability_call_count:
        raise DuplicateTestEvaluationError("Test probability evaluation was already consumed.")
    verify_multiclass_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    features = test_partition.features
    target = test_partition.target
    guard.evaluated = True
    guard.probability_call_count += 1
    raw_probabilities = np.asarray(fitted_pipeline.predict_proba(features), dtype=float)
    estimator_order = list(_jsonable(fitted_pipeline.named_steps["model"].classes_))
    output_order = list(contract.target_classes)
    if set(estimator_order) != set(output_order):
        raise FinalizationContractError("Estimator probability classes differ from target contract.")
    indices = [estimator_order.index(label) for label in output_order]
    probabilities = raw_probabilities[:, indices].copy()
    if probabilities.shape != (test_partition.row_count, len(output_order)):
        raise FinalizationContractError("Test probability matrix shape differs from contract.")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise FinalizationContractError("Test probabilities are invalid.")
    predictions = np.asarray(output_order, dtype=object)[np.argmax(probabilities, axis=1)]
    computed = compute_multiclass_metrics(
        y_true=target,
        y_pred=predictions,
        target_classes=output_order,
        probabilities=probabilities,
        probability_class_order=output_order,
    )
    generalization = compute_multiclass_generalization_review(
        validation_evidence=validation_evidence,
        test_evidence=computed,
    )
    pairs = compare_multiclass_confusion_pairs(
        validation_confusion=validation_evidence["confusion_matrix"],
        test_confusion=computed["confusion_matrix"],
        target_classes=output_order,
    )
    sensitivity = compute_multiclass_repeated_profile_sensitivity(
        final_training_features=final_training_features,
        test_features=features,
        y_test=target,
        predictions=predictions,
        probabilities=probabilities,
        target_classes=output_order,
        official_metrics=computed["metrics"],
    )
    probability_bytes = pd.DataFrame(probabilities, columns=output_order).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return MulticlassFinalEvaluation(
        metrics=computed["metrics"],
        per_class=tuple(computed["per_class"]),
        confusion_matrix=computed["confusion_matrix"],
        estimator_class_order=tuple(estimator_order),
        output_class_order=tuple(output_order),
        validation_to_test=generalization,
        confusion_pair_comparison=pairs,
        repeated_profile_sensitivity=sensitivity,
        probability_sha256=sha256_bytes(probability_bytes),
    )


def describe_multiclass_fitted_pipeline(
    *,
    pipeline: Pipeline,
    contract: MulticlassFrozenFinalizationContract,
) -> dict[str, Any]:
    """Return a fitted-state descriptor that can be recomputed after reload."""

    verify_multiclass_pipeline_contract(pipeline, contract=contract, require_fitted=True)
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    return _jsonable(
        {
            "pipeline_class": f"{pipeline.__class__.__module__}.{pipeline.__class__.__name__}",
            "steps": list(pipeline.named_steps),
            "preprocess_class": f"{preprocess.__class__.__module__}.{preprocess.__class__.__name__}",
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "selected_hyperparameters": dict(contract.hyperparameters),
            "random_state": contract.random_state,
            "feature_order": list(contract.feature_columns),
            "preprocessing_contract": dict(contract.preprocessing_contract),
            "imbalance_policy": dict(contract.imbalance_policy),
            "transformed_feature_names": preprocess.get_feature_names_out().tolist(),
            "estimator_class_order": _jsonable(model.classes_),
            "output_class_order": list(contract.target_classes),
            "fitted_state": {
                "n_iter": int(model.n_iter_),
                "do_early_stopping": bool(model.do_early_stopping_),
                "is_fitted": True,
            },
            "runtime_versions": runtime_versions(),
        }
    )


def serialize_multiclass_pipeline_to_staging(
    *, pipeline: Pipeline, staging_path: str | Path
) -> str:
    if not isinstance(pipeline, Pipeline) or not _is_fitted(pipeline):
        raise SerializationValidationError("Only a fitted multiclass Pipeline can be serialized.")
    path = Path(staging_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return sha256_file(path)


def validate_multiclass_serialized_pipeline(
    *,
    staging_path: str | Path,
    expected_sha256: str,
    contract: MulticlassFrozenFinalizationContract,
    reference_pipeline: Pipeline,
    validation_sample: pd.DataFrame,
) -> Pipeline:
    path = Path(staging_path)
    if sha256_file(path) != expected_sha256:
        raise SerializationValidationError("Serialized multiclass pipeline SHA-256 mismatch.")
    loaded = joblib.load(path)
    verify_multiclass_pipeline_contract(loaded, contract=contract, require_fitted=True)
    sample = validation_sample.loc[:, list(contract.feature_columns)].copy(deep=True)
    expected_predictions = reference_pipeline.predict(sample)
    loaded_predictions = loaded.predict(sample)
    if not np.array_equal(expected_predictions, loaded_predictions):
        raise SerializationValidationError("Round-trip multiclass predictions differ.")
    expected_probabilities = reference_pipeline.predict_proba(sample)
    loaded_probabilities = loaded.predict_proba(sample)
    if not np.allclose(expected_probabilities, loaded_probabilities, rtol=0.0, atol=0.0):
        raise SerializationValidationError("Round-trip multiclass probabilities differ.")
    descriptor = describe_multiclass_fitted_pipeline(pipeline=reference_pipeline, contract=contract)
    loaded_descriptor = describe_multiclass_fitted_pipeline(pipeline=loaded, contract=contract)
    if compute_fitted_model_fingerprint(descriptor) != compute_fitted_model_fingerprint(loaded_descriptor):
        raise SerializationValidationError("Round-trip fitted-state fingerprint differs.")
    return loaded


def validate_multiclass_inference_input(
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    frame = (
        value.copy(deep=True)
        if isinstance(value, pd.DataFrame)
        else pd.DataFrame([value.to_dict() if isinstance(value, pd.Series) else dict(value)])
    )
    required = list(bundle["required_input_columns"])
    if list(frame.columns) != required:
        raise FinalizationContractError(
            f"Inference input columns/order mismatch. expected={required}, observed={list(frame.columns)}"
        )
    if frame.isna().any().any():
        raise FinalizationContractError("Missing required inference values must be rejected.")
    for column in required:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise FinalizationContractError(f"Inference feature must be finite numeric: {column}")
        frame[column] = converted
    return frame


def smoke_predict_multiclass_bundle(
    pipeline: Pipeline,
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Technical v2 smoke inference; this is not Notebook 05's demo workflow."""

    _validate_multiclass_json_artifact("inference-bundle.json", bundle)
    frame = validate_multiclass_inference_input(value, bundle=bundle)
    raw = np.asarray(pipeline.predict_proba(frame), dtype=float)
    estimator_order = list(_jsonable(pipeline.named_steps["model"].classes_))
    output_order = list(bundle["output_class_order"])
    if estimator_order != list(bundle["estimator_class_order"]):
        raise SerializationValidationError("Loaded estimator class order differs from bundle.")
    if set(estimator_order) != set(output_order):
        raise SerializationValidationError("Loaded estimator classes differ from output contract.")
    ordered = raw[:, [estimator_order.index(label) for label in output_order]]
    if ordered.shape[1] != len(output_order):
        raise SerializationValidationError("Smoke probability column count differs.")
    if not np.isfinite(ordered).all() or not np.allclose(ordered.sum(axis=1), 1.0, atol=1e-8):
        raise SerializationValidationError("Smoke probabilities are invalid.")
    predicted = np.asarray(output_order, dtype=object)[np.argmax(ordered, axis=1)]
    rows = []
    for index, label in enumerate(predicted):
        rows.append(
            {
                "predicted_class": label,
                "class_order": list(output_order),
                "class_probabilities": ordered[index].astype(float).tolist(),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


def validate_multiclass_upstream_lineage_metadata_only(
    *,
    project_root: str | Path,
    preparation_handoff_path: str | Path,
    model_selection_handoff_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate upstream bytes and contracts without parsing the test CSV."""

    root = Path(project_root).resolve()
    preparation_relative = _require_relative_path(
        preparation_handoff_path, field="preparation_handoff_path"
    )
    preparation_path = root / preparation_relative
    preparation_handoff = _load_json(preparation_path)
    if preparation_handoff.get("schema_version") != "preparation-handoff.v1":
        raise UpstreamHandoffError("Preparation handoff schema is invalid.")
    components = preparation_handoff.get("components", {})
    required_components = {
        "preparation_manifest",
        "feature_manifest",
        "split_manifest",
        "quality_evidence",
    }
    if set(components) != required_components:
        raise UpstreamHandoffError("Preparation handoff components are incomplete.")
    manifests: dict[str, dict[str, Any]] = {"preparation_handoff": preparation_handoff}
    for name in sorted(required_components):
        reference = components[name]
        relative = _require_relative_path(reference["path"], field=f"components.{name}.path")
        path = root / relative
        if sha256_file(path) != reference["sha256"]:
            raise UpstreamHandoffError(f"Preparation component hash mismatch: {name}")
        manifests[name] = _load_json(path)
    split = manifests["split_manifest"]
    for partition in ("train", "validation", "test"):
        relative = _require_relative_path(
            split["partition_paths"][partition], field=f"partition_paths.{partition}"
        )
        if sha256_file(root / relative) != split["partition_sha256"][partition]:
            raise UpstreamHandoffError(f"Prepared partition hash mismatch: {partition}")
    prepared_manifest = manifests["preparation_manifest"]
    prepared_relative = _require_relative_path(
        prepared_manifest["prepared_path"], field="prepared_path"
    )
    if sha256_file(root / prepared_relative) != prepared_manifest["prepared_sha256"]:
        raise UpstreamHandoffError("Prepared dataset hash mismatch.")

    selection_relative = _require_relative_path(
        model_selection_handoff_path, field="model_selection_handoff_path"
    )
    selection_path = root / selection_relative
    selection = _load_json(selection_path)
    validate_multiclass_finalization_contract(selection)
    manifest_path = selection_path.parent / "model-selection-manifest.json"
    selection_manifest = _load_json(manifest_path)
    if selection_manifest.get("schema_version") != "model-selection-manifest.v2":
        raise UpstreamHandoffError("Model-selection manifest schema is invalid.")
    for filename, fingerprint in selection_manifest.get("artifact_fingerprints", {}).items():
        if sha256_file(selection_path.parent / filename) != fingerprint.get("byte_sha256"):
            raise UpstreamHandoffError(f"Model-selection artifact hash mismatch: {filename}")
    preparation_reference = selection.get("preparation_handoff_reference", {})
    if preparation_reference.get("path") != preparation_relative:
        raise UpstreamHandoffError("Model-selection preparation lineage path differs.")
    if sha256_file(preparation_path) != preparation_reference.get("byte_sha256"):
        raise UpstreamHandoffError("Model-selection preparation lineage hash differs.")
    feature = manifests["feature_manifest"]
    if selection.get("available_feature_columns") != feature.get("feature_columns"):
        raise UpstreamHandoffError("Feature contract differs between handoffs.")
    if selection.get("target_classes") != feature.get("target_classes"):
        raise UpstreamHandoffError("Target class order differs between handoffs.")
    return _deepcopy(manifests), _deepcopy(selection)


_validate_upstream_handoff_contracts_v1 = validate_upstream_handoff_contracts


def validate_upstream_handoff_contracts(
    *,
    project_root: str | Path,
    model_selection_handoff_path: str | Path,
    preparation_paths: Mapping[str, str | Path] | None = None,
    preparation_handoff_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load v1 or v2 upstream contracts while preserving the v1 API."""

    root = Path(project_root).resolve()
    selection_relative = _require_relative_path(
        model_selection_handoff_path, field="model_selection_handoff_path"
    )
    selection_preview = _load_json(root / selection_relative)
    if selection_preview.get("schema_version") == "model-selection-handoff.v1":
        if preparation_paths is None:
            raise UpstreamHandoffError("Binary v1 requires the four preparation paths.")
        return _validate_upstream_handoff_contracts_v1(
            project_root=root,
            preparation_paths=preparation_paths,
            model_selection_handoff_path=selection_relative,
        )
    if selection_preview.get("schema_version") != "model-selection-handoff.v2":
        raise FinalizationContractError("Unsupported model-selection handoff schema.")
    if preparation_handoff_path is None:
        preparation_handoff_path = selection_preview.get("preparation_handoff_reference", {}).get("path")
    if not isinstance(preparation_handoff_path, (str, Path)):
        raise UpstreamHandoffError("Multiclass preparation handoff path is missing.")
    _, selection = validate_multiclass_upstream_lineage_metadata_only(
        project_root=root,
        preparation_handoff_path=preparation_handoff_path,
        model_selection_handoff_path=selection_relative,
    )
    preparation = load_and_validate_preparation_handoff(
        project_root=root,
        preparation_handoff_path=preparation_handoff_path,
    )
    return preparation, selection


def build_multiclass_final_test_evidence(
    *,
    contract: MulticlassFrozenFinalizationContract,
    test_partition: MulticlassTestPartitionData,
    evaluation: MulticlassFinalEvaluation,
    validation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "final-test-evidence.v2",
        "artifact_type": "final_test_evidence",
        "dataset_slug": contract.dataset_slug,
        "problem_type": contract.problem_type,
        "partition": "test",
        "partition_path": test_partition.partition_path,
        "partition_sha256": test_partition.partition_sha256,
        "row_count": test_partition.row_count,
        "class_counts": dict(test_partition.class_counts),
        "frozen_finalization_contract_fingerprint": contract.fingerprint,
        "metrics": _deepcopy(evaluation.metrics),
        "per_class": _deepcopy(list(evaluation.per_class)),
        "confusion_matrix": _deepcopy(evaluation.confusion_matrix),
        "estimator_class_order": list(evaluation.estimator_class_order),
        "output_class_order": list(evaluation.output_class_order),
        "decision_rule": contract.decision_rule,
        "selected_validation_evidence": {
            "metrics": _deepcopy(validation_evidence["metrics"]),
            "per_class": _deepcopy(validation_evidence["per_class"]),
            "confusion_matrix": _deepcopy(validation_evidence["confusion_matrix"]),
        },
        "validation_to_test": _deepcopy(evaluation.validation_to_test),
        "confusion_pair_comparison": _deepcopy(evaluation.confusion_pair_comparison),
        "repeated_profile_sensitivity": _deepcopy(evaluation.repeated_profile_sensitivity),
        "probability_matrix_sha256_aggregate_only": evaluation.probability_sha256,
        "test_loaded_only_after_final_fit": True,
        "test_probability_evaluation_count": evaluation.test_probability_evaluation_count,
        "test_partition_evaluation_count": evaluation.test_probability_evaluation_count,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "test_used_for_feature_selection": False,
        "test_used_for_preprocessing_selection": False,
        "test_used_for_imbalance_policy_selection": False,
        "no_post_test_adjustment": True,
        "individual_rows_persisted": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
    }


def build_multiclass_inference_bundle(
    *,
    contract: MulticlassFrozenFinalizationContract,
    fitted_pipeline: Pipeline,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    model_state_descriptor: Mapping[str, Any],
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
    upstream_references: Mapping[str, Any],
    final_artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    verify_multiclass_pipeline_contract(fitted_pipeline, contract=contract, require_fitted=True)
    estimator_order = list(_jsonable(fitted_pipeline.named_steps["model"].classes_))
    output_order = list(contract.target_classes)
    if set(estimator_order) != set(output_order):
        raise FinalizationContractError("Estimator class set differs from inference output contract.")
    expected_schema = [
        {
            "name": column,
            "role": "numerical",
            "required": True,
            "expected_dtype": expected_input_dtypes[column],
            "missing_value_behavior": "reject",
        }
        for column in contract.feature_columns
    ]
    return {
        "schema_version": "inference-bundle.v2",
        "artifact_type": "inference_bundle",
        "bundle_version": "2.0.0",
        "dataset_slug": contract.dataset_slug,
        "problem_type": contract.problem_type,
        "model_id": contract.model_id,
        "model_family": contract.model_family,
        "model_artifact_path": _require_relative_path(
            model_artifact_path, field="model_artifact_path"
        ),
        "model_artifact_format": "joblib",
        "model_artifact_sha256": model_artifact_sha256,
        "model_state_fingerprint": model_state_fingerprint,
        "model_state_descriptor": _deepcopy(model_state_descriptor),
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "feature_policy": contract.feature_policy,
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns_excluded": list(contract.identifier_columns),
        "target_column": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding_metadata_only": dict(contract.target_encoding),
        "target_semantics": contract.target_semantics,
        "estimator_class_order": estimator_order,
        "output_class_order": output_order,
        "decision_rule": contract.decision_rule,
        "preprocessing_contract": dict(contract.preprocessing_contract),
        "imbalance_policy": dict(contract.imbalance_policy),
        "expected_input_schema": expected_schema,
        "expected_input_dtypes": _deepcopy(expected_input_dtypes),
        "required_input_columns": list(contract.feature_columns),
        "prohibited_input_columns": [*contract.identifier_columns, contract.target_column],
        "missing_value_policy": _deepcopy(missing_value_policy),
        "runtime_version_requirements": runtime_versions(),
        "inference_output_contract": {
            "predicted_class": {"type": "string", "allowed_values": output_order},
            "class_order": output_order,
            "class_probabilities": {
                "type": "array",
                "length": len(output_order),
                "aligned_to": "class_order",
                "finite": True,
                "row_sum": 1.0,
            },
            "decision_rule": contract.decision_rule,
            "binary_threshold": "not_applicable",
            "operational_prediction_available": False,
        },
        "lineage": _deepcopy(upstream_references),
        "artifact_set": {
            "final_model_manifest_path": final_artifact_paths["final-model-manifest.json"],
            "final_test_evidence_path": final_artifact_paths["final-test-evidence.json"],
            "final_model_handoff_path": final_artifact_paths["final-model-handoff.json"],
        },
        "test_partition_reference": {
            "path": contract.test_partition_path,
            "byte_sha256": contract.test_partition_sha256,
            "row_count": contract.test_partition_row_count,
        },
        "readiness": {
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "serialization_reload_validated": True,
            "inference_smoke_test_completed": True,
            "educational_inference_demo_ready": True,
            "operational_modeling_ready": False,
        },
        "limitations": [
            "Educational static-snapshot model; production validity is unconfirmed.",
            "Production feature availability, drift monitoring, SLOs, retraining, and deployment validation are absent.",
            "ShapeFactor2 source provenance remains unresolved despite frozen predictive inclusion.",
        ],
        "security_note": "Load joblib only from a trusted source after verifying its exact SHA-256.",
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
    }


def build_multiclass_final_model_manifest(
    *,
    contract: MulticlassFrozenFinalizationContract,
    upstream_references: Mapping[str, Any],
    training_data: MulticlassFinalTrainingData,
    test_partition: MulticlassTestPartitionData,
    fit_duration_seconds: float,
    model_artifact_path: str,
    model_artifact_sha256: str,
    model_state_fingerprint: str,
    model_state_descriptor: Mapping[str, Any],
    final_artifact_paths: Mapping[str, str],
    final_artifact_fingerprints: Mapping[str, Mapping[str, str]],
    analysis_conclusions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-manifest.v2",
        "artifact_type": "final_model_manifest",
        "dataset_slug": contract.dataset_slug,
        "problem_type": contract.problem_type,
        "upstream_references": _deepcopy(upstream_references),
        "frozen_finalization_contract": contract.as_dict(),
        "frozen_finalization_contract_fingerprint": contract.fingerprint,
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "feature_policy": contract.feature_policy,
        "feature_columns": list(contract.feature_columns),
        "numerical_features": list(contract.numerical_features),
        "categorical_features": list(contract.categorical_features),
        "identifier_columns": list(contract.identifier_columns),
        "preprocessing_contract": dict(contract.preprocessing_contract),
        "imbalance_policy": dict(contract.imbalance_policy),
        "target_column": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding_metadata_only": dict(contract.target_encoding),
        "target_semantics": contract.target_semantics,
        "decision_rule": contract.decision_rule,
        "estimator_class_order": _deepcopy(model_state_descriptor["estimator_class_order"]),
        "output_class_order": list(contract.target_classes),
        "final_training_partitions": list(contract.training_partitions),
        "final_training_row_count": training_data.row_count,
        "final_training_class_counts": dict(training_data.class_counts),
        "test_partition_reference": {
            "path": test_partition.partition_path,
            "byte_sha256": test_partition.partition_sha256,
            "row_count": test_partition.row_count,
            "class_counts": dict(test_partition.class_counts),
        },
        "test_access_policy": {
            "loaded_after_final_fit": True,
            "test_evaluation_count": 1,
            "used_for_adjustment": False,
        },
        "model_artifact_path": _require_relative_path(
            model_artifact_path, field="model_artifact_path"
        ),
        "artifact_format": "joblib",
        "model_artifact_byte_sha256": model_artifact_sha256,
        "model_state_fingerprint": model_state_fingerprint,
        "model_state_descriptor": _deepcopy(model_state_descriptor),
        "fit_duration_seconds": float(fit_duration_seconds),
        "runtime_versions": runtime_versions(),
        "final_artifact_paths": _deepcopy(final_artifact_paths),
        "final_artifact_fingerprints": _deepcopy(final_artifact_fingerprints),
        "analysis_conclusions": {
            "shape_factor_2": _deepcopy(analysis_conclusions["shape_factor_2"]),
            "confirmed_derived_feature_ablation": _deepcopy(
                analysis_conclusions["confirmed_derived_feature_ablation"]
            ),
        },
        "readiness": {
            "preparation_handoff_validated": True,
            "model_selection_handoff_validated": True,
            "frozen_finalization_contract_validated": True,
            "final_training_completed": True,
            "final_model_trained": True,
            "test_partition_opened_after_final_fit": True,
            "final_test_evaluation_completed": True,
            "test_partition_evaluation_count": 1,
            "test_used_for_adjustment": False,
            "model_artifact_materialized": True,
            "model_bundle_materialized": True,
            "serialization_reload_validated": True,
            "inference_smoke_test_completed": True,
            "educational_final_model_completed": True,
            "educational_inference_demo_ready": True,
            "final_model_handoff_ready": True,
            "operational_modeling_ready": False,
        },
        "limitations": [
            "Educational final model completed; production approval is not claimed.",
            "Operational validity remains unconfirmed.",
            "ShapeFactor2 source provenance remains unresolved; predictive sensitivity does not resolve source provenance.",
            "No production data contract, monitoring, SLO, retraining, or deployment validation exists.",
        ],
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
    }


def build_multiclass_final_model_handoff(
    *,
    contract: MulticlassFrozenFinalizationContract,
    upstream_references: Mapping[str, Any],
    final_references: Mapping[str, Mapping[str, str]],
    evaluation: MulticlassFinalEvaluation,
    analysis_conclusions: Mapping[str, Any],
    runtime_requirements: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "final-model-handoff.v2",
        "artifact_type": "final_model_handoff",
        "dataset_slug": contract.dataset_slug,
        "problem_type": contract.problem_type,
        "upstream_references": _deepcopy(upstream_references),
        "final_references": _deepcopy(final_references),
        "frozen_finalization_contract": contract.as_dict(),
        "frozen_finalization_contract_fingerprint": contract.fingerprint,
        "model_state_fingerprint": final_references["model_artifact"]["semantic_sha256"],
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "selected_hyperparameters": dict(contract.hyperparameters),
        "estimator_random_state": contract.random_state,
        "feature_policy": contract.feature_policy,
        "feature_order": list(contract.feature_columns),
        "preprocessing": dict(contract.preprocessing_contract),
        "imbalance_policy": dict(contract.imbalance_policy),
        "target_column": contract.target_column,
        "target_classes": list(contract.target_classes),
        "target_encoding_metadata_only": dict(contract.target_encoding),
        "target_semantics": contract.target_semantics,
        "decision_rule": contract.decision_rule,
        "estimator_class_order": list(evaluation.estimator_class_order),
        "output_class_order": list(evaluation.output_class_order),
        "final_training_partitions": list(contract.training_partitions),
        "final_evaluation_partition": contract.evaluation_partition,
        "final_test_metrics": _deepcopy(evaluation.metrics),
        "final_test_per_class": _deepcopy(list(evaluation.per_class)),
        "final_test_confusion_matrix": _deepcopy(evaluation.confusion_matrix),
        "generalization_evidence": _deepcopy(evaluation.validation_to_test),
        "confusion_pair_evidence": _deepcopy(evaluation.confusion_pair_comparison),
        "repeated_profile_sensitivity": _deepcopy(evaluation.repeated_profile_sensitivity),
        "analysis_conclusions": {
            "shape_factor_2": _deepcopy(analysis_conclusions["shape_factor_2"]),
            "confirmed_derived_feature_ablation": _deepcopy(
                analysis_conclusions["confirmed_derived_feature_ablation"]
            ),
        },
        "runtime_requirements": _deepcopy(runtime_requirements),
        "notebook_05_instructions": [
            "Validate final-model-handoff.v2 and inference-bundle.v2 in a new process.",
            "Verify every sibling hash and the model SHA-256 before trusted joblib loading.",
            "Do not refit, retune, resplit, or access test.",
            "Require the 16 numerical features in the frozen order and reject missing values.",
            "Return predicted_class plus seven probabilities aligned to class_order.",
            "Keep binary thresholds and positive-class semantics not applicable.",
        ],
        "preparation_handoff_validated": True,
        "model_selection_handoff_validated": True,
        "frozen_finalization_contract_validated": True,
        "final_training_completed": True,
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "test_partition_opened_after_final_fit": True,
        "final_test_evaluation_completed": True,
        "test_partition_evaluated": True,
        "test_partition_evaluation_count": evaluation.test_probability_evaluation_count,
        "test_partition_used_for_adjustment": False,
        "test_partition_used_for_model_selection": False,
        "test_partition_used_for_feature_selection": False,
        "test_partition_used_for_hyperparameter_selection": False,
        "test_partition_used_for_preprocessing_selection": False,
        "test_partition_used_for_imbalance_policy_selection": False,
        "no_model_selection_decision_changed_after_test": True,
        "serialization_reload_validated": True,
        "inference_smoke_test_completed": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "api_implemented": False,
    }


def _validate_multiclass_json_artifact(
    filename: str, payload: Mapping[str, Any]
) -> None:
    if filename not in MULTICLASS_FINAL_SCHEMAS:
        raise FinalizationContractError(f"Unsupported final artifact filename: {filename}")
    schema, artifact_type = MULTICLASS_FINAL_SCHEMAS[filename]
    if payload.get("schema_version") != schema or payload.get("artifact_type") != artifact_type:
        raise FinalizationContractError(f"Invalid multiclass schema/type for {filename}.")
    if payload.get("problem_type") != "multiclass_classification":
        raise FinalizationContractError(f"Invalid multiclass problem_type for {filename}.")
    if payload.get("operational_modeling_ready") is not False:
        raise FinalizationContractError("Operational modeling readiness must remain false.")
    if payload.get("operational_validity") != "unconfirmed":
        raise FinalizationContractError("Operational validity must remain unconfirmed.")
    _validate_paths_recursively(payload)
    if filename == "inference-bundle.json":
        prohibited = {
            "positive_class",
            "negative_class",
            "positive_encoded_label",
            "positive_class_probability",
            "educational_decision_threshold",
            "operational_threshold",
        }
        if prohibited & set(payload):
            raise FinalizationContractError("Multiclass bundle contains binary operational fields.")
        classes = payload.get("target_classes")
        if payload.get("output_class_order") != classes or len(classes or ()) < 3:
            raise FinalizationContractError("Multiclass bundle output class order is invalid.")
        if set(payload.get("estimator_class_order", ())) != set(classes):
            raise FinalizationContractError("Multiclass estimator class set is invalid.")
        output = payload.get("inference_output_contract", {})
        if output.get("binary_threshold") != "not_applicable":
            raise FinalizationContractError("Multiclass binary threshold must be not_applicable.")
    if filename == "final-test-evidence.json":
        if payload.get("test_partition_evaluation_count") != 1:
            raise FinalizationContractError("Final test evaluation count must equal one.")
        if payload.get("no_post_test_adjustment") is not True:
            raise FinalizationContractError("Final test cannot trigger adjustment.")
        rendered = canonical_json_bytes(payload)
        for prohibited in (b"individual_predictions", b"row_predictions", b"individual_probabilities"):
            if prohibited in rendered:
                raise FinalizationContractError("Final test evidence contains row-level outputs.")
    if filename == "final-model-handoff.json":
        required_true = {
            "preparation_handoff_validated",
            "model_selection_handoff_validated",
            "frozen_finalization_contract_validated",
            "final_training_completed",
            "final_model_trained",
            "test_partition_opened_after_final_fit",
            "final_test_evaluation_completed",
            "model_artifact_materialized",
            "model_bundle_materialized",
            "serialization_reload_validated",
            "inference_smoke_test_completed",
            "final_model_handoff_ready",
            "educational_inference_demo_ready",
            "no_model_selection_decision_changed_after_test",
        }
        if any(payload.get(key) is not True for key in required_true):
            raise FinalizationContractError("Multiclass final handoff readiness is incomplete.")
        if payload.get("test_partition_evaluation_count") != 1:
            raise FinalizationContractError("Final test evaluation count must equal one.")
        if payload.get("test_partition_used_for_adjustment") is not False:
            raise FinalizationContractError("Test cannot be used for adjustment.")


def _multiclass_contract_from_bundle(
    bundle: Mapping[str, Any],
) -> MulticlassFrozenFinalizationContract:
    lineage = bundle.get("lineage", {})
    model_selection = lineage.get("model_selection", {})
    test = bundle["test_partition_reference"]
    classes = list(bundle["target_classes"])
    encoding = bundle["target_encoding_metadata_only"]
    contract = MulticlassFrozenFinalizationContract(
        dataset_slug=str(bundle["dataset_slug"]),
        problem_type=str(bundle["problem_type"]),
        model_selection_handoff_path=str(model_selection["path"]),
        model_selection_handoff_sha256=str(model_selection["byte_sha256"]),
        model_id=str(bundle["model_id"]),
        model_family=str(bundle["model_family"]),
        hyperparameters=tuple(sorted(bundle["selected_hyperparameters"].items())),
        random_state=int(bundle["estimator_random_state"]),
        feature_policy=str(bundle["feature_policy"]),
        feature_columns=tuple(bundle["feature_columns"]),
        numerical_features=tuple(bundle["numerical_features"]),
        categorical_features=tuple(bundle["categorical_features"]),
        identifier_columns=tuple(bundle["identifier_columns_excluded"]),
        target_column=str(bundle["target_column"]),
        target_classes=tuple(classes),
        target_encoding=tuple((label, int(encoding[label])) for label in classes),
        target_semantics=str(bundle["target_semantics"]),
        preprocessing_contract=tuple(sorted(bundle["preprocessing_contract"].items())),
        imbalance_policy=tuple(sorted(bundle["imbalance_policy"].items())),
        decision_rule=str(bundle["decision_rule"]),
        training_partitions=("train", "validation"),
        evaluation_partition="test",
        test_partition_path=str(test["path"]),
        test_partition_sha256=str(test["byte_sha256"]),
        test_partition_row_count=int(test["row_count"]),
    )
    validate_multiclass_frozen_model_contract(contract)
    return contract


def _validate_multiclass_complete_set(
    output_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    output = Path(output_directory).resolve()
    if inspect_final_artifact_set(output) != "complete":
        raise ArtifactConflictError("Multiclass final artifact set is not complete.")
    payloads = {filename: _load_json(output / filename) for filename in MULTICLASS_FINAL_SCHEMAS}
    for filename, payload in payloads.items():
        _validate_multiclass_json_artifact(filename, payload)
    handoff = payloads["final-model-handoff.json"]
    bundle = payloads["inference-bundle.json"]
    manifest = payloads["final-model-manifest.json"]
    evidence = payloads["final-test-evidence.json"]
    if len({item.get("dataset_slug") for item in payloads.values()}) != 1:
        raise ArtifactConflictError("Dataset slug differs within multiclass artifact set.")
    model_relative = PurePosixPath(
        _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    )
    expected_model_path = (output / "final-pipeline.joblib").resolve()
    inferred_root = expected_model_path
    for _ in model_relative.parts:
        inferred_root = inferred_root.parent
    if (inferred_root / model_relative).resolve() != expected_model_path:
        raise ArtifactConflictError("Bundle model path does not identify final-pipeline.joblib.")
    model_hash = sha256_file(expected_model_path)
    if model_hash != bundle.get("model_artifact_sha256"):
        raise ArtifactConflictError("Model bytes differ from multiclass bundle.")
    for name, reference in handoff.get("final_references", {}).items():
        relative = _require_relative_path(reference["path"], field=f"final_references.{name}.path")
        absolute = (inferred_root / relative).resolve()
        try:
            absolute.relative_to(inferred_root)
        except ValueError as exc:
            raise ArtifactConflictError("Final reference escapes project root.") from exc
        if not absolute.is_file():
            raise ArtifactConflictError(f"Final reference integrity mismatch: {name}")
        reference_byte_matches = sha256_file(absolute) == reference.get("byte_sha256")
        reference_semantic_matches = False
        if absolute.suffix == ".json":
            try:
                reference_payload = _load_json(absolute)
            except (OSError, json.JSONDecodeError):
                reference_payload = None
            if isinstance(reference_payload, Mapping):
                reference_semantic_matches = _semantic_fingerprint_matches_declared(
                    reference_payload,
                    reference.get("semantic_sha256"),
                )
        if not reference_byte_matches and not reference_semantic_matches:
            raise ArtifactConflictError(f"Final reference integrity mismatch: {name}")
    fingerprints = manifest.get("final_artifact_fingerprints", {})
    expected = {
        "final-pipeline.joblib": (model_hash, bundle["model_state_fingerprint"]),
        "final-test-evidence.json": (
            sha256_file(output / "final-test-evidence.json"),
            semantic_fingerprint(evidence),
        ),
        "inference-bundle.json": (
            sha256_file(output / "inference-bundle.json"),
            semantic_fingerprint(bundle),
        ),
    }
    for filename, (byte_hash, semantic_hash) in expected.items():
        declared = fingerprints.get(filename, {})
        semantic_matches = declared.get("semantic_sha256") == semantic_hash
        if filename == "final-test-evidence.json":
            semantic_matches = semantic_matches or _semantic_fingerprint_matches_declared(
                evidence,
                declared.get("semantic_sha256"),
            )
        elif filename == "inference-bundle.json":
            semantic_matches = semantic_matches or _semantic_fingerprint_matches_declared(
                bundle,
                declared.get("semantic_sha256"),
            )
        byte_matches = declared.get("byte_sha256") == byte_hash
        if filename == "final-pipeline.joblib":
            valid_fingerprint = byte_matches and semantic_matches
        else:
            valid_fingerprint = semantic_matches
        if not valid_fingerprint:
            raise ArtifactConflictError(f"Manifest fingerprint mismatch: {filename}")
    if handoff.get("model_state_fingerprint") != bundle.get("model_state_fingerprint"):
        raise ArtifactConflictError("Model-state fingerprint differs between handoff and bundle.")
    if handoff.get("frozen_finalization_contract_fingerprint") != manifest.get(
        "frozen_finalization_contract_fingerprint"
    ) or evidence.get("frozen_finalization_contract_fingerprint") != handoff.get(
        "frozen_finalization_contract_fingerprint"
    ):
        raise ArtifactConflictError("Frozen finalization fingerprints are inconsistent.")
    return handoff, bundle, manifest, evidence


def validate_existing_multiclass_finalization_equivalence(
    *,
    output_directory: str | Path,
    contract: MulticlassFrozenFinalizationContract,
) -> bool:
    handoff, bundle, _, _ = _validate_multiclass_complete_set(output_directory)
    checks = {
        "dataset_slug": contract.dataset_slug,
        "selected_model_id": contract.model_id,
        "selected_model_family": contract.model_family,
        "frozen_finalization_contract_fingerprint": contract.fingerprint,
    }
    for key, expected in checks.items():
        if handoff.get(key) != expected:
            raise ArtifactConflictError(f"Existing multiclass artifact differs at {key}.")
    if handoff.get("feature_order") != list(contract.feature_columns):
        raise ArtifactConflictError("Existing multiclass feature order is divergent.")
    if handoff.get("selected_hyperparameters") != dict(contract.hyperparameters):
        raise ArtifactConflictError("Existing multiclass hyperparameters are divergent.")
    if handoff.get("test_partition_evaluation_count") != 1:
        raise ArtifactConflictError("Existing final test evaluation count must remain one.")
    if bundle.get("output_class_order") != list(contract.target_classes):
        raise ArtifactConflictError("Existing output class order is divergent.")
    return True


def write_multiclass_final_model_artifacts(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    pipeline: Pipeline,
    contract: MulticlassFrozenFinalizationContract,
    training_data: MulticlassFinalTrainingData,
    test_partition: MulticlassTestPartitionData,
    evaluation: MulticlassFinalEvaluation,
    validation_evidence: Mapping[str, Any],
    fit_duration_seconds: float,
    upstream_references: Mapping[str, Any],
    expected_input_dtypes: Mapping[str, str],
    missing_value_policy: Mapping[str, Any],
    analysis_conclusions: Mapping[str, Any],
) -> ArtifactWriteResult:
    """Atomically materialize the v2 joblib plus four schema-aware JSONs."""

    root = Path(project_root).resolve()
    relative_output = _require_relative_path(output_directory, field="output_directory")
    output = (root / relative_output).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise FinalizationContractError("Output directory escapes project root.") from exc
    state = inspect_final_artifact_set(output)
    if state == "partial":
        raise ArtifactConflictError("Partial final artifact set detected; refusing repair.")
    if state == "complete":
        validate_existing_multiclass_finalization_equivalence(
            output_directory=output, contract=contract
        )
        byte_hashes = {
            name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES
        }
        semantic_hashes = {
            name: semantic_fingerprint(_load_json(output / name))
            for name in MULTICLASS_FINAL_SCHEMAS
        }
        semantic_hashes["final-pipeline.joblib"] = _load_json(
            output / "inference-bundle.json"
        )["model_state_fingerprint"]
        return ArtifactWriteResult(output, (), (), True, byte_hashes, semantic_hashes)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".final-model-v2-staging-", dir=output.parent))
    backup_root = Path(tempfile.mkdtemp(prefix=".final-model-v2-backup-", dir=output.parent))
    staging = staging_root / output.name
    staging.mkdir(parents=True, exist_ok=True)
    promoted: list[str] = []
    backed_up: list[str] = []
    try:
        descriptor = describe_multiclass_fitted_pipeline(
            pipeline=pipeline, contract=contract
        )
        state_fingerprint = compute_fitted_model_fingerprint(descriptor)
        model_staging = staging / "final-pipeline.joblib"
        model_hash = serialize_multiclass_pipeline_to_staging(
            pipeline=pipeline, staging_path=model_staging
        )
        sample = training_data.features.iloc[: min(32, training_data.row_count)].copy(deep=True)
        reloaded = validate_multiclass_serialized_pipeline(
            staging_path=model_staging,
            expected_sha256=model_hash,
            contract=contract,
            reference_pipeline=pipeline,
            validation_sample=sample,
        )
        paths = {
            name: f"{relative_output}/{name}" for name in FINAL_ARTIFACT_FILENAMES
        }
        evidence = build_multiclass_final_test_evidence(
            contract=contract,
            test_partition=test_partition,
            evaluation=evaluation,
            validation_evidence=validation_evidence,
        )
        bundle = build_multiclass_inference_bundle(
            contract=contract,
            fitted_pipeline=pipeline,
            model_artifact_path=paths["final-pipeline.joblib"],
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fingerprint,
            model_state_descriptor=descriptor,
            expected_input_dtypes=expected_input_dtypes,
            missing_value_policy=missing_value_policy,
            upstream_references=upstream_references,
            final_artifact_paths=paths,
        )
        smoke = smoke_predict_multiclass_bundle(reloaded, sample.iloc[:8], bundle=bundle)
        if len(smoke) != min(8, len(sample)):
            raise SerializationValidationError("Multiclass serialization smoke row count differs.")

        fingerprints: dict[str, dict[str, str]] = {
            "final-pipeline.joblib": {
                "byte_sha256": model_hash,
                "semantic_sha256": state_fingerprint,
            }
        }
        for filename, payload in {
            "final-test-evidence.json": evidence,
            "inference-bundle.json": bundle,
        }.items():
            _validate_multiclass_json_artifact(filename, payload)
            content = canonical_json_text(payload).encode("utf-8")
            (staging / filename).write_bytes(content)
            fingerprints[filename] = {
                "byte_sha256": sha256_bytes(content),
                "semantic_sha256": semantic_fingerprint(payload),
            }
        manifest = build_multiclass_final_model_manifest(
            contract=contract,
            upstream_references=upstream_references,
            training_data=training_data,
            test_partition=test_partition,
            fit_duration_seconds=fit_duration_seconds,
            model_artifact_path=paths["final-pipeline.joblib"],
            model_artifact_sha256=model_hash,
            model_state_fingerprint=state_fingerprint,
            model_state_descriptor=descriptor,
            final_artifact_paths=paths,
            final_artifact_fingerprints=fingerprints,
            analysis_conclusions=analysis_conclusions,
        )
        _validate_multiclass_json_artifact("final-model-manifest.json", manifest)
        manifest_content = canonical_json_text(manifest).encode("utf-8")
        (staging / "final-model-manifest.json").write_bytes(manifest_content)
        fingerprints["final-model-manifest.json"] = {
            "byte_sha256": sha256_bytes(manifest_content),
            "semantic_sha256": semantic_fingerprint(manifest),
        }
        final_references = {
            "model_artifact": {
                "path": paths["final-pipeline.joblib"],
                **fingerprints["final-pipeline.joblib"],
            },
            "final_model_manifest": {
                "path": paths["final-model-manifest.json"],
                **fingerprints["final-model-manifest.json"],
            },
            "final_test_evidence": {
                "path": paths["final-test-evidence.json"],
                **fingerprints["final-test-evidence.json"],
            },
            "inference_bundle": {
                "path": paths["inference-bundle.json"],
                **fingerprints["inference-bundle.json"],
            },
        }
        handoff = build_multiclass_final_model_handoff(
            contract=contract,
            upstream_references=upstream_references,
            final_references=final_references,
            evaluation=evaluation,
            analysis_conclusions=analysis_conclusions,
            runtime_requirements=runtime_versions(),
        )
        _validate_multiclass_json_artifact("final-model-handoff.json", handoff)
        handoff_content = canonical_json_text(handoff).encode("utf-8")
        (staging / "final-model-handoff.json").write_bytes(handoff_content)
        fingerprints["final-model-handoff.json"] = {
            "byte_sha256": sha256_bytes(handoff_content),
            "semantic_sha256": semantic_fingerprint(handoff),
        }
        for filename in MULTICLASS_FINAL_SCHEMAS:
            _validate_multiclass_json_artifact(filename, _load_json(staging / filename))
        if sha256_file(model_staging) != bundle["model_artifact_sha256"]:
            raise SerializationValidationError("Staged bundle/model hash mismatch.")

        output.mkdir(parents=True, exist_ok=True)
        for filename in FINAL_ARTIFACT_FILENAMES:
            destination = output / filename
            if destination.exists():
                backup = backup_root / filename
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(filename)
            os.replace(staging / filename, destination)
            promoted.append(filename)

        _validate_multiclass_complete_set(output)
        loaded_bundle = load_and_validate_inference_bundle(
            project_root=root, bundle_path=paths["inference-bundle.json"]
        )
        trusted = load_trusted_pipeline_from_bundle(
            project_root=root, bundle=loaded_bundle
        )
        trusted_smoke = smoke_predict_multiclass_bundle(
            trusted, sample.iloc[:8], bundle=loaded_bundle
        )
        if trusted_smoke["predicted_class"].tolist() != smoke["predicted_class"].tolist():
            raise SerializationValidationError("Post-promotion smoke predictions differ.")
        load_and_validate_final_model_handoff(
            project_root=root, handoff_path=paths["final-model-handoff.json"]
        )
        byte_hashes = {
            name: sha256_file(output / name) for name in FINAL_ARTIFACT_FILENAMES
        }
        semantic_hashes = {
            **{
                name: values["semantic_sha256"]
                for name, values in fingerprints.items()
            }
        }
        return ArtifactWriteResult(
            output,
            tuple(promoted),
            (),
            False,
            byte_hashes,
            semantic_hashes,
        )
    except Exception:
        for filename in reversed(promoted):
            destination = output / filename
            if destination.exists():
                destination.unlink()
        for filename in reversed(backed_up):
            backup = backup_root / filename
            destination = output / filename
            if backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


_load_and_validate_inference_bundle_v1 = load_and_validate_inference_bundle
_load_and_validate_final_model_handoff_v1 = load_and_validate_final_model_handoff
_load_trusted_pipeline_from_bundle_v1 = load_trusted_pipeline_from_bundle


def load_and_validate_inference_bundle(
    *, project_root: str | Path, bundle_path: str | Path
) -> dict[str, Any]:
    """Load either inference-bundle.v1 or genuine multiclass v2."""

    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle_path, field="bundle_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Inference bundle not found: {relative}")
    preview = _load_json(path)
    if preview.get("schema_version") == "inference-bundle.v1":
        return _load_and_validate_inference_bundle_v1(
            project_root=root, bundle_path=relative
        )
    if preview.get("schema_version") != "inference-bundle.v2":
        raise FinalizationContractError("Unsupported inference bundle schema.")
    _, bundle, _, _ = _validate_multiclass_complete_set(path.parent)
    return _deepcopy(bundle)


def load_and_validate_final_model_handoff(
    *, project_root: str | Path, handoff_path: str | Path
) -> dict[str, Any]:
    """Load either final-model-handoff.v1 or multiclass v2 with sibling integrity."""

    root = Path(project_root).resolve()
    relative = _require_relative_path(handoff_path, field="handoff_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Final model handoff not found: {relative}")
    preview = _load_json(path)
    if preview.get("schema_version") == "final-model-handoff.v1":
        return _load_and_validate_final_model_handoff_v1(
            project_root=root, handoff_path=relative
        )
    if preview.get("schema_version") != "final-model-handoff.v2":
        raise FinalizationContractError("Unsupported final model handoff schema.")
    handoff, _, _, _ = _validate_multiclass_complete_set(path.parent)
    return _deepcopy(handoff)


def load_and_validate_final_model_manifest(
    *, project_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(manifest_path, field="manifest_path")
    path = root / relative
    payload = _load_json(path)
    if payload.get("schema_version") == "final-model-manifest.v2":
        _, _, manifest, _ = _validate_multiclass_complete_set(path.parent)
        return _deepcopy(manifest)
    _validate_json_artifact("final-model-manifest.json", payload)
    return _deepcopy(payload)


def load_and_validate_final_test_evidence(
    *, project_root: str | Path, evidence_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = _require_relative_path(evidence_path, field="evidence_path")
    path = root / relative
    payload = _load_json(path)
    if payload.get("schema_version") == "final-test-evidence.v2":
        _, _, _, evidence = _validate_multiclass_complete_set(path.parent)
        return _deepcopy(evidence)
    _validate_json_artifact("final-test-evidence.json", payload)
    return _deepcopy(payload)


def load_trusted_pipeline_from_bundle(
    *, project_root: str | Path, bundle: Mapping[str, Any]
) -> Pipeline:
    """Verify v1/v2 bytes first, then validate the fitted state after joblib load."""

    if bundle.get("schema_version") == "inference-bundle.v1":
        return _load_trusted_pipeline_from_bundle_v1(
            project_root=project_root, bundle=bundle
        )
    _validate_multiclass_json_artifact("inference-bundle.json", bundle)
    root = Path(project_root).resolve()
    relative = _require_relative_path(bundle["model_artifact_path"], field="model_artifact_path")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {relative}")
    if sha256_file(path) != bundle["model_artifact_sha256"]:
        raise UntrustedArtifactError("Refusing to load multiclass joblib with divergent SHA-256.")
    loaded = joblib.load(path)
    contract = _multiclass_contract_from_bundle(bundle)
    verify_multiclass_pipeline_contract(loaded, contract=contract, require_fitted=True)
    descriptor = describe_multiclass_fitted_pipeline(pipeline=loaded, contract=contract)
    fingerprint = compute_fitted_model_fingerprint(descriptor)
    if descriptor != bundle.get("model_state_descriptor"):
        raise SerializationValidationError("Loaded multiclass fitted-state descriptor differs.")
    if fingerprint != bundle.get("model_state_fingerprint"):
        raise SerializationValidationError("Loaded multiclass fitted-state fingerprint differs.")
    return loaded


_dispatch_bundle_v12 = load_and_validate_inference_bundle
_dispatch_handoff_v12 = load_and_validate_final_model_handoff
_dispatch_manifest_v12 = load_and_validate_final_model_manifest
_dispatch_evidence_v12 = load_and_validate_final_test_evidence
_dispatch_pipeline_v12 = load_trusted_pipeline_from_bundle

def load_and_validate_inference_bundle(*, project_root, bundle_path):
    root=Path(project_root).resolve(); rel=_require_relative_path(bundle_path,field="bundle_path")
    if _load_json(root/rel).get("schema_version") != "inference-bundle.v3": return _dispatch_bundle_v12(project_root=root,bundle_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[1])

def load_and_validate_final_model_handoff(*, project_root, handoff_path):
    root=Path(project_root).resolve(); rel=_require_relative_path(handoff_path,field="handoff_path")
    if _load_json(root/rel).get("schema_version") != "final-model-handoff.v3": return _dispatch_handoff_v12(project_root=root,handoff_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[0])

def load_and_validate_final_model_manifest(*, project_root, manifest_path):
    root=Path(project_root).resolve(); rel=_require_relative_path(manifest_path,field="manifest_path")
    if _load_json(root/rel).get("schema_version") != "final-model-manifest.v3": return _dispatch_manifest_v12(project_root=root,manifest_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[2])

def load_and_validate_final_test_evidence(*, project_root, evidence_path):
    root=Path(project_root).resolve(); rel=_require_relative_path(evidence_path,field="evidence_path")
    if _load_json(root/rel).get("schema_version") != "final-test-evidence.v3": return _dispatch_evidence_v12(project_root=root,evidence_path=rel)
    return _deepcopy(_validate_regression_complete_set((root/rel).parent)[3])

def load_trusted_pipeline_from_bundle(*, project_root, bundle):
    if bundle.get("schema_version") != "inference-bundle.v3": return _dispatch_pipeline_v12(project_root=project_root,bundle=bundle)
    root=Path(project_root).resolve(); rel=_require_relative_path(bundle["model_artifact_path"],field="model_artifact_path"); path=root/rel
    if sha256_file(path) != bundle.get("model_artifact_sha256"): raise UntrustedArtifactError("Refusing to load regression joblib with divergent SHA-256.")
    loaded=joblib.load(path)
    if not _is_fitted(loaded) or loaded.named_steps["model"].__class__.__name__ != bundle["model_contract"]["family"]: raise SerializationValidationError("Trusted regression pipeline contract mismatch.")
    params=loaded.named_steps["model"].get_params(deep=False)
    if any(params.get(k)!=v for k,v in _regression_model_params(bundle).items()): raise SerializationValidationError("Regression model parameters differ.")
    descriptor=describe_regression_fitted_pipeline(pipeline=loaded,contract={
        "selected_hyperparameters":bundle["model_contract"]["selected_hyperparameters"],
        "selected_estimator_fixed_constructor_parameters":bundle["model_contract"]["fixed_constructor_parameters"],
        "feature_order":bundle["feature_order"],"preprocessing":bundle["preprocessing_contract"]})
    if (descriptor != bundle.get("model_state_descriptor")
            or compute_regression_model_state_fingerprint(descriptor) != bundle.get("model_state_fingerprint")):
        raise SerializationValidationError("Regression fitted-state descriptor or fingerprint differs.")
    return loaded
