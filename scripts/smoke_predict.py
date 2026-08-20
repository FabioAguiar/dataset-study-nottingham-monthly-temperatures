"""Safe, educational inference helpers for validated sklearn bundles.

The module deliberately contains no dataset-specific constants, dataset reads, network
access, persistence, fitting, or implicit preprocessing. All behavior is derived from
an explicitly supplied inference bundle, final-model handoff, manifest, and independent
in-memory inputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
from typing import Any, Callable, Mapping, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from scripts.finalize_model import (
    load_and_validate_final_model_handoff,
    load_and_validate_final_model_manifest,
    load_and_validate_inference_bundle,
    load_trusted_pipeline_from_bundle,
)


class InferenceContractError(ValueError):
    """Raised when final inference artifacts disagree with their declared contract."""


class RuntimeCompatibilityError(InferenceContractError):
    """Raised when the current process is unsafe for deserializing the model."""

    def __init__(self, message: str, *, report: "RuntimeCompatibilityReport") -> None:
        super().__init__(message)
        self.report = report


class InferenceInputError(InferenceContractError):
    """Raised when an independent inference input violates the bundle contract."""


class TrustedModelSourceError(InferenceContractError):
    """Raised when trusted provenance was not explicitly confirmed."""


class RuntimeCompatibilityWarning(UserWarning):
    """Warning emitted for a load-safe Python patch-only difference."""


@dataclass(frozen=True)
class RuntimeComponentReport:
    """Deterministic compatibility result for one runtime component."""

    component: str
    expected: str
    observed: str
    compatible: bool
    status: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "expected": self.expected,
            "observed": self.observed,
            "compatible": self.compatible,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RuntimeCompatibilityReport:
    """Immutable, deterministic runtime compatibility report."""

    mode: str
    compatible: bool
    components: tuple[RuntimeComponentReport, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "compatible": self.compatible,
            "components": [component.as_dict() for component in self.components],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class InputNormalizationResult:
    """Defensive normalized input plus aggregate, non-PII diagnostics."""

    dataframe: pd.DataFrame
    input_materializations_applied: tuple[tuple[str, int], ...]
    unknown_categories_report: tuple[tuple[str, tuple[Any, ...]], ...]

    def materializations_dict(self) -> dict[str, int]:
        return dict(self.input_materializations_applied)

    def unknown_categories_dict(self) -> dict[str, list[Any]]:
        return {
            column: [deepcopy(value) for value in values]
            for column, values in self.unknown_categories_report
        }


_RUNTIME_COMPONENTS = ("python", "pandas", "scikit_learn", "joblib")
_VERSION_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _normalized_runtime_mapping(versions: Mapping[str, Any]) -> dict[str, str]:
    aliases = {
        "scikit-learn": "scikit_learn",
        "sklearn": "scikit_learn",
        "scikit_learn": "scikit_learn",
    }
    normalized: dict[str, str] = {}
    for key, value in versions.items():
        canonical = aliases.get(str(key), str(key))
        normalized[canonical] = str(value)
    missing = [name for name in _RUNTIME_COMPONENTS if name not in normalized]
    if missing:
        raise InferenceContractError(
            "Runtime version mapping is missing components: " + ", ".join(missing)
        )
    return normalized


def _version_triplet(value: str, *, component: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.match(value.strip())
    if not match:
        raise InferenceContractError(
            f"Invalid {component} version in runtime contract: {value!r}"
        )
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def current_runtime_versions() -> dict[str, str]:
    """Return versions for the current process without changing warning filters."""

    return {
        "python": platform.python_version(),
        "pandas": str(pd.__version__),
        "scikit_learn": str(sklearn.__version__),
        "joblib": str(joblib.__version__),
    }


def validate_runtime_compatibility(
    expected_versions: Mapping[str, Any],
    *,
    observed_versions: Mapping[str, Any] | None = None,
    mode: str = "exact",
    raise_on_incompatible: bool | None = None,
) -> RuntimeCompatibilityReport:
    """Compare runtime versions in ``exact`` or deserialization ``load_safe`` mode.

    ``exact`` compares all four components byte-for-byte and returns a report by
    default. ``load_safe`` requires an exact pandas, scikit-learn, and joblib match,
    plus a Python major/minor match. A Python patch-only difference is compatible but
    emits :class:`RuntimeCompatibilityWarning`. Unsafe ``load_safe`` results raise by
    default so the caller cannot reach a joblib loader accidentally.
    """

    if mode not in {"exact", "load_safe"}:
        raise ValueError("mode must be 'exact' or 'load_safe'.")
    expected = _normalized_runtime_mapping(expected_versions)
    observed = _normalized_runtime_mapping(
        current_runtime_versions() if observed_versions is None else observed_versions
    )
    if raise_on_incompatible is None:
        raise_on_incompatible = mode == "load_safe"

    component_reports: list[RuntimeComponentReport] = []
    warning_messages: list[str] = []
    for component in _RUNTIME_COMPONENTS:
        expected_value = expected[component]
        observed_value = observed[component]
        if mode == "exact" or component != "python":
            compatible = observed_value == expected_value
            status = "compatible" if compatible else "incompatible"
            detail = "exact match" if compatible else "exact version mismatch"
        else:
            expected_triplet = _version_triplet(expected_value, component=component)
            observed_triplet = _version_triplet(observed_value, component=component)
            compatible = observed_triplet[:2] == expected_triplet[:2]
            if not compatible:
                status = "incompatible"
                detail = "Python major/minor mismatch"
            elif observed_value == expected_value:
                status = "compatible"
                detail = "exact match"
            else:
                status = "warning"
                detail = "Python patch differs; major/minor is load-safe"
                message = (
                    "Python patch differs from the bundle: "
                    f"expected={expected_value}, observed={observed_value}."
                )
                warning_messages.append(message)
                warnings.warn(message, RuntimeCompatibilityWarning, stacklevel=2)
        component_reports.append(
            RuntimeComponentReport(
                component=component,
                expected=expected_value,
                observed=observed_value,
                compatible=compatible,
                status=status,
                detail=detail,
            )
        )

    report = RuntimeCompatibilityReport(
        mode=mode,
        compatible=all(item.compatible for item in component_reports),
        components=tuple(component_reports),
        warnings=tuple(warning_messages),
    )
    if raise_on_incompatible and not report.compatible:
        mismatches = ", ".join(
            f"{item.component}(expected={item.expected}, observed={item.observed})"
            for item in report.components
            if not item.compatible
        )
        raise RuntimeCompatibilityError(
            "Runtime is not safe for model deserialization: " + mismatches,
            report=report,
        )
    return report


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceContractError(f"{field} must be an object.")
    return value


def _schema_version(
    payload: Mapping[str, Any], *, artifact: str, v1_schema: str, v2_schema: str,
    v3_schema: str | None = None,
) -> str:
    """Return a supported schema, retaining legacy schema-less v1 test fixtures."""

    observed = payload.get("schema_version")
    if observed is None:
        return v1_schema
    supported = {v1_schema, v2_schema} | ({v3_schema} if v3_schema else set())
    if observed not in supported:
        raise InferenceContractError(
            f"Unsupported {artifact} schema: {observed!r}."
        )
    return str(observed)


def _inference_bundle_schema(bundle: Mapping[str, Any]) -> str:
    return _schema_version(
        bundle,
        artifact="inference bundle",
        v1_schema="inference-bundle.v1",
        v2_schema="inference-bundle.v2",
        v3_schema="inference-bundle.v3",
    )


def _final_handoff_schema(handoff: Mapping[str, Any]) -> str:
    return _schema_version(
        handoff,
        artifact="final-model handoff",
        v1_schema="final-model-handoff.v1",
        v2_schema="final-model-handoff.v2",
        v3_schema="final-model-handoff.v3",
    )


def _validate_matching_schema_generations(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any]
) -> str:
    handoff_schema = _final_handoff_schema(handoff)
    bundle_schema = _inference_bundle_schema(bundle)
    generations = {
        "final-model-handoff.v1": "v1",
        "final-model-handoff.v2": "v2",
        "inference-bundle.v1": "v1",
        "inference-bundle.v2": "v2",
        "final-model-handoff.v3": "v3",
        "inference-bundle.v3": "v3",
    }
    if generations[handoff_schema] != generations[bundle_schema]:
        raise InferenceContractError(
            "Final-model handoff and inference bundle use different schema generations."
        )
    return generations[bundle_schema]


def _validate_inference_readiness_v1(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    """Validate educational readiness while preserving all operational limitations."""

    handoff_requirements = {
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "final_test_evaluation_completed": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_sealed_at_input": True,
        "test_partition_evaluated": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "test_partition_used_for_model_selection": False,
        "test_partition_used_for_threshold_selection": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
        "operational_threshold": "unresolved",
        "api_implemented": False,
    }
    for field, expected in handoff_requirements.items():
        observed = handoff.get(field)
        if observed != expected:
            raise InferenceContractError(
                f"Final-model handoff readiness mismatch for {field}: "
                f"expected={expected!r}, observed={observed!r}"
            )

    readiness = _require_mapping(bundle.get("readiness"), field="bundle.readiness")
    bundle_requirements = {
        "educational_inference_demo_ready": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "operational_modeling_ready": False,
    }
    for field, expected in bundle_requirements.items():
        observed = readiness.get(field)
        if observed != expected:
            raise InferenceContractError(
                f"Inference-bundle readiness mismatch for {field}: "
                f"expected={expected!r}, observed={observed!r}"
            )
    for field, expected in {
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "temporal_contract_status": "unresolved",
        "feature_inference_availability": "unconfirmed",
    }.items():
        if bundle.get(field) != expected:
            raise InferenceContractError(
                f"Inference-bundle limitation mismatch for {field}: "
                f"expected={expected!r}, observed={bundle.get(field)!r}"
            )
    output_contract = _require_mapping(
        bundle.get("output_contract"), field="bundle.output_contract"
    )
    if output_contract.get("operational_prediction_available") is not False:
        raise InferenceContractError(
            "The bundle must preserve operational_prediction_available=false."
        )


def validate_multiclass_inference_readiness(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    """Validate the educational-only readiness contract of genuine v2 artifacts."""

    if _validate_matching_schema_generations(handoff, bundle) != "v2":
        raise InferenceContractError("Multiclass readiness requires v2 artifacts.")
    handoff_requirements = {
        "educational_final_model_completed": True,
        "final_model_trained": True,
        "final_test_evaluation_completed": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "final_model_handoff_ready": True,
        "educational_inference_demo_ready": True,
        "test_partition_evaluation_count": 1,
        "test_partition_used_for_adjustment": False,
        "no_model_selection_decision_changed_after_test": True,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "api_implemented": False,
    }
    for field, expected in handoff_requirements.items():
        observed = handoff.get(field)
        if observed != expected:
            raise InferenceContractError(
                f"Multiclass handoff readiness mismatch for {field}: "
                f"expected={expected!r}, observed={observed!r}"
            )

    readiness = _require_mapping(bundle.get("readiness"), field="bundle.readiness")
    for field, expected in {
        "educational_inference_demo_ready": True,
        "model_artifact_materialized": True,
        "model_bundle_materialized": True,
        "serialization_reload_validated": True,
        "inference_smoke_test_completed": True,
        "operational_modeling_ready": False,
    }.items():
        observed = readiness.get(field)
        if observed != expected:
            raise InferenceContractError(
                f"Multiclass bundle readiness mismatch for {field}: "
                f"expected={expected!r}, observed={observed!r}"
            )
    if bundle.get("operational_modeling_ready") is not False:
        raise InferenceContractError("Operational modeling readiness must remain false.")
    if bundle.get("operational_validity") != "unconfirmed":
        raise InferenceContractError("Operational validity must remain unconfirmed.")
    output = _require_mapping(
        bundle.get("inference_output_contract"),
        field="bundle.inference_output_contract",
    )
    if output.get("operational_prediction_available") is not False:
        raise InferenceContractError(
            "The multiclass bundle must preserve operational_prediction_available=false."
        )


def validate_inference_readiness(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    """Dispatch educational readiness validation by artifact schema generation."""

    generation = _validate_matching_schema_generations(handoff, bundle)
    if generation == "v1":
        _validate_inference_readiness_v1(handoff, bundle)
        return
    if generation == "v2":
        validate_multiclass_inference_readiness(handoff, bundle)
        return
    validate_continuous_inference_readiness(handoff, bundle)


def validate_continuous_inference_readiness(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    """Validate fail-closed demo readiness and limitations for v3."""
    if _validate_matching_schema_generations(handoff, bundle) != "v3":
        raise InferenceContractError("Continuous readiness requires v3 artifacts.")
    hr = _require_mapping(handoff.get("readiness"), field="handoff.readiness")
    br = _require_mapping(bundle.get("readiness"), field="bundle.readiness")
    for field, expected in {"inference_demo_ready": True, "operational_modeling_ready": False, "operational_validity": "unconfirmed"}.items():
        if hr.get(field) != expected or br.get(field) != expected:
            raise InferenceContractError(f"Continuous readiness mismatch for {field}.")
    for field, expected in {"final_fit_count": 1, "test_partition_evaluation_count": 1, "test_prediction_call_count": 1, "test_partition_used_for_adjustment": False, "no_model_selection_decision_changed_after_test": True}.items():
        if hr.get(field) != expected:
            raise InferenceContractError(f"Continuous handoff mismatch for {field}.")


def _validate_bundle_handoff_alignment_v1(
    handoff: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate model identity, feature, target, threshold, hash, and path alignment."""

    pairs = (
        ("dataset_slug", "dataset_slug"),
        ("selected_model_id", "model_id"),
        ("selected_model_family", "model_family"),
        ("model_state_fingerprint", "model_state_fingerprint"),
        ("feature_order", "feature_columns"),
        ("target_encoding", "target_encoding"),
        ("positive_class", "positive_class"),
        ("educational_threshold", "educational_decision_threshold"),
    )
    for handoff_field, bundle_field in pairs:
        if handoff.get(handoff_field) != bundle.get(bundle_field):
            raise InferenceContractError(
                "Handoff/bundle mismatch: "
                f"{handoff_field}={handoff.get(handoff_field)!r}, "
                f"{bundle_field}={bundle.get(bundle_field)!r}"
            )

    references = _require_mapping(
        handoff.get("final_references"), field="handoff.final_references"
    )
    model_reference = _require_mapping(
        references.get("model_artifact"),
        field="handoff.final_references.model_artifact",
    )
    if model_reference.get("path") != bundle.get("model_artifact_path"):
        raise InferenceContractError("Model artifact path differs between handoff and bundle.")
    if model_reference.get("byte_sha256") != bundle.get("model_artifact_sha256"):
        raise InferenceContractError("Model artifact SHA-256 differs between handoff and bundle.")
    if model_reference.get("semantic_sha256") != bundle.get("model_state_fingerprint"):
        raise InferenceContractError(
            "Model state fingerprint differs between handoff and bundle."
        )

    if manifest is None:
        return
    manifest_pairs = (
        ("dataset_slug", "dataset_slug"),
        ("selected_model_id", "model_id"),
        ("selected_model_family", "model_family"),
        ("fitted_state_semantic_fingerprint", "model_state_fingerprint"),
        ("feature_columns", "feature_columns"),
        ("target_encoding", "target_encoding"),
        ("educational_threshold", "educational_decision_threshold"),
        ("model_artifact_path", "model_artifact_path"),
        ("model_artifact_byte_sha256", "model_artifact_sha256"),
    )
    for manifest_field, bundle_field in manifest_pairs:
        if manifest.get(manifest_field) != bundle.get(bundle_field):
            raise InferenceContractError(
                "Manifest/bundle mismatch: "
                f"{manifest_field}={manifest.get(manifest_field)!r}, "
                f"{bundle_field}={bundle.get(bundle_field)!r}"
            )


def validate_multiclass_bundle_handoff_alignment(
    handoff: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate v2 identity, input, class-order, model, and runtime alignment."""

    if _validate_matching_schema_generations(handoff, bundle) != "v2":
        raise InferenceContractError("Multiclass alignment requires v2 artifacts.")
    pairs = (
        ("dataset_slug", "dataset_slug"),
        ("problem_type", "problem_type"),
        ("selected_model_id", "model_id"),
        ("selected_model_family", "model_family"),
        ("model_state_fingerprint", "model_state_fingerprint"),
        ("feature_order", "feature_columns"),
        ("target_column", "target_column"),
        ("target_classes", "target_classes"),
        ("target_semantics", "target_semantics"),
        ("selected_hyperparameters", "selected_hyperparameters"),
        ("preprocessing", "preprocessing_contract"),
        ("imbalance_policy", "imbalance_policy"),
        ("decision_rule", "decision_rule"),
        ("estimator_class_order", "estimator_class_order"),
        ("output_class_order", "output_class_order"),
    )
    for handoff_field, bundle_field in pairs:
        if handoff.get(handoff_field) != bundle.get(bundle_field):
            raise InferenceContractError(
                "Multiclass handoff/bundle mismatch: "
                f"{handoff_field}={handoff.get(handoff_field)!r}, "
                f"{bundle_field}={bundle.get(bundle_field)!r}"
            )
    classes = list(bundle.get("target_classes", ()))
    estimator_order = list(bundle.get("estimator_class_order", ()))
    output_order = list(bundle.get("output_class_order", ()))
    if len(classes) < 3 or len(set(map(str, classes))) != len(classes):
        raise InferenceContractError("Multiclass target classes must be unique.")
    if set(estimator_order) != set(classes) or set(output_order) != set(classes):
        raise InferenceContractError("Multiclass estimator/output class sets differ.")
    output_contract = _require_mapping(
        bundle.get("inference_output_contract"),
        field="bundle.inference_output_contract",
    )
    if output_contract.get("class_order") != output_order:
        raise InferenceContractError("Output contract class order differs from bundle.")
    if output_contract.get("decision_rule") != bundle.get("decision_rule"):
        raise InferenceContractError("Output contract decision rule differs from bundle.")

    references = _require_mapping(
        handoff.get("final_references"), field="handoff.final_references"
    )
    model_reference = _require_mapping(
        references.get("model_artifact"),
        field="handoff.final_references.model_artifact",
    )
    for reference_field, bundle_field in (
        ("path", "model_artifact_path"),
        ("byte_sha256", "model_artifact_sha256"),
        ("semantic_sha256", "model_state_fingerprint"),
    ):
        if model_reference.get(reference_field) != bundle.get(bundle_field):
            raise InferenceContractError(
                f"Multiclass model reference differs at {reference_field}."
            )

    if manifest is None:
        return
    manifest_pairs = (
        ("dataset_slug", "dataset_slug"),
        ("problem_type", "problem_type"),
        ("selected_model_id", "model_id"),
        ("selected_model_family", "model_family"),
        ("model_state_fingerprint", "model_state_fingerprint"),
        ("feature_columns", "feature_columns"),
        ("target_column", "target_column"),
        ("target_classes", "target_classes"),
        ("selected_hyperparameters", "selected_hyperparameters"),
        ("preprocessing_contract", "preprocessing_contract"),
        ("imbalance_policy", "imbalance_policy"),
        ("decision_rule", "decision_rule"),
        ("estimator_class_order", "estimator_class_order"),
        ("output_class_order", "output_class_order"),
        ("model_artifact_path", "model_artifact_path"),
        ("model_artifact_byte_sha256", "model_artifact_sha256"),
        ("runtime_versions", "runtime_version_requirements"),
    )
    for manifest_field, bundle_field in manifest_pairs:
        if manifest.get(manifest_field) != bundle.get(bundle_field):
            raise InferenceContractError(
                "Multiclass manifest/bundle mismatch: "
                f"{manifest_field}={manifest.get(manifest_field)!r}, "
                f"{bundle_field}={bundle.get(bundle_field)!r}"
            )
    if manifest.get("model_state_descriptor") != bundle.get(
        "model_state_descriptor"
    ):
        raise InferenceContractError("Manifest fitted-state descriptor differs from bundle.")


def validate_bundle_handoff_alignment(
    handoff: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Dispatch bundle/handoff/manifest alignment by schema generation."""

    generation = _validate_matching_schema_generations(handoff, bundle)
    if generation == "v1":
        _validate_bundle_handoff_alignment_v1(
            handoff, bundle, manifest=manifest
        )
        return
    if generation == "v2":
        validate_multiclass_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
        return
    validate_continuous_bundle_handoff_alignment(handoff, bundle, manifest=manifest)


def validate_continuous_bundle_handoff_alignment(
    handoff: Mapping[str, Any], bundle: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None
) -> None:
    """Validate v3 identity, contracts, references, and runtime alignment."""
    if _validate_matching_schema_generations(handoff, bundle) != "v3":
        raise InferenceContractError("Continuous alignment requires v3 artifacts.")
    for field in ("dataset_slug", "problem_type", "selected_model_id", "selected_model_family", "feature_order", "target_contract", "prediction_contract", "preprocessing_contract"):
        if handoff.get(field) != bundle.get(field):
            raise InferenceContractError(f"Continuous handoff/bundle mismatch at {field}.")
    model_ref = _require_mapping(handoff.get("model_artifact_reference"), field="handoff.model_artifact_reference")
    if (model_ref.get("path"), model_ref.get("sha256"), model_ref.get("state_fingerprint")) != (bundle.get("model_artifact_path"), bundle.get("model_artifact_sha256"), bundle.get("model_state_fingerprint")):
        raise InferenceContractError("Continuous model reference differs from bundle.")
    siblings = _require_mapping(handoff.get("sibling_references"), field="handoff.sibling_references")
    for ref_name, sibling_name in (("bundle_reference", "inference-bundle.json"), ("manifest_reference", "final-model-manifest.json")):
        if handoff.get(ref_name) != siblings.get(sibling_name):
            raise InferenceContractError(f"Continuous sibling reference differs for {sibling_name}.")
    if manifest is None:
        return
    model_contract = _require_mapping(bundle.get("model_contract"), field="bundle.model_contract")
    for field in ("dataset_slug", "problem_type", "selected_model_id", "selected_model_family", "target_contract", "preprocessing_contract"):
        if manifest.get(field) != bundle.get(field):
            raise InferenceContractError(f"Continuous manifest/bundle mismatch at {field}.")
    if manifest.get("selected_hyperparameters") != model_contract.get("selected_hyperparameters"):
        raise InferenceContractError("Continuous selected hyperparameters differ.")
    artifact = _require_mapping(manifest.get("model_artifact"), field="manifest.model_artifact")
    if (artifact.get("path"), artifact.get("byte_sha256"), artifact.get("state_fingerprint")) != (bundle.get("model_artifact_path"), bundle.get("model_artifact_sha256"), bundle.get("model_state_fingerprint")):
        raise InferenceContractError("Continuous manifest model reference differs.")
    runtime = _require_mapping(manifest.get("runtime_versions"), field="manifest.runtime_versions")
    for component, version in _require_mapping(bundle.get("runtime_compatibility"), field="bundle.runtime_compatibility").items():
        if runtime.get(component) != version:
            raise InferenceContractError(f"Continuous runtime mismatch for {component}.")


def _portable_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise InferenceContractError(f"{field} must be a non-empty relative path.")
    raw = value.strip()
    windows = PureWindowsPath(raw)
    if windows.is_absolute() or bool(windows.drive) or raw.startswith(("\\\\", "//")):
        raise InferenceContractError(f"{field} must not be absolute.")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise InferenceContractError(f"{field} must not be absolute.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise InferenceContractError(f"{field} contains path traversal or invalid segments.")
    return path


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifact_before_load(
    *,
    project_root: str | Path,
    bundle: Mapping[str, Any],
    handoff: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve and hash the model without deserializing it."""

    relative = _portable_relative_path(
        bundle.get("model_artifact_path"), field="model_artifact_path"
    )
    root = Path(project_root).resolve()
    absolute = (root.joinpath(*relative.parts)).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise InferenceContractError("Model artifact path escapes the project root.") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"Model artifact not found: {relative.as_posix()}")
    expected_hash = str(bundle.get("model_artifact_sha256", ""))
    observed_hash = _sha256_file(absolute)
    if not expected_hash or observed_hash != expected_hash:
        raise TrustedModelSourceError(
            "Model artifact SHA-256 mismatch: "
            f"expected={expected_hash or '<missing>'}, observed={observed_hash}"
        )
    validate_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    return absolute


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InferenceContractError(f"{label} must contain a JSON object.")
    return payload


def _manifest_from_handoff(
    *, project_root: Path, handoff: Mapping[str, Any]
) -> dict[str, Any]:
    if _final_handoff_schema(handoff) == "final-model-handoff.v3":
        reference = _require_mapping(handoff.get("manifest_reference"), field="handoff.manifest_reference")
        hash_field = "sha256"
    else:
        references = _require_mapping(handoff.get("final_references"), field="handoff.final_references")
        reference = _require_mapping(references.get("final_model_manifest"), field="handoff.final_references.final_model_manifest")
        hash_field = "byte_sha256"
    relative = _portable_relative_path(reference.get("path"), field="final model manifest path")
    path = (project_root.joinpath(*relative.parts)).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise InferenceContractError("Final model manifest path escapes project root.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Final model manifest not found: {relative.as_posix()}")
    observed = _sha256_file(path)
    expected = str(reference.get(hash_field, ""))
    if observed != expected:
        raise InferenceContractError(
            "Final model manifest SHA-256 mismatch: "
            f"expected={expected}, observed={observed}"
        )
    if _final_handoff_schema(handoff) in {"final-model-handoff.v2", "final-model-handoff.v3"}:
        return load_and_validate_final_model_manifest(
            project_root=project_root,
            manifest_path=relative.as_posix(),
        )
    return _load_json_object(path, label="final model manifest")


def load_validated_inference_pipeline(
    *,
    project_root: str | Path,
    handoff_path: str | Path,
    bundle_path: str | Path,
    trusted_source: bool,
    observed_runtime_versions: Mapping[str, Any] | None = None,
    loader: Callable[..., Pipeline] = load_trusted_pipeline_from_bundle,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], RuntimeCompatibilityReport]:
    """Run all gates in order, then delegate to the trusted finalization loader."""

    root = Path(project_root).resolve()
    handoff = load_and_validate_final_model_handoff(
        project_root=root, handoff_path=handoff_path
    )
    bundle = load_and_validate_inference_bundle(
        project_root=root, bundle_path=bundle_path
    )
    validate_inference_readiness(handoff, bundle)
    manifest = _manifest_from_handoff(project_root=root, handoff=handoff)
    validate_bundle_handoff_alignment(handoff, bundle, manifest=manifest)
    validate_model_artifact_before_load(project_root=root, bundle=bundle, handoff=handoff, manifest=manifest)
    runtime_contract = manifest.get("runtime_versions") if _inference_bundle_schema(bundle) == "inference-bundle.v3" else bundle.get("runtime_version_requirements")
    report = validate_runtime_compatibility(
        _require_mapping(runtime_contract, field="runtime version contract"),
        observed_versions=observed_runtime_versions,
        mode="load_safe",
        raise_on_incompatible=True,
    )
    if trusted_source is not True:
        raise TrustedModelSourceError(
            "trusted_source=True is required before deserializing the model."
        )
    pipeline = loader(project_root=root, bundle=bundle)
    validate_loaded_pipeline_contract(pipeline, bundle=bundle, manifest=manifest)
    return pipeline, deepcopy(handoff), deepcopy(bundle), report


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _expected_dtype_mapping(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(
        bundle.get("expected_input_dtypes"), field="bundle.expected_input_dtypes"
    )


def _as_input_dataframe(value: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        index = pd.Index([value.name]) if value.name is not None else pd.RangeIndex(1)
        return pd.DataFrame([value.copy(deep=True).to_dict()], index=index)
    if isinstance(value, Mapping):
        return pd.DataFrame([deepcopy(dict(value))])
    raise InferenceInputError("Input must be a Mapping, pandas.Series, or pandas.DataFrame.")


def _condition_matches(value: Any, expected: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric) and numeric == float(expected)
    return value == expected


def apply_declared_missing_value_policy(
    dataframe: pd.DataFrame,
    policy_or_bundle: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[tuple[str, int], ...]]:
    """Apply only explicitly declared deterministic blank-materialization rules."""

    frame = dataframe.copy(deep=True)
    policy = policy_or_bundle.get("missing_value_policy", policy_or_bundle)
    policy = _require_mapping(policy, field="missing_value_policy")
    rules = policy.get("preparation_rules", ())
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise InferenceContractError("missing_value_policy.preparation_rules must be a list.")
    materializations: dict[str, int] = {}
    for position, raw_rule in enumerate(rules):
        rule = _require_mapping(raw_rule, field=f"preparation_rules[{position}]")
        required = {
            "column",
            "condition_column",
            "condition_value",
            "blank_replacement",
            "strip_strings",
        }
        missing = sorted(required.difference(rule))
        if missing:
            raise InferenceContractError(
                "Missing-value rule is incomplete: " + ", ".join(missing)
            )
        column = str(rule["column"])
        condition_column = str(rule["condition_column"])
        if column not in frame.columns or condition_column not in frame.columns:
            raise InferenceInputError(
                f"Missing-value rule references unavailable columns: {column}, {condition_column}."
            )
        count = 0
        column_values = frame[column].astype(object).copy(deep=True)
        for row_position in range(len(frame)):
            value = column_values.iloc[row_position]
            if pd.isna(value):
                raise InferenceInputError(
                    f"Column {column} contains an undeclared generic missing value."
                )
            normalized = value
            if bool(rule["strip_strings"]) and isinstance(value, str):
                normalized = value.strip()
                column_values.iloc[row_position] = normalized
            if isinstance(normalized, str) and normalized == "":
                condition_value = frame.iloc[row_position][condition_column]
                if not _condition_matches(condition_value, rule["condition_value"]):
                    raise InferenceInputError(
                        f"Blank {column} is allowed only when {condition_column} "
                        f"equals {rule['condition_value']!r}."
                    )
                column_values.iloc[row_position] = deepcopy(rule["blank_replacement"])
                count += 1
        frame[column] = column_values
        if count:
            materializations[column] = materializations.get(column, 0) + count
    return frame, tuple(sorted(materializations.items()))


def _coerce_string(series: pd.Series, *, column: str) -> pd.Series:
    if series.isna().any():
        raise InferenceInputError(f"Column {column} contains missing values.")
    try:
        return series.map(str).astype("string")
    except Exception as exc:
        raise InferenceInputError(f"Column {column} cannot be coerced to string.") from exc


def _coerce_numeric(series: pd.Series, *, column: str, integer: bool) -> pd.Series:
    if series.isna().any():
        raise InferenceInputError(f"Column {column} contains missing values.")
    converted = pd.to_numeric(series, errors="coerce")
    invalid_count = int(converted.isna().sum())
    if invalid_count:
        raise InferenceInputError(
            f"Column {column} contains {invalid_count} invalid numeric conversion(s)."
        )
    values = converted.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise InferenceInputError(f"Column {column} contains non-finite values.")
    if integer:
        if not np.equal(values, np.floor(values)).all():
            raise InferenceInputError(f"Column {column} requires integer values.")
        return pd.Series(values.astype("int64"), index=series.index, name=series.name)
    return pd.Series(values.astype("float64"), index=series.index, name=series.name)


def report_unknown_input_categories(
    dataframe: pd.DataFrame,
    bundle: Mapping[str, Any],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Report unknown categorical values without changing input or fitted vocabulary."""

    categorical = bundle.get("categorical_features")
    if not isinstance(categorical, Sequence) or isinstance(categorical, (str, bytes)):
        raise InferenceContractError("bundle.categorical_features must be a list.")
    if len(categorical) == 0:
        return ()
    vocabularies = _require_mapping(
        bundle.get("fitted_categorical_vocabularies"),
        field="bundle.fitted_categorical_vocabularies",
    )
    report: list[tuple[str, tuple[Any, ...]]] = []
    for column in sorted(str(value) for value in categorical):
        if column not in dataframe.columns:
            raise InferenceInputError(f"Categorical feature is missing: {column}")
        declared = list(deepcopy(vocabularies.get(column, [])))
        unknown: list[Any] = []
        for raw in dataframe[column].tolist():
            value = _python_scalar(raw)
            if not any(value == expected for expected in declared):
                if not any(value == prior for prior in unknown):
                    unknown.append(deepcopy(value))
        if unknown:
            unknown.sort(key=lambda value: (type(value).__name__, repr(value)))
            report.append((column, tuple(unknown)))
    return tuple(report)


def _validate_multiclass_input_contract(bundle: Mapping[str, Any]) -> None:
    features = list(bundle.get("feature_columns", ()))
    required = list(bundle.get("required_input_columns", ()))
    numerical = list(bundle.get("numerical_features", ()))
    categorical = list(bundle.get("categorical_features", ()))
    if not features or required != features or numerical != features or categorical:
        raise InferenceContractError(
            "V2 input contract must declare one ordered, required, numerical feature set."
        )
    dtypes = _expected_dtype_mapping(bundle)
    if list(dtypes) != features and set(dtypes) != set(features):
        raise InferenceContractError("V2 expected dtype fields differ from feature columns.")
    if any(dtypes.get(column) not in {"numeric", "integer"} for column in features):
        raise InferenceContractError("V2 numerical inputs require numeric/integer dtypes.")
    schema = bundle.get("expected_input_schema")
    if not isinstance(schema, Sequence) or isinstance(schema, (str, bytes)):
        raise InferenceContractError("bundle.expected_input_schema must be a list.")
    if len(schema) != len(features):
        raise InferenceContractError("V2 input schema length differs from feature count.")
    for column, raw in zip(features, schema, strict=True):
        entry = _require_mapping(raw, field=f"expected_input_schema[{column}]")
        expected = {
            "name": column,
            "role": "numerical",
            "required": True,
            "expected_dtype": dtypes[column],
            "missing_value_behavior": "reject",
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise InferenceContractError(
                f"V2 input schema entry differs for {column}."
            )
    policy = _require_mapping(
        bundle.get("missing_value_policy"), field="bundle.missing_value_policy"
    )
    if policy.get("strategy") != "reject_missing_required_values":
        raise InferenceContractError("V2 missing-value strategy must reject missing values.")
    if policy.get("learned_imputation_in_final_pipeline") is not False:
        raise InferenceContractError("V2 final pipeline must not declare learned imputation.")


def normalize_inference_input(
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    reject_extra_columns: bool = True,
) -> InputNormalizationResult:
    """Validate, defensively copy, materialize declared blanks, and coerce inputs."""

    if _inference_bundle_schema(bundle) == "inference-bundle.v2":
        _validate_multiclass_input_contract(bundle)
    frame = _as_input_dataframe(value)
    if frame.empty:
        raise InferenceInputError("Inference input must contain at least one row.")
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise InferenceInputError(
            "Inference input contains duplicate columns: "
            + ", ".join(sorted(map(str, duplicates)))
        )
    features = bundle.get("feature_columns")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise InferenceContractError("bundle.feature_columns must be a list.")
    feature_columns = tuple(str(column) for column in features)
    prohibited = tuple(str(column) for column in bundle.get("prohibited_input_columns", ()))
    present_prohibited = sorted(set(frame.columns).intersection(prohibited))
    if present_prohibited:
        raise InferenceInputError(
            "Prohibited input columns are present: " + ", ".join(present_prohibited)
        )
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise InferenceInputError("Missing required input columns: " + ", ".join(missing))
    extras = [column for column in frame.columns if column not in feature_columns]
    if reject_extra_columns and extras:
        raise InferenceInputError(
            "Unexpected input columns are not accepted: "
            + ", ".join(sorted(map(str, extras)))
        )
    frame = frame.loc[:, list(feature_columns)].copy(deep=True)
    frame, materializations = apply_declared_missing_value_policy(frame, bundle)

    expected_dtypes = _expected_dtype_mapping(bundle)
    coerced = pd.DataFrame(index=frame.index.copy())
    for column in feature_columns:
        expected = expected_dtypes.get(column)
        if expected == "string":
            coerced[column] = _coerce_string(frame[column], column=column)
        elif expected == "integer":
            coerced[column] = _coerce_numeric(
                frame[column], column=column, integer=True
            )
        elif expected == "numeric":
            coerced[column] = _coerce_numeric(
                frame[column], column=column, integer=False
            )
        else:
            raise InferenceContractError(
                f"Unsupported expected dtype for {column}: {expected!r}"
            )
    unknown = report_unknown_input_categories(coerced, bundle)
    return InputNormalizationResult(
        dataframe=coerced.copy(deep=True),
        input_materializations_applied=materializations,
        unknown_categories_report=unknown,
    )


def _fitted_encoder(preprocess: ColumnTransformer) -> OneHotEncoder:
    try:
        encoder = preprocess.named_transformers_["categorical"]
    except Exception as exc:
        raise InferenceContractError(
            "Loaded pipeline lacks a fitted categorical transformer."
        ) from exc
    if not isinstance(encoder, OneHotEncoder):
        raise InferenceContractError("Categorical transformer must be OneHotEncoder.")
    return encoder


def _validate_loaded_pipeline_contract_v1(
    pipeline: Any,
    *,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate the exact fitted pipeline contract without fitting or transforming data."""

    if not isinstance(pipeline, Pipeline):
        raise InferenceContractError("Loaded artifact must be an sklearn Pipeline.")
    if list(pipeline.named_steps) != ["preprocess", "model"]:
        raise InferenceContractError("Pipeline steps must be exactly preprocess then model.")
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not isinstance(preprocess, ColumnTransformer):
        raise InferenceContractError("Pipeline preprocess step must be ColumnTransformer.")
    expected_model_family = bundle.get("model_family")
    if not isinstance(expected_model_family, str) or not expected_model_family:
        raise InferenceContractError("Bundle model_family must be a non-empty string.")
    if model.__class__.__name__ != expected_model_family:
        raise InferenceContractError(
            "Pipeline model family differs from bundle: "
            f"expected={expected_model_family}, observed={model.__class__.__name__}"
        )
    try:
        check_is_fitted(pipeline)
        check_is_fitted(preprocess)
        check_is_fitted(model)
    except Exception as exc:
        raise InferenceContractError("Loaded pipeline must already be fitted.") from exc

    if bundle.get("preprocessing_embedded") is not True:
        raise InferenceContractError("Bundle must declare preprocessing_embedded=true.")
    if preprocess.remainder != "drop" or float(preprocess.sparse_threshold) != 0.0:
        raise InferenceContractError(
            "ColumnTransformer must use remainder='drop' and dense output."
        )
    configured = {
        name: (transformer, list(columns))
        for name, transformer, columns in preprocess.transformers
    }
    numerical = list(bundle.get("numerical_features", ()))
    categorical = list(bundle.get("categorical_features", ()))
    numerical_transformer, numerical_columns = configured.get("numerical", (None, []))
    if numerical_transformer != "passthrough" or numerical_columns != numerical:
        raise InferenceContractError("Numerical passthrough contract differs from bundle.")
    _, categorical_columns = configured.get("categorical", (None, []))
    if categorical_columns != categorical:
        raise InferenceContractError("Categorical feature order differs from bundle.")
    encoder = _fitted_encoder(preprocess)
    if encoder.handle_unknown != "ignore" or encoder.drop is not None:
        raise InferenceContractError("OneHotEncoder unknown/drop policy differs from bundle.")
    if hasattr(encoder, "sparse_output") and encoder.sparse_output is not False:
        raise InferenceContractError("OneHotEncoder output must be dense.")

    feature_columns = list(bundle.get("feature_columns", ()))
    observed_features = list(getattr(pipeline, "feature_names_in_", ()))
    if observed_features != feature_columns:
        raise InferenceContractError("Loaded pipeline feature order differs from bundle.")
    transformed = list(preprocess.get_feature_names_out())
    if transformed != list(bundle.get("transformed_feature_names", ())):
        raise InferenceContractError("Transformed feature names differ from bundle.")

    vocabularies = _require_mapping(
        bundle.get("fitted_categorical_vocabularies"),
        field="bundle.fitted_categorical_vocabularies",
    )
    if len(encoder.categories_) != len(categorical):
        raise InferenceContractError("Fitted categorical vocabulary count differs from bundle.")
    for column, observed_values in zip(categorical, encoder.categories_, strict=True):
        observed = [_python_scalar(value) for value in observed_values.tolist()]
        expected = list(vocabularies.get(column, ()))
        if observed != expected:
            raise InferenceContractError(
                f"Fitted categorical vocabulary differs for {column}."
            )

    classes = [_python_scalar(value) for value in np.asarray(model.classes_).tolist()]
    target_encoding = _require_mapping(
        bundle.get("target_encoding"), field="bundle.target_encoding"
    )
    expected_classes = list(target_encoding.values())
    if classes != expected_classes:
        raise InferenceContractError(
            f"Estimator classes differ: expected={expected_classes}, observed={classes}"
        )
    if bundle.get("positive_encoded_label") not in classes:
        raise InferenceContractError("Positive encoded label is absent from estimator classes.")

    params = model.get_params(deep=False)
    hyperparameters = _require_mapping(
        bundle.get("selected_hyperparameters"),
        field="bundle.selected_hyperparameters",
    )
    for key, expected in hyperparameters.items():
        parameter = str(key).removeprefix("model__")
        if params.get(parameter) != expected:
            raise InferenceContractError(
                f"Loaded model hyperparameter differs for {parameter}."
            )
    if "random_state" in params and params["random_state"] != bundle.get(
        "estimator_random_state"
    ):
        raise InferenceContractError("Loaded model random_state differs from bundle.")
    if model.__class__.__name__ != bundle.get("model_family"):
        raise InferenceContractError("Loaded model family differs from bundle.")

    if manifest is not None:
        descriptor = _require_mapping(
            manifest.get("fitted_state_descriptor"),
            field="manifest.fitted_state_descriptor",
        )
        if descriptor.get("steps") != ["preprocess", "model"]:
            raise InferenceContractError("Manifest fitted-state step descriptor differs.")
        if descriptor.get("feature_order") != feature_columns:
            raise InferenceContractError("Manifest fitted-state feature order differs.")
        if descriptor.get("transformed_feature_names") != transformed:
            raise InferenceContractError(
                "Manifest transformed feature names differ from loaded pipeline."
            )
        if descriptor.get("categorical_vocabularies") != dict(vocabularies):
            raise InferenceContractError(
                "Manifest categorical vocabularies differ from inference bundle."
            )
        if manifest.get("fitted_state_semantic_fingerprint") != bundle.get(
            "model_state_fingerprint"
        ):
            raise InferenceContractError("Manifest model-state fingerprint differs.")


def validate_multiclass_loaded_pipeline_contract(
    pipeline: Any,
    *,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate a fitted numerical-only v2 pipeline without fitting or transforming."""

    if _inference_bundle_schema(bundle) != "inference-bundle.v2":
        raise InferenceContractError("Multiclass pipeline validation requires a v2 bundle.")
    if not isinstance(pipeline, Pipeline):
        raise InferenceContractError("Loaded artifact must be an sklearn Pipeline.")
    if list(pipeline.named_steps) != ["preprocess", "model"]:
        raise InferenceContractError("Pipeline steps must be exactly preprocess then model.")
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not isinstance(preprocess, ColumnTransformer):
        raise InferenceContractError("Pipeline preprocess step must be ColumnTransformer.")
    expected_model_family = bundle.get("model_family")
    if model.__class__.__name__ != expected_model_family:
        raise InferenceContractError(
            "Pipeline model family differs from multiclass bundle."
        )
    try:
        check_is_fitted(pipeline)
        check_is_fitted(preprocess)
        check_is_fitted(model)
    except Exception as exc:
        raise InferenceContractError("Loaded pipeline must already be fitted.") from exc

    feature_columns = list(bundle.get("feature_columns", ()))
    numerical = list(bundle.get("numerical_features", ()))
    categorical = list(bundle.get("categorical_features", ()))
    if not feature_columns or numerical != feature_columns:
        raise InferenceContractError(
            "Multiclass numerical feature order must equal the full feature order."
        )
    if categorical:
        raise InferenceContractError(
            "This v2 numerical-only contract must have zero categorical features."
        )
    if preprocess.remainder != "drop" or float(preprocess.sparse_threshold) != 0.0:
        raise InferenceContractError(
            "ColumnTransformer must use remainder='drop' and dense output."
        )
    configured = {
        name: (transformer, list(columns))
        for name, transformer, columns in preprocess.transformers
    }
    if set(configured) != {"numerical"}:
        raise InferenceContractError(
            "Multiclass pipeline must contain only the numerical transformer."
        )
    numerical_transformer, numerical_columns = configured["numerical"]
    if numerical_transformer != "passthrough" or numerical_columns != numerical:
        raise InferenceContractError("Numerical passthrough contract differs from bundle.")
    observed_features = list(getattr(pipeline, "feature_names_in_", ()))
    if observed_features != feature_columns:
        raise InferenceContractError("Loaded pipeline feature order differs from bundle.")

    descriptor = _require_mapping(
        bundle.get("model_state_descriptor"), field="bundle.model_state_descriptor"
    )
    transformed = list(preprocess.get_feature_names_out())
    expected_transformed = list(descriptor.get("transformed_feature_names", ()))
    if transformed != expected_transformed:
        raise InferenceContractError("Transformed feature names differ from bundle.")
    if descriptor.get("steps") != ["preprocess", "model"]:
        raise InferenceContractError("Fitted-state step descriptor differs.")
    if descriptor.get("feature_order") != feature_columns:
        raise InferenceContractError("Fitted-state feature order differs.")
    if descriptor.get("preprocessing_contract") != bundle.get(
        "preprocessing_contract"
    ):
        raise InferenceContractError("Fitted-state preprocessing contract differs.")

    classes = [_python_scalar(value) for value in np.asarray(model.classes_).tolist()]
    estimator_order = list(bundle.get("estimator_class_order", ()))
    output_order = list(bundle.get("output_class_order", ()))
    if classes != estimator_order:
        raise InferenceContractError(
            f"Estimator class order differs: expected={estimator_order}, observed={classes}"
        )
    if len(output_order) < 3 or set(classes) != set(output_order):
        raise InferenceContractError("Estimator class set differs from output class set.")
    if descriptor.get("estimator_class_order") != classes:
        raise InferenceContractError("Fitted-state estimator class order differs.")
    if descriptor.get("output_class_order") != output_order:
        raise InferenceContractError("Fitted-state output class order differs.")

    params = model.get_params(deep=False)
    hyperparameters = _require_mapping(
        bundle.get("selected_hyperparameters"),
        field="bundle.selected_hyperparameters",
    )
    for key, expected in hyperparameters.items():
        parameter = str(key).removeprefix("model__")
        if parameter not in params or params[parameter] != expected:
            raise InferenceContractError(
                f"Loaded model hyperparameter differs for {parameter}."
            )
    if "random_state" in params and params["random_state"] != bundle.get(
        "estimator_random_state"
    ):
        raise InferenceContractError("Loaded model random_state differs from bundle.")
    if descriptor.get("random_state") != bundle.get("estimator_random_state"):
        raise InferenceContractError("Fitted-state random_state differs from bundle.")
    fingerprint = bundle.get("model_state_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise InferenceContractError("Model-state fingerprint is missing from bundle.")

    if manifest is not None:
        if manifest.get("model_state_descriptor") != dict(descriptor):
            raise InferenceContractError(
                "Manifest fitted-state descriptor differs from loaded bundle."
            )
        if manifest.get("model_state_fingerprint") != fingerprint:
            raise InferenceContractError("Manifest model-state fingerprint differs.")


def validate_loaded_pipeline_contract(
    pipeline: Any,
    *,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Dispatch fitted-pipeline validation by inference-bundle schema."""

    schema = _inference_bundle_schema(bundle)
    if schema == "inference-bundle.v1":
        _validate_loaded_pipeline_contract_v1(
            pipeline, bundle=bundle, manifest=manifest
        )
        return
    if schema == "inference-bundle.v2":
        validate_multiclass_loaded_pipeline_contract(pipeline, bundle=bundle, manifest=manifest)
        return
    validate_continuous_loaded_pipeline_contract(pipeline, bundle=bundle, manifest=manifest)


def _validate_continuous_contract(bundle: Mapping[str, Any]) -> tuple[list[str], str]:
    features = list(bundle.get("feature_order", ()))
    dtypes = _require_mapping(bundle.get("input_feature_dtypes"), field="bundle.input_feature_dtypes")
    if not features or len(features) != len(set(features)) or set(dtypes) != set(features):
        raise InferenceContractError("Continuous feature/dtype contract is invalid.")
    for feature in features:
        spec = _require_mapping(dtypes[feature], field=f"input_feature_dtypes.{feature}")
        if spec.get("dtype") not in {"float64", "int64"} or spec.get("role") != "numerical_feature":
            raise InferenceContractError(f"Unsupported continuous dtype contract for {feature}.")
    prediction = _require_mapping(bundle.get("prediction_contract"), field="bundle.prediction_contract")
    target = _require_mapping(bundle.get("target_contract"), field="bundle.target_contract")
    unit = prediction.get("unit")
    if prediction.get("type") != "continuous_numeric" or prediction.get("scale") != "original_target_scale":
        raise InferenceContractError("Continuous prediction contract is invalid.")
    if not isinstance(unit, str) or not unit.strip() or unit != target.get("unit"):
        raise InferenceContractError("Prediction and target units must be equal and non-empty.")
    return features, unit


def normalize_continuous_inference_input(
    value: Mapping[str, Any] | pd.Series | pd.DataFrame, *, bundle: Mapping[str, Any]
) -> pd.DataFrame:
    """Strictly validate ordered v3 input without silently reordering it."""
    if _inference_bundle_schema(bundle) != "inference-bundle.v3":
        raise InferenceContractError("Continuous input requires a v3 bundle.")
    features, _ = _validate_continuous_contract(bundle)
    frame = _as_input_dataframe(value)
    if frame.empty:
        raise InferenceInputError("Inference input must contain at least one row.")
    if frame.columns.duplicated().any():
        raise InferenceInputError("Inference input contains duplicate columns.")
    missing = [name for name in features if name not in frame.columns]
    extras = [name for name in frame.columns if name not in features]
    if missing:
        raise InferenceInputError("Missing required input columns: " + ", ".join(missing))
    if extras:
        raise InferenceInputError("Unexpected input columns are not accepted: " + ", ".join(map(str, extras)))
    if list(frame.columns) != features:
        raise InferenceInputError("Input feature order differs from bundle.feature_order.")
    result = pd.DataFrame(index=frame.index.copy())
    specs = _require_mapping(bundle["input_feature_dtypes"], field="bundle.input_feature_dtypes")
    for feature in features:
        try:
            numeric = pd.to_numeric(frame[feature], errors="raise")
        except (TypeError, ValueError) as exc:
            raise InferenceInputError(f"Input column {feature} must be numeric.") from exc
        values = np.asarray(numeric, dtype=float)
        if not np.isfinite(values).all():
            raise InferenceInputError(f"Input column {feature} must contain finite values.")
        dtype = _require_mapping(specs[feature], field=f"input_feature_dtypes.{feature}")["dtype"]
        if dtype == "int64" and not np.equal(values, np.trunc(values)).all():
            raise InferenceInputError(f"Input column {feature} must contain integral values.")
        try:
            result[feature] = numeric.astype(dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InferenceInputError(f"Input column {feature} cannot be represented as {dtype}.") from exc
    return result


def validate_continuous_loaded_pipeline_contract(
    pipeline: Any, *, bundle: Mapping[str, Any], manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate the fitted continuous v3 pipeline without fitting it."""
    features, _ = _validate_continuous_contract(bundle)
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["preprocess", "model"]:
        raise InferenceContractError("Pipeline steps must be exactly preprocess then model.")
    preprocess, model = pipeline.named_steps.values()
    if not isinstance(preprocess, ColumnTransformer):
        raise InferenceContractError("Pipeline preprocess step must be ColumnTransformer.")
    try:
        check_is_fitted(pipeline); check_is_fitted(preprocess); check_is_fitted(model)
    except Exception as exc:
        raise InferenceContractError("Loaded pipeline must already be fitted.") from exc
    contract = _require_mapping(bundle.get("model_contract"), field="bundle.model_contract")
    descriptor = _require_mapping(bundle.get("model_state_descriptor"), field="bundle.model_state_descriptor")
    if model.__class__.__name__ != contract.get("family") or list(getattr(pipeline, "feature_names_in_", ())) != features or int(getattr(pipeline, "n_features_in_", -1)) != len(features):
        raise InferenceContractError("Loaded continuous model identity/features differ from bundle.")
    indicators = _require_mapping(descriptor.get("fitted_state_indicators"), field="descriptor.fitted_state_indicators")
    if descriptor.get("step_names") != ["preprocess", "model"] or descriptor.get("feature_order") != features or descriptor.get("preprocessing_contract") != bundle.get("preprocessing_contract") or indicators.get("n_features_in_") != len(features):
        raise InferenceContractError("Continuous fitted-state descriptor differs.")
    params = model.get_params(deep=False)
    expected_params = dict(_require_mapping(contract.get("fixed_constructor_parameters"), field="model_contract.fixed_constructor_parameters"))
    for key, expected in _require_mapping(contract.get("selected_hyperparameters"), field="model_contract.selected_hyperparameters").items():
        expected_params[str(key).removeprefix("model__")] = expected
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            raise InferenceContractError(f"Continuous model parameter differs for {key}.")
    if manifest is not None:
        artifact = _require_mapping(manifest.get("model_artifact"), field="manifest.model_artifact")
        if artifact.get("state_descriptor") != dict(descriptor) or artifact.get("state_fingerprint") != bundle.get("model_state_fingerprint"):
            raise InferenceContractError("Continuous manifest fitted state differs.")


def predict_continuous_batch(pipeline: Pipeline, value: Mapping[str, Any] | pd.Series | pd.DataFrame, *, bundle: Mapping[str, Any], runtime_report: RuntimeCompatibilityReport | None = None) -> pd.Series:
    """Make exactly one continuous predict call for a validated batch."""
    if runtime_report is not None and not runtime_report.compatible:
        raise RuntimeCompatibilityError("Runtime is not compatible for prediction.", report=runtime_report)
    validate_continuous_loaded_pipeline_contract(pipeline, bundle=bundle)
    frame = normalize_continuous_inference_input(value, bundle=bundle)
    raw = np.asarray(pipeline.predict(frame))
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or len(raw) != len(frame):
        raise InferenceContractError("Continuous prediction shape must be one value per row.")
    try:
        values = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise InferenceContractError("Continuous predictions must be numeric.") from exc
    if not np.isfinite(values).all():
        raise InferenceContractError("Continuous predictions must be finite.")
    return pd.Series(values, index=frame.index.copy(), name="prediction")


def continuous_output_to_frame(output: pd.Series, *, bundle: Mapping[str, Any], identifier_name: str = "example_id") -> pd.DataFrame:
    """Present identifiers, predictions, and the contract-derived unit."""
    _, unit = _validate_continuous_contract(bundle)
    return pd.DataFrame({identifier_name: output.index.copy(), "prediction": output.to_numpy(copy=True), "unit": unit})


def resolve_positive_probability_column(
    pipeline: Pipeline, *, bundle: Mapping[str, Any]
) -> int:
    """Resolve the positive probability column from fitted estimator classes_."""

    if not isinstance(pipeline, Pipeline) or "model" not in pipeline.named_steps:
        raise InferenceContractError("A fitted sklearn Pipeline is required.")
    classes = [
        _python_scalar(value)
        for value in np.asarray(pipeline.named_steps["model"].classes_).tolist()
    ]
    positive = bundle.get("positive_encoded_label")
    matches = [index for index, value in enumerate(classes) if value == positive]
    if len(matches) != 1:
        raise InferenceContractError(
            "Positive encoded label cannot be resolved uniquely from estimator classes_."
        )
    return matches[0]


def resolve_multiclass_probability_columns(
    pipeline: Pipeline, *, bundle: Mapping[str, Any]
) -> tuple[int, ...]:
    """Resolve estimator probability columns into the declared v2 output order."""

    if _inference_bundle_schema(bundle) != "inference-bundle.v2":
        raise InferenceContractError("Multiclass probability mapping requires v2.")
    if not isinstance(pipeline, Pipeline) or "model" not in pipeline.named_steps:
        raise InferenceContractError("A fitted sklearn Pipeline is required.")
    classes = [
        _python_scalar(value)
        for value in np.asarray(pipeline.named_steps["model"].classes_).tolist()
    ]
    estimator_order = list(bundle.get("estimator_class_order", ()))
    output_order = list(bundle.get("output_class_order", ()))
    if classes != estimator_order:
        raise InferenceContractError(
            "Loaded estimator class order differs from the bundle declaration."
        )
    if len(output_order) < 3 or set(classes) != set(output_order):
        raise InferenceContractError(
            "Estimator classes differ from the multiclass output contract."
        )
    return tuple(classes.index(label) for label in output_order)


def build_multiclass_inference_output(
    raw_probabilities: Any,
    *,
    estimator_predictions: Sequence[Any],
    index: pd.Index,
    pipeline: Pipeline,
    bundle: Mapping[str, Any],
    normalization: InputNormalizationResult,
    runtime_report: RuntimeCompatibilityReport | None = None,
) -> pd.DataFrame:
    """Validate, remap, and build the educational v2 core output contract."""

    if _inference_bundle_schema(bundle) != "inference-bundle.v2":
        raise InferenceContractError("Multiclass output construction requires v2.")
    if bundle.get("decision_rule") != "argmax_class_score_or_probability":
        raise InferenceContractError("Unsupported multiclass decision rule.")
    output_contract = _require_mapping(
        bundle.get("inference_output_contract"),
        field="bundle.inference_output_contract",
    )
    output_order = list(bundle.get("output_class_order", ()))
    probability_contract = _require_mapping(
        output_contract.get("class_probabilities"),
        field="bundle.inference_output_contract.class_probabilities",
    )
    if output_contract.get("class_order") != output_order:
        raise InferenceContractError("Output class order differs from output contract.")
    if probability_contract.get("length") != len(output_order):
        raise InferenceContractError("Probability length differs from output class count.")
    if probability_contract.get("aligned_to") != "class_order":
        raise InferenceContractError("Probabilities must be aligned to class_order.")

    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(index):
        raise InferenceContractError("predict_proba returned an unexpected shape.")
    columns = resolve_multiclass_probability_columns(pipeline, bundle=bundle)
    if raw.shape[1] != len(columns):
        raise InferenceContractError("predict_proba class count differs from bundle.")
    ordered = raw[:, list(columns)].copy()
    if ordered.shape != (len(index), len(output_order)):
        raise InferenceContractError("Remapped probability shape differs from contract.")
    tolerance = 1e-12
    if not np.isfinite(ordered).all():
        raise InferenceContractError("Multiclass probabilities must be finite.")
    if (ordered < -tolerance).any() or (ordered > 1.0 + tolerance).any():
        raise InferenceContractError("Multiclass probabilities must be in [0, 1].")
    if not np.allclose(ordered.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise InferenceContractError("Multiclass probability rows must sum to one.")

    argmax_positions = np.argmax(ordered, axis=1)
    predicted = [output_order[int(position)] for position in argmax_positions]
    estimator_labels = [_python_scalar(value) for value in estimator_predictions]
    if len(estimator_labels) != len(index):
        raise InferenceContractError("Estimator prediction count differs from input rows.")
    if estimator_labels != predicted:
        raise InferenceContractError(
            "Estimator predictions disagree with remapped probability argmax."
        )
    fingerprint = bundle.get("model_state_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise InferenceContractError("Model-state fingerprint is missing from bundle.")

    records: list[dict[str, Any]] = []
    for position, label in enumerate(predicted):
        probabilities = [float(value) for value in ordered[position].tolist()]
        records.append(
            {
                "predicted_class": _python_scalar(label),
                "class_order": deepcopy(output_order),
                "class_probabilities": probabilities,
                "top_probability": float(max(probabilities)),
                "operational_prediction_available": False,
                "unknown_categories_report": normalization.unknown_categories_dict(),
                "input_materializations_applied": normalization.materializations_dict(),
                "runtime_compatibility_confirmed": bool(
                    runtime_report.compatible if runtime_report is not None else False
                ),
                "runtime_compatibility_report": (
                    runtime_report.as_dict() if runtime_report is not None else None
                ),
                "model_state_fingerprint": fingerprint,
            }
        )
    output = pd.DataFrame(records, index=index.copy())
    if len(output) != len(index):
        raise InferenceContractError("Output row count differs from input row count.")
    for column in output.columns:
        output[column] = output[column].astype(object)
    return output


def predict_multiclass_batch(
    pipeline: Pipeline,
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    runtime_report: RuntimeCompatibilityReport | None = None,
    validate_pipeline: bool = True,
) -> pd.DataFrame:
    """Normalize inputs and run one safe, contract-aligned v2 batch inference."""

    if _inference_bundle_schema(bundle) != "inference-bundle.v2":
        raise InferenceContractError("predict_multiclass_batch requires a v2 bundle.")
    if validate_pipeline:
        validate_multiclass_loaded_pipeline_contract(pipeline, bundle=bundle)
    normalization = normalize_inference_input(value, bundle=bundle)
    inference_frame = normalization.dataframe.copy(deep=True)
    probabilities = pipeline.predict_proba(inference_frame.copy(deep=True))
    predictions = pipeline.predict(inference_frame.copy(deep=True))
    return build_multiclass_inference_output(
        probabilities,
        estimator_predictions=predictions,
        index=inference_frame.index,
        pipeline=pipeline,
        bundle=bundle,
        normalization=normalization,
        runtime_report=runtime_report,
    )


def predict_multiclass(
    pipeline: Pipeline,
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    runtime_report: RuntimeCompatibilityReport | None = None,
    validate_pipeline: bool = True,
) -> dict[str, Any]:
    """Run one v2 inference and return an unambiguous Python-native result."""

    output = predict_multiclass_batch(
        pipeline,
        value,
        bundle=bundle,
        runtime_report=runtime_report,
        validate_pipeline=validate_pipeline,
    )
    if len(output) != 1:
        raise InferenceInputError("Single-row inference requires exactly one input row.")
    row = output.iloc[0]
    return {
        "predicted_class": _python_scalar(row["predicted_class"]),
        "class_order": deepcopy(row["class_order"]),
        "class_probabilities": deepcopy(row["class_probabilities"]),
        "top_probability": float(row["top_probability"]),
        "operational_prediction_available": False,
        "unknown_categories_report": deepcopy(row["unknown_categories_report"]),
        "input_materializations_applied": deepcopy(
            row["input_materializations_applied"]
        ),
        "runtime_compatibility_confirmed": bool(
            row["runtime_compatibility_confirmed"]
        ),
        "runtime_compatibility_report": deepcopy(row["runtime_compatibility_report"]),
        "model_state_fingerprint": str(row["model_state_fingerprint"]),
    }


def multiclass_output_to_frame(output: pd.DataFrame) -> pd.DataFrame:
    """Create a presentation table without changing the v2 core output contract."""

    required = {"predicted_class", "class_order", "class_probabilities"}
    if not required.issubset(output.columns) or output.empty:
        raise InferenceContractError("A non-empty multiclass core output is required.")
    class_order = list(output.iloc[0]["class_order"])
    records: list[dict[str, Any]] = []
    for _, row in output.iterrows():
        row_order = list(row["class_order"])
        probabilities = list(row["class_probabilities"])
        if row_order != class_order or len(probabilities) != len(class_order):
            raise InferenceContractError("Multiclass output rows use inconsistent classes.")
        record = {"predicted_class": row["predicted_class"]}
        record.update(
            {
                f"probability_{label}": float(probability)
                for label, probability in zip(class_order, probabilities, strict=True)
            }
        )
        records.append(record)
    return pd.DataFrame(records, index=output.index.copy())


def build_inference_output(
    probabilities: Sequence[float],
    *,
    index: pd.Index,
    bundle: Mapping[str, Any],
    normalization: InputNormalizationResult,
    runtime_report: RuntimeCompatibilityReport | None = None,
) -> pd.DataFrame:
    """Build the educational-only output contract without persistence."""

    threshold = float(bundle.get("educational_decision_threshold"))
    if not 0.0 <= threshold <= 1.0:
        raise InferenceContractError("Educational threshold must be in [0, 1].")
    target_encoding = _require_mapping(
        bundle.get("target_encoding"), field="bundle.target_encoding"
    )
    reverse = {_python_scalar(value): str(label) for label, value in target_encoding.items()}
    positive = _python_scalar(bundle.get("positive_encoded_label"))
    if positive not in reverse:
        raise InferenceContractError("Positive encoded label is absent from target encoding.")
    model_fingerprint = str(bundle.get("model_state_fingerprint", ""))
    if not model_fingerprint:
        raise InferenceContractError("Model-state fingerprint is missing from bundle.")

    records: list[dict[str, Any]] = []
    for raw_probability in probabilities:
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise InferenceContractError("Predicted probability must be finite and in [0, 1].")
        encoded = int(probability >= threshold)
        if encoded not in reverse:
            raise InferenceContractError("Educational prediction lacks a reverse target label.")
        records.append(
            {
                "positive_class_probability": probability,
                "educational_prediction_encoded": int(encoded),
                "educational_prediction_label": reverse[encoded],
                "educational_threshold": threshold,
                "operational_prediction_available": False,
                "unknown_categories_report": normalization.unknown_categories_dict(),
                "input_materializations_applied": normalization.materializations_dict(),
                "runtime_compatibility_confirmed": bool(
                    runtime_report.compatible if runtime_report is not None else False
                ),
                "runtime_compatibility_report": (
                    runtime_report.as_dict() if runtime_report is not None else None
                ),
                "model_state_fingerprint": model_fingerprint,
            }
        )
    if len(records) != len(index):
        raise InferenceContractError("Output row count differs from input row count.")
    output = pd.DataFrame(records, index=index.copy())
    for column in output.columns:
        output[column] = output[column].astype(object)
    return output


def predict_educational_batch(
    pipeline: Pipeline,
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    runtime_report: RuntimeCompatibilityReport | None = None,
    validate_pipeline: bool = True,
) -> pd.DataFrame:
    """Normalize independent inputs and call ``predict_proba`` once for the batch."""

    if validate_pipeline:
        validate_loaded_pipeline_contract(pipeline, bundle=bundle)
    normalization = normalize_inference_input(value, bundle=bundle)
    probabilities = np.asarray(
        pipeline.predict_proba(normalization.dataframe.copy(deep=True)), dtype=float
    )
    if probabilities.ndim != 2 or probabilities.shape[0] != len(normalization.dataframe):
        raise InferenceContractError("predict_proba returned an unexpected shape.")
    positive_column = resolve_positive_probability_column(pipeline, bundle=bundle)
    if positive_column >= probabilities.shape[1]:
        raise InferenceContractError("Positive probability column is outside output shape.")
    positive_probabilities = probabilities[:, positive_column].copy()
    return build_inference_output(
        [float(value) for value in positive_probabilities],
        index=normalization.dataframe.index,
        bundle=bundle,
        normalization=normalization,
        runtime_report=runtime_report,
    )


def predict_educational(
    pipeline: Pipeline,
    value: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    bundle: Mapping[str, Any],
    runtime_report: RuntimeCompatibilityReport | None = None,
    validate_pipeline: bool = True,
) -> dict[str, Any]:
    """Run one educational inference and return Python scalar output values."""

    output = predict_educational_batch(
        pipeline,
        value,
        bundle=bundle,
        runtime_report=runtime_report,
        validate_pipeline=validate_pipeline,
    )
    if len(output) != 1:
        raise InferenceInputError("Single-row inference requires exactly one input row.")
    row = output.iloc[0]
    return {
        "positive_class_probability": float(row["positive_class_probability"]),
        "educational_prediction_encoded": int(row["educational_prediction_encoded"]),
        "educational_prediction_label": str(row["educational_prediction_label"]),
        "educational_threshold": float(row["educational_threshold"]),
        "operational_prediction_available": False,
        "unknown_categories_report": deepcopy(row["unknown_categories_report"]),
        "input_materializations_applied": deepcopy(
            row["input_materializations_applied"]
        ),
        "runtime_compatibility_confirmed": bool(
            row["runtime_compatibility_confirmed"]
        ),
        "runtime_compatibility_report": deepcopy(row["runtime_compatibility_report"]),
        "model_state_fingerprint": str(row["model_state_fingerprint"]),
    }


__all__ = [
    "InferenceContractError",
    "RuntimeCompatibilityError",
    "InferenceInputError",
    "TrustedModelSourceError",
    "RuntimeCompatibilityWarning",
    "RuntimeComponentReport",
    "RuntimeCompatibilityReport",
    "InputNormalizationResult",
    "current_runtime_versions",
    "validate_runtime_compatibility",
    "validate_inference_readiness",
    "validate_multiclass_inference_readiness",
    "validate_continuous_inference_readiness",
    "validate_bundle_handoff_alignment",
    "validate_multiclass_bundle_handoff_alignment",
    "validate_continuous_bundle_handoff_alignment",
    "validate_model_artifact_before_load",
    "load_validated_inference_pipeline",
    "validate_loaded_pipeline_contract",
    "validate_multiclass_loaded_pipeline_contract",
    "validate_continuous_loaded_pipeline_contract",
    "normalize_continuous_inference_input",
    "predict_continuous_batch",
    "continuous_output_to_frame",
    "normalize_inference_input",
    "apply_declared_missing_value_policy",
    "report_unknown_input_categories",
    "resolve_positive_probability_column",
    "resolve_multiclass_probability_columns",
    "predict_educational",
    "predict_educational_batch",
    "build_inference_output",
    "predict_multiclass",
    "predict_multiclass_batch",
    "build_multiclass_inference_output",
    "multiclass_output_to_frame",
]
