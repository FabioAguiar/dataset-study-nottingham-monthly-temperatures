from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_data import (
    ContinuousRegressionSplitPolicy,
    DatasetValidationError,
    HandoffValidationError,
    PartitionValidationError,
    build_feature_manifest,
    build_preparation_handoff_manifest,
    build_preparation_manifest,
    build_quality_evidence,
    build_split_manifest,
    fingerprint_dataframe,
    fingerprint_dataframe_csv,
    fingerprint_file,
    load_and_validate_preparation_handoff,
    prepare_tabular_dataset,
    split_continuous_regression_dataset,
    validate_prepared_dataset,
    validate_raw_dataset,
    validate_regression_partitions,
    write_preparation_artifacts,
)

SLUG = "synthetic-regression"
FEATURES = ("input_a", "input_b")
TARGET = "response"
MANIFEST_ROOT = Path("artifacts/preparation") / SLUG
HANDOFF_PATH = MANIFEST_ROOT / "preparation-handoff.json"


def _frame() -> pd.DataFrame:
    values = np.arange(40, dtype=float)
    return pd.DataFrame({"input_a": values + 0.25, "input_b": values * 2 + 1, "response": values * 0.7 + 3})


def _readiness() -> dict[str, object]:
    return {
        "educational_model_selection_ready": True, "test_partition_sealed": True,
        "test_partition_evaluated": False, "model_selected": False,
        "final_model_trained": False, "operational_modeling_ready": False,
        "operational_validity": "unconfirmed", "temporal_contract_status": "resolved_static_snapshot",
        "feature_inference_availability": "unconfirmed",
    }


def _json_write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> dict[str, Path]:
    frame = _frame()
    source_path = Path("data/raw") / SLUG / "dataset.csv"
    absolute_source = tmp_path / source_path
    absolute_source.parent.mkdir(parents=True)
    frame.to_csv(absolute_source, index=False)
    prepared = prepare_tabular_dataset(frame)
    validation_kwargs = dict(
        column_order=tuple(frame.columns), identifier_columns=(), feature_columns=FEATURES,
        target_column=TARGET, target_classes=(), categorical_expected_values={},
        expected_types={column: "numeric" for column in frame.columns},
        problem_type="continuous_regression",
    )
    raw_report = validate_raw_dataset(frame, **validation_kwargs)
    prepared_report = validate_prepared_dataset(
        frame, prepared.dataframe, authorized_changed_columns=(), **validation_kwargs
    )
    policy = ContinuousRegressionSplitPolicy(
        "shuffled_random_snapshot", "educational_benchmark", .70, .15, .15, 42, True
    )
    partitions = split_continuous_regression_dataset(prepared.dataframe, policy=policy)
    partition_report = validate_regression_partitions(
        prepared.dataframe, partitions, identifier_columns=(), target_column=TARGET
    )
    prepared_path = Path("data/processed") / SLUG / "prepared.csv"
    split_root = Path("data/processed") / SLUG / "splits" / "shuffled-70-15-15-seed-42"
    partition_paths = {name: split_root / f"{name}.csv" for name in ("train", "validation", "test")}
    partition_sha = {name: fingerprint_dataframe_csv(partition) for name, partition in partitions.as_mapping().items()}
    readiness = _readiness()
    exploration_path = Path("artifacts/exploration") / SLUG / "exploration-handoff.json"
    exploration = {
        "schema_version": "exploration-handoff.v1", "artifact_type": "exploration_handoff",
        "dataset_slug": SLUG,
        "source": {"dataset_id": 999, "sha256": fingerprint_file(absolute_source)},
        "prediction_contract": {"problem_type": "continuous_regression", "target_column": TARGET,
                                "target_classes": [], "target_semantics": "Continuous / quantitative",
                                "target_unit": "kWh"},
        "feature_contract": {"feature_columns": list(FEATURES), "identifier_columns": []},
    }
    absolute_exploration = tmp_path / exploration_path
    absolute_exploration.parent.mkdir(parents=True)
    _json_write(absolute_exploration, exploration)
    lineage = {
        "path": exploration_path.as_posix(), "schema_version": "exploration-handoff.v1",
        "dataset_slug": SLUG, "sha256": fingerprint_file(absolute_exploration),
        "source_dataset_id": 999, "source_dataset_sha256": fingerprint_file(absolute_source),
    }
    prep = build_preparation_manifest(
        dataset_slug=SLUG, source_path=source_path, source_sha256=fingerprint_file(absolute_source),
        prepared_path=prepared_path, prepared_sha256=fingerprint_dataframe_csv(prepared.dataframe),
        raw_report=raw_report, prepared_report=prepared_report, preparation=prepared,
        raw_fingerprint_before=fingerprint_dataframe(frame), raw_fingerprint_after=fingerprint_dataframe(frame),
        source_sha256_after=fingerprint_file(absolute_source), deterministic_rules=[], readiness=readiness,
        upstream_exploration=lineage,
    )
    feature = build_feature_manifest(
        dataset_slug=SLUG, identifier_columns=(), feature_columns=FEATURES,
        numerical_features=FEATURES, categorical_features=(), categorical_expected_values={},
        target_column=TARGET, target_classes=(), expected_dtypes={},
        preprocessing_contract={"learned_transformations_fitted_in_notebook_02": False},
        prohibited_predictors=(TARGET,), problem_type="continuous_regression",
        target_semantics="Continuous / quantitative", target_unit="kWh",
        prediction_output="Continuous numeric value on the original target scale",
    )
    split = build_split_manifest(
        dataset_slug=SLUG, policy=policy, partitions=partitions, validation=partition_report,
        partition_paths=partition_paths, partition_sha256=partition_sha,
    )
    quality = build_quality_evidence(
        dataset_slug=SLUG, raw_report=raw_report, prepared_report=prepared_report,
        partition_report=partition_report, preparation=prepared,
        fingerprints={"prepared_sha256": fingerprint_dataframe_csv(prepared.dataframe),
                      "partition_sha256": partition_sha}, readiness=readiness,
        preservation_checks={"rows_removed": 0, "values_changed": 0},
    )
    paths = {name: MANIFEST_ROOT / f"{name.replace('_', '-')}.json" for name in (
        "preparation_manifest", "feature_manifest", "split_manifest", "quality_evidence"
    )}
    components = {"preparation_manifest": prep, "feature_manifest": feature,
                  "split_manifest": split, "quality_evidence": quality}
    handoff = build_preparation_handoff_manifest(
        dataset_slug=SLUG, component_paths=paths, component_payloads=components,
        readiness=readiness, upstream_exploration=lineage,
    )
    write_preparation_artifacts(
        project_root=tmp_path,
        csv_artifacts={prepared_path: prepared.dataframe,
                       **{partition_paths[name]: part for name, part in partitions.as_mapping().items()}},
        json_artifacts={**{paths[name]: payload for name, payload in components.items()}, HANDOFF_PATH: handoff},
    )
    return {"root": tmp_path, "handoff": tmp_path / HANDOFF_PATH, "upstream": absolute_exploration,
            "prepared": tmp_path / prepared_path,
            **{name: tmp_path / path for name, path in partition_paths.items()},
            **{name: tmp_path / path for name, path in paths.items()}}


def _load(bundle):
    return load_and_validate_preparation_handoff(project_root=bundle["root"], preparation_handoff_path=HANDOFF_PATH)


def _mutate_component(bundle, name, mutate):
    payload = json.loads(bundle[name].read_text())
    mutate(payload)
    _json_write(bundle[name], payload)
    handoff = json.loads(bundle["handoff"].read_text())
    handoff["components"][name]["sha256"] = fingerprint_file(bundle[name])
    _json_write(bundle["handoff"], handoff)


def _mutate_upstream(bundle, mutate):
    payload = json.loads(bundle["upstream"].read_text())
    mutate(payload)
    _json_write(bundle["upstream"], payload)
    handoff = json.loads(bundle["handoff"].read_text())
    handoff["upstream_exploration"]["sha256"] = fingerprint_file(bundle["upstream"])
    _json_write(bundle["handoff"], handoff)


def test_valid_continuous_handoff_reloads_without_resplit_or_model_fit(bundle):
    loaded = _load(bundle)
    assert len(loaded.prepared) == 40
    assert tuple(map(len, (loaded.train, loaded.validation, loaded.test))) == (28, 6, 6)


@pytest.mark.parametrize("component", ["preparation_manifest", "feature_manifest", "split_manifest", "quality_evidence"])
def test_component_sha_mismatch_fails(bundle, component):
    with bundle[component].open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(HandoffValidationError, match="component fingerprint"):
        _load(bundle)


@pytest.mark.parametrize("name", ["prepared", "train", "validation", "test"])
def test_csv_fingerprint_mismatch_fails(bundle, name):
    with bundle[name].open("a", encoding="utf-8") as stream:
        stream.write("corrupt\n")
    with pytest.raises(HandoffValidationError, match="fingerprint"):
        _load(bundle)


def test_upstream_sha_mismatch_fails(bundle):
    with bundle["upstream"].open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(HandoffValidationError, match="Upstream exploration fingerprint"):
        _load(bundle)


@pytest.mark.parametrize("field,value,message", [
    ("dataset_slug", "other", "dataset_slug"),
    ("schema_version", "exploration-handoff.v0", "schema_version"),
])
def test_upstream_lineage_contract_mutations_fail(bundle, field, value, message):
    _mutate_upstream(bundle, lambda payload: payload.__setitem__(field, value))
    with pytest.raises(HandoffValidationError, match=message):
        _load(bundle)


@pytest.mark.parametrize("contract,key,value,message", [
    ("feature_contract", "feature_columns", ["input_b", "input_a"], "feature order"),
    ("prediction_contract", "target_column", "other", "target column"),
    ("prediction_contract", "target_semantics", "Ordinal", "semantics"),
    ("prediction_contract", "target_unit", "MWh", "unit"),
    ("prediction_contract", "problem_type", "multiclass_classification", "problem_type"),
])
def test_upstream_downstream_contract_mismatch_fails(bundle, contract, key, value, message):
    _mutate_upstream(bundle, lambda payload: payload[contract].__setitem__(key, value))
    with pytest.raises(HandoffValidationError, match=message):
        _load(bundle)


def test_split_problem_type_mismatch_fails(bundle):
    _mutate_component(bundle, "split_manifest", lambda payload: payload.__setitem__("problem_type", "binary_classification"))
    with pytest.raises(HandoffValidationError, match="problem_type"):
        _load(bundle)


def test_dataset_slug_mismatch_across_components_fails(bundle):
    _mutate_component(bundle, "quality_evidence", lambda payload: payload.__setitem__("dataset_slug", "other"))
    with pytest.raises(HandoffValidationError, match="dataset_slug"):
        _load(bundle)


def test_partition_row_count_inconsistency_fails(bundle):
    _mutate_component(bundle, "split_manifest", lambda payload: payload["row_counts"].__setitem__("train", 27))
    with pytest.raises(HandoffValidationError, match="row count"):
        _load(bundle)


def test_partition_membership_overlap_or_coverage_corruption_fails(bundle):
    def corrupt(payload):
        payload["membership"]["validation"][0] = payload["membership"]["train"][0]
    _mutate_component(bundle, "split_manifest", corrupt)
    with pytest.raises((HandoffValidationError, DatasetValidationError, PartitionValidationError)):
        _load(bundle)


@pytest.mark.parametrize("container,key,value", [
    ("readiness", "test_partition_sealed", False),
    ("readiness", "test_partition_evaluated", True),
    ("readiness", "model_selected", True),
    ("readiness", "final_model_trained", True),
])
def test_preparation_stage_readiness_corruption_fails(bundle, container, key, value):
    handoff = json.loads(bundle["handoff"].read_text())
    handoff[container][key] = value
    _json_write(bundle["handoff"], handoff)
    with pytest.raises(HandoffValidationError, match="readiness|sealed|unevaluated|model"):
        _load(bundle)


def test_returns_defensive_copies_and_reload_isolated(bundle):
    first = _load(bundle)
    changed = first.prepared
    changed.loc[0, "input_a"] = -999
    assert first.prepared.loc[0, "input_a"] != -999
    assert _load(bundle).prepared.loc[0, "input_a"] != -999


def test_component_path_escape_is_rejected(bundle):
    handoff = json.loads(bundle["handoff"].read_text())
    handoff["components"]["feature_manifest"]["path"] = "../feature-manifest.json"
    _json_write(bundle["handoff"], handoff)
    with pytest.raises((HandoffValidationError, ValueError)):
        _load(bundle)
