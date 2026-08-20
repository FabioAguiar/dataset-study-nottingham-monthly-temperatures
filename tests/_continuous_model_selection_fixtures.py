from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_data import (
    ContinuousRegressionSplitPolicy,
    build_feature_manifest,
    build_preparation_handoff_manifest,
    build_preparation_manifest,
    build_quality_evidence,
    build_split_manifest,
    fingerprint_dataframe,
    fingerprint_dataframe_csv,
    fingerprint_file,
    prepare_tabular_dataset,
    split_continuous_regression_dataset,
    validate_prepared_dataset,
    validate_raw_dataset,
    validate_regression_partitions,
    write_preparation_artifacts,
)
from scripts.select_models import write_regression_model_selection_artifacts


SLUG = "synthetic-regression"
FEATURES = ("input_a", "input_b")
TARGET = "response"
PREPARATION_HANDOFF = Path("artifacts/preparation") / SLUG / "preparation-handoff.json"
SELECTION_HANDOFF = Path("artifacts/model-selection") / SLUG / "model-selection-handoff.json"


def _readiness():
    return {
        "educational_model_selection_ready": True,
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "model_selected": False,
        "final_model_trained": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "temporal_contract_status": "resolved_static_snapshot",
        "feature_inference_availability": "unconfirmed",
    }


def build_synthetic_preparation(root: Path):
    values = np.arange(40, dtype=float)
    frame = pd.DataFrame({
        "input_a": values + 0.25,
        "input_b": values * 2 + 1,
        "response": values * 0.7 + 3,
    })
    source_path = Path("data/raw") / SLUG / "source.csv"
    absolute_source = root / source_path
    absolute_source.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(absolute_source, index=False)
    prepared = prepare_tabular_dataset(frame)
    validation_args = dict(
        column_order=tuple(frame.columns), identifier_columns=(), feature_columns=FEATURES,
        target_column=TARGET, target_classes=(), categorical_expected_values={},
        expected_types={column: "numeric" for column in frame.columns},
        problem_type="continuous_regression",
    )
    raw_report = validate_raw_dataset(frame, **validation_args)
    prepared_report = validate_prepared_dataset(
        frame, prepared.dataframe, authorized_changed_columns=(), **validation_args
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
    partition_sha = {
        name: fingerprint_dataframe_csv(partition)
        for name, partition in partitions.as_mapping().items()
    }
    exploration_path = Path("artifacts/exploration") / SLUG / "exploration-handoff.json"
    exploration = {
        "schema_version": "exploration-handoff.v1", "artifact_type": "exploration_handoff",
        "dataset_slug": SLUG,
        "source": {"dataset_id": 999, "sha256": fingerprint_file(absolute_source)},
        "prediction_contract": {
            "problem_type": "continuous_regression", "target_column": TARGET,
            "target_classes": [], "target_semantics": "Continuous / quantitative", "target_unit": "kWh",
        },
        "feature_contract": {"feature_columns": list(FEATURES), "identifier_columns": []},
    }
    exploration_absolute = root / exploration_path
    exploration_absolute.parent.mkdir(parents=True, exist_ok=True)
    import json
    exploration_absolute.write_text(json.dumps(exploration, sort_keys=True) + "\n")
    lineage = {
        "path": exploration_path.as_posix(), "schema_version": "exploration-handoff.v1",
        "dataset_slug": SLUG, "sha256": fingerprint_file(exploration_absolute),
        "source_dataset_id": 999, "source_dataset_sha256": fingerprint_file(absolute_source),
    }
    readiness = _readiness()
    preparation_manifest = build_preparation_manifest(
        dataset_slug=SLUG, source_path=source_path, source_sha256=fingerprint_file(absolute_source),
        prepared_path=prepared_path, prepared_sha256=fingerprint_dataframe_csv(prepared.dataframe),
        raw_report=raw_report, prepared_report=prepared_report, preparation=prepared,
        raw_fingerprint_before=fingerprint_dataframe(frame), raw_fingerprint_after=fingerprint_dataframe(frame),
        source_sha256_after=fingerprint_file(absolute_source), deterministic_rules=[],
        readiness=readiness, upstream_exploration=lineage,
    )
    feature_manifest = build_feature_manifest(
        dataset_slug=SLUG, identifier_columns=(), feature_columns=FEATURES,
        numerical_features=FEATURES, categorical_features=(), categorical_expected_values={},
        target_column=TARGET, target_classes=(), expected_dtypes={},
        preprocessing_contract={"learned_transformations_fitted_in_notebook_02": False},
        prohibited_predictors=(TARGET,), problem_type="continuous_regression",
        target_semantics="Continuous / quantitative", target_unit="kWh",
        prediction_output="Continuous numeric value on the original target scale",
    )
    split_manifest = build_split_manifest(
        dataset_slug=SLUG, policy=policy, partitions=partitions, validation=partition_report,
        partition_paths=partition_paths, partition_sha256=partition_sha,
    )
    quality_evidence = build_quality_evidence(
        dataset_slug=SLUG, raw_report=raw_report, prepared_report=prepared_report,
        partition_report=partition_report, preparation=prepared,
        fingerprints={"prepared_sha256": fingerprint_dataframe_csv(prepared.dataframe),
                      "partition_sha256": partition_sha}, readiness=readiness,
        preservation_checks={"rows_removed": 0, "values_changed": 0},
    )
    manifest_root = PREPARATION_HANDOFF.parent
    paths = {
        "preparation_manifest": manifest_root / "preparation-manifest.json",
        "feature_manifest": manifest_root / "feature-manifest.json",
        "split_manifest": manifest_root / "split-manifest.json",
        "quality_evidence": manifest_root / "quality-evidence.json",
    }
    components = {
        "preparation_manifest": preparation_manifest, "feature_manifest": feature_manifest,
        "split_manifest": split_manifest, "quality_evidence": quality_evidence,
    }
    handoff = build_preparation_handoff_manifest(
        dataset_slug=SLUG, component_paths=paths, component_payloads=components,
        readiness=readiness, upstream_exploration=lineage,
    )
    write_preparation_artifacts(
        project_root=root,
        csv_artifacts={prepared_path: prepared.dataframe,
                       **{partition_paths[n]: p for n, p in partitions.as_mapping().items()}},
        json_artifacts={**{paths[n]: p for n, p in components.items()}, PREPARATION_HANDOFF: handoff},
    )
    return {
        "root": root, "handoff_path": PREPARATION_HANDOFF, "handoff": handoff,
        "components": components, "paths": paths, "partitions": partitions,
        "partition_paths": partition_paths, "partition_sha": partition_sha,
        "prepared_path": prepared_path,
    }


def synthetic_selection_artifacts(preparation):
    target = {"column": TARGET, "semantics": "Continuous / quantitative", "unit": "kWh",
              "prediction_output": "continuous numeric value on the original target scale"}
    cv = {"strategy": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42,
          "fit_partition": "train_only"}
    params = {"model__alpha": 0.1}
    metrics = {"mae": 0.1, "rmse": 0.12, "r2": 0.99, "medae": 0.1}
    selection = {
        "selected_model_id": "synthetic_ridge", "selected_model_family": "Ridge",
        "ranking": [], "practical_tie": False, "finalists": ["synthetic_ridge"],
        "criteria_applied": ["validation_mae"], "selection_rationale": "Synthetic deterministic evidence.",
    }
    hashes = {
        "prepared_sha256": preparation["components"]["preparation_manifest"]["prepared_sha256"],
        "train_sha256": preparation["partition_sha"]["train"],
        "validation_sha256": preparation["partition_sha"]["validation"],
        "test_sha256_integrity_reference_only": preparation["partition_sha"]["test"],
        **{f"{name}_sha256": preparation["handoff"]["components"][name]["sha256"]
           for name in ("preparation_manifest", "feature_manifest", "split_manifest", "quality_evidence")},
    }
    prep_ref = {"path": PREPARATION_HANDOFF.as_posix(), "schema_version": "preparation-handoff.v1",
                "sha256": fingerprint_file(preparation["root"] / PREPARATION_HANDOFF)}
    manifest = {
        "schema_version": "model-selection-manifest.v3", "artifact_type": "model_selection_manifest",
        "dataset_slug": SLUG, "problem_type": "continuous_regression", "target_contract": target,
        "feature_contract": {"available_features": list(FEATURES), "selected_features": list(FEATURES),
                             "selected_feature_policy": "all_features"},
        "model_selection_contract": {"primary_metric": "mae", "primary_metric_direction": "lower_is_better",
                                     "primary_metric_unit": "kWh"},
        "cv_contract": cv, "candidate_families": [{"model_id": "synthetic_ridge", "family": "Ridge",
                                                      "search_space": {"model__alpha": [0.1]}}],
        "preparation_handoff_reference": prep_ref, "preparation_artifact_hashes": hashes,
        "test_partition_sealed": True, "test_partition_evaluated": False,
    }
    candidates = {
        "schema_version": "candidate-results.v3", "artifact_type": "candidate_results",
        "dataset_slug": SLUG, "problem_type": "continuous_regression",
        "primary_metric_contract": {"name": "mae", "direction": "lower_is_better", "unit": "kWh"},
        "family_searches": [{"model_id": "synthetic_ridge", "family": "Ridge",
                             "selected_hyperparameters": params}], "selection": selection,
        "test_partition_evaluated": False,
    }
    validation = {
        "schema_version": "validation-evidence.v3", "artifact_type": "validation_evidence",
        "dataset_slug": SLUG, "problem_type": "continuous_regression", "target_contract": target,
        "primary_metric": {"name": "mae", "direction": "lower_is_better", "unit": "kWh"},
        "models": {"synthetic_ridge": metrics}, "test_partition_evaluated": False,
    }
    analysis = {
        "schema_version": "selection-analysis.v3", "artifact_type": "selection_analysis",
        "dataset_slug": SLUG, "problem_type": "continuous_regression",
        "selection_rationale": selection["selection_rationale"], "test_partition_evaluated": False,
    }
    ready_true = ("preparation_handoff_validated", "frozen_partitions_respected", "regression_cv_completed",
                  "candidate_models_evaluated", "feature_policy_frozen", "selected_candidate_frozen",
                  "regression_metric_contract_frozen", "model_selection_handoff_reloadable",
                  "test_partition_sealed", "final_model_training_ready")
    readiness = {name: True for name in ready_true}
    readiness.update({name: False for name in ("test_partition_evaluated", "final_model_trained",
                                               "model_artifact_materialized", "model_bundle_materialized",
                                               "operational_modeling_ready")})
    handoff = {
        "schema_version": "model-selection-handoff.v3", "artifact_type": "model_selection_handoff",
        "dataset_slug": SLUG, "problem_type": "continuous_regression", "target_contract": target,
        "available_feature_columns": list(FEATURES), "selected_feature_columns": list(FEATURES),
        "selected_feature_policy": "all_features", "primary_metric": "mae",
        "primary_metric_direction": "lower_is_better", "cv_contract": cv,
        "selected_model_id": "synthetic_ridge", "selected_model_family": "Ridge",
        "selected_hyperparameters": params, "preparation_handoff_reference": prep_ref,
        "preparation_artifact_hashes": hashes, "test_partition_sealed": True,
        "test_partition_evaluated": False, "final_model_training_ready": True,
        "final_model_trained": False, "model_artifact": None, "model_artifact_materialized": False,
        "bundle": None, "model_bundle_materialized": False, "readiness": readiness,
        "final_training_instructions": {
            "notebook": "notebooks/04_final_model_and_bundle.ipynb",
            "reconstruct_pipeline_from_contract": True, "fit_partitions": ["train", "validation"],
            "final_evaluation_partition": "test", "access_test_only_after_contract_freeze_and_final_fit": True,
            "evaluate_test_once": True, "do_not_retune": True, "do_not_change_feature_policy": True,
            "do_not_change_hyperparameters": True, "do_not_change_preprocessing": True,
            "target_scale": "original kWh scale", "prediction_type": "continuous_numeric",
        },
    }
    cv_results = pd.DataFrame([{
        "phase": "family_search", "model_id": "synthetic_ridge", "family": "Ridge",
        "candidate_index": 0, "parameters": '{"model__alpha":0.1}', "rank_mae": 1,
        "mean_cv_mae": .1, "std_cv_mae": .01, "mean_cv_rmse": .12, "std_cv_rmse": .01,
        "mean_cv_r2": .99, "std_cv_r2": .01, "mean_cv_medae": .1, "std_cv_medae": .01,
    }])
    return {"model-selection-manifest.json": manifest, "candidate-results.json": candidates,
            "cross-validation-results.csv": cv_results, "validation-evidence.json": validation,
            "selection-analysis.json": analysis, "model-selection-handoff.json": handoff}


def build_synthetic_project(root: Path):
    preparation = build_synthetic_preparation(root)
    artifacts = synthetic_selection_artifacts(preparation)
    write_regression_model_selection_artifacts(
        output_directory=root / SELECTION_HANDOFF.parent, artifacts=artifacts
    )
    return preparation, artifacts
