from __future__ import annotations

import copy
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import scripts.select_models as sm
from scripts.prepare_data import _technical_occurrence_membership_keys


CLASSES = ("SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA")
FEATURES = ("f1", "f2", "ShapeFactor2", "derived")
ENCODING = {label: index for index, label in enumerate(CLASSES)}


def _multiclass_frame(rows_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for class_index, label in enumerate(CLASSES):
        for row_index in range(rows_per_class):
            rows.append(
                {
                    "f1": class_index * 2.0 + (row_index % 5) * 0.05,
                    "f2": class_index * -0.7 + (row_index % 7) * 0.03,
                    "ShapeFactor2": class_index * 0.1 + row_index * 0.001,
                    "derived": class_index * 2.0 + (row_index % 5) * 0.05,
                    "Class": label,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _multiclass_frame()
    train = frame.groupby("Class", sort=False).head(20).reset_index(drop=True)
    validation = frame.groupby("Class", sort=False).tail(10).reset_index(drop=True)
    return train, validation


@pytest.fixture()
def roles(partitions) -> sm.MulticlassPartitionRoles:
    train, validation = partitions
    return sm.validate_multiclass_feature_partition_roles(
        train=train,
        validation=validation,
        feature_columns=FEATURES,
        identifier_columns=(),
        target_column="Class",
        target_classes=CLASSES,
        target_encoding=ENCODING,
    )


def _contract(**changes):
    contract = {
        "problem_type": "multiclass_classification",
        "target_semantics": "nominal_unordered",
        "primary_metric": "macro_f1",
        "refit_metric": "macro_f1",
        "cv": {"strategy": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "dummy_macro_f1_margin": 0.02,
        "practical_tie_tolerance": 0.002,
        "decision_rule": "argmax_class_score_or_probability",
        "positive_class": None,
        "binary_threshold": {"status": "not_applicable", "value": None},
        "operational_threshold": {"status": "not_applicable", "value": None},
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "operational_validity": "unconfirmed",
    }
    contract.update(changes)
    return contract


@pytest.mark.parametrize(
    "changes",
    [
        {"problem_type": "binary_classification"},
        {"target_semantics": "ordinal"},
        {"primary_metric": "accuracy"},
        {"refit_metric": "weighted_f1"},
        {"positive_class": "BOMBAY"},
        {"binary_threshold": {"status": "unresolved", "value": None}},
        {"operational_threshold": {"status": "not_applicable", "value": 0.5}},
        {"test_partition_sealed": False},
        {"test_partition_evaluated": True},
    ],
)
def test_multiclass_contract_rejects_binary_or_unsealed_semantics(changes):
    with pytest.raises(sm.ModelSelectionContractError):
        sm.validate_multiclass_model_selection_contract(_contract(**changes))


def test_multiclass_contract_is_defensive_and_uses_macro_f1():
    source = _contract()
    result = sm.validate_multiclass_model_selection_contract(source)
    result["cv"]["random_state"] = 99
    assert source["cv"]["random_state"] == 42
    assert result["primary_metric"] == "macro_f1"
    assert sm.build_multiclass_scoring_contract()["macro_f1"] == "f1_macro"


def test_seven_nominal_classes_and_no_positive_class(roles):
    assert tuple(roles.y_train.unique()) == CLASSES
    assert _contract()["positive_class"] is None
    assert _contract()["binary_threshold"]["status"] == "not_applicable"


def test_role_validation_preserves_readable_labels_and_does_not_mutate(partitions):
    train, validation = partitions
    before = train.copy(deep=True)
    result = sm.validate_multiclass_feature_partition_roles(
        train=train, validation=validation, feature_columns=FEATURES,
        identifier_columns=(), target_column="Class", target_classes=CLASSES,
        target_encoding=ENCODING,
    )
    changed = result.x_train
    changed.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(train, before)
    assert result.x_train.iloc[0, 0] != 999
    assert result.y_train.dtype == train["Class"].dtype


def test_missing_class_in_partition_is_rejected(partitions):
    train, validation = partitions
    validation = validation[validation["Class"] != "BOMBAY"]
    with pytest.raises(sm.FeatureRoleError, match="coverage"):
        sm.validate_multiclass_feature_partition_roles(
            train=train, validation=validation, feature_columns=FEATURES,
            identifier_columns=(), target_column="Class", target_classes=CLASSES,
            target_encoding=ENCODING,
        )


def test_five_fold_stratification_has_complete_class_coverage(roles):
    cv = sm.build_cross_validation(n_splits=5, shuffle=True, random_state=42)
    rows = sm.describe_multiclass_cv_folds(
        cv=cv, x=roles.x_train, y=roles.y_train, target_classes=CLASSES
    )
    assert len(rows) == 5
    assert all(row["all_classes_present"] for row in rows)
    assert all(set(row["validation_class_counts"]) == set(CLASSES) for row in rows)


def test_logistic_scaling_is_fold_safe_and_tree_pipelines_do_not_scale():
    logistic = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="lbfgs"), numerical_features=FEATURES,
        categorical_features=(), scale_numerical=True,
    )
    assert isinstance(logistic, Pipeline)
    assert isinstance(logistic.named_steps["preprocess"].transformers[0][1], StandardScaler)
    assert not hasattr(logistic.named_steps["preprocess"], "transformers_")
    for estimator in (
        DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5),
        HistGradientBoostingClassifier(max_iter=5),
    ):
        pipeline = sm.build_candidate_pipeline(
            estimator=estimator, numerical_features=FEATURES,
            categorical_features=(), scale_numerical=False,
        )
        assert pipeline.named_steps["preprocess"].transformers[0][1] == "passthrough"


def test_critical_selection_apis_accept_neither_validation_nor_test():
    for function in (
        sm.run_multiclass_model_search,
        sm.evaluate_multiclass_feature_policy_cv,
    ):
        names = tuple(inspect.signature(function).parameters)
        assert not any("validation" in name.lower() or "test" in name.lower() for name in names)
    assert "test" not in " ".join(inspect.signature(sm.evaluate_multiclass_candidates_on_validation).parameters).lower()


def test_dummy_logistic_and_tree_candidates_evaluate_multiclass(roles):
    candidates = (
        DummyClassifier(strategy="prior"),
        LogisticRegression(solver="lbfgs", max_iter=500),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1),
        HistGradientBoostingClassifier(max_iter=20, random_state=42),
    )
    for estimator in candidates:
        pipeline = sm.build_candidate_pipeline(
            estimator=estimator, numerical_features=FEATURES,
            categorical_features=(),
            scale_numerical=isinstance(estimator, LogisticRegression),
        )
        pipeline.fit(roles.x_train, roles.y_train)
        result = sm.evaluate_multiclass_classifier(
            estimator=pipeline, x=roles.x_validation,
            y_true=roles.y_validation, target_classes=CLASSES,
        )
        assert len(result["per_class"]) == 7
        assert result["confusion_matrix"]["class_order"] == list(CLASSES)
        assert result["metrics"]["log_loss"] is not None


def test_probability_class_order_is_validated():
    y = pd.Series(CLASSES)
    probabilities = np.eye(7)
    with pytest.raises(sm.ModelSelectionError, match="class order"):
        sm.compute_multiclass_metrics(
            y_true=y, y_pred=y, target_classes=CLASSES,
            probabilities=probabilities, probability_class_order=tuple(reversed(CLASSES)),
        )


def test_class_weight_search_is_deterministic(roles):
    pipeline = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="lbfgs", max_iter=500, random_state=42),
        numerical_features=FEATURES, categorical_features=(), scale_numerical=True,
    )
    kwargs = dict(
        model_id="logistic", family="LogisticRegression", pipeline=pipeline,
        search_strategy="GridSearchCV",
        search_space={"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        x_train=roles.x_train, y_train=roles.y_train,
        scoring=sm.build_multiclass_scoring_contract(),
        cv=sm.build_cross_validation(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
    )
    first = sm.run_multiclass_model_search(**kwargs)
    second = sm.run_multiclass_model_search(**kwargs)
    assert first.best_parameters == second.best_parameters
    assert first.search.best_score_ == pytest.approx(second.search.best_score_)
    summary, table = sm.summarize_multiclass_search_results(first, n_splits=5)
    assert summary["candidate_count"] == 4
    assert len(summary["fold_metrics"]) == 5
    assert "mean_cv_macro_f1" in table


def test_feature_policy_and_shape_factor_2_ablation_use_train_only(roles):
    cv = sm.build_cross_validation(n_splits=5, shuffle=True, random_state=42)
    scoring = sm.build_multiclass_scoring_contract()
    results = {}
    for policy, columns in {
        "all_features": FEATURES,
        "without_shape_factor_2": tuple(c for c in FEATURES if c != "ShapeFactor2"),
        "without_confirmed_derived": tuple(c for c in FEATURES if c != "derived"),
    }.items():
        summary, pipeline = sm.evaluate_multiclass_feature_policy_cv(
            model_id=f"tree__{policy}", family="DecisionTreeClassifier",
            estimator=DecisionTreeClassifier(random_state=42),
            selected_hyperparameters={"model__max_depth": 5},
            feature_policy=policy, feature_columns=columns, scale_features=False,
            x_train=roles.x_train, y_train=roles.y_train, cv=cv, scoring=scoring, n_jobs=1,
        )
        results[policy] = summary
        assert isinstance(pipeline, Pipeline)
        assert not hasattr(pipeline.named_steps["preprocess"], "transformers_")
    assert results["without_shape_factor_2"]["feature_count"] == len(FEATURES) - 1
    assert results["without_confirmed_derived"]["feature_count"] == len(FEATURES) - 1


def _evaluation(metrics):
    return {"metrics": metrics}


def test_candidate_ranking_uses_balanced_accuracy_then_minimum_recall():
    cv = {
        "a": {"family": "RandomForestClassifier", "feature_policy": "all_features", "cv_macro_f1_std": 0.01},
        "b": {"family": "LogisticRegression", "feature_policy": "all_features", "cv_macro_f1_std": 0.02},
    }
    validation = {
        "a": _evaluation({"macro_f1": 0.900, "balanced_accuracy": 0.90, "minimum_per_class_recall": 0.70, "log_loss": 0.2}),
        "b": _evaluation({"macro_f1": 0.899, "balanced_accuracy": 0.91, "minimum_per_class_recall": 0.60, "log_loss": 0.2}),
    }
    result = sm.select_multiclass_candidate_model(
        cv_summaries=cv, validation_evaluations=validation,
        dummy_validation_metrics={"macro_f1": 0.1}, dummy_macro_f1_margin=0.02,
        practical_tie_tolerance=0.002,
        simplicity_order=["LogisticRegression", "RandomForestClassifier"],
    )
    assert result["practical_tie"] is True
    assert result["selected_model_id"] == "b"
    validation["a"]["metrics"]["balanced_accuracy"] = 0.91
    validation["a"]["metrics"]["minimum_per_class_recall"] = 0.75
    result = sm.select_multiclass_candidate_model(
        cv_summaries=cv, validation_evaluations=validation,
        dummy_validation_metrics={"macro_f1": 0.1}, dummy_macro_f1_margin=0.02,
        practical_tie_tolerance=0.002,
        simplicity_order=["LogisticRegression", "RandomForestClassifier"],
    )
    assert result["selected_model_id"] == "a"


def test_candidate_must_exceed_dummy_macro_f1():
    with pytest.raises(sm.NoEligibleCandidateError):
        sm.select_multiclass_candidate_model(
            cv_summaries={"a": {"family": "DecisionTreeClassifier", "cv_macro_f1_std": 0.01}},
            validation_evaluations={"a": _evaluation({
                "macro_f1": 0.11, "balanced_accuracy": 0.2,
                "minimum_per_class_recall": 0.0, "log_loss": 2.0,
            })},
            dummy_validation_metrics={"macro_f1": 0.10},
            dummy_macro_f1_margin=0.02, practical_tie_tolerance=0.002,
            simplicity_order=["DecisionTreeClassifier"],
        )


def test_barbunya_cali_hypothesis_reports_observed_pair_ranking():
    counts = np.eye(7, dtype=int) * 10
    counts[1, 3] = 7
    counts[3, 1] = 6
    normalized = counts / counts.sum(axis=1, keepdims=True)
    evaluation = {"confusion_matrix": {"class_order": list(CLASSES), "counts": counts.tolist(), "row_normalized": normalized.tolist()}}
    result = sm.analyze_overlap_confusion_hypothesis(
        evaluation=evaluation, target_classes=CLASSES, focal_pair=("BARBUNYA", "CALI")
    )
    assert result["status"] == "supported"
    assert result["focal_pair_rank"] == 1


def test_repeated_profile_sensitivity_is_non_destructive(roles):
    x_train = roles.x_train
    x_validation = roles.x_validation
    x_validation.iloc[0] = x_train.iloc[0]
    y_validation = roles.y_validation
    predictions = y_validation.tolist()
    probabilities = np.zeros((len(y_validation), 7))
    for index, label in enumerate(y_validation):
        probabilities[index, CLASSES.index(label)] = 1.0
    before = x_validation.copy(deep=True)
    result = sm.analyze_repeated_profile_sensitivity(
        train_features=x_train, validation_features=x_validation,
        y_validation=y_validation, predictions=predictions,
        target_classes=CLASSES, probabilities=probabilities,
    )
    assert result["repeated_profile_validation_row_count"] >= 1
    assert result["validation_row_count"] == len(x_validation)
    pd.testing.assert_frame_equal(x_validation, before)
    assert "does not prove" in result["interpretation"]


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return sm.sha256_file(path)


def _materialize_preparation_boundary(root: Path) -> None:
    feature_columns = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRatio",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "Roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
    ]
    target_encoding = {label: index for index, label in enumerate(CLASSES)}
    frame = pd.DataFrame(
        [
            {**{column: float(i + offset) for column in feature_columns}, "Class": label}
            for offset, label in enumerate(CLASSES)
            for i in range(3)
        ],
        columns=[*feature_columns, "Class"],
    )
    processed = root / "data/processed/dry-bean"
    split_dir = processed / "splits/stratified-70-15-15-seed-42"
    processed.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = processed / "prepared.csv"
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"
    train = frame.groupby("Class", sort=False).head(1).reset_index(drop=True)
    validation = frame.groupby("Class", sort=False).nth(1).reset_index(drop=True)
    test = frame.groupby("Class", sort=False).nth(2).reset_index(drop=True)
    frame.to_csv(prepared_path, index=False)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    source_membership = _technical_occurrence_membership_keys(frame)
    partition_membership = {
        "train": source_membership[0::3],
        "validation": source_membership[1::3],
        "test": source_membership[2::3],
    }

    artifact_root = root / "artifacts/preparation/dry-bean"
    feature_manifest = {
        "schema_version": "feature-manifest.v2",
        "artifact_type": "feature_manifest",
        "dataset_slug": "dry-bean",
        "feature_columns": feature_columns,
        "numerical_features": feature_columns,
        "categorical_features": [],
        "identifier_columns": [],
        "target_column": "Class",
        "target_classes": list(CLASSES),
        "target_encoding_contract": target_encoding,
        "target_contract": {
            "semantics": "nominal_unordered",
            "ordered_class_contract": list(CLASSES),
        },
        "positive_target_class": None,
        "problem_type": "multiclass_classification",
    }
    split_manifest = {
        "schema_version": "split-manifest.v2",
        "artifact_type": "split_manifest",
        "dataset_slug": "dry-bean",
        "partition_paths": {
            "train": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/train.csv",
            "validation": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/validation.csv",
            "test": "data/processed/dry-bean/splits/stratified-70-15-15-seed-42/test.csv",
        },
        "partition_sha256": {
            "train": sm.sha256_file(train_path),
            "validation": sm.sha256_file(validation_path),
            "test": sm.sha256_file(test_path),
        },
        "row_counts": {"train": 7, "validation": 7, "test": 7},
        "split_method": "synthetic_stratified_snapshot",
        "rounding_method": "synthetic",
        "membership": {
            name: list(values) for name, values in partition_membership.items()
        },
        "membership_kind": "technical_row_occurrence",
        "membership_semantics": "synthetic test fixture",
        "operational_validity": "unconfirmed",
        "operational_modeling_ready": False,
        "educational_model_selection_ready": True,
        "prevalence_tolerance": 1.0,
    }
    preparation_manifest = {
        "schema_version": "preparation-manifest.v1",
        "artifact_type": "preparation_manifest",
        "dataset_slug": "dry-bean",
        "source_path": "data/raw/dry-bean/dataset.csv",
        "prepared_path": "data/processed/dry-bean/prepared.csv",
        "prepared_sha256": sm.sha256_file(prepared_path),
        "source_row_count": 21,
        "prepared_row_count": 21,
        "source_column_count": 17,
        "prepared_column_count": 17,
        "column_order": [*feature_columns, "Class"],
        "readiness": {
            "educational_model_selection_ready": True,
            "test_partition_evaluated": False,
        },
    }
    quality_evidence = {
        "schema_version": "quality-evidence.v1",
        "artifact_type": "preparation_quality_evidence",
        "dataset_slug": "dry-bean",
        "fingerprint_checks": {
            "prepared_sha256": sm.sha256_file(prepared_path),
            "partition_sha256": split_manifest["partition_sha256"],
        },
        "readiness": {
            "educational_model_selection_ready": True,
            "test_partition_evaluated": False,
        },
    }
    component_payloads = {
        "feature_manifest": (
            "artifacts/preparation/dry-bean/feature-manifest.json",
            feature_manifest,
        ),
        "preparation_manifest": (
            "artifacts/preparation/dry-bean/preparation-manifest.json",
            preparation_manifest,
        ),
        "quality_evidence": (
            "artifacts/preparation/dry-bean/quality-evidence.json",
            quality_evidence,
        ),
        "split_manifest": (
            "artifacts/preparation/dry-bean/split-manifest.json",
            split_manifest,
        ),
    }
    components = {}
    for name, (relative, payload) in component_payloads.items():
        components[name] = {
            "path": relative,
            "sha256": _write_json(root / relative, payload),
            "schema_version": payload["schema_version"],
        }
    handoff = {
        "schema_version": "preparation-handoff.v1",
        "artifact_type": "preparation_handoff",
        "dataset_slug": "dry-bean",
        "components": components,
        "readiness": {"educational_model_selection_ready": True, "test_partition_sealed": True},
    }
    _write_json(artifact_root / "preparation-handoff.json", handoff)


def _v2_artifact_set(root: Path) -> dict[str, object]:
    prep_path = root / "artifacts/preparation/dry-bean/preparation-handoff.json"
    prep_handoff = json.loads(prep_path.read_text())
    feature = json.loads((root / prep_handoff["components"]["feature_manifest"]["path"]).read_text())
    split = json.loads((root / prep_handoff["components"]["split_manifest"]["path"]).read_text())
    selected_id = "hist__all_features"
    metrics = {
        "macro_f1": 0.92, "balanced_accuracy": 0.91, "macro_recall": 0.91,
        "weighted_f1": 0.93, "accuracy": 0.93,
        "minimum_per_class_recall": 0.84, "log_loss": 0.22, "row_count": 2042,
    }
    evaluation = {
        "metrics": metrics,
        "per_class": [{"class": label, "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10} for label in feature["target_classes"]],
        "confusion_matrix": {"class_order": feature["target_classes"], "counts": np.eye(7, dtype=int).tolist(), "row_normalized": np.eye(7).tolist()},
        "decision_rule": "argmax_class_score_or_probability",
    }
    readiness = {
        "preparation_handoff_validated": True, "frozen_partitions_respected": True,
        "multiclass_cv_completed": True, "candidate_models_evaluated": True,
        "feature_policy_evaluated": True, "imbalance_policy_frozen": True,
        "selected_candidate_frozen": True, "multiclass_decision_rule_frozen": True,
        "model_selection_handoff_reloadable": True, "test_partition_sealed": True,
        "test_partition_evaluated": False, "final_model_training_ready": True,
        "final_model_trained": False, "model_artifact_materialized": False,
        "model_bundle_materialized": False, "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
    }
    prep_ref = {"path": "artifacts/preparation/dry-bean/preparation-handoff.json", "byte_sha256": sm.sha256_file(prep_path), "schema_version": "preparation-handoff.v1"}
    prep_hashes = {
        **{f"{name}_sha256": ref["sha256"] for name, ref in prep_handoff["components"].items()},
        "prepared_sha256": json.loads((root / prep_handoff["components"]["preparation_manifest"]["path"]).read_text())["prepared_sha256"],
        "train_sha256": split["partition_sha256"]["train"],
        "validation_sha256": split["partition_sha256"]["validation"],
        "test_sha256_integrity_reference_only": split["partition_sha256"]["test"],
    }
    handoff = sm.build_multiclass_model_selection_handoff(
        dataset_slug="dry-bean", preparation_handoff_reference=prep_ref,
        preparation_artifact_hashes=prep_hashes, target_column=feature["target_column"],
        target_classes=feature["target_classes"], target_encoding=feature["target_encoding_contract"],
        available_feature_columns=feature["feature_columns"],
        selected_feature_columns=feature["feature_columns"], selected_feature_policy="all_features",
        selected_model_id=selected_id, selected_model_family="HistGradientBoostingClassifier",
        selected_hyperparameters={"model__max_iter": 150},
        selected_preprocessing_contract={"numerical_scaling": "none"},
        selected_imbalance_policy={"strategy": "none", "class_weight": None, "resampling": "none"},
        cv_contract={"strategy": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42, "fit_partition": "train_only"},
        random_seeds={"cross_validation": 42}, primary_metric="macro_f1",
        secondary_metrics=["balanced_accuracy", "macro_recall"],
        selected_cv_evidence={"model_id": selected_id, "family": "HistGradientBoostingClassifier", "feature_policy": "all_features", "cv_macro_f1_mean": 0.91, "cv_macro_f1_std": 0.01},
        selected_validation_evidence=evaluation, selection_rationale="deterministic",
        tie_break_rationale={"practical_tie": False, "tolerance": 0.002, "order": []},
        analysis_conclusions={"shape_factor_2": {"provenance_status": "unresolved"}},
        final_training_instructions={"fit_partitions": ["train", "validation"], "final_evaluation_partition": "test", "do_not_retune": True},
        readiness=readiness,
    )
    selection = {"selected_model_id": selected_id, "selected_model_family": "HistGradientBoostingClassifier"}
    manifest = sm.build_multiclass_model_selection_manifest(
        dataset_slug="dry-bean", preparation_handoff_reference=prep_ref,
        preparation_artifact_hashes=prep_hashes, model_selection_contract=_contract(),
        candidate_families=[{"model_id": "hist", "family": "HistGradientBoostingClassifier"}],
        feature_policies={"all_features": feature["feature_columns"]},
        cv_contract={"strategy": "StratifiedKFold", "n_splits": 5},
        scoring_contract={"primary_metric": "macro_f1"}, search_contract={"validation_in_search": False},
        random_seeds={"cross_validation": 42},
        artifact_paths={name: f"artifacts/model-selection/dry-bean/{name}" for name in sm.MULTICLASS_ARTIFACT_FILENAMES},
        readiness=readiness, limitations=["educational only"],
    )
    return {
        "model-selection-manifest.json": manifest,
        "candidate-results.json": {"schema_version": "candidate-results.v2", "artifact_type": "candidate_results", "selection": selection},
        "cross-validation-results.csv": pd.DataFrame([{"phase": "family_search", "model_id": selected_id, "family": "HistGradientBoostingClassifier", "feature_policy": "all_features", "search_strategy": "GridSearchCV", "parameters": "{}", "mean_cv_macro_f1": 0.91}]),
        "validation-evidence.json": {"schema_version": "validation-evidence.v2", "artifact_type": "validation_evidence", "dataset_slug": "dry-bean", "models": {selected_id: evaluation}, "selection": selection, "test_partition_evaluated": False},
        "selection-analysis.json": {"schema_version": "selection-analysis.v2", "artifact_type": "selection_analysis", "dataset_slug": "dry-bean", "test_partition_evaluated": False},
        "model-selection-handoff.json": handoff,
    }


def test_v2_atomic_write_loader_and_no_final_model(tmp_path):
    _materialize_preparation_boundary(tmp_path)
    output = tmp_path / "artifacts/model-selection/dry-bean"
    artifacts = _v2_artifact_set(tmp_path)
    result = sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=artifacts)
    assert set(result.created) == set(sm.MULTICLASS_ARTIFACT_FILENAMES)
    assert not (output / "threshold-analysis.json").exists()
    loaded = sm.load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/dry-bean/model-selection-handoff.json",
    )
    assert loaded["schema_version"] == "model-selection-handoff.v2"
    assert loaded["positive_class"] is None
    assert loaded["test_partition_evaluated"] is False
    assert loaded["final_model_trained"] is False
    assert loaded["model_artifact"] is None and loaded["bundle"] is None


def test_v2_semantic_idempotence_and_conflict(tmp_path):
    _materialize_preparation_boundary(tmp_path)
    output = tmp_path / "artifacts/model-selection/dry-bean"
    first = _v2_artifact_set(tmp_path)
    sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=first)
    second = _v2_artifact_set(tmp_path)
    second["candidate-results.json"]["duration_seconds"] = 999
    result = sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=second)
    assert result.idempotent is True
    legacy_text = _v2_artifact_set(tmp_path)
    for filename in ("model-selection-handoff.json", "selection-analysis.json"):
        payload = legacy_text[filename]
        rendered = json.loads(
            json.dumps(payload).replace(
                "Repeated-profile evidence does not prove duplicate identity or leakage",
                "repeated profiles do not prove duplicate identity or leakage",
            )
        )
        legacy_text[filename] = rendered
    legacy_text["model-selection-manifest.json"]["limitations"] = [
        (
            "Repeated-profile analysis is sensitivity evidence and does not prove duplicate identity or leakage."
            if value == "Repeated-profile evidence does not prove duplicate identity or leakage."
            else value
        )
        for value in legacy_text["model-selection-manifest.json"]["limitations"]
    ]
    result = sm.write_multiclass_model_selection_artifacts(
        output_directory=output, artifacts=legacy_text
    )
    assert result.idempotent is True
    stale_manifest = json.loads((output / "model-selection-manifest.json").read_text())
    stale_manifest["artifact_fingerprints"]["model-selection-handoff.json"][
        "semantic_sha256"
    ] = "0" * 64
    stale_manifest["artifact_fingerprints"]["selection-analysis.json"][
        "byte_sha256"
    ] = "1" * 64
    stale_manifest["self_semantic_sha256"] = "2" * 64
    (output / "model-selection-manifest.json").write_text(
        json.dumps(stale_manifest, indent=2, sort_keys=True)
    )
    refreshed = sm.write_multiclass_model_selection_artifacts(
        output_directory=output, artifacts=_v2_artifact_set(tmp_path)
    )
    assert refreshed.idempotent is False
    assert "model-selection-manifest.json" in refreshed.replaced
    divergent = _v2_artifact_set(tmp_path)
    divergent["model-selection-handoff.json"]["selected_feature_policy"] = "different"
    with pytest.raises(sm.ArtifactConflictError, match="divergent"):
        sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=divergent)


def test_v2_partial_set_and_fingerprint_mismatch_fail_closed(tmp_path):
    _materialize_preparation_boundary(tmp_path)
    output = tmp_path / "artifacts/model-selection/dry-bean"
    output.mkdir(parents=True)
    (output / "candidate-results.json").write_text("{}")
    with pytest.raises(sm.ArtifactConflictError, match="Partial"):
        sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=_v2_artifact_set(tmp_path))
    shutil.rmtree(output)
    sm.write_multiclass_model_selection_artifacts(output_directory=output, artifacts=_v2_artifact_set(tmp_path))
    path = output / "selection-analysis.json"
    payload = json.loads(path.read_text())
    payload["changed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(sm.ModelSelectionHandoffError, match="fingerprint mismatch"):
        sm.load_and_validate_model_selection_handoff(
            project_root=tmp_path,
            handoff_path="artifacts/model-selection/dry-bean/model-selection-handoff.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positive_class", "BOMBAY"),
        ("binary_threshold", {"status": "unresolved", "value": None}),
        ("target_classes", list(reversed(CLASSES))),
        ("available_feature_columns", ["wrong"]),
        ("test_partition_evaluated", True),
        ("final_model_trained", True),
        ("model_artifact", "model.joblib"),
    ],
)
def test_v2_loader_rejects_cross_contract_inconsistency(tmp_path, field, value):
    _materialize_preparation_boundary(tmp_path)
    output = tmp_path / "artifacts/model-selection/dry-bean"
    artifacts = _v2_artifact_set(tmp_path)
    artifacts["model-selection-handoff.json"][field] = value
    sm.write_multiclass_model_selection_artifacts(
        output_directory=output, artifacts=artifacts
    )
    with pytest.raises(sm.ModelSelectionHandoffError):
        sm.load_and_validate_model_selection_handoff(
            project_root=tmp_path,
            handoff_path="artifacts/model-selection/dry-bean/model-selection-handoff.json",
        )


def test_v1_binary_handoff_remains_loadable(tmp_path):
    directory = tmp_path / "artifacts/model-selection/binary"
    directory.mkdir(parents=True)
    handoff = {
        "schema_version": "model-selection-handoff.v1",
        "artifact_type": "model_selection_handoff",
        "dataset_slug": "synthetic-binary",
        "target_classes": ["No", "Yes"],
        "target_encoding": {"No": 0, "Yes": 1},
        "positive_class": "Yes",
        "selected_model_id": "hist_gradient_boosting",
        "selected_model_family": "HistGradientBoostingClassifier",
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_model_trained": False,
        "model_artifact": None,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "operational_modeling_ready": False,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
        "readiness": {
            "educational_model_selection_completed": True,
            "educational_final_candidate_selected": True,
            "educational_threshold_selected": True,
            "model_selection_handoff_ready": True,
            "final_model_training_ready": True,
        },
    }
    payloads = {
        "candidate-results.json": {
            "schema_version": "candidate-results.v1",
            "artifact_type": "candidate_results",
            "dataset_slug": "synthetic-binary",
        },
        "cross-validation-results.csv": pd.DataFrame([{"model_id": "hist_gradient_boosting"}]),
        "validation-evidence.json": {
            "schema_version": "validation-evidence.v1",
            "artifact_type": "validation_evidence",
            "dataset_slug": "synthetic-binary",
        },
        "threshold-analysis.json": {
            "schema_version": "threshold-analysis.v1",
            "artifact_type": "threshold_analysis",
            "dataset_slug": "synthetic-binary",
        },
        "model-selection-handoff.json": handoff,
    }
    fingerprints = {}
    for filename, payload in payloads.items():
        path = directory / filename
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
            loaded = pd.read_csv(path)
        else:
            _write_json(path, payload)
            loaded = payload
        fingerprints[filename] = {
            "byte_sha256": sm.sha256_file(path),
            "semantic_sha256": sm._semantic_fingerprint_value(filename, loaded),
        }
    _write_json(
        directory / "model-selection-manifest.json",
        {
            "schema_version": "model-selection-manifest.v1",
            "artifact_type": "model_selection_manifest",
            "dataset_slug": "synthetic-binary",
            "artifact_fingerprints": fingerprints,
        },
    )
    loaded = sm.load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/binary/model-selection-handoff.json",
    )
    assert loaded["schema_version"] == "model-selection-handoff.v1"
    assert loaded["positive_class"] is not None


def test_notebook_03_is_clean_and_contains_no_binary_threshold_workflow():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/03_model_selection_and_evaluation.ipynb").read_text())
    code_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None and cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "threshold-analysis.json" not in code_text
    assert "X_test" not in code_text and "y_test" not in code_text
    assert "preparation.test" not in code_text
    assert "final-pipeline.joblib" not in code_text
    assert "average_precision" not in code_text
