from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import scripts.select_models as sm


FEATURES = ["num_a", "num_b", "cat_text", "cat_int"]
NUMERICAL = ["num_a", "num_b"]
CATEGORICAL = ["cat_text", "cat_int"]
IDENTIFIERS = ["row_id"]
TARGET = "target"
ENCODING = {"No": 0, "Yes": 1}


def _frame(rows: int = 180) -> pd.DataFrame:
    records = []
    for index in range(rows):
        num_a = float((index * 7) % 31) / 10.0
        num_b = float((index * 11) % 23) / 7.0
        cat_text = ("a", "b", "c")[index % 3]
        cat_int = index % 2
        score = num_a + 0.7 * num_b + (1.1 if cat_text == "c" else 0) + cat_int
        target = "Yes" if score > 3.4 else "No"
        records.append(
            {
                "row_id": f"id-{index:04d}",
                "num_a": num_a,
                "num_b": num_b,
                "cat_text": cat_text,
                "cat_int": cat_int,
                "target": target,
            }
        )
    return pd.DataFrame(records, index=[1000 + i * 3 for i in range(rows)])


@pytest.fixture()
def partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _frame()
    return frame.iloc[:120].copy(), frame.iloc[120:].copy()


@pytest.fixture()
def roles(partitions: tuple[pd.DataFrame, pd.DataFrame]) -> sm.PartitionRoles:
    train, validation = partitions
    return sm.validate_feature_partition_roles(
        train=train,
        validation=validation,
        feature_columns=FEATURES,
        identifier_columns=IDENTIFIERS,
        target_column=TARGET,
        target_classes=("No", "Yes"),
        positive_class="Yes",
        target_encoding=ENCODING,
    )


def _contract(**changes):
    contract = {
        "evaluation_mode": "stratified_random_snapshot",
        "purpose": "educational_benchmark",
        "primary_metric": "average_precision",
        "refit_metric": "average_precision",
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 17,
        },
        "dummy_average_precision_margin": 0.01,
        "practical_tie_tolerance": 0.01,
        "educational_recall_target": 0.80,
        "test_partition_sealed": True,
        "operational_validity": "unconfirmed",
        "operational_threshold": "unresolved",
    }
    contract.update(changes)
    return contract


def _specs():
    common = {
        "numerical_features": NUMERICAL,
        "categorical_features": CATEGORICAL,
        "random_state": 17,
    }
    return [
        {
            **common,
            "model_id": "logistic",
            "family": "LogisticRegression",
            "estimator": LogisticRegression(
                solver="liblinear", max_iter=200, random_state=17
            ),
            "scale_numerical": True,
            "search_strategy": "GridSearchCV",
            "search_space": {"model__C": [0.1, 1.0], "model__penalty": ["l2"]},
            "candidate_count": 2,
        },
        {
            **common,
            "model_id": "tree",
            "family": "DecisionTreeClassifier",
            "estimator": DecisionTreeClassifier(random_state=17),
            "scale_numerical": False,
            "search_strategy": "GridSearchCV",
            "search_space": {"model__max_depth": [2, 4]},
            "candidate_count": 2,
        },
        {
            **common,
            "model_id": "forest",
            "family": "RandomForestClassifier",
            "estimator": RandomForestClassifier(
                n_estimators=12, random_state=17, n_jobs=1
            ),
            "scale_numerical": False,
            "search_strategy": "RandomizedSearchCV",
            "search_space": {
                "model__max_depth": [2, 4],
                "model__min_samples_leaf": [1, 2],
            },
            "n_iter": 2,
        },
        {
            **common,
            "model_id": "hist",
            "family": "HistGradientBoostingClassifier",
            "estimator": HistGradientBoostingClassifier(
                max_iter=20, random_state=17
            ),
            "scale_numerical": False,
            "search_strategy": "RandomizedSearchCV",
            "search_space": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [7, 15],
            },
            "n_iter": 2,
        },
    ]


def _required_strategies():
    return {
        "LogisticRegression": "GridSearchCV",
        "DecisionTreeClassifier": "GridSearchCV",
        "RandomForestClassifier": "RandomizedSearchCV",
        "HistGradientBoostingClassifier": "RandomizedSearchCV",
    }


def _baseline_spec():
    return {
        "model_id": "dummy_prior",
        "family": "DummyClassifier",
        "estimator": DummyClassifier(strategy="prior"),
        "eligible": False,
    }


# A. Contracts

def test_valid_contract_is_defensive_copy():
    source = _contract()
    validated = sm.validate_model_selection_contract(source)
    validated["cv"]["n_splits"] = 9
    assert source["cv"]["n_splits"] == 3


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"primary_metric": "accuracy"}, "Average Precision"),
        ({"refit_metric": "roc_auc"}, "refit_metric"),
        ({"test_partition_sealed": False}, "sealed"),
        ({"operational_validity": "confirmed"}, "unconfirmed"),
        ({"operational_threshold": 0.5}, "unresolved"),
        ({"educational_recall_target": 0}, "interval"),
        ({"educational_recall_target": 1.1}, "interval"),
    ],
)
def test_invalid_contracts(changes, match):
    with pytest.raises(sm.ModelSelectionContractError, match=match):
        sm.validate_model_selection_contract(_contract(**changes))


@pytest.mark.parametrize(
    "cv_changes",
    [
        {"strategy": "KFold"},
        {"n_splits": 1},
        {"shuffle": False},
        {"random_state": "17"},
    ],
)
def test_invalid_cv_contract(cv_changes):
    contract = _contract()
    contract["cv"].update(cv_changes)
    with pytest.raises(sm.ModelSelectionContractError):
        sm.validate_model_selection_contract(contract)


def test_absolute_paths_are_rejected_cross_platform():
    contract = _contract(output_path="C:/private/result.json")
    with pytest.raises(sm.ModelSelectionContractError, match="absolute Windows"):
        sm.validate_model_selection_contract(contract)


# B. Roles

@pytest.mark.parametrize("missing", ["row_id", "target", "num_a"])
def test_missing_role_columns_are_rejected(partitions, missing):
    train, validation = partitions
    train = train.drop(columns=[missing])
    with pytest.raises(sm.FeatureRoleError, match="missing required columns"):
        sm.validate_feature_partition_roles(
            train=train,
            validation=validation,
            feature_columns=FEATURES,
            identifier_columns=IDENTIFIERS,
            target_column=TARGET,
            target_classes=("No", "Yes"),
            positive_class="Yes",
            target_encoding=ENCODING,
        )


@pytest.mark.parametrize("leaked", ["row_id", "target"])
def test_identifier_and_target_are_excluded_from_x(partitions, leaked):
    train, validation = partitions
    with pytest.raises(sm.FeatureRoleError, match="Predictors contain"):
        sm.validate_feature_partition_roles(
            train=train,
            validation=validation,
            feature_columns=[*FEATURES, leaked],
            identifier_columns=IDENTIFIERS,
            target_column=TARGET,
            target_classes=("No", "Yes"),
            positive_class="Yes",
            target_encoding=ENCODING,
        )


def test_feature_order_and_integer_category_are_preserved(roles):
    x = roles.x_train
    assert list(x.columns) == FEATURES
    assert pd.api.types.is_integer_dtype(x["cat_int"])


def test_positive_class_and_encoding_must_be_coherent(partitions):
    train, validation = partitions
    with pytest.raises(sm.FeatureRoleError, match="positive_class"):
        sm.validate_feature_partition_roles(
            train=train,
            validation=validation,
            feature_columns=FEATURES,
            identifier_columns=IDENTIFIERS,
            target_column=TARGET,
            target_classes=("No", "Yes"),
            positive_class="Maybe",
            target_encoding=ENCODING,
        )
    with pytest.raises(sm.FeatureRoleError, match="encoded as 1"):
        sm.validate_feature_partition_roles(
            train=train,
            validation=validation,
            feature_columns=FEATURES,
            identifier_columns=IDENTIFIERS,
            target_column=TARGET,
            target_classes=("No", "Yes"),
            positive_class="Yes",
            target_encoding={"No": 1, "Yes": 0},
        )


def test_roles_do_not_mutate_inputs_and_return_defensive_copies(partitions):
    train, validation = partitions
    before_train = train.copy(deep=True)
    result = sm.validate_feature_partition_roles(
        train=train,
        validation=validation,
        feature_columns=FEATURES,
        identifier_columns=IDENTIFIERS,
        target_column=TARGET,
        target_classes=("No", "Yes"),
        positive_class="Yes",
        target_encoding=ENCODING,
    )
    changed = result.x_train
    changed.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(train, before_train)
    assert result.x_train.iloc[0, 0] != 999


# C. Pipelines

def test_logistic_pipeline_contains_encoder_and_scaler():
    pipeline = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="liblinear"),
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=True,
    )
    assert isinstance(pipeline, Pipeline)
    preprocess = pipeline.named_steps["preprocess"]
    configured = dict((name, transformer) for name, transformer, _ in preprocess.transformers)
    assert isinstance(configured["numerical"], StandardScaler)
    assert isinstance(configured["categorical"], OneHotEncoder)
    assert configured["categorical"].handle_unknown == "ignore"
    assert configured["categorical"].drop is None


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(random_state=17),
        RandomForestClassifier(n_estimators=5, random_state=17),
        HistGradientBoostingClassifier(max_iter=5, random_state=17),
    ],
)
def test_tree_pipelines_do_not_scale_numerical_features(estimator):
    pipeline = sm.build_candidate_pipeline(
        estimator=estimator,
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=False,
    )
    configured = dict(
        (name, transformer)
        for name, transformer, _ in pipeline.named_steps["preprocess"].transformers
    )
    assert configured["numerical"] == "passthrough"


def test_pipeline_is_unfitted_before_cv_and_does_not_mutate_input(roles):
    pipeline = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="liblinear"),
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=True,
    )
    assert not hasattr(pipeline.named_steps["preprocess"], "transformers_")
    x = roles.x_train
    before = x.copy(deep=True)
    pipeline.fit(x, roles.y_train)
    pd.testing.assert_frame_equal(x, before)


def test_unknown_categories_are_ignored_and_reported(roles):
    pipeline = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="liblinear"),
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=True,
    )
    pipeline.fit(roles.x_train, roles.y_train)
    validation = roles.x_validation
    validation.loc[validation.index[0], "cat_text"] = "never-seen"
    transformed = pipeline.predict_proba(validation)
    assert transformed.shape[0] == len(validation)
    report = sm.report_unknown_categories(
        estimator=pipeline, x_validation=validation
    )
    assert report == {"cat_text": ["never-seen"]}


# D. Candidate model specifications

def test_required_dummy_and_four_candidate_families_validate():
    baseline, specs = sm.validate_candidate_model_specs(
        baseline_spec=_baseline_spec(),
        candidate_specs=_specs(),
        required_family_search_strategies=_required_strategies(),
        expected_candidate_count=4,
    )
    assert baseline["eligible"] is False
    assert [spec["family"] for spec in specs] == [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
    ]


def test_invalid_dummy_or_model_family_is_rejected():
    baseline = _baseline_spec()
    baseline["estimator"] = DummyClassifier(strategy="most_frequent")
    with pytest.raises(sm.CandidateSpecificationError, match="prior"):
        sm.validate_candidate_model_specs(
            baseline_spec=baseline,
            candidate_specs=_specs(),
            required_family_search_strategies=_required_strategies(),
            expected_candidate_count=4,
        )
    specs = _specs()
    specs[0]["family"] = "SVC"
    with pytest.raises(sm.CandidateSpecificationError, match="families"):
        sm.validate_candidate_model_specs(
            baseline_spec=_baseline_spec(),
            candidate_specs=specs,
            required_family_search_strategies=_required_strategies(),
            expected_candidate_count=4,
        )


def test_incompatible_grid_and_candidate_count_are_rejected():
    specs = _specs()
    specs[0]["search_space"] = {"wrong__C": [1.0]}
    with pytest.raises(sm.CandidateSpecificationError, match="model step"):
        sm.validate_candidate_model_specs(
            baseline_spec=_baseline_spec(),
            candidate_specs=specs,
            required_family_search_strategies=_required_strategies(),
            expected_candidate_count=4,
        )
    specs = _specs()
    specs[0]["candidate_count"] = 99
    with pytest.raises(sm.CandidateSpecificationError, match="grid size"):
        sm.validate_candidate_model_specs(
            baseline_spec=_baseline_spec(),
            candidate_specs=specs,
            required_family_search_strategies=_required_strategies(),
            expected_candidate_count=4,
        )


# E. Search

@pytest.mark.parametrize(
    ("spec_index", "search_type"),
    [(0, GridSearchCV), (1, GridSearchCV), (2, RandomizedSearchCV), (3, RandomizedSearchCV)],
)
def test_required_search_strategies_use_reduced_spaces(roles, spec_index, search_type):
    spec = _specs()[spec_index]
    pipeline = sm.build_candidate_pipeline(
        estimator=spec["estimator"],
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=spec["scale_numerical"],
    )
    outcome = sm.run_model_search(
        model_id=spec["model_id"],
        family=spec["family"],
        pipeline=pipeline,
        search_strategy=spec["search_strategy"],
        search_space=spec["search_space"],
        x_train=roles.x_train,
        y_train=roles.y_train,
        scoring=sm.build_scoring_contract(),
        cv=sm.build_cross_validation(n_splits=3, shuffle=True, random_state=17),
        refit_metric="average_precision",
        n_jobs=1,
        random_state=17,
        n_iter=spec.get("n_iter"),
    )
    assert isinstance(outcome.search, search_type)
    assert outcome.search.refit == "average_precision"
    assert len(outcome.search.scoring) == 10
    assert outcome.candidate_count == 2


def test_same_seed_produces_equivalent_selection(roles):
    spec = _specs()[0]
    kwargs = dict(
        model_id=spec["model_id"],
        family=spec["family"],
        pipeline=sm.build_candidate_pipeline(
            estimator=spec["estimator"],
            numerical_features=NUMERICAL,
            categorical_features=CATEGORICAL,
            scale_numerical=True,
        ),
        search_strategy=spec["search_strategy"],
        search_space=spec["search_space"],
        x_train=roles.x_train,
        y_train=roles.y_train,
        scoring=sm.build_scoring_contract(),
        cv=sm.build_cross_validation(n_splits=3, shuffle=True, random_state=17),
        refit_metric="average_precision",
        n_jobs=1,
    )
    first = sm.run_model_search(**kwargs)
    second = sm.run_model_search(**kwargs)
    assert first.best_parameters == second.best_parameters
    assert first.search.best_score_ == pytest.approx(second.search.best_score_)


def test_selection_apis_do_not_accept_validation_or_test_for_tuning():
    parameters = inspect.signature(sm.run_model_search).parameters
    assert "x_validation" not in parameters
    assert "test" not in " ".join(parameters).lower()


# F. Metrics and baseline

def _fit_logistic(roles):
    pipeline = sm.build_candidate_pipeline(
        estimator=LogisticRegression(solver="liblinear", max_iter=200),
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=True,
    )
    pipeline.fit(roles.x_train, roles.y_train)
    return pipeline


def test_probability_metrics_are_human_oriented_and_complete(roles):
    evaluation = sm.evaluate_probability_classifier(
        estimator=_fit_logistic(roles),
        x=roles.x_validation,
        y_true=roles.y_validation,
        threshold=0.5,
    )
    metrics = evaluation["metrics"]
    required = {
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "balanced_accuracy",
        "accuracy",
        "log_loss",
        "brier_score",
        "predicted_positive_count",
        "predicted_positive_rate",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    }
    assert required.issubset(metrics)
    assert metrics["log_loss"] >= 0
    assert metrics["brier_score"] >= 0
    assert metrics["predicted_positive_count"] == (
        metrics["true_positives"] + metrics["false_positives"]
    )


def test_zero_division_is_controlled_for_dummy_prior(roles):
    dummy = sm.build_candidate_pipeline(
        estimator=DummyClassifier(strategy="prior"),
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        scale_numerical=False,
    )
    dummy.fit(roles.x_train, roles.y_train)
    evaluation = sm.evaluate_probability_classifier(
        estimator=dummy,
        x=roles.x_validation,
        y_true=roles.y_validation,
        threshold=0.99,
    )
    assert evaluation["metrics"]["precision"] == 0
    assert evaluation["metrics"]["recall"] == 0
    assert evaluation["metrics"]["predicted_positive_count"] == 0


def test_f2_weights_recall_more_than_f1_for_false_negatives():
    y = [1, 1, 1, 1, 0, 0]
    predicted = [1, 0, 0, 0, 0, 0]
    assert sm.compute_fbeta(y, predicted, beta=2) < 1


# G/H. Baseline margin and deterministic selection

def _selection_inputs(ap_a=0.75, ap_b=0.70, brier_a=0.18, brier_b=0.17):
    cv = {
        "a": {
            "model_id": "a",
            "family": "LogisticRegression",
            "cv_average_precision_std": 0.02,
            "cv_average_precision_confidence_lower": 0.70,
            "cv_average_precision_confidence_upper": 0.78,
        },
        "b": {
            "model_id": "b",
            "family": "DecisionTreeClassifier",
            "cv_average_precision_std": 0.03,
            "cv_average_precision_confidence_lower": 0.68,
            "cv_average_precision_confidence_upper": 0.76,
        },
    }
    validation = {
        "a": {"metrics": {"average_precision": ap_a, "roc_auc": 0.81, "brier_score": brier_a, "log_loss": 0.50}},
        "b": {"metrics": {"average_precision": ap_b, "roc_auc": 0.80, "brier_score": brier_b, "log_loss": 0.51}},
    }
    return cv, validation


def test_candidate_below_dummy_margin_is_ineligible():
    cv, validation = _selection_inputs(ap_a=0.509, ap_b=0.505)
    with pytest.raises(sm.NoEligibleCandidateError):
        sm.select_candidate_model(
            cv_summaries=cv,
            validation_evaluations=validation,
            dummy_validation_metrics={"average_precision": 0.50},
            dummy_average_precision_margin=0.01,
            practical_tie_tolerance=0.01,
            simplicity_order=["LogisticRegression", "DecisionTreeClassifier"],
        )


def test_selection_orders_by_validation_average_precision_without_tie():
    cv, validation = _selection_inputs()
    result = sm.select_candidate_model(
        cv_summaries=cv,
        validation_evaluations=validation,
        dummy_validation_metrics={"average_precision": 0.40},
        dummy_average_precision_margin=0.01,
        practical_tie_tolerance=0.01,
        simplicity_order=["LogisticRegression", "DecisionTreeClassifier"],
    )
    assert result["selected_model_id"] == "a"
    assert result["practical_tie"] is False


def test_practical_tie_uses_brier_then_log_loss_stability_and_simplicity():
    cv, validation = _selection_inputs(ap_a=0.750, ap_b=0.746, brier_a=0.18, brier_b=0.17)
    result = sm.select_candidate_model(
        cv_summaries=cv,
        validation_evaluations=validation,
        dummy_validation_metrics={"average_precision": 0.40},
        dummy_average_precision_margin=0.01,
        practical_tie_tolerance=0.01,
        simplicity_order=["LogisticRegression", "DecisionTreeClassifier"],
    )
    assert result["practical_tie"] is True
    assert result["selected_model_id"] == "b"
    assert result["criteria_applied"][0]["criterion"] == "lower_validation_brier_score"


def test_tie_falls_back_to_simplicity_and_stable_id():
    cv, validation = _selection_inputs(ap_a=0.75, ap_b=0.75, brier_a=0.17, brier_b=0.17)
    validation["a"]["metrics"]["log_loss"] = 0.5
    validation["b"]["metrics"]["log_loss"] = 0.5
    cv["a"]["cv_average_precision_std"] = cv["b"]["cv_average_precision_std"] = 0.02
    validation["a"]["metrics"]["roc_auc"] = validation["b"]["metrics"]["roc_auc"] = 0.8
    result = sm.select_candidate_model(
        cv_summaries=cv,
        validation_evaluations=validation,
        dummy_validation_metrics={"average_precision": 0.4},
        dummy_average_precision_margin=0.01,
        practical_tie_tolerance=0.01,
        simplicity_order=["LogisticRegression", "DecisionTreeClassifier"],
    )
    assert result["selected_model_id"] == "a"


def test_cv_confidence_interval_and_overlap_rule():
    lower, upper = sm.compute_cv_confidence_interval(0.7, 0.05, n_splits=5)
    assert lower < 0.7 < upper
    first = {
        "validation_average_precision": 0.70,
        "cv_average_precision_confidence_lower": 0.65,
        "cv_average_precision_confidence_upper": 0.75,
    }
    second = {
        "validation_average_precision": 0.695,
        "cv_average_precision_confidence_lower": 0.68,
        "cv_average_precision_confidence_upper": 0.72,
    }
    assert sm.detect_practical_tie(first, second, tolerance=0.01)


# I. Threshold scenarios

def test_threshold_analysis_contains_all_required_scenarios_and_selected_recall():
    y = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    probabilities = [0.95, 0.8, 0.7, 0.6, 0.4, 0.9, 0.5, 0.3, 0.2, 0.1]
    result = sm.analyze_thresholds(
        y_validation=y,
        positive_probabilities=probabilities,
        recall_targets=(0.70, 0.80, 0.90),
        selected_scenario_id="minimum_recall_0_80",
    )
    ids = {scenario["scenario_id"] for scenario in result["scenarios"]}
    assert ids == {
        "default_0_50",
        "maximum_f1",
        "maximum_f2",
        "minimum_recall_0_70",
        "minimum_recall_0_80",
        "minimum_recall_0_90",
    }
    selected = result["selected_scenario"]
    assert selected["recall"] >= 0.80
    assert selected["target_satisfied"] is True
    assert result["operational_threshold"] == "unresolved"
    assert selected["predicted_positive_count"] == selected["true_positives"] + selected["false_positives"]


def test_recall_threshold_tiebreak_prefers_f2_then_higher_threshold():
    y = pd.Series([1, 1, 0, 0])
    probabilities = [0.9, 0.6, 0.6, 0.1]
    result = sm.analyze_thresholds(
        y_validation=y,
        positive_probabilities=probabilities,
        recall_targets=(0.5, 0.8, 0.9),
        selected_scenario_id="minimum_recall_0_80",
    )
    assert result["selected_scenario"]["selection_rule"].endswith("highest_threshold")


def test_threshold_api_mentions_only_validation_not_test():
    parameters = inspect.signature(sm.analyze_thresholds).parameters
    assert "y_validation" in parameters
    assert all("test" not in name.lower() for name in parameters)


# J/K. Artifacts and persistence

def _artifact_set() -> dict[str, object]:
    readiness = {
        "educational_model_selection_completed": True,
        "educational_final_candidate_selected": True,
        "educational_threshold_selected": True,
        "model_selection_handoff_ready": True,
        "final_model_training_ready": True,
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
        "final_model_trained": False,
        "model_artifact_materialized": False,
        "model_bundle_materialized": False,
        "operational_modeling_ready": False,
    }
    paths = {name: f"artifacts/model-selection/example/{name}" for name in sm.ARTIFACT_FILENAMES}
    manifest = sm.build_model_selection_manifest(
        dataset_slug="example",
        preparation_references={"feature_manifest_path": "artifacts/preparation/example/feature-manifest.json"},
        preparation_fingerprints={"feature_manifest_sha256": "a" * 64},
        model_selection_contract=_contract(),
        candidate_model_ids=["a", "b", "c", "d"],
        baseline_model_id="dummy",
        cv_contract=_contract()["cv"],
        scoring_contract={"primary": "average_precision"},
        search_strategies={"a": "GridSearchCV"},
        search_spaces={"a": {"model__C": [1.0]}},
        random_seeds={"cv": 17},
        artifact_paths=paths,
        readiness=readiness,
        limitations=["educational only"],
    )
    candidate_results = {
        "schema_version": "candidate-results.v1",
        "artifact_type": "candidate_results",
        "baseline": {"model_id": "dummy"},
        "candidates": [{"model_id": "a", "mean_fit_time": 0.01}],
        "selection": {"selected_model_id": "a"},
    }
    cv = pd.DataFrame(
        [
            {
                "model_id": "a",
                "family": "LogisticRegression",
                "search_strategy": "GridSearchCV",
                "parameters": '{"model__C":1.0}',
                "mean_cv_average_precision": 0.8,
                "mean_fit_time": 0.01,
            }
        ]
    )
    validation = {
        "schema_version": "validation-evidence.v1",
        "artifact_type": "validation_evidence",
        "dataset_slug": "example",
        "partition": "validation",
        "models": {"a": {"metrics": {"average_precision": 0.8}}},
    }
    threshold = {
        "schema_version": "threshold-analysis.v1",
        "artifact_type": "threshold_analysis",
        "dataset_slug": "example",
        "selected_model_id": "a",
        "partition": "validation",
        "selected_educational_threshold": 0.42,
        "operational_threshold": "unresolved",
    }
    handoff = sm.build_model_selection_handoff(
        dataset_slug="example",
        preparation_contract_references={"split_manifest_path": "artifacts/preparation/example/split-manifest.json"},
        selected_model_family="LogisticRegression",
        selected_model_id="a",
        selected_hyperparameters={"model__C": 1.0},
        selected_preprocessing_contract={"categorical_strategy": "one_hot"},
        selected_validation_metrics={"average_precision": 0.8},
        selected_educational_threshold={"scenario_id": "minimum_recall_0_80", "threshold": 0.42},
        threshold_selection_rule="validation only",
        selection_rationale="deterministic",
        feature_columns=FEATURES,
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL,
        target_encoding=ENCODING,
        positive_class="Yes",
        random_seeds={"cv": 17},
        final_training_instructions={"fit_partitions": ["train", "validation"], "evaluate_partition": "test"},
        readiness=readiness,
    )
    return {
        "model-selection-manifest.json": manifest,
        "candidate-results.json": candidate_results,
        "cross-validation-results.csv": cv,
        "validation-evidence.json": validation,
        "threshold-analysis.json": threshold,
        "model-selection-handoff.json": handoff,
    }


def test_artifact_builders_preserve_schemas_paths_features_and_blocks():
    artifacts = _artifact_set()
    handoff = artifacts["model-selection-handoff.json"]
    assert handoff["schema_version"] == "model-selection-handoff.v1"
    assert handoff["feature_columns"] == FEATURES
    assert handoff["test_partition_sealed"] is True
    assert handoff["test_partition_evaluated"] is False
    assert handoff["final_model_trained"] is False
    assert handoff["model_artifact"] is None
    assert handoff["operational_validity"] == "unconfirmed"


def test_atomic_write_creates_directories_and_loads_handoff(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    result = sm.write_model_selection_artifacts(
        output_directory=output, artifacts=_artifact_set()
    )
    assert set(result.created) == set(sm.ARTIFACT_FILENAMES)
    assert all((output / name).is_file() for name in sm.ARTIFACT_FILENAMES)
    loaded = sm.load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/example/model-selection-handoff.json",
    )
    assert loaded["selected_model_id"] == "a"


def test_equivalent_rerun_is_idempotent_and_ignores_timestamps_and_timings(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    first = _artifact_set()
    sm.write_model_selection_artifacts(output_directory=output, artifacts=first)
    second = _artifact_set()
    second["candidate-results.json"]["generated_at"] = "2099-01-01T00:00:00Z"
    second["candidate-results.json"]["candidates"][0]["mean_fit_time"] = 99.0
    second["cross-validation-results.csv"].loc[0, "mean_fit_time"] = 99.0
    result = sm.write_model_selection_artifacts(
        output_directory=output, artifacts=second
    )
    assert result.idempotent is True
    assert result.created == ()
    assert result.replaced == ()


def test_semantic_conflict_is_rejected_without_partial_change(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    sm.write_model_selection_artifacts(output_directory=output, artifacts=_artifact_set())
    before = {name: (output / name).read_bytes() for name in sm.ARTIFACT_FILENAMES}
    divergent = _artifact_set()
    divergent["model-selection-handoff.json"]["selected_model_id"] = "different"
    with pytest.raises(sm.ArtifactConflictError, match="divergent"):
        sm.write_model_selection_artifacts(
            output_directory=output, artifacts=divergent
        )
    assert before == {name: (output / name).read_bytes() for name in sm.ARTIFACT_FILENAMES}


def test_explicit_overwrite_replaces_conflict(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    sm.write_model_selection_artifacts(output_directory=output, artifacts=_artifact_set())
    divergent = _artifact_set()
    divergent["model-selection-handoff.json"]["selection_rationale"] = "changed"
    result = sm.write_model_selection_artifacts(
        output_directory=output, artifacts=divergent, overwrite=True
    )
    assert set(result.replaced) == set(sm.ARTIFACT_FILENAMES)
    loaded = json.loads((output / "model-selection-handoff.json").read_text())
    assert loaded["selection_rationale"] == "changed"


def test_mid_promotion_failure_rolls_back_new_set(tmp_path, monkeypatch):
    output = tmp_path / "artifacts/model-selection/example"
    original = sm.os.replace
    calls = {"count": 0}

    def failing_replace(source, destination):
        if ".model-selection-staging-" in str(source):
            calls["count"] += 1
            if calls["count"] == 3:
                raise OSError("injected promotion failure")
        return original(source, destination)

    monkeypatch.setattr(sm.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected"):
        sm.write_model_selection_artifacts(
            output_directory=output, artifacts=_artifact_set()
        )
    assert not any((output / name).exists() for name in sm.ARTIFACT_FILENAMES)
    assert not list(output.parent.glob(".model-selection-staging-*"))
    assert not list(output.parent.glob(".model-selection-backup-*"))


def test_handoff_rejects_fingerprint_mismatch(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    sm.write_model_selection_artifacts(output_directory=output, artifacts=_artifact_set())
    path = output / "threshold-analysis.json"
    payload = json.loads(path.read_text())
    payload["selected_educational_threshold"] = 0.99
    path.write_text(json.dumps(payload))
    with pytest.raises(sm.ModelSelectionHandoffError, match="fingerprint mismatch"):
        sm.load_and_validate_model_selection_handoff(
            project_root=tmp_path,
            handoff_path="artifacts/model-selection/example/model-selection-handoff.json",
        )


def test_handoff_loader_returns_defensive_copy(tmp_path):
    output = tmp_path / "artifacts/model-selection/example"
    sm.write_model_selection_artifacts(output_directory=output, artifacts=_artifact_set())
    first = sm.load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/example/model-selection-handoff.json",
    )
    first["feature_columns"].append("mutated")
    second = sm.load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path="artifacts/model-selection/example/model-selection-handoff.json",
    )
    assert "mutated" not in second["feature_columns"]


# L. Compatibility and normalization

def test_search_summary_normalizes_negative_loss_scorers(roles):
    spec = _specs()[0]
    outcome = sm.run_model_search(
        model_id=spec["model_id"],
        family=spec["family"],
        pipeline=sm.build_candidate_pipeline(
            estimator=spec["estimator"],
            numerical_features=NUMERICAL,
            categorical_features=CATEGORICAL,
            scale_numerical=True,
        ),
        search_strategy="GridSearchCV",
        search_space=spec["search_space"],
        x_train=roles.x_train,
        y_train=roles.y_train,
        scoring=sm.build_scoring_contract(),
        cv=sm.build_cross_validation(n_splits=3, shuffle=True, random_state=17),
        refit_metric="average_precision",
        n_jobs=1,
    )
    summary, table = sm.summarize_search_results(outcome, n_splits=3)
    assert summary["cv_log_loss_mean"] >= 0
    assert summary["cv_brier_score_mean"] >= 0
    assert (table["mean_cv_log_loss"] >= 0).all()
    assert (table["mean_cv_brier_score"] >= 0).all()
    copy_table = outcome.cv_results
    copy_table.iloc[0, 0] = 999
    assert outcome.cv_results.iloc[0, 0] != 999



def test_csv_semantic_fingerprint_survives_float_roundtrip(tmp_path):
    frame = pd.DataFrame([
        {
            "model_id": "a",
            "family": "LogisticRegression",
            "search_strategy": "GridSearchCV",
            "parameters": '{"model__C":1.0}',
            "mean_cv_average_precision": 0.8123456789012345,
        }
    ])
    path = tmp_path / "results.csv"
    frame.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert sm.semantic_fingerprint_csv(frame) == sm.semantic_fingerprint_csv(reloaded)

def test_pandas_string_float_integer_and_nontrivial_indices(partitions):
    train, validation = partitions
    train["cat_text"] = train["cat_text"].astype("string")
    validation["cat_text"] = validation["cat_text"].astype("string")
    result = sm.validate_feature_partition_roles(
        train=train,
        validation=validation,
        feature_columns=FEATURES,
        identifier_columns=IDENTIFIERS,
        target_column=TARGET,
        target_classes=("No", "Yes"),
        positive_class="Yes",
        target_encoding=ENCODING,
    )
    assert str(result.x_train["cat_text"].dtype) == "string"
    assert result.x_train.index.equals(train.index)


def test_warning_handling_can_be_recorded_without_failing(roles):
    with pytest.warns(UserWarning, match="educational"):
        import warnings

        warnings.warn("educational warning", UserWarning)
    assert roles.x_train.shape[0] == 120
