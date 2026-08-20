import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.select_models import (
    FeatureRoleError,
    ModelSelectionContractError,
    analyze_regression_repeated_profile_sensitivity,
    analyze_regression_target_extreme_sensitivity,
    build_candidate_pipeline,
    build_regression_cross_validation,
    build_regression_scoring_contract,
    compute_regression_metrics,
    describe_regression_cv_folds,
    select_regression_candidate_model,
    validate_regression_feature_partition_roles,
    validate_regression_model_selection_contract,
)


def contract():
    return {"problem_type": "continuous_regression", "target_semantics": "Continuous / quantitative",
            "target_unit": "MPa", "primary_metric": "mae",
            "primary_metric_direction": "lower_is_better", "refit_metric": "mae",
            "cv": {"strategy": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
            "test_partition_sealed": True, "test_partition_evaluated": False}


def frames():
    train = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "b": [5., 4., 3., 2., 1.],
                          "target": [2.3, 4.1, 6.2, 8.4, 10.5]})
    validation = pd.DataFrame({"a": [1., 8., 9.], "b": [5., 2., 1.],
                               "target": [2.4, 15.8, 18.1]})
    return train, validation


@pytest.mark.parametrize("field,value", [
    ("problem_type", "binary_classification"), ("target_semantics", "nominal"),
    ("primary_metric", "r2"), ("primary_metric_direction", "higher_is_better"),
    ("test_partition_sealed", False), ("test_partition_evaluated", True),
])
def test_contract_fails_closed(field, value):
    value_contract = contract(); value_contract[field] = value
    with pytest.raises(ModelSelectionContractError):
        validate_regression_model_selection_contract(value_contract)


def test_contract_and_cv_are_exact_and_deterministic():
    assert validate_regression_model_selection_contract(contract()) == contract()
    cv1, cv2 = build_regression_cross_validation(), build_regression_cross_validation()
    assert (cv1.n_splits, cv1.shuffle, cv1.random_state) == (5, True, 42)
    x = pd.DataFrame({"a": range(20)}); y = pd.Series(np.arange(20.0))
    assert [v.tolist() for _, v in cv1.split(x)] == [v.tolist() for _, v in cv2.split(x)]
    rows = describe_regression_cv_folds(cv=cv1, x=x, y=y)
    assert all(r["diagnostic_only"] and not r["used_for_fold_assignment"] for r in rows)


def test_partition_roles_are_ordered_continuous_and_defensive():
    train, validation = frames()
    roles = validate_regression_feature_partition_roles(train=train, validation=validation,
        feature_columns=["b", "a"], identifier_columns=[], target_column="target")
    assert list(roles.x_train) == ["b", "a"] and "target" not in roles.x_train
    changed = roles.x_train; changed.iloc[0, 0] = -999
    assert roles.x_train.iloc[0, 0] == 5
    assert "test" not in inspect.signature(validate_regression_feature_partition_roles).parameters
    train.loc[0, "target"] = np.inf
    with pytest.raises(FeatureRoleError):
        validate_regression_feature_partition_roles(train=train, validation=validation,
            feature_columns=["a", "b"], identifier_columns=[], target_column="target")


def test_scoring_and_pipeline_keep_positive_human_errors_and_fold_safe_scaling():
    assert build_regression_scoring_contract() == {"mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error", "r2": "r2", "medae": "neg_median_absolute_error"}
    pipeline = build_candidate_pipeline(estimator=Ridge(), numerical_features=["a", "b"],
        categorical_features=[], scale_numerical=True)
    scaler = pipeline.named_steps["preprocess"].transformers[0][1]
    assert isinstance(scaler, StandardScaler) and not hasattr(scaler, "mean_")
    assert DummyRegressor(strategy="median").strategy == "median"


def test_known_regression_metrics_and_diagnostics():
    result = compute_regression_metrics([1., 2., 3., 4.], [1., 1., 5., 4.])
    assert result["mae"] == pytest.approx(.75)
    assert result["rmse"] == pytest.approx(np.sqrt(1.25))
    assert result["medae"] == pytest.approx(.5)
    assert result["residual_mean"] == pytest.approx(-.25)
    assert result["max_absolute_error"] == 2


def summaries():
    return {key: {"family": key, "cv_mae_std": std} for key, std in
            (("a", .4), ("b", .3), ("c", .2))}


def evaluation(mae, rmse, medae, r2):
    return {"metrics": {"mae": mae, "rmse": rmse, "medae": medae, "r2": r2}}


def test_selection_eligibility_and_tie_break_are_deterministic():
    evaluations = {"a": evaluation(4.00, 5.1, 3.0, .7),
                   "b": evaluation(4.05, 5.0, 3.0, .7),
                   "c": evaluation(5.0, 5.0, 3.0, .7)}
    kwargs = dict(cv_summaries=summaries(), validation_evaluations=evaluations,
                  baseline_validation_metrics={"mae": 5.0}, practical_tie_tolerance=.10)
    first = select_regression_candidate_model(**kwargs)
    assert first == select_regression_candidate_model(**kwargs)
    assert first["selected_model_id"] == "b" and not next(r for r in first["ranking"] if r["model_id"] == "c")["eligible"]


def test_sensitivities_are_diagnostic_and_train_derived():
    train, validation = frames(); pred = [2., 16., 18.]
    repeated = analyze_regression_repeated_profile_sensitivity(train_features=train[["a", "b"]],
        validation_features=validation[["a", "b"]], y_validation=validation["target"], predictions=pred)
    assert repeated["repeated_profile_validation_row_count"] == 1
    assert repeated["proven_duplicate_identity"] is False and repeated["used_for_selection"] is False
    extreme1 = analyze_regression_target_extreme_sensitivity(y_train=train["target"],
        y_validation=validation["target"], predictions=pred)
    shifted = validation["target"] * 100
    extreme2 = analyze_regression_target_extreme_sensitivity(y_train=train["target"],
        y_validation=shifted, predictions=pred)
    assert extreme1["train_derived_lower_fence"] == extreme2["train_derived_lower_fence"]
    assert extreme1["used_for_selection"] is False


def test_official_notebook_is_clean_and_sealed_in_code():
    notebook = json.loads(open("notebooks/03_model_selection_and_evaluation.ipynb", encoding="utf-8").read())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for cell in (c for c in notebook["cells"] if c["cell_type"] == "code"):
        assert cell["execution_count"] is None and cell["outputs"] == []
    for forbidden in ("X_test", "y_test", "threshold-analysis.json",
                      "average_precision", "macro_f1", "DummyClassifier", "StratifiedKFold",
                      "target_classes", "positive_class", "final-pipeline.joblib"):
        assert forbidden not in code
    assert "load_and_validate_preparation_for_model_selection" in code
    assert "load_and_validate_preparation_handoff" not in code
