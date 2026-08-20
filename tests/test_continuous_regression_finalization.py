import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from scripts.finalize_model import (
    FinalizationContractError,
    RegressionFrozenFinalizationContract,
    assemble_regression_final_training_data,
    compute_regression_model_state_fingerprint,
    describe_regression_fitted_pipeline,
    reconstruct_regression_selected_pipeline,
)
from scripts.select_models import compute_regression_metrics


FEATURES = ["a", "b"]


def contract(family="HistGradientBoostingRegressor", *, scale=False):
    fixed = {
        "Ridge": {"alpha": 1.0},
        "DecisionTreeRegressor": {"random_state": 42, "max_depth": None},
        "RandomForestRegressor": {"random_state": 42, "n_estimators": 5, "n_jobs": 1},
        "HistGradientBoostingRegressor": {"random_state": 42, "max_iter": 10},
    }[family]
    selected = {"model__alpha": 2.0} if family == "Ridge" else {}
    payload = {
        "selected_model_family": family,
        "selected_estimator_fixed_constructor_parameters": fixed,
        "selected_hyperparameters": selected,
        "feature_order": FEATURES,
        "preprocessing": {"type": "Pipeline", "categorical_features": [],
                          "numerical_features": FEATURES, "scale_numerical": scale,
                          "scaler": "StandardScaler" if scale else None},
    }
    return RegressionFrozenFinalizationContract(tuple(payload.items()))


@pytest.mark.parametrize("family", ["Ridge", "DecisionTreeRegressor",
    "RandomForestRegressor", "HistGradientBoostingRegressor"])
def test_exact_supported_family_reconstruction_is_unfitted(family):
    pipeline = reconstruct_regression_selected_pipeline(contract(family, scale=family == "Ridge"))
    assert pipeline.named_steps["model"].__class__.__name__ == family
    with pytest.raises(Exception):
        check_is_fitted(pipeline)
    assert (pipeline.named_steps["preprocess"].transformers[0][1].__class__.__name__ == "StandardScaler") == (family == "Ridge")


def test_selected_parameters_override_fixed_values():
    pipeline = reconstruct_regression_selected_pipeline(contract("Ridge", scale=True))
    assert pipeline.named_steps["model"].alpha == 2.0


@pytest.mark.parametrize("mutation", ["family", "fixed", "selected", "preprocess"])
def test_reconstruction_fails_closed(mutation):
    data = contract().as_dict()
    if mutation == "family": data["selected_model_family"] = "SVR"
    elif mutation == "fixed": data["selected_estimator_fixed_constructor_parameters"]["bogus"] = 1
    elif mutation == "selected": data["selected_hyperparameters"]["model__bogus"] = 1
    else: data["preprocessing"]["categorical_features"] = ["a"]
    bad = RegressionFrozenFinalizationContract(tuple(data.items()))
    with pytest.raises(FinalizationContractError):
        reconstruct_regression_selected_pipeline(bad)


def test_training_assembly_has_no_test_argument_and_preserves_order():
    assert "test" not in inspect.signature(assemble_regression_final_training_data).parameters
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [5.0, 6.0]})
    validation = pd.DataFrame({"a": [7.0], "b": [8.0], "y": [9.0]})
    x, y = assemble_regression_final_training_data(train=train, validation=validation,
        feature_columns=FEATURES, target_column="y")
    assert x.to_dict("records") == [{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 4.0}, {"a": 7.0, "b": 8.0}]
    assert y.tolist() == [5.0, 6.0, 9.0]


@pytest.mark.parametrize("target", [[1.0, np.nan], [1.0, np.inf], ["x", "y"]])
def test_training_target_must_be_numeric_complete_finite(target):
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": target})
    with pytest.raises(FinalizationContractError):
        assemble_regression_final_training_data(train=frame, validation=frame,
            feature_columns=FEATURES, target_column="y")


def test_known_regression_metrics_and_diagnostics():
    metrics = compute_regression_metrics([1, 2, 4], [0, 2, 7])
    residuals = np.array([1.0, 0.0, -3.0]); absolute = np.abs(residuals)
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert metrics["medae"] == 1.0
    assert metrics["residual_mean"] == pytest.approx(residuals.mean())
    assert metrics["residual_standard_deviation"] == pytest.approx(residuals.std(ddof=1))
    assert metrics["max_absolute_error"] == 3.0
    for q in (50, 90, 95): assert metrics[f"absolute_error_p{q}"] == pytest.approx(np.quantile(absolute, q/100))


def test_descriptor_and_semantic_fingerprint_survive_equivalent_fit():
    c = contract("Ridge", scale=True)
    frame = pd.DataFrame({"a": [1., 2., 3.], "b": [3., 2., 1.]})
    pipeline = reconstruct_regression_selected_pipeline(c).fit(frame, [1., 2., 4.])
    descriptor = describe_regression_fitted_pipeline(pipeline=pipeline, contract=c)
    assert descriptor["model_class"] == "Ridge"
    assert compute_regression_model_state_fingerprint(descriptor) == compute_regression_model_state_fingerprint(json.loads(json.dumps(descriptor)))


def test_official_notebook_is_clean_and_continuous_only():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/04_final_model_and_bundle.ipynb").read_text())
    code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code").lower()
    assert all(c.get("execution_count") is None and c.get("outputs") == [] for c in notebook["cells"] if c["cell_type"] == "code")
    assert "run_regression_finalization" in code
    assert "argmax" not in code and "positive_class" not in code and "threshold" not in code
