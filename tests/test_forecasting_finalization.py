import dataclasses
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import scripts.forecasting_finalization as ff
from scripts.forecasting_model_selection import frozen_specification_catalog, seasonal_trend_ols_forecast
from scripts.forecasting_preparation import seasonal_mase_scale


def spec():
    return next(x for x in frozen_specification_catalog() if x["candidate_id"] == "seasonal_trend_ols")


def development():
    idx = pd.period_range("1920-01", "1938-12", freq="M", name="period")
    y = 40 + .03*np.arange(228) + 8*np.sin(2*np.pi*np.arange(228)/12)
    return pd.Series(y, index=idx, name="temperature")


def write_series(path, series):
    pd.DataFrame({"period": series.index.astype(str), "temperature": series.values}).to_csv(path, index=False)
    return ff.sha256_file(path)


def contract(tmp_path, dev=None, hold=None):
    dev = development() if dev is None else dev
    hold = pd.Series(np.arange(12.)+50, index=pd.period_range("1939-01", periods=12, freq="M", name="period"), name="temperature") if hold is None else hold
    dp, hp = tmp_path/"development.csv", tmp_path/"final-holdout.csv"
    ds, hs = write_series(dp, dev), write_series(hp, hold)
    return ff.FrozenForecastingFinalizationContract(
        "nottem", "seasonal_trend_ols", "candidate", "DeterministicSeasonalTrendOLS",
        spec(), str(dp), ds, str(hp), hs,
        {"mae": 1., "rmse": 1.2, "seasonal_mase_12": .5},
        "artifacts/model-selection/nottem/model-selection-handoff.json", "a"*64, "b"*64,
        {"path":"prep.json","sha256":"c"*64,"schema_version":"forecasting-preparation-handoff.v1"})


def fitted(tmp_path):
    c = contract(tmp_path); g = ff.ForecastingFinalizationGuard()
    return c, g, ff.reconstruct_and_fit_selected_forecasting_model(c, development(), g)


def test_exact_ols_reconstruction_and_oracle(tmp_path):
    c, g, model = fitted(tmp_path)
    future = pd.period_range("1939-01", periods=12, freq="M", name="period")
    oracle = seasonal_trend_ols_forecast(development(), future)
    assert model.design_columns == ff.DESIGN_COLUMNS
    assert len(model.coefficients) == 13 and model.training_start_ordinal == development().index[0].ordinal
    assert g.final_fit_count == 1
    np.testing.assert_allclose(model.forecast_periods(future), oracle, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mutator", [
    lambda s: pd.concat([s, pd.Series([1.], index=pd.PeriodIndex(["1939-01"], freq="M"))]),
    lambda s: s.iloc[:-1],
    lambda s: s.drop(s.index[10]),
    lambda s: s.iloc[::-1],
    lambda s: s.mask(np.arange(len(s)) == 5, np.inf),
])
def test_development_rejects_scope_chronology_and_nonfinite(tmp_path, mutator):
    s=mutator(development()); p=tmp_path/"d.csv"; sha=write_series(p,s)
    with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_full_development(p, expected_sha256=sha)


def test_development_hash_is_authenticated(tmp_path):
    p=tmp_path/"d.csv"; write_series(p,development())
    with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_full_development(p, expected_sha256="0"*64)


def test_guard_freeze_open_and_duplicate_rules(tmp_path):
    c=contract(tmp_path); g=ff.ForecastingFinalizationGuard()
    with pytest.raises(ff.ForecastingHoldoutAccessError): ff.open_final_holdout_once(c.holdout_path, expected_sha256=c.holdout_sha256, guard=g)
    model=ff.reconstruct_and_fit_selected_forecasting_model(c,development(),g); g.mark_frozen()
    ff.open_final_holdout_once(c.holdout_path, expected_sha256=c.holdout_sha256, guard=g)
    with pytest.raises(ff.ForecastingDuplicateFinalEvaluationError): ff.open_final_holdout_once(c.holdout_path, expected_sha256=c.holdout_sha256, guard=g)
    with pytest.raises(ff.ForecastingFinalizationContractError): g.register_fit()
    assert g.post_holdout_fit_count == 1 and model.frozen


def test_holdout_hash_checked_before_read_csv(tmp_path, monkeypatch):
    c,g,model=fitted(tmp_path); g.mark_frozen(); calls=[]
    original=pd.read_csv
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:(calls.append(a[0]),original(*a,**k))[1])
    with pytest.raises(ff.ForecastingFinalizationContractError):
        ff.open_final_holdout_once(c.holdout_path,expected_sha256="0"*64,guard=g)
    assert calls == []


def test_forecast_independent_of_holdout_and_metrics(tmp_path):
    c,g,model=fitted(tmp_path); g.mark_frozen()
    future=pd.period_range("1939-01",periods=12,freq="M",name="period")
    a=model.forecast_periods(future); b=model.forecast_periods(future)
    np.testing.assert_array_equal(a,b)
    h1=pd.Series(np.zeros(12),index=future); h2=pd.Series(np.ones(12)*999,index=future)
    g1=ff.ForecastingFinalizationGuard(1,True,1,1); rows1,m1,d1=ff.evaluate_final_forecast_once(a,h1,development=development(),guard=g1)
    g2=ff.ForecastingFinalizationGuard(1,True,1,1); rows2,m2,d2=ff.evaluate_final_forecast_once(a,h2,development=development(),guard=g2)
    assert d1 == d2 == seasonal_mase_scale(development(),period=12)
    assert len(rows1)==12 and math.isclose(m1["rmse"],np.sqrt(np.mean([(x["y_pred"]-x["y_true"])**2 for x in rows1])))
    assert m2["mae"] > m1["mae"]  # bad performance is recorded, not gated


def test_duplicate_forecast_and_evaluation(tmp_path):
    c,g,m=fitted(tmp_path); g.mark_frozen(); h=ff.open_final_holdout_once(c.holdout_path,expected_sha256=c.holdout_sha256,guard=g)
    p=ff.generate_final_forecast_once(m,origin="1938-12",guard=g)
    ff.evaluate_final_forecast_once(p,h,development=development(),guard=g)
    with pytest.raises(ff.ForecastingDuplicateFinalEvaluationError): ff.generate_final_forecast_once(m,origin="1938-12",guard=g)
    with pytest.raises(ff.ForecastingDuplicateFinalEvaluationError): ff.evaluate_final_forecast_once(p,h,development=development(),guard=g)


def test_joblib_roundtrip_hash_before_load_and_state_tamper(tmp_path,monkeypatch):
    c,g,m=fitted(tmp_path); p=tmp_path/"model.joblib"; sha=ff.serialize_frozen_forecasting_model_to_staging(m,p)
    loaded=ff.validate_serialized_forecasting_model(p,expected_byte_sha256=sha,expected_state_fingerprint=m.model_state_semantic_fingerprint)
    np.testing.assert_array_equal(loaded.coefficients,m.coefficients)
    called=[]; original=joblib.load; monkeypatch.setattr(joblib,"load",lambda *a,**k:(called.append(1),original(*a,**k))[1])
    with pytest.raises(ff.ForecastingArtifactConflictError): ff.validate_serialized_forecasting_model(p,expected_byte_sha256="0"*64,expected_state_fingerprint=m.model_state_semantic_fingerprint)
    assert called == []
    bad=dataclasses.replace(m,coefficients=(m.coefficients[0]+1,)+m.coefficients[1:])
    joblib.dump(bad,p); badsha=ff.sha256_file(p)
    with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_serialized_forecasting_model(p,expected_byte_sha256=badsha,expected_state_fingerprint=m.model_state_semantic_fingerprint)


def test_inference_history_one_row_and_contract_rejections(tmp_path):
    c,g,m=fitted(tmp_path)
    one=pd.DataFrame({"period":["1938-12"],"temperature":[50.]})
    out=m.predict_from_history(one)
    assert list(out)==["period","forecast"] and len(out)==12 and out.period.iloc[0]=="1939-01"
    bad=[pd.DataFrame({"temperature":[1.]}),pd.DataFrame({"period":["1938-12"],"temperature":[np.inf]}),
         pd.DataFrame({"period":["1938-11"],"temperature":[1.]}),
         pd.DataFrame({"period":["1938-12","1938-12"],"temperature":[1.,2.]}),
         pd.DataFrame({"period":["1938-12","1939-02"],"temperature":[1.,2.]})]
    for frame in bad:
        with pytest.raises(ff.ForecastingFinalizationContractError): m.predict_from_history(frame)


def test_json_fingerprint_and_evidence_tamper(tmp_path):
    c,g,m=fitted(tmp_path); g.mark_frozen(); h=ff.open_final_holdout_once(c.holdout_path,expected_sha256=c.holdout_sha256,guard=g)
    p=ff.generate_final_forecast_once(m,origin="1938-12",guard=g); rows,metrics,d=ff.evaluate_final_forecast_once(p,h,development=development(),guard=g)
    e=ff.build_forecasting_final_test_evidence(c,m,rows,metrics,d,g); path=tmp_path/"e.json"; path.write_bytes(ff._canonical(e))
    ff.load_and_validate_forecasting_final_test_evidence(project_root=tmp_path,evidence_path=path)
    e["forecasts"][0]["abs_error"] += 1; e["semantic_fingerprint"]=ff.semantic_fingerprint(e); path.write_bytes(ff._canonical(e))
    with pytest.raises(ff.ForecastingFinalizationContractError): ff.load_and_validate_forecasting_final_test_evidence(project_root=tmp_path,evidence_path=path)


def test_artifact_states_and_path_escape(tmp_path):
    out=tmp_path/"out"; assert ff._artifact_state(out)=="absent"; out.mkdir(); (out/ff.FINAL_FILENAMES[0]).write_text("x")
    assert ff._artifact_state(out)=="partial"
    with pytest.raises(ff.ForecastingArtifactConflictError): ff._confined(tmp_path,"../escape")


def test_entry_boundary_rejects_flags_and_unsupported_winner(tmp_path):
    base={"dataset_slug":"nottem","problem_type":"time_series_forecasting","forecasting_mode":"univariate",
          "readiness":{"final_model_training_ready":True,"final_holdout_sealed":True,"selected_specification_frozen":True,
          "model_selection_handoff_reloadable":True,"final_model_trained":False,"model_artifact_materialized":False,
          "model_bundle_materialized":False,"final_holdout_evaluated":False,"operational_modeling_ready":False}}
    with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_forecasting_finalization_contract(base,handoff_path=tmp_path/"missing")
    for key in ("final_model_training_ready",):
        bad=json.loads(json.dumps(base)); bad["readiness"][key]=False
        with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_forecasting_finalization_contract(bad,handoff_path=tmp_path/"x")
    for key in ("final_model_trained","model_artifact_materialized"):
        bad=json.loads(json.dumps(base)); bad["readiness"][key]=True
        with pytest.raises(ff.ForecastingFinalizationContractError): ff.validate_forecasting_finalization_contract(bad,handoff_path=tmp_path/"x")


def test_real_complete_set_rerun_is_read_free_and_fit_free(monkeypatch):
    root=Path(__file__).parents[1]
    if not (root/"artifacts/models/nottem/final-model-handoff.json").is_file(): pytest.skip("runtime final set unavailable")
    original_read_csv=pd.read_csv
    def guarded_read_csv(path,*args,**kwargs):
        if Path(path).name == "final-holdout.csv": pytest.fail("idempotent rerun read holdout")
        return original_read_csv(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_csv",guarded_read_csv)
    monkeypatch.setattr(ff.OLS,"fit",lambda *a,**k:pytest.fail("idempotent rerun fit"))
    result=ff.run_forecasting_finalization(project_root=root)
    assert result.status=="reused_equivalent" and result.guard=={"final_fit_count":0,"holdout_open_count":0,"final_evaluation_count":0}


def test_rerun_resolves_relative_handoff_against_project_root(monkeypatch, tmp_path):
    root=Path(__file__).parents[1].resolve()
    if not (root/"artifacts/models/nottem/final-model-handoff.json").is_file(): pytest.skip("runtime final set unavailable")
    monkeypatch.chdir(tmp_path)
    result=ff.run_forecasting_finalization(project_root=root)
    assert result.status == "reused_equivalent"


# ======================================================================================
# Notebook 04 -> Notebook 05 boundary hardening: semantic authentication tamper matrix.
#
# These tests never call run_forecasting_finalization to *generate* artifacts (no real
# fit, no real holdout open). They build one fully valid synthetic five-artifact set
# with a FrozenForecastingModel already fitted by the existing finalization primitives,
# then tamper exactly one semantically meaningful field at a time, recompute every
# hash/fingerprint an attacker with full filesystem + hashing access could recompute
# (a "cryptographically self-consistent but semantically false" artifact set), and
# assert that scripts.forecasting_finalization.load_and_validate_forecasting_final_model_handoff
# still rejects it. This is the loader Notebook 05 will call.
# ======================================================================================

_UPSTREAM_TRAINING_INSTRUCTIONS = {
    "notebook": "notebooks/04_final_model_and_bundle.ipynb",
    "training_scope": "full development 1920-01 -> 1938-12",
    "reconstruct_exact_selected_specification": True,
    "freeze_before_final_holdout_open": True,
    "final_forecast_horizon": 12,
    "final_evaluation_period": "1939-01 -> 1939-12",
    "open_holdout_only_after_final_spec_fit": True,
    "evaluate_holdout_once": True,
    "do_not_retune": True,
    "do_not_change_candidate": True,
    "do_not_change_hyperparameters": True,
    "do_not_change_transform_policy": True,
    "target_scale": "original degrees Fahrenheit",
}


def _authenticated_upstream_selection_dict(dev_sha, hold_sha):
    """Minimal dict accepted by ff.validate_forecasting_finalization_contract.

    Standing in for a fully authenticated Notebook 03 handoff: constructing the real
    upstream artifact chain (preparation + exploration + full model-selection sibling
    set) is Notebook 03's own test responsibility (tests/test_forecasting_model_selection.py,
    out of scope here). This dict is fed through the *real* contract validator, so it must
    satisfy every field that validator actually checks.
    """
    return {
        "dataset_slug": "nottem", "problem_type": "time_series_forecasting", "forecasting_mode": "univariate",
        "readiness": {"final_model_training_ready": True, "final_holdout_sealed": True,
                      "selected_specification_frozen": True, "model_selection_handoff_reloadable": True,
                      "final_model_trained": False, "model_artifact_materialized": False,
                      "model_bundle_materialized": False, "final_holdout_evaluated": False,
                      "operational_modeling_ready": False},
        "selected_specification": spec(), "selected_candidate_id": "seasonal_trend_ols",
        "selected_family": "DeterministicSeasonalTrendOLS", "selected_role": "candidate",
        "selection": {"selected_candidate_id": "seasonal_trend_ols"},
        "final_training_instructions": dict(_UPSTREAM_TRAINING_INSTRUCTIONS),
        "development": {"start": "1920-01", "end": "1938-12", "row_count": 228,
                        "path": "development.csv", "sha256": dev_sha},
        "sealed_final_holdout_metadata_only": {"start": "1939-01", "end": "1939-12", "row_count": 12,
                                               "sealed": True, "evaluated": False,
                                               "path": "final-holdout.csv", "sha256": hold_sha},
        "evaluation": {"selected_pooled_metrics": {"mae": 1.0, "rmse": 1.2, "seasonal_mase_12": 0.5}},
        "semantic_fingerprint": "b" * 64,
        "upstream_preparation_handoff": {"path": "prep.json", "sha256": "c" * 64,
                                         "schema_version": "forecasting-preparation-handoff.v1"},
    }


def full_artifact_set(tmp_path, monkeypatch):
    """Build one complete, internally-consistent five-artifact final set in tmp_path.

    Uses the project's own finalization primitives (reconstruct/fit/freeze/evaluate,
    build_forecasting_final_test_evidence, build_forecasting_inference_bundle,
    build_forecasting_final_model_manifest, build_forecasting_final_model_handoff) so the
    baseline is exactly what a real Notebook 04 run would authenticate -- no parallel
    hand-rolled artifact shape. Only the upstream Notebook 03 handoff is stood in for
    (via monkeypatch) rather than fully reconstructed, per the finalization-boundary
    test scope.
    """
    dev = development()
    hold = pd.Series(np.arange(12.) + 50, index=pd.period_range("1939-01", periods=12, freq="M", name="period"), name="temperature")
    dev_path, hold_path = tmp_path / "development.csv", tmp_path / "final-holdout.csv"
    dev_sha, hold_sha = write_series(dev_path, dev), write_series(hold_path, hold)

    upstream_path = tmp_path / "model-selection-handoff.json"
    upstream_path.write_bytes(b'{"placeholder": true}')

    selected = _authenticated_upstream_selection_dict(dev_sha, hold_sha)
    monkeypatch.setattr(
        ff, "load_and_validate_forecasting_model_selection_handoff",
        lambda *, project_root, handoff_path, expected_dataset_slug: selected,
    )

    c = ff.validate_forecasting_finalization_contract(selected, handoff_path=upstream_path)
    g = ff.ForecastingFinalizationGuard()
    model = ff.freeze_forecasting_model(ff.reconstruct_and_fit_selected_forecasting_model(c, dev, g))
    g.mark_frozen()
    holdout_series = ff.open_final_holdout_once(hold_path, expected_sha256=hold_sha, guard=g)
    predictions = ff.generate_final_forecast_once(model, origin="1938-12", guard=g)
    rows, metrics, denominator = ff.evaluate_final_forecast_once(predictions, holdout_series, development=dev, guard=g)
    evidence = ff.build_forecasting_final_test_evidence(c, model, rows, metrics, denominator, g)

    out = tmp_path / "out"
    out.mkdir()
    model_path = out / "final-pipeline.joblib"
    model_sha = ff.serialize_frozen_forecasting_model_to_staging(model, model_path)

    rel = lambda name: f"out/{name}"
    bundle = ff.build_forecasting_inference_bundle(c, model, model_path=rel("final-pipeline.joblib"), model_sha256=model_sha)
    manifest = ff.build_forecasting_final_model_manifest(c, model, g, model_path=rel("final-pipeline.joblib"), model_sha256=model_sha)
    (out / "final-test-evidence.json").write_bytes(ff._canonical(evidence))
    (out / "inference-bundle.json").write_bytes(ff._canonical(bundle))
    (out / "final-model-manifest.json").write_bytes(ff._canonical(manifest))

    refs = {
        "final-pipeline.joblib": ff._reference(rel("final-pipeline.joblib"), model_sha, model.model_state_semantic_fingerprint),
        "final-model-manifest.json": ff._reference(rel("final-model-manifest.json"), ff.sha256_file(out / "final-model-manifest.json"), manifest["semantic_fingerprint"]),
        "final-test-evidence.json": ff._reference(rel("final-test-evidence.json"), ff.sha256_file(out / "final-test-evidence.json"), evidence["semantic_fingerprint"]),
        "inference-bundle.json": ff._reference(rel("inference-bundle.json"), ff.sha256_file(out / "inference-bundle.json"), bundle["semantic_fingerprint"]),
    }
    handoff = ff.build_forecasting_final_model_handoff(c, model, evidence, bundle, refs)
    (out / "final-model-handoff.json").write_bytes(ff._canonical(handoff))

    return {"tmp_path": tmp_path, "out": out, "contract": c, "model": model, "guard": g,
            "manifest": manifest, "evidence": evidence, "bundle": bundle, "handoff": handoff, "refs": refs}


def _load(fx):
    return ff.load_and_validate_forecasting_final_model_handoff(
        project_root=fx["tmp_path"], handoff_path=fx["out"] / "final-model-handoff.json")


def test_full_artifact_set_baseline_is_accepted(tmp_path, monkeypatch):
    """Sanity check: the untampered synthetic set the tamper matrix mutates must load cleanly."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    handoff = _load(fx)
    assert handoff["selected_model"]["candidate_id"] == "seasonal_trend_ols"
    assert handoff["readiness"]["operational_modeling_ready"] is False


_JSON_ARTIFACT_NAMES = {
    "manifest": "final-model-manifest.json",
    "evidence": "final-test-evidence.json",
    "bundle": "inference-bundle.json",
}


def _set_path(container, path, value):
    obj = container
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _finalize_tamper(fx, changed_keys):
    """Re-sign every mutated artifact, cascade its new hash/fingerprint into the handoff's
    sibling references, then re-sign the handoff itself -- exactly what an attacker with
    full write + hashing access (but no ability to refit the frozen model) could do."""
    for key in changed_keys:
        name = _JSON_ARTIFACT_NAMES[key]
        payload = fx[key]
        payload["semantic_fingerprint"] = ff.semantic_fingerprint(payload)
        (fx["out"] / name).write_bytes(ff._canonical(payload))
        sha = ff.sha256_file(fx["out"] / name)
        fx["refs"][name] = ff._reference(fx["refs"][name]["path"], sha, payload["semantic_fingerprint"])
    fx["handoff"]["final_references"] = dict(fx["refs"])
    fx["handoff"]["semantic_fingerprint"] = ff.semantic_fingerprint(fx["handoff"])
    (fx["out"] / "final-model-handoff.json").write_bytes(ff._canonical(fx["handoff"]))


def _apply_tamper(fx, artifact_key, path, value):
    _set_path(fx[artifact_key], path, value)
    _finalize_tamper(fx, [] if artifact_key == "handoff" else [artifact_key])


MANIFEST_TAMPERS = [
    ("dataset_slug", ("dataset_slug",), "tampered-dataset"),
    ("problem_type", ("problem_type",), "regression"),
    ("forecasting_mode", ("forecasting_mode",), "multivariate"),
    ("upstream_path", ("upstream", "model_selection_handoff", "path"), "other.json"),
    ("upstream_schema", ("upstream", "model_selection_handoff", "schema_version"), "x"),
    ("upstream_sha", ("upstream", "model_selection_handoff", "sha256"), "0" * 64),
    ("upstream_fp", ("upstream", "model_selection_handoff", "semantic_fingerprint"), "0" * 64),
    ("selected_candidate", ("selected_model", "selected_candidate_id"), "naive_last_value"),
    ("selected_role", ("selected_model", "selected_role"), "primary_baseline"),
    ("selected_family", ("selected_model", "selected_family"), "NaiveLastValue"),
    ("selected_spec", ("selected_model", "selected_specification", "trend_policy"), "quadratic"),
    ("training_scope", ("training", "scope"), "full series including 1939"),
    ("training_dev_path", ("training", "development_path"), "other-dev.csv"),
    ("training_dev_sha", ("training", "development_sha256"), "0" * 64),
    ("training_start", ("training", "start"), "1919-01"),
    ("training_end", ("training", "end"), "1939-12"),
    ("training_observations", ("training", "observations"), 240),
    ("training_fit_count", ("training", "final_fit_count"), 2),
    ("target_column", ("target", "column"), "temp"),
    ("target_semantics", ("target", "semantics"), "x"),
    ("target_unit", ("target", "unit"), "degrees Celsius"),
    ("target_frequency", ("target", "frequency"), "D"),
    ("model_path", ("model_state", "model_artifact_path"), "other.joblib"),
    ("model_sha", ("model_state", "model_artifact_byte_sha256"), "0" * 64),
    ("model_fp", ("model_state", "model_state_semantic_fingerprint"), "0" * 64),
    ("model_fitted_false", ("model_state", "fitted"), False),
    ("model_frozen_false", ("model_state", "frozen"), False),
    ("forecast_horizon", ("forecast", "horizon"), 6),
    ("multi_step_strategy", ("forecast", "multi_step_strategy"), "recursive"),
    ("holdout_frozen_flag", ("holdout_boundary", "model_frozen_before_holdout_open"), False),
    ("holdout_open_count", ("holdout_boundary", "holdout_open_count"), 2),
    ("holdout_path", ("holdout_boundary", "path"), "other-holdout.csv"),
    ("holdout_sha", ("holdout_boundary", "sha256"), "0" * 64),
    ("holdout_start", ("holdout_boundary", "start"), "1940-01"),
    ("holdout_end", ("holdout_boundary", "end"), "1940-12"),
    ("holdout_rows", ("holdout_boundary", "row_count"), 6),
    ("eval_completed", ("final_evaluation", "completed"), False),
    ("eval_count", ("final_evaluation", "evaluation_count"), 2),
    ("eval_adjustment", ("final_evaluation", "used_for_adjustment"), True),
    ("eval_retuning", ("final_evaluation", "used_for_retuning"), True),
    ("eval_selection", ("final_evaluation", "used_for_model_selection"), True),
    ("ready_trained", ("readiness", "final_model_trained"), False),
    ("ready_materialized", ("readiness", "model_artifact_materialized"), False),
    ("ready_holdout_eval", ("readiness", "final_holdout_evaluated"), False),
    ("ready_bundle", ("readiness", "inference_bundle_materialized"), False),
    ("ready_handoff", ("readiness", "final_model_handoff_ready"), False),
    ("ready_demo", ("readiness", "inference_demo_ready"), False),
    ("ready_operational", ("readiness", "operational_modeling_ready"), True),
]


@pytest.mark.parametrize("case", MANIFEST_TAMPERS, ids=[c[0] for c in MANIFEST_TAMPERS])
def test_manifest_tamper_matrix_rejected(tmp_path, monkeypatch, case):
    _, path, value = case
    fx = full_artifact_set(tmp_path, monkeypatch)
    _apply_tamper(fx, "manifest", path, value)
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


EVIDENCE_TAMPERS = [
    ("dataset_slug", ("dataset_slug",), "tampered"),
    ("problem_type", ("problem_type",), "regression"),
    ("forecasting_mode", ("forecasting_mode",), "multivariate"),
    ("origin", ("final_forecast_origin",), "1939-01"),
    ("period_start", ("final_evaluation_period", "start"), "1940-01"),
    ("period_end", ("final_evaluation_period", "end"), "1940-12"),
    ("horizon", ("forecast_horizon",), 6),
    ("holdout_sha", ("holdout_byte_sha256",), "0" * 64),
    ("model_state_fp", ("model_state_semantic_fingerprint",), "0" * 64),
    ("denominator", ("final_seasonal_mase_denominator",), 999.0),
    ("formula_mae", ("metric_formulas", "mae"), "sum(abs(y_pred-y_true))"),
    ("formula_rmse", ("metric_formulas", "rmse"), "mean((y_pred-y_true)^2)"),
    ("formula_mase", ("metric_formulas", "seasonal_mase_12"), "sum(abs_error/denominator)"),
    ("row_horizon", ("forecasts", 0, "horizon"), 2),
    ("row_period", ("forecasts", 0, "forecast_period"), "1940-01"),
    ("row_ypred", ("forecasts", 0, "y_pred"), 999.0),
    ("row_error", ("forecasts", 0, "error"), 999.0),
    ("row_abs_error", ("forecasts", 0, "abs_error"), 999.0),
    ("row_squared_error", ("forecasts", 0, "squared_error"), 999.0),
    ("row_scale", ("forecasts", 0, "seasonal_mase_scale"), 999.0),
    ("row_scaled_abs_error", ("forecasts", 0, "scaled_abs_error"), 999.0),
    ("metrics_mae", ("metrics", "mae"), 999.0),
    ("metrics_rmse", ("metrics", "rmse"), 999.0),
    ("metrics_mase", ("metrics", "seasonal_mase_12"), 999.0),
    ("ref_metrics_mae", ("model_selection_reference_metrics", "mae"), 999.0),
    ("deltas_mae", ("generalization_deltas", "mae"), 999.0),
    ("fit_count", ("final_fit_count",), 2),
    ("call_count", ("final_forecast_call_count",), 2),
    ("eval_count", ("final_evaluation_count",), 2),
    ("frozen_flag", ("model_frozen_before_holdout_open",), False),
    ("adjustment", ("holdout_used_for_adjustment",), True),
    ("retuning", ("holdout_used_for_retuning",), True),
    ("selection", ("holdout_used_for_model_selection",), True),
    ("candidate_changed", ("candidate_changed_after_holdout",), True),
    ("spec_changed", ("specification_changed_after_holdout",), True),
]


@pytest.mark.parametrize("case", EVIDENCE_TAMPERS, ids=[c[0] for c in EVIDENCE_TAMPERS])
def test_evidence_tamper_matrix_rejected(tmp_path, monkeypatch, case):
    _, path, value = case
    fx = full_artifact_set(tmp_path, monkeypatch)
    _apply_tamper(fx, "evidence", path, value)
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


BUNDLE_TAMPERS = [
    ("dataset_slug", ("dataset_slug",), "tampered"),
    ("problem_type", ("problem_type",), "regression"),
    ("forecasting_mode", ("forecasting_mode",), "multivariate"),
    ("candidate", ("model", "selected_candidate_id"), "naive_last_value"),
    ("family", ("model", "selected_family"), "NaiveLastValue"),
    ("spec", ("model", "selected_specification", "trend_policy"), "quadratic"),
    ("model_path", ("model", "model_artifact_path"), "other.joblib"),
    ("model_sha", ("model", "model_artifact_sha256"), "0" * 64),
    ("model_fp", ("model", "model_state_semantic_fingerprint"), "0" * 64),
    ("model_format", ("model", "model_artifact_format"), "pickle"),
    ("model_kind", ("model", "model_artifact_kind"), "sklearn_pipeline"),
    ("target_column", ("target", "column"), "temp"),
    ("target_semantics", ("target", "semantics"), "x"),
    ("target_unit", ("target", "unit"), "C"),
    ("target_dtype", ("target", "dtype"), "string"),
    ("frequency", ("time", "expected_frequency"), "D"),
    ("seasonal_period", ("time", "seasonal_period"), 4),
    ("time_horizon", ("time", "forecast_horizon"), 6),
    ("time_origin", ("time", "forecast_origin"), "first period"),
    ("input_kind", ("input_contract", "kind"), "other"),
    ("input_repr", ("input_contract", "representation"), "other"),
    ("input_cols", ("input_contract", "required_columns"), ["period"]),
    ("input_monthly", ("input_contract", "period_requirements", "monthly"), False),
    ("input_monotonic", ("input_contract", "period_requirements", "monotonic_increasing"), False),
    ("input_unique", ("input_contract", "period_requirements", "unique"), False),
    ("input_contiguous", ("input_contract", "period_requirements", "contiguous"), False),
    ("input_dtype", ("input_contract", "temperature_requirements", "dtype"), "string"),
    ("input_no_missing", ("input_contract", "temperature_requirements", "no_missing"), False),
    ("input_exog", ("input_contract", "exogenous_predictors"), "some"),
    ("input_refit", ("input_contract", "refit_on_input"), True),
    ("input_hist_required", ("input_contract", "history_required"), False),
    ("input_hist_role", ("input_contract", "history_role"), "other"),
    ("input_refit_flag", ("input_contract", "historical_values_used_to_refit"), True),
    ("input_update_flag", ("input_contract", "historical_values_used_to_update_coefficients"), True),
    ("input_origin", ("input_contract", "forecast_origin"), "first period"),
    ("input_min_hist", ("input_contract", "minimum_history_observations"), 0),
    ("input_supplied_end", ("input_contract", "supplied_history_end_minimum"), "1900-01"),
    ("output_rows", ("output_contract", "row_count"), 6),
    ("output_cols", ("output_contract", "columns"), ["period"]),
    ("output_periods", ("output_contract", "periods"), "other"),
    ("output_dtype", ("output_contract", "forecast_dtype"), "string"),
    ("output_unit", ("output_contract", "unit"), "C"),
    ("verify_sha", ("security", "verify_joblib_sha_before_load"), False),
    ("trusted_local", ("security", "trusted_local_artifact_only"), False),
    ("ready_demo", ("readiness", "inference_demo_ready"), False),
    ("ready_trained", ("readiness", "final_model_trained"), False),
    ("ready_frozen", ("readiness", "model_frozen"), False),
    ("ready_operational", ("readiness", "operational_modeling_ready"), True),
]


@pytest.mark.parametrize("case", BUNDLE_TAMPERS, ids=[c[0] for c in BUNDLE_TAMPERS])
def test_bundle_tamper_matrix_rejected(tmp_path, monkeypatch, case):
    _, path, value = case
    fx = full_artifact_set(tmp_path, monkeypatch)
    _apply_tamper(fx, "bundle", path, value)
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


HANDOFF_TAMPERS = [
    ("dataset_slug", ("dataset_slug",), "tampered"),
    ("problem_type", ("problem_type",), "regression"),
    ("forecasting_mode", ("forecasting_mode",), "multivariate"),
    ("upstream_path", ("upstream", "model_selection_handoff", "path"), "other.json"),
    ("upstream_sha", ("upstream", "model_selection_handoff", "sha256"), "0" * 64),
    ("upstream_fp", ("upstream", "model_selection_handoff", "semantic_fingerprint"), "0" * 64),
    ("ref_path", ("final_references", "final-model-manifest.json", "path"), "other.json"),
    ("ref_sha", ("final_references", "final-model-manifest.json", "byte_sha256"), "0" * 64),
    ("ref_fp", ("final_references", "final-model-manifest.json", "semantic_fingerprint"), "0" * 64),
    ("candidate", ("selected_model", "candidate_id"), "naive_last_value"),
    ("role", ("selected_model", "role"), "primary_baseline"),
    ("family", ("selected_model", "family"), "NaiveLastValue"),
    ("spec", ("selected_model", "selected_specification", "trend_policy"), "quadratic"),
    ("training_scope", ("training", "scope"), "full series including 1939"),
    ("training_rows", ("training", "rows"), 240),
    ("training_fit_count", ("training", "fit_count"), 2),
    ("eval_origin", ("final_evaluation", "origin"), "1939-01"),
    ("eval_period_start", ("final_evaluation", "period", "start"), "1940-01"),
    ("eval_period_end", ("final_evaluation", "period", "end"), "1940-12"),
    ("eval_horizon", ("final_evaluation", "horizon"), 6),
    ("eval_count", ("final_evaluation", "evaluation_count"), 2),
    ("eval_metrics", ("final_evaluation", "metrics", "mae"), 999.0),
    ("inference_input", ("inference", "input_contract", "kind"), "other"),
    ("inference_output", ("inference", "output_contract", "row_count"), 6),
    ("inference_ref", ("inference", "model_artifact_reference", "byte_sha256"), "0" * 64),
    ("boundary_frozen", ("boundary", "model_frozen_before_holdout_open"), False),
    ("boundary_evaluated", ("boundary", "holdout_evaluated"), False),
    ("boundary_count", ("boundary", "holdout_evaluation_count"), 2),
    ("boundary_adjustment", ("boundary", "holdout_used_for_adjustment"), True),
    ("boundary_no_retune", ("boundary", "no_retuning_after_final_evaluation"), False),
    ("boundary_no_change", ("boundary", "no_model_change_after_final_evaluation"), False),
    ("readiness_operational", ("readiness", "operational_modeling_ready"), True),
    ("instructions_truncated", ("notebook_05_instructions",), ["validate final-model handoff"]),
]


@pytest.mark.parametrize("case", HANDOFF_TAMPERS, ids=[c[0] for c in HANDOFF_TAMPERS])
def test_handoff_tamper_matrix_rejected(tmp_path, monkeypatch, case):
    _, path, value = case
    fx = full_artifact_set(tmp_path, monkeypatch)
    _apply_tamper(fx, "handoff", path, value)
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


def test_mandatory_manifest_reassigned_attack_is_rejected(tmp_path, monkeypatch):
    """Reproduces the exact pre-fix gap: dataset_slug tampered, training scope widened
    to include 1939, and operational_modeling_ready flipped true -- all reassigned with a
    freshly recomputed semantic_fingerprint, sibling ref, and handoff re-sign."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    fx["manifest"]["dataset_slug"] = "tampered-dataset"
    fx["manifest"]["training"]["scope"] = "full series including 1939"
    fx["manifest"]["readiness"]["operational_modeling_ready"] = True
    _finalize_tamper(fx, ["manifest"])
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


def test_coordinated_attack_b_fake_handoff_training_is_rejected(tmp_path, monkeypatch):
    """Section-59-B coordinated attack: handoff.training.scope widened to include 1939,
    handoff reassigned, nothing else touched."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    fx["handoff"]["training"]["scope"] = "full series including 1939"
    _finalize_tamper(fx, [])
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


def test_coordinated_attack_c_fake_final_period_is_rejected(tmp_path, monkeypatch):
    """Section-59-C coordinated attack: evidence and handoff final-evaluation periods are
    altered consistently with each other, hashes/fingerprints updated, but the frozen
    model is untouched -- rejected against the canonical 1939-01..1939-12 contract."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    fx["evidence"]["final_evaluation_period"] = {"start": "1940-01", "end": "1940-12"}
    fx["handoff"]["final_evaluation"]["period"] = {"start": "1940-01", "end": "1940-12"}
    _finalize_tamper(fx, ["evidence"])
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


def test_coordinated_attack_d_fake_inference_contract_is_rejected(tmp_path, monkeypatch):
    """Section-59-D coordinated attack: bundle input/output contract altered, handoff
    inference block updated consistently, both reassigned -- rejected against the
    canonical bundle contract (bundle's own intrinsic contract check)."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    fx["bundle"]["input_contract"]["minimum_history_observations"] = 0
    fx["handoff"]["inference"]["input_contract"]["minimum_history_observations"] = 0
    _finalize_tamper(fx, ["bundle"])
    with pytest.raises(ff.ForecastingFinalizationError):
        _load(fx)


def test_resigned_final_prediction_attack_is_rejected_via_model_replay(tmp_path, monkeypatch):
    """Sections 28 and 59-E: evidence y_pred tampered with every dependent equation,
    metric, delta, and handoff metric consistently recomputed and re-signed, while the
    frozen model itself is left untouched. The frozen model's own forecast replay must
    still disagree with the (now-false) persisted predictions."""
    fx = full_artifact_set(tmp_path, monkeypatch)
    evidence = fx["evidence"]
    row = evidence["forecasts"][0]
    row["y_pred"] = row["y_pred"] + 5.0
    row["error"] = row["y_pred"] - row["y_true"]
    row["abs_error"] = abs(row["error"])
    row["squared_error"] = row["error"] ** 2
    row["scaled_abs_error"] = row["abs_error"] / row["seasonal_mase_scale"]
    metrics = ff._metrics_from_rows(evidence["forecasts"])
    evidence["metrics"] = metrics
    evidence["generalization_deltas"] = {
        k: metrics[k] - evidence["model_selection_reference_metrics"][k] for k in metrics
    }
    fx["handoff"]["final_evaluation"]["metrics"] = dict(metrics)
    _finalize_tamper(fx, ["evidence"])
    with pytest.raises(ff.ForecastingFinalizationError, match="replay"):
        _load(fx)


def test_final_model_manifest_manifest_intrinsic_loader_alone_rejects_identity_tamper(tmp_path):
    """The manifest auxiliary loader must reject semantically false identity/training/
    readiness content on its own (intrinsic validation), independent of the main
    cross-artifact handoff loader -- this is the exact function the pre-fix audit found
    accepted a re-signed tampered manifest."""
    fx_root = tmp_path
    c = ff.FrozenForecastingFinalizationContract(
        "nottem", "seasonal_trend_ols", "candidate", "DeterministicSeasonalTrendOLS", spec(),
        str(fx_root / "development.csv"), "d" * 64, str(fx_root / "final-holdout.csv"), "e" * 64,
        {"mae": 1., "rmse": 1.2, "seasonal_mase_12": .5},
        "artifacts/model-selection/nottem/model-selection-handoff.json", "a" * 64, "b" * 64,
        {"path": "prep.json", "sha256": "c" * 64, "schema_version": "forecasting-preparation-handoff.v1"},
    )
    g = ff.ForecastingFinalizationGuard()
    model = ff.reconstruct_and_fit_selected_forecasting_model(c, development(), g)
    manifest = ff.build_forecasting_final_model_manifest(
        c, model, g, model_path="artifacts/models/nottem/final-pipeline.joblib", model_sha256="f" * 64)
    manifest["dataset_slug"] = "tampered-dataset"
    manifest["training"]["scope"] = "full series including 1939"
    manifest["readiness"]["operational_modeling_ready"] = True
    manifest["semantic_fingerprint"] = ff.semantic_fingerprint(manifest)
    path = tmp_path / "final-model-manifest.json"
    path.write_bytes(ff._canonical(manifest))
    with pytest.raises(ff.ForecastingFinalizationError):
        ff.load_and_validate_forecasting_final_model_manifest(project_root=tmp_path, manifest_path=path)


def test_real_artifacts_pass_hardened_loader_read_only(monkeypatch):
    """Validates the real runtime artifact set (not regenerated, not modified) against
    the hardened main loader in a fresh call, asserting: zero OLS fits, zero
    final-holdout.csv reads, zero re-evaluations, and that the trusted frozen-model
    replay reproduces every persisted final-evidence prediction."""
    root = Path(__file__).parents[1]
    if not (root / "artifacts/models/nottem/final-model-handoff.json").is_file():
        pytest.skip("runtime final set unavailable")
    fit_calls = []
    monkeypatch.setattr(ff.OLS, "fit", lambda self, *a, **k: fit_calls.append(1) or pytest.fail("real-artifact load triggered a fit"))
    original_read_csv = pd.read_csv
    holdout_reads = []
    def guarded_read_csv(path, *a, **k):
        if Path(path).name == "final-holdout.csv":
            holdout_reads.append(path)
            pytest.fail("real-artifact load reopened the final holdout")
        return original_read_csv(path, *a, **k)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

    handoff = ff.load_and_validate_forecasting_final_model_handoff(project_root=root)

    assert fit_calls == [] and holdout_reads == []
    assert handoff["selected_model"]["candidate_id"] == "seasonal_trend_ols"
    assert handoff["selected_model"]["family"] == "DeterministicSeasonalTrendOLS"
    assert handoff["training"] == {"scope": "full development 1920-01 -> 1938-12", "rows": 228, "fit_count": 1}
    assert handoff["final_evaluation"]["origin"] == "1938-12"
    assert handoff["final_evaluation"]["period"] == {"start": "1939-01", "end": "1939-12"}
    assert handoff["final_evaluation"]["horizon"] == 12
    assert handoff["readiness"]["operational_modeling_ready"] is False
    assert handoff["readiness"]["inference_demo_ready"] is True

    model = ff.load_trusted_forecasting_model_from_bundle(
        project_root=root, bundle_path=root / "artifacts/models/nottem/inference-bundle.json")
    future = pd.period_range("1939-01", periods=12, freq="M", name="period")
    replay = model.forecast_periods(future).to_numpy(float)
    evidence = ff.load_and_validate_forecasting_final_test_evidence(
        project_root=root, evidence_path=root / "artifacts/models/nottem/final-test-evidence.json")
    persisted = np.array([row["y_pred"] for row in evidence["forecasts"]], dtype=float)
    assert np.allclose(replay, persisted, rtol=1e-12, atol=1e-12)
