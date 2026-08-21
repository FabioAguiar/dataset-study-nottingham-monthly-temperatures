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
