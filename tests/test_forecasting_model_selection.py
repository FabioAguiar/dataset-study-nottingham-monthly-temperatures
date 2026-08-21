from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.forecasting_model_selection as fms


def synthetic_series() -> pd.Series:
    idx = pd.period_range("1920-01", "1938-12", freq="M", name="period")
    t = np.arange(228)
    return pd.Series(50 + .01*t + 10*np.sin(2*np.pi*t/12), index=idx, name="temperature")


def schedule(series=None):
    s = synthetic_series() if series is None else series
    rows=[]
    from scripts.forecasting_preparation import seasonal_mase_scale
    for i in range(9):
        train=s.iloc[:120+i*12]; future=s.iloc[120+i*12:132+i*12]
        rows.append({"fold":i+1,"train_start":"1920-01","train_end_forecast_origin":str(train.index[-1]),
          "training_observations":len(train),"complete_training_cycles":len(train)//12,
          "validation_start":str(future.index[0]),"validation_end":str(future.index[-1]),"validation_observations":12,
          "seasonal_mase_scale_from_training":seasonal_mase_scale(train)})
    return {"mode":"expanding_window","initial_training_months":120,"forecast_horizon":12,
      "origin_step_months":12,"fold_count":9,"validation_forecast_count":108,
      "validation_targets_overlap":False,"schedule":rows}


def frame():
    s=synthetic_series(); return pd.DataFrame({"period":s.index.astype(str),"temperature":s.values})


def metric_result(cid, mae, mase=1, rmse=2, std=3, long=4, rank=0, eligible=True):
    return {"candidate_id":cid,"eligible":eligible,"pooled_mae":mae,"pooled_seasonal_mase":mase,
            "pooled_rmse":rmse,"fold_mae_std":std,"long_horizon_mae_h7_h12":long,"complexity_rank":rank}


def test_catalog_is_exact_frozen_and_serializable():
    c=fms.frozen_specification_catalog(); fms.validate_catalog(c)
    assert len(c)==10 and sum(x["role"].endswith("baseline") for x in c)==2
    assert len({x["candidate_id"] for x in c})==len({x["complexity_rank"] for x in c})==10
    assert "random" not in json.dumps(c).lower() and "holdout" not in json.dumps(c).lower()


def test_baselines_are_training_only_and_exact():
    s=synthetic_series().iloc[:120]; future=pd.period_range(s.index[-1]+1,periods=12,freq="M",name="period")
    np.testing.assert_array_equal(fms.seasonal_naive_forecast(s,future),s.iloc[-12:])
    assert (fms.naive_last_value_forecast(s,future)==s.iloc[-1]).all()
    changed=synthetic_series().copy(); changed.loc[future] += 999
    np.testing.assert_array_equal(fms.seasonal_naive_forecast(s,future),fms.seasonal_naive_forecast(s,future))


@pytest.mark.parametrize("cid",["seasonal_trend_ols","holt_winters_additive_no_trend","holt_winters_additive_trend","holt_winters_additive_damped_trend","autoreg_lag_1_12_ct","autoreg_lag_1_2_12_ct","sarima_100_100_12","sarima_100_011_12"])
def test_learned_forecasters_shape_finite_index_repeatable(cid):
    spec=next(x for x in fms.frozen_specification_catalog() if x["candidate_id"]==cid)
    train=synthetic_series().iloc[:180]; future=pd.period_range(train.index[-1]+1,periods=12,freq="M",name="period")
    with pytest.warns(Warning) if cid.startswith("sarima") else __import__('contextlib').nullcontext():
        a,_=fms.forecast_specification(spec,train,future)
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore'); b,_=fms.forecast_specification(spec,train,future)
    assert len(a)==12 and a.index.equals(future) and np.isfinite(a).all()
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-9)
    assert train.index[-1] < future[0]


def test_ols_future_calendar_design_changes_months():
    train=synthetic_series().iloc[:120]; future=pd.period_range(train.index[-1]+1,periods=12,freq="M",name="period")
    pred=fms.seasonal_trend_ols_forecast(train,future)
    assert pred.nunique()>6 and pred.index.month.tolist()==list(range(1,13))


def test_backtest_geometry_and_metrics():
    spec=fms.frozen_specification_catalog()[0]
    rows,audit=fms.backtest_specification(spec,synthetic_series(),schedule())
    assert len(rows)==108 and rows.fold.nunique()==9 and set(rows.horizon)==set(range(1,13))
    assert rows.forecast_origin.min()=="1929-12" and rows.forecast_origin.max()=="1937-12"
    assert rows.forecast_period.min()=="1930-01" and rows.forecast_period.max()=="1938-12"
    assert (pd.PeriodIndex(rows.forecast_period,freq="M") == pd.PeriodIndex(rows.forecast_origin,freq="M") + rows.horizon.to_numpy()).all()
    result=fms.aggregate_metrics(spec,rows,audit)
    assert result["complete"] and result["folds_completed"]==9 and len(result["horizon_mae"])==12
    assert result["pooled_rmse"]==pytest.approx(np.sqrt(rows.squared_error.mean()))
    assert result["fold_mae_std"]==pytest.approx(np.std([x["fold_mae"] for x in result["fold_summaries"]],ddof=0))
    assert result["short_horizon_mae_h1_h6"]==pytest.approx(rows[rows.horizon<=6].abs_error.mean())


def test_pooled_rmse_is_not_mean_fold_rmse():
    errors=np.concatenate([np.repeat(float(fold),12) for fold in range(1,10)])
    rows=pd.DataFrame({"fold":np.repeat(range(1,10),12),"abs_error":errors,
      "squared_error":errors**2,"scaled_abs_error":errors,"horizon":list(range(1,13))*9})
    audit=[{"failure":None,"warnings":[]} for _ in range(9)]
    r=fms.aggregate_metrics(fms.frozen_specification_catalog()[0],rows,audit)
    assert r["pooled_rmse"] != pytest.approx(np.mean([x["fold_rmse"] for x in r["fold_summaries"]]))


@pytest.mark.parametrize("field,values,winner",[
 ("mae",(1,1.06),"a"),("mase",(1,2),"a"),("rmse",(1,2),"a"),("std",(1,2),"a"),("long",(1,2),"a"),("rank",(0,1),"a")])
def test_selection_tie_breaks(field,values,winner):
    a=metric_result("a",1,1,1,1,1,0); b=metric_result("b",1.01,1,1,1,1,0)
    key={"mae":"pooled_mae","mase":"pooled_seasonal_mase","rmse":"pooled_rmse","std":"fold_mae_std","long":"long_horizon_mae_h7_h12","rank":"complexity_rank"}[field]
    a[key],b[key]=values
    assert fms.select_winner([a,b])["selected_candidate_id"]==winner


def test_selection_lexical_and_baseline_can_win():
    a=metric_result("seasonal_naive_12",1,rank=0); b=metric_result("z",1,rank=0)
    assert fms.select_winner([b,a])["selected_candidate_id"]=="seasonal_naive_12"


def test_incomplete_never_ranks():
    assert fms.select_winner([metric_result("bad",0,eligible=False),metric_result("ok",2)])["selected_candidate_id"]=="ok"


def test_failure_policy_and_baseline_block(monkeypatch):
    def fail(*args): raise RuntimeError("fit failed")
    monkeypatch.setattr(fms,"forecast_specification",fail)
    candidate=fms.frozen_specification_catalog()[2]
    rows,audit=fms.backtest_specification(candidate,synthetic_series(),schedule())
    assert rows.empty and not fms.aggregate_metrics(candidate,rows,audit)["eligible"]
    with pytest.raises(fms.ForecastingModelSelectionError,match="Baseline"):
        fms.backtest_specification(fms.frozen_specification_catalog()[0],synthetic_series(),schedule())


def test_nonfinite_and_wrong_length_are_ineligible(monkeypatch):
    future_result=lambda training,future:(pd.Series([np.nan]*12,index=future),None)
    monkeypatch.setattr(fms,"forecast_specification",future_result)
    spec=fms.frozen_specification_catalog()[2]
    rows,audit=fms.backtest_specification(spec,synthetic_series(),schedule())
    assert not fms.aggregate_metrics(spec,rows,audit)["eligible"]


def test_development_and_mase_tamper_rejected():
    bad=frame(); bad.loc[0,"period"]="1939-01"
    with pytest.raises(fms.ForecastingModelSelectionError): fms.validate_development_series(bad)
    bc=schedule(); bc["schedule"][0]["seasonal_mase_scale_from_training"] += .01
    with pytest.raises(fms.ForecastingModelSelectionError,match="authentication"):
        fms.fold_slices(synthetic_series(),bc)


def test_notebook_structure_and_holdout_safety():
    p=Path(__file__).parents[1]/"notebooks/03_model_selection_and_evaluation.ipynb"
    nb=json.loads(p.read_text()); code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    assert "load_and_validate_forecasting_preparation_handoff" in code
    assert "forecasting_model_selection" in code and "write_forecasting_model_selection_artifacts" in code
    assert "load_and_validate_forecasting_model_selection_handoff" in code
    prohibited=["train_test_split","KFold(","StratifiedKFold","shuffle=True","GridSearchCV","RandomizedSearchCV","auto_arima","Prophet","final-holdout.csv","get_rdataset"]
    assert not any(x in code for x in prohibited)


def test_path_escape_and_required_names(tmp_path):
    with pytest.raises(fms.ForecastingModelSelectionError): fms._confined(tmp_path,"../escape")
    with pytest.raises(fms.ForecastingModelSelectionError,match="six"):
        fms.write_forecasting_model_selection_artifacts(project_root=tmp_path,artifacts={})


def test_runtime_versions_are_non_scientific_metadata():
    left={"schema_version":"x.v1","runtime_versions":{"python":"3.11","pandas":"2.2"},"result":1}
    right={"schema_version":"x.v1","runtime_versions":{"python":"3.12","pandas":"3.0"},"result":1}
    assert fms.semantic_fingerprint(left)==fms.semantic_fingerprint(right)


def test_runtime_metadata_difference_reuses_valid_artifacts(tmp_path, monkeypatch):
    base=tmp_path/"artifacts/model-selection/nottem"; base.mkdir(parents=True)
    artifacts={name:({"schema_version":"x.v1","value":1} if name.endswith(".json") else pd.DataFrame({"value":[1]})) for name in fms.REQUIRED_FILENAMES}
    artifacts["model-selection-manifest.json"]["runtime_versions"]={"python":"old"}
    # Materialize once without invoking the real cross-artifact loader, whose
    # preparation boundary is intentionally outside this focused writer test.
    first=fms.write_forecasting_model_selection_artifacts(project_root=tmp_path,artifacts=artifacts)
    assert first.status=="created"
    changed=json.loads(json.dumps({k:v for k,v in artifacts.items() if k!="cross-validation-results.csv"}))
    changed["cross-validation-results.csv"]=artifacts["cross-validation-results.csv"].copy()
    changed["model-selection-manifest.json"]["runtime_versions"]={"python":"new"}
    monkeypatch.setattr(fms,"load_and_validate_forecasting_model_selection_handoff",lambda **kwargs:{})
    monkeypatch.setattr(fms,"_equivalent_evidence_csv",lambda *args,**kwargs:(True,"equivalent"))
    second=fms.write_forecasting_model_selection_artifacts(project_root=tmp_path,artifacts=changed)
    assert second.status=="reused_equivalent"


def test_evidence_csv_equivalence_ignores_float_text_rendering(tmp_path):
    columns=["candidate_id","role","family","fold","forecast_origin","forecast_period","horizon",
      "y_true","y_pred","error","abs_error","squared_error","seasonal_mase_scale","scaled_abs_error",
      "fit_status","forecast_status","warning_count"]
    row=["x","candidate","Family",1,"1929-12","1930-01",1,1.0,1.0000000000001,
      1e-13,1e-13,1e-26,2.0,5e-14,"success","success",0]
    current=pd.DataFrame([row],columns=columns)
    path=tmp_path/"evidence.csv"
    path.write_text(current.to_csv(index=False,float_format="%.14g"),encoding="utf-8")
    assert fms._equivalent_evidence_csv(path,current)[0]


def test_evidence_csv_equivalence_rejects_scientific_change(tmp_path):
    current=pd.DataFrame({"candidate_id":["x"],"fold":[1],"horizon":[1],"warning_count":[0],
      "y_true":[1.0],"y_pred":[2.0],"error":[1.0],"abs_error":[1.0],"squared_error":[1.0],
      "seasonal_mase_scale":[2.0],"scaled_abs_error":[0.5]})
    path=tmp_path/"evidence.csv"; changed=current.copy(); changed.loc[0,"y_pred"]=2.01
    changed.to_csv(path,index=False)
    assert not fms._equivalent_evidence_csv(path,current)[0]


def test_evidence_csv_equivalence_is_row_order_independent(tmp_path):
    current=pd.DataFrame({"candidate_id":["b","a"],"role":["candidate"]*2,"family":["F"]*2,
      "fold":[1,1],"forecast_origin":["1929-12"]*2,"forecast_period":["1930-01"]*2,
      "horizon":[1,1],"y_true":[1.,2.],"y_pred":[1.,2.],"error":[0.,0.],
      "abs_error":[0.,0.],"squared_error":[0.,0.],"seasonal_mase_scale":[2.,2.],
      "scaled_abs_error":[0.,0.],"fit_status":["success"]*2,"forecast_status":["success"]*2,
      "warning_count":[0,0]})
    path=tmp_path/"evidence.csv"; current.iloc[::-1].to_csv(path,index=False)
    assert fms._equivalent_evidence_csv(path,current)[0]


def test_ineligible_partial_forecast_output_is_non_selection_metadata(tmp_path):
    current=pd.DataFrame({"candidate_id":["failed","eligible"],"role":["candidate"]*2,"family":["F"]*2,
      "fold":[1,1],"forecast_origin":["1929-12"]*2,"forecast_period":["1930-01"]*2,
      "horizon":[1,1],"y_true":[1.,1.],"y_pred":[2.,2.],"error":[1.,1.],
      "abs_error":[1.,1.],"squared_error":[1.,1.],"seasonal_mase_scale":[2.,2.],
      "scaled_abs_error":[.5,.5],"fit_status":["success"]*2,"forecast_status":["success"]*2,
      "warning_count":[0,0]})
    persisted=current.copy(); persisted.loc[persisted.candidate_id=="failed",["y_pred","error","abs_error","squared_error","scaled_abs_error"]]=[9.,8.,8.,64.,4.]
    path=tmp_path/"evidence.csv"; persisted.to_csv(path,index=False)
    assert fms._equivalent_evidence_csv(path,current,ineligible_candidate_ids=["failed"])[0]
    assert not fms._equivalent_evidence_csv(path,current)[0]


def test_full_pipeline_never_reads_holdout(monkeypatch):
    original=pd.read_csv
    def guarded(path,*a,**k):
        if "final-holdout.csv" in str(path): raise AssertionError("holdout opened")
        return original(path,*a,**k)
    monkeypatch.setattr(pd,"read_csv",guarded)
    evidence,summaries,audits=fms.run_all_backtests(frame(),schedule())
    assert evidence.forecast_period.max()=="1938-12" and fms.select_winner(summaries)["selected_candidate_id"]
