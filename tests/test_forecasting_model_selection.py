from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _authenticated_baseline_evidence():
    series=synthetic_series(); contract=schedule()
    rows,audits=fms.backtest_specification(
        fms.frozen_specification_catalog()[0],series,contract)
    preparation=SimpleNamespace(
        development=frame(),backtesting_contract=contract)
    return rows,audits,preparation


@pytest.mark.parametrize("column",[
    "error","abs_error","squared_error","scaled_abs_error",
])
def test_oos_row_equation_tampering_is_rejected(column):
    rows,_,preparation=_authenticated_baseline_evidence()
    rows.loc[0,column] += .01
    with pytest.raises(fms.ForecastingModelSelectionError,match=column):
        fms._authenticate_oos_evidence(
            rows,preparation.development,preparation,
            fms.frozen_specification_catalog())


def test_oos_truth_attack_rejected_even_with_consistent_derived_columns():
    rows,_,preparation=_authenticated_baseline_evidence()
    rows.loc[0,"y_true"] += 1
    error=rows.loc[0,"y_pred"]-rows.loc[0,"y_true"]
    rows.loc[0,["error","abs_error","squared_error","scaled_abs_error"]]=[
        error,abs(error),error**2,
        abs(error)/rows.loc[0,"seasonal_mase_scale"]]
    with pytest.raises(fms.ForecastingModelSelectionError,match="development"):
        fms._authenticate_oos_evidence(
            rows,preparation.development,preparation,
            fms.frozen_specification_catalog())


def test_oos_mase_scale_attack_rejected_against_preparation():
    rows,_,preparation=_authenticated_baseline_evidence()
    rows.loc[0,"seasonal_mase_scale"] += 1
    rows.loc[0,"scaled_abs_error"] = (
        rows.loc[0,"abs_error"]/rows.loc[0,"seasonal_mase_scale"])
    with pytest.raises(fms.ForecastingModelSelectionError,match="preparation"):
        fms._authenticate_oos_evidence(
            rows,preparation.development,preparation,
            fms.frozen_specification_catalog())


def test_recomputed_metric_contract_has_all_scientific_formulas():
    errors=np.array([0.,2.,4.,6.]*27)
    rows=pd.DataFrame({
        "fold":np.repeat(range(1,10),12),"horizon":list(range(1,13))*9,
        "abs_error":np.abs(errors),"squared_error":errors**2,
        "scaled_abs_error":np.abs(errors)/2,
    })
    audits=[{"failure":None,"warnings":[]} for _ in range(9)]
    result=fms.aggregate_metrics(
        fms.frozen_specification_catalog()[0],rows,audits)
    assert result["pooled_mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert result["pooled_rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result["pooled_seasonal_mase"] == pytest.approx(np.mean(np.abs(errors)/2))
    assert result["fold_summaries"][0]["fold_mae"] == pytest.approx(np.mean(np.abs(errors[:12])))
    assert result["fold_summaries"][0]["fold_rmse"] == pytest.approx(np.sqrt(np.mean(errors[:12]**2)))
    assert result["fold_summaries"][0]["fold_seasonal_mase"] == pytest.approx(np.mean(np.abs(errors[:12])/2))
    assert result["fold_mae_std"] == pytest.approx(np.std(
        [x["fold_mae"] for x in result["fold_summaries"]],ddof=0))
    assert result["horizon_mae"]["1"] == pytest.approx(rows[rows.horizon==1].abs_error.mean())
    assert result["short_horizon_mae_h1_h6"] == pytest.approx(rows[rows.horizon<=6].abs_error.mean())
    assert result["long_horizon_mae_h7_h12"] == pytest.approx(rows[rows.horizon>=7].abs_error.mean())


def test_fake_winner_attack_is_rejected_by_oos_reconciliation():
    rows,audits,_=_authenticated_baseline_evidence()
    real=fms.aggregate_metrics(fms.frozen_specification_catalog()[0],rows,audits)
    fake=json.loads(json.dumps(real)); fake["pooled_mae"]=-999
    fake_selection=fms.select_winner([fake])
    replayed=fms.select_winner([real])
    assert fake_selection["selected_candidate_id"] == replayed["selected_candidate_id"]
    with pytest.raises(fms.ForecastingModelSelectionError,match="OOS evidence"):
        fms._same_science(fake,real,"candidate-results.specifications[0]")


def _synthetic_model_selection_set(root: Path, monkeypatch):
    development=frame(); series=synthetic_series(); contract=schedule(series)
    dev_path=root/"data/processed/nottem/development.csv"
    hold_path=root/"data/processed/nottem/final-holdout.csv"
    dev_path.parent.mkdir(parents=True)
    development.to_csv(dev_path,index=False)
    hold=pd.DataFrame({
        "period":pd.period_range("1939-01",periods=12,freq="M").astype(str),
        "temperature":np.arange(12,dtype=float)})
    hold.to_csv(hold_path,index=False)
    dev_ref={"path":"data/processed/nottem/development.csv",
        "sha256":fms.sha256_file(dev_path),"row_count":228,
        "start":"1920-01","end":"1938-12"}
    hold_ref={"path":"data/processed/nottem/final-holdout.csv",
        "sha256":fms.sha256_file(hold_path),"row_count":12,
        "start":"1939-01","end":"1939-12","sealed":True,
        "evaluated":False,"exposed_to_model_selection":False}
    prediction={"forecast_horizon":12,"forecasting_mode":"univariate",
        "frequency":"monthly","index_type":"monthly_period",
        "prediction_output":"point_forecast","problem_type":"time_series_forecasting",
        "source_exogenous_predictors":[],"target_classes":[],
        "target_column":"temperature","target_semantics":"Monthly mean air temperature",
        "target_unit":"degrees Fahrenheit"}
    evaluation={"forecast_intervals_required":False,
        "horizon_wise_diagnostic":"mae_h1_to_h12",
        "percentage_error_metrics":"excluded","point_forecasts_required":True,
        "primary_baseline":"seasonal_naive_12","primary_metric":"mae",
        "seasonal_mase_period":12,"secondary_baseline":"naive_last_value",
        "secondary_metrics":["rmse","seasonal_mase_12"]}
    prep_payload={"schema_version":"forecasting-preparation-handoff.v1",
        "prepared_data":{"development":dev_ref,"sealed_final_holdout":hold_ref},
        "prediction_contract":prediction,"evaluation_contract":evaluation,
        "backtesting_contract":contract}
    prep_path=root/"artifacts/preparation/nottem/preparation-handoff.json"
    prep_path.parent.mkdir(parents=True)
    prep_path.write_text(json.dumps(prep_payload,sort_keys=True),encoding="utf-8")
    authenticated=SimpleNamespace(development=development,
        prediction_contract=prediction,evaluation_contract=evaluation,
        backtesting_contract=contract,sealed_holdout_integrity=hold_ref)
    monkeypatch.setattr(
        fms,"load_and_validate_forecasting_preparation_handoff",
        lambda **kwargs: authenticated)
    catalog=fms.frozen_specification_catalog()
    offsets={spec["candidate_id"]:1.0+spec["complexity_rank"]/10 for spec in catalog}
    offsets["seasonal_trend_ols"]=.1
    offsets["seasonal_naive_12"]=.5
    offsets["naive_last_value"]=2.0
    rows=[]; audits={}
    for spec in catalog:
        cid=spec["candidate_id"]; audits[cid]=[]
        for item in contract["schedule"]:
            fold=item["fold"]; scale=item["seasonal_mase_scale_from_training"]
            audits[cid].append({"fold":fold,"fit_status":"success",
                "forecast_status":"success","warnings":[],"failure":None})
            origin=pd.Period(item["train_end_forecast_origin"],freq="M")
            for horizon in range(1,13):
                period=origin+horizon; truth=float(series.loc[period])
                error=offsets[cid]+horizon/1000
                rows.append({"candidate_id":cid,"role":spec["role"],
                    "family":spec["family"],"fold":fold,
                    "forecast_origin":str(origin),"forecast_period":str(period),
                    "horizon":horizon,"y_true":truth,"y_pred":truth+error,
                    "error":error,"abs_error":abs(error),
                    "squared_error":error**2,"seasonal_mase_scale":scale,
                    "scaled_abs_error":abs(error)/scale,
                    "fit_status":"success","forecast_status":"success",
                    "warning_count":0})
    evidence=pd.DataFrame(rows,columns=fms._OOS_COLUMNS)
    summaries=fms._recompute_candidate_summaries(evidence,catalog,audits)
    artifacts=fms.build_artifacts(project_root=root,
        preparation_handoff_path="artifacts/preparation/nottem/preparation-handoff.json",
        preparation_payload=prep_payload,evidence=evidence,
        summaries=summaries,audits=audits)
    fms.write_forecasting_model_selection_artifacts(
        project_root=root,artifacts=artifacts)
    path=root/"artifacts/model-selection/nottem/model-selection-handoff.json"
    assert fms.load_and_validate_forecasting_model_selection_handoff(
        project_root=root,handoff_path=path.relative_to(root))
    return path


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.write_bytes(fms._canonical(payload))


def _resign_set(handoff_path: Path, changed_sibling: str | None=None):
    handoff=_json(handoff_path)
    if changed_sibling:
        sibling_path=handoff_path.parent/changed_sibling
        payload=_json(sibling_path)
        _write_json(sibling_path,payload)
        ref=handoff["sibling_artifacts"][changed_sibling]
        ref["sha256"]=fms.sha256_file(sibling_path)
        ref["semantic_fingerprint"]=fms.semantic_fingerprint(payload)
    handoff["semantic_fingerprint"]=fms.semantic_fingerprint(handoff)
    _write_json(handoff_path,handoff)


def _tamper_sibling(handoff_path: Path, filename: str, mutation):
    path=handoff_path.parent/filename; payload=_json(path)
    mutation(payload); _write_json(path,payload)
    _resign_set(handoff_path,filename)


def _tamper_handoff(handoff_path: Path, mutation):
    payload=_json(handoff_path); mutation(payload)
    payload["semantic_fingerprint"]=fms.semantic_fingerprint(payload)
    _write_json(handoff_path,payload)


def _must_reject(root: Path, handoff_path: Path):
    with pytest.raises(fms.ForecastingModelSelectionError):
        fms.load_and_validate_forecasting_model_selection_handoff(
            project_root=root,handoff_path=handoff_path.relative_to(root))


def test_fake_winner_attack_fully_resigned_keeps_oos_unchanged(tmp_path,monkeypatch):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    csv_path=path.parent/"cross-validation-results.csv"
    original_csv=csv_path.read_bytes()
    candidate_path=path.parent/"candidate-results.json"
    candidate=_json(candidate_path)
    fake=next(x for x in candidate["specifications"]
              if x["candidate_id"]=="naive_last_value")
    for key in ("pooled_mae","pooled_rmse","pooled_seasonal_mase",
                "fold_mae_std","short_horizon_mae_h1_h6",
                "long_horizon_mae_h7_h12"):
        fake[key]=0.0
    fake["horizon_mae"]={str(i):0.0 for i in range(1,13)}
    fake["delta_mae_vs_seasonal_naive"]=-999.0
    fake["relative_mae_improvement_pct"]=100.0
    fake["deterministic_rank"]=1
    real_id=candidate["selection"]["selected_candidate_id"]
    real=next(x for x in candidate["specifications"] if x["candidate_id"]==real_id)
    real["deterministic_rank"]=2
    selection={"best_raw_mae":0.0,
        "practical_tie_tolerance_f":fms.PRACTICAL_MAE_TIE_TOLERANCE_F,
        "finalists":["naive_last_value"],"tie_break_order":fms.TIE_BREAK_ORDER,
        "selected_candidate_id":"naive_last_value",
        "ranking":["naive_last_value"]+
            [x for x in candidate["selection"]["ranking"] if x!="naive_last_value"]}
    candidate["selection"]=selection; _write_json(candidate_path,candidate)
    validation_path=path.parent/"validation-evidence.json"
    validation=_json(validation_path)
    validation["pooled_summaries"]=candidate["specifications"]
    validation["selected_candidate_id"]="naive_last_value"
    _write_json(validation_path,validation)
    analysis_path=path.parent/"selection-analysis.json"
    analysis=_json(analysis_path); analysis["selection"]=selection
    analysis["baseline_comparison"]=[{"candidate_id":x["candidate_id"],
        "delta_mae_vs_seasonal_naive":x.get("delta_mae_vs_seasonal_naive"),
        "relative_mae_improvement_pct":x.get("relative_mae_improvement_pct")}
        for x in candidate["specifications"]]
    analysis["stability_comparison"]={x["candidate_id"]:x.get("fold_mae_std")
        for x in candidate["specifications"]}
    analysis["long_horizon_comparison"]={x["candidate_id"]:
        x.get("long_horizon_mae_h7_h12") for x in candidate["specifications"]}
    analysis["scientific_interpretation"]="naive_last_value is selected by the frozen practical-tie rule."
    _write_json(analysis_path,analysis)
    handoff=_json(path); spec=fms.frozen_specification_catalog()[1]
    handoff.update({"selected_candidate_id":"naive_last_value",
        "selected_role":spec["role"],"selected_family":spec["family"],
        "selected_specification":spec,"selection":selection})
    handoff["evaluation"].update({"selected_pooled_metrics":{
        "mae":0.0,"rmse":0.0,"seasonal_mase_12":0.0},
        "selected_fold_mae_std":0.0,
        "selected_horizon_mae":fake["horizon_mae"],
        "delta_mae_vs_primary_baseline":-999.0})
    handoff["final_training_instructions"]["fit_once_on_full_development_if_required"]=False
    _write_json(path,handoff)
    for name in ("candidate-results.json","validation-evidence.json",
                 "selection-analysis.json"):
        _resign_set(path,name)
    _resign_set(path)
    assert csv_path.read_bytes()==original_csv
    _must_reject(tmp_path,path)


_INSTRUCTION_TAMPERS=[
    ("notebook","other.ipynb"),
    ("training_scope","full series 1920-01 -> 1939-12"),
    ("reconstruct_exact_selected_specification",False),
    ("fit_once_on_full_development_if_required",False),
    ("freeze_before_final_holdout_open",False),("final_forecast_horizon",6),
    ("final_evaluation_period","1938-01 -> 1938-12"),
    ("open_holdout_only_after_final_spec_fit",False),
    ("evaluate_holdout_once",False),("do_not_retune",False),
    ("do_not_change_candidate",False),("do_not_change_hyperparameters",False),
    ("do_not_change_transform_policy",False),("target_scale","Celsius"),
]


@pytest.mark.parametrize("field,value",_INSTRUCTION_TAMPERS)
def test_resigned_final_training_instruction_tampering(
        tmp_path,monkeypatch,field,value):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_handoff(path,lambda p:p["final_training_instructions"].__setitem__(
        field,value))
    _must_reject(tmp_path,path)


_READY_TRUE=["preparation_handoff_validated","development_only_model_selection",
    "temporal_backtesting_completed","frozen_backtesting_contract_respected",
    "baselines_evaluated","candidate_catalog_frozen_before_evaluation",
    "candidate_models_evaluated","selected_specification_frozen",
    "metric_contract_frozen","model_selection_handoff_reloadable",
    "final_holdout_sealed","final_model_training_ready"]
_READY_FALSE=["final_holdout_evaluated","final_model_trained",
    "model_artifact_materialized","model_bundle_materialized",
    "operational_modeling_ready"]


@pytest.mark.parametrize("field",_READY_TRUE)
def test_resigned_required_true_readiness_tampering(tmp_path,monkeypatch,field):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_handoff(path,lambda p:p["readiness"].__setitem__(field,False))
    _must_reject(tmp_path,path)


@pytest.mark.parametrize("field",_READY_FALSE)
def test_resigned_required_false_readiness_tampering(tmp_path,monkeypatch,field):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_handoff(path,lambda p:p["readiness"].__setitem__(field,True))
    _must_reject(tmp_path,path)


def test_resigned_premature_model_artifact_is_rejected(tmp_path,monkeypatch):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_handoff(path,lambda p:p.__setitem__(
        "model_artifact",{"path":"artifacts/models/nottem/model.joblib"}))
    _must_reject(tmp_path,path)


_MANIFEST_CASES=["preparation_handoff","development_integrity",
    "sealed_holdout_integrity_metadata_only","prediction_contract",
    "evaluation_contract","backtesting_contract","baselines",
    "frozen_candidate_catalog","primary_metric","tolerance","tie_break_order",
    "artifact_paths","final_holdout_sealed","final_holdout_evaluated",
    "final_model_trained","model_artifact_materialized"]


@pytest.mark.parametrize("case",_MANIFEST_CASES)
def test_resigned_manifest_tampering(tmp_path,monkeypatch,case):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    def mutate(p):
        if case in ("preparation_handoff","development_integrity",
                    "sealed_holdout_integrity_metadata_only"):
            p[case]["path"]="other"
        elif case in ("prediction_contract","evaluation_contract",
                      "backtesting_contract"):
            p[case][next(iter(p[case]))]="other"
        elif case in ("baselines","frozen_candidate_catalog"):
            p[case][0]["family"]="Other"
        elif case=="primary_metric":
            p["selection_contract"]["primary_metric"]="rmse"
        elif case=="tolerance":
            p["selection_contract"]["practical_mae_tie_tolerance_f"]=1.0
        elif case=="tie_break_order":
            p["selection_contract"]["tie_break_order"]=[]
        elif case=="artifact_paths":
            p["artifact_paths"]["candidate-results.json"]="other"
        elif case=="final_holdout_sealed": p[case]=False
        else: p[case]=True
    _tamper_sibling(path,"model-selection-manifest.json",mutate)
    _must_reject(tmp_path,path)


_VALIDATION_CASES=["evaluation_protocol","fold_count",
    "forecasts_per_complete_specification","metric_formulas",
    "seasonal_mase_scaling_policy","pooled_summaries",
    "selected_candidate_id","max_validation_period",
    "final_holdout_evaluated","fold_audits"]


@pytest.mark.parametrize("case",_VALIDATION_CASES)
def test_resigned_validation_evidence_tampering(tmp_path,monkeypatch,case):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    def mutate(p):
        if case=="evaluation_protocol": p[case]="random_cv"
        elif case in ("fold_count","forecasts_per_complete_specification"):
            p[case]+=1
        elif case=="metric_formulas": p[case]["mae"]="wrong"
        elif case=="seasonal_mase_scaling_policy": p[case]="wrong"
        elif case=="pooled_summaries": p[case][0]["pooled_mae"]+=1
        elif case=="selected_candidate_id": p[case]="naive_last_value"
        elif case=="max_validation_period": p[case]="1939-01"
        elif case=="final_holdout_evaluated": p[case]=True
        else: p[case]["seasonal_naive_12"][0]["fit_status"]="failed"
    _tamper_sibling(path,"validation-evidence.json",mutate)
    _must_reject(tmp_path,path)


_ANALYSIS_CASES=["catalog_frozen_before_evaluation","selection",
    "baseline_comparison","stability_comparison","long_horizon_comparison",
    "failed_or_ineligible","no_1939_values_influenced_selection",
    "scientific_interpretation"]


@pytest.mark.parametrize("case",_ANALYSIS_CASES)
def test_resigned_selection_analysis_tampering(tmp_path,monkeypatch,case):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    def mutate(p):
        if case=="catalog_frozen_before_evaluation": p[case]=False
        elif case=="selection": p[case]["selected_candidate_id"]="naive_last_value"
        elif case=="baseline_comparison": p[case][0]["delta_mae_vs_seasonal_naive"]=1
        elif case=="stability_comparison":
            p[case]["seasonal_naive_12"]+=1
        elif case=="long_horizon_comparison":
            p[case]["seasonal_naive_12"]+=1
        elif case=="failed_or_ineligible": p[case]=["seasonal_naive_12"]
        elif case=="no_1939_values_influenced_selection": p[case]=False
        else: p[case]="naive_last_value is the selected winner."
    _tamper_sibling(path,"selection-analysis.json",mutate)
    _must_reject(tmp_path,path)


_SPEC_FIELDS=["candidate_id","family","role","complexity_rank","constructor",
    "fixed_hyperparameters","preprocessing_policy","differencing_policy",
    "trend_policy","seasonal_policy","multi_step_strategy","exogenous_policy"]


@pytest.mark.parametrize("field",["selected_candidate_id","selected_role",
    "selected_family","selected_specification"]+_SPEC_FIELDS)
def test_resigned_selected_winner_contract_tampering(
        tmp_path,monkeypatch,field):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    def mutate(p):
        if field=="selected_candidate_id": p[field]="naive_last_value"
        elif field=="selected_role": p[field]="primary_baseline"
        elif field=="selected_family": p[field]="Other"
        elif field=="selected_specification": p[field]={}
        elif field=="complexity_rank": p["selected_specification"][field]=999
        elif field=="fixed_hyperparameters":
            p["selected_specification"][field]={"wrong":True}
        else: p["selected_specification"][field]="wrong"
    _tamper_handoff(path,mutate)
    _must_reject(tmp_path,path)


_METRIC_CASES=["mae","rmse","seasonal_mase_12","selected_fold_mae_std",
    "selected_horizon_mae","seasonal_naive_metrics",
    "delta_mae_vs_primary_baseline"]


@pytest.mark.parametrize("case",_METRIC_CASES)
def test_resigned_selected_metric_tampering(tmp_path,monkeypatch,case):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    def mutate(p):
        evaluation=p["evaluation"]
        if case in ("mae","rmse","seasonal_mase_12"):
            evaluation["selected_pooled_metrics"][case]+=1
        elif case=="selected_fold_mae_std": evaluation[case]+=1
        elif case=="selected_horizon_mae": evaluation[case]["1"]+=1
        elif case=="seasonal_naive_metrics":
            evaluation[case]["pooled_mae"]+=1
        else: evaluation[case]+=1
    _tamper_handoff(path,mutate)
    _must_reject(tmp_path,path)


@pytest.mark.parametrize("field,value",[
    ("mode","rolling"),("initial_training_months",121),
    ("forecast_horizon",11),("origin_step_months",11),
    ("fold_count",8),("validation_forecast_count",107)])
def test_resigned_backtest_summary_tampering(
        tmp_path,monkeypatch,field,value):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_handoff(path,lambda p:p["backtest"].__setitem__(field,value))
    _must_reject(tmp_path,path)


@pytest.mark.parametrize("field,value",[
    ("practical_mae_tie_tolerance_f",1.0),("tie_break_order",[])])
def test_resigned_selection_tampering(tmp_path,monkeypatch,field,value):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    _tamper_sibling(path,"candidate-results.json",
        lambda p:p["selection"].__setitem__(field,value))
    _must_reject(tmp_path,path)


def test_training_scope_is_derived_from_authenticated_upstream_metadata(
        tmp_path,monkeypatch):
    path=_synthetic_model_selection_set(tmp_path,monkeypatch)
    prep_path=tmp_path/"artifacts/preparation/nottem/preparation-handoff.json"
    prep=_json(prep_path); prep["prepared_data"]["development"]["start"]="1921-01"
    _write_json(prep_path,prep)
    handoff=_json(path)
    handoff["upstream_preparation_handoff"]["sha256"]=fms.sha256_file(prep_path)
    handoff["semantic_fingerprint"]=fms.semantic_fingerprint(handoff)
    _write_json(path,handoff)
    _must_reject(tmp_path,path)
