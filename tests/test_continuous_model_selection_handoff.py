import copy
import json
from pathlib import Path
import pytest

from scripts.select_models import (
    ArtifactConflictError,
    REGRESSION_ARTIFACT_FILENAMES,
    load_and_validate_model_selection_handoff,
    validate_regression_model_selection_contract,
    write_regression_model_selection_artifacts,
)
import scripts.select_models as sm
from tests._continuous_model_selection_fixtures import (
    SELECTION_HANDOFF, build_synthetic_preparation, synthetic_selection_artifacts,
)


def _artifacts(root):
    return synthetic_selection_artifacts(build_synthetic_preparation(root))


def test_contract_unit_is_generic():
    validated = validate_regression_model_selection_contract({
        "problem_type": "continuous_regression",
        "target_semantics": "Continuous / quantitative",
        "target_unit": "kWh",
        "primary_metric": "mae",
        "primary_metric_direction": "lower_is_better",
        "refit_metric": "mae",
        "cv": {"strategy": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "test_partition_sealed": True,
        "test_partition_evaluated": False,
    })
    assert validated["target_unit"] == "kWh"


def test_valid_write_reload_and_equivalent_reuse(tmp_path):
    out = tmp_path / SELECTION_HANDOFF.parent
    artifacts = _artifacts(tmp_path)
    first = write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    assert not first.idempotent
    second = write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    assert second.idempotent
    # The public loader needs the authenticated preparation lineage too.
    loaded = load_and_validate_model_selection_handoff(
        project_root=tmp_path,
        handoff_path=SELECTION_HANDOFF,
    )
    assert loaded["selected_model_id"] == "synthetic_ridge"


@pytest.mark.parametrize("mutation", [
    "target", "target_unit", "metric", "cv", "family", "search_grid",
    "feature", "winner", "readiness",
])
def test_writer_fails_closed_for_scientific_manifest_changes(tmp_path, mutation):
    artifacts = _artifacts(tmp_path)
    out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    changed = _artifacts(tmp_path)
    manifest = changed["model-selection-manifest.json"]
    if mutation == "target": manifest["target_contract"]["column"] = "other"
    elif mutation == "target_unit": manifest["target_contract"]["unit"] = "kg"
    elif mutation == "metric": manifest["model_selection_contract"]["primary_metric"] = "rmse"
    elif mutation == "cv": manifest["cv_contract"]["n_splits"] = 4
    elif mutation == "family": manifest["candidate_families"][0]["family"] = "OtherRegressor"
    elif mutation == "search_grid": manifest["candidate_families"][0]["search_space"] = {"model__alpha": [7]}
    elif mutation == "feature": manifest["feature_contract"]["selected_feature_policy"] = "subset"
    elif mutation == "winner": changed["model-selection-handoff.json"]["selected_model_id"] = "ridge"
    else: changed["model-selection-handoff.json"]["readiness"]["final_model_training_ready"] = False
    with pytest.raises(ArtifactConflictError):
        write_regression_model_selection_artifacts(output_directory=out, artifacts=changed)


def test_writer_reuses_volatile_metadata_only(tmp_path):
    artifacts = _artifacts(tmp_path); out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    changed = _artifacts(tmp_path)
    changed["model-selection-manifest.json"]["generated_at"] = "2099-01-01T00:00:00Z"
    assert write_regression_model_selection_artifacts(
        output_directory=out, artifacts=changed
    ).idempotent


def test_writer_reuses_equivalent_artifacts_across_runtime_versions(tmp_path):
    artifacts = _artifacts(tmp_path)
    out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    changed = _artifacts(tmp_path)
    changed["model-selection-manifest.json"]["runtime_versions"] = {
        "python": "different-kernel",
        "pandas": "different-runtime",
        "scikit_learn": "different-runtime",
        "platform": "different-platform",
    }
    result = write_regression_model_selection_artifacts(
        output_directory=out, artifacts=changed, overwrite=False
    )
    assert result.idempotent


def test_writer_rejects_partial_set(tmp_path):
    out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts(tmp_path))
    (out / "selection-analysis.json").unlink()
    with pytest.raises(ArtifactConflictError):
        write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts(tmp_path))


def test_writer_atomic_rollback_on_promotion_failure(tmp_path, monkeypatch):
    out = tmp_path / "selection"
    original = sm.os.replace
    promotion_count = 0

    def fail_during_promotion(source, destination):
        nonlocal promotion_count
        if Path(destination).parent == out:
            promotion_count += 1
            if promotion_count == 2:
                raise OSError("injected promotion failure")
        return original(source, destination)

    monkeypatch.setattr(sm.os, "replace", fail_during_promotion)
    with pytest.raises(OSError, match="injected"):
        write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts(tmp_path))
    assert not any((out / name).exists() for name in REGRESSION_ARTIFACT_FILENAMES)


def test_runtime_loader_is_defensive(tmp_path):
    artifacts = _artifacts(tmp_path)
    write_regression_model_selection_artifacts(
        output_directory=tmp_path / SELECTION_HANDOFF.parent, artifacts=artifacts
    )
    one = load_and_validate_model_selection_handoff(project_root=tmp_path, handoff_path=SELECTION_HANDOFF)
    one["target_contract"]["unit"] = "kg"
    two = load_and_validate_model_selection_handoff(project_root=tmp_path, handoff_path=SELECTION_HANDOFF)
    assert two["target_contract"]["unit"] == "kWh"


@pytest.mark.parametrize("filename", REGRESSION_ARTIFACT_FILENAMES)
def test_loader_rejects_corrupted_v3_component(tmp_path, filename):
    out = tmp_path / SELECTION_HANDOFF.parent
    write_regression_model_selection_artifacts(output_directory=out, artifacts=_artifacts(tmp_path))
    path = out / filename
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(Exception):
        load_and_validate_model_selection_handoff(
            project_root=tmp_path,
            handoff_path=out.relative_to(tmp_path) / "model-selection-handoff.json",
        )


@pytest.mark.parametrize("field,value", [
    ("selected_model_id", "ridge"),
    ("selected_model_family", "OtherRegressor"),
    ("selected_hyperparameters", {"model__alpha": 1.0}),
    ("selected_feature_columns", ["Age"]),
    ("primary_metric", "rmse"),
    ("primary_metric_direction", "higher_is_better"),
    ("test_partition_sealed", False),
    ("test_partition_evaluated", True),
    ("final_model_training_ready", False),
    ("final_model_trained", True),
    ("model_artifact_materialized", True),
    ("model_bundle_materialized", True),
])
def test_loader_rejects_divergent_handoff_contract(tmp_path, field, value):
    artifacts = _artifacts(tmp_path)
    artifacts["model-selection-handoff.json"][field] = value
    out = tmp_path / SELECTION_HANDOFF.parent
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    with pytest.raises(Exception):
        load_and_validate_model_selection_handoff(
            project_root=tmp_path,
            handoff_path=out.relative_to(tmp_path) / "model-selection-handoff.json",
        )


def test_loader_rejects_path_escape(tmp_path):
    artifacts = _artifacts(tmp_path)
    artifacts["model-selection-handoff.json"]["preparation_handoff_reference"]["path"] = "../escape.json"
    out = tmp_path / "selection"
    write_regression_model_selection_artifacts(output_directory=out, artifacts=artifacts)
    with pytest.raises(Exception):
        load_and_validate_model_selection_handoff(
            project_root=tmp_path, handoff_path=out.relative_to(tmp_path) / "model-selection-handoff.json"
        )
