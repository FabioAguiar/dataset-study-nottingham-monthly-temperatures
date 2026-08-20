import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.prepare_data import load_and_validate_preparation_for_model_selection
from scripts.select_models import load_and_validate_model_selection_handoff
from tests._continuous_model_selection_fixtures import (
    PREPARATION_HANDOFF, SELECTION_HANDOFF, build_synthetic_project,
)


def _read_names(monkeypatch):
    original = pd.read_csv
    calls = []

    def spy(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    return calls


def test_safe_preparation_loader_hashes_but_never_parses_prepared_or_test(tmp_path, monkeypatch):
    preparation, _ = build_synthetic_project(tmp_path)
    calls = _read_names(monkeypatch)
    loaded = load_and_validate_preparation_for_model_selection(
        project_root=tmp_path, preparation_handoff_path=PREPARATION_HANDOFF
    )
    assert calls.count("train.csv") == 1
    assert calls.count("validation.csv") == 1
    assert calls.count("prepared.csv") == 0
    assert calls.count("test.csv") == 0
    assert loaded.sealed_test_integrity_reference["sha256"]
    assert loaded.sealed_test_integrity_reference["sha256"] == preparation["partition_sha"]["test"]
    assert loaded.sealed_test_integrity_reference["row_count"] == len(preparation["partitions"].test)
    assert loaded.prepared_integrity_reference["sha256"]
    assert loaded.manifests["preparation_handoff"]["consumer_contract"]["test_partition_sealed"] is True
    assert loaded.manifests["preparation_handoff"]["consumer_contract"]["test_partition_evaluated"] is False
    assert not hasattr(loaded, "test") and not hasattr(loaded, "prepared")


def test_v3_selection_loader_never_parses_prepared_or_test(tmp_path, monkeypatch):
    build_synthetic_project(tmp_path)
    calls = _read_names(monkeypatch)
    loaded = load_and_validate_model_selection_handoff(
        project_root=tmp_path, handoff_path=SELECTION_HANDOFF
    )
    assert loaded["schema_version"] == "model-selection-handoff.v3"
    assert calls.count("train.csv") == 1
    assert calls.count("validation.csv") == 1
    assert "prepared.csv" not in calls and "test.csv" not in calls


def test_safe_frames_are_defensive_and_isolated(tmp_path):
    build_synthetic_project(tmp_path)
    loaded = load_and_validate_preparation_for_model_selection(
        project_root=tmp_path, preparation_handoff_path=PREPARATION_HANDOFF
    )
    first = loaded.train
    first.iloc[0, 0] = -999
    assert loaded.train.iloc[0, 0] != -999
    validation = loaded.validation
    validation.iloc[0, 0] = -999
    assert loaded.validation.iloc[0, 0] != -999


def test_fresh_process_v3_reload_rejects_any_prepared_or_test_parse(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    build_synthetic_project(project)
    hook = tmp_path / "hook"
    hook.mkdir()
    sitecustomize = hook / "sitecustomize.py"
    sitecustomize.write_text(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "_original = pd.read_csv\n"
        "def _sealed(path, *args, **kwargs):\n"
        "    if Path(path).name in {'prepared.csv', 'test.csv'}:\n"
        "        raise AssertionError(f'forbidden parse: {Path(path).name}')\n"
        "    return _original(path, *args, **kwargs)\n"
        "pd.read_csv = _sealed\n",
        encoding="utf-8",
    )
    code = (
        "from scripts.select_models import load_and_validate_model_selection_handoff as load; "
        f"p=load(project_root={str(project)!r}, handoff_path={str(SELECTION_HANDOFF)!r}); "
        "assert p['schema_version']=='model-selection-handoff.v3'; "
        "assert p['selected_model_id']=='synthetic_ridge'; "
        "assert p['test_partition_sealed'] is True; "
        "print('v3-reload-ok')"
    )
    environment = os.environ.copy()
    root = str(__file__).rsplit("/tests/", 1)[0]
    environment["PYTHONPATH"] = os.pathsep.join((str(hook), root))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=project, env=environment,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v3-reload-ok"
