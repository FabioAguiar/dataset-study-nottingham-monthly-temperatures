import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import pytest

from scripts.download_data import (
    DatasetAcquisition,
    DatasetDownloadError,
    acquire_uci_dataset,
    discover_dataset_files,
    resolve_project_path,
)


def test_discover_dataset_files_excludes_hidden_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / "raw" / "sample"
    hidden_dir = dataset_dir / ".complete"
    hidden_dir.mkdir(parents=True)
    (dataset_dir / "dataset.csv").write_text("a\n1\n", encoding="utf-8")
    (hidden_dir / "bundle.complete").write_text("ok", encoding="utf-8")

    files = discover_dataset_files(dataset_dir)

    assert [path.name for path in files] == ["dataset.csv"]


def test_require_one_file_uses_explicit_selector(tmp_path: Path) -> None:
    project = tmp_path / "study"
    destination = project / "data" / "raw" / "sample"
    destination.mkdir(parents=True)
    source_file = destination / "dataset.csv"
    source_file.write_text("a\n1\n", encoding="utf-8")

    acquisition = DatasetAcquisition(
        source_kind="kaggle",
        source_reference="owner/sample",
        destination=destination,
        resolved_path=destination,
        files=(source_file,),
        project_root=project,
    )

    assert acquisition.require_one_file("dataset.csv") == source_file
    assert acquisition.display_destination == "data/raw/sample"


def test_resolve_project_path_rejects_outside_destination(
    tmp_path: Path,
) -> None:
    project = tmp_path / "study"
    project.mkdir()

    with pytest.raises(ValueError, match="inside the project root"):
        resolve_project_path("../outside", project_root=project)



def test_acquire_uci_dataset_materializes_official_python_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "study"
    project.mkdir()

    original = pd.DataFrame(
        {
            "Area": [28395, 28734],
            "Class": ["SEKER", "SEKER"],
        }
    )
    variables = pd.DataFrame(
        {
            "name": ["Area", "Class"],
            "role": ["Feature", "Target"],
        }
    )
    dataset = SimpleNamespace(
        data=SimpleNamespace(original=original),
        metadata={
            "uci_id": 602,
            "name": "Dry Bean",
            "dataset_doi": "10.24432/C50S4B",
        },
        variables=variables,
    )

    fake_module = SimpleNamespace(
        fetch_ucirepo=lambda *, id: dataset if id == 602 else None
    )
    monkeypatch.setitem(sys.modules, "ucimlrepo", fake_module)

    acquisition = acquire_uci_dataset(
        dataset_id=602,
        destination="data/raw/dry-bean",
        project_root=project,
    )

    assert acquisition.source_kind == "uci"
    assert acquisition.display_destination == "data/raw/dry-bean"
    assert set(acquisition.relative_files) == {
        "data/raw/dry-bean/dataset.csv",
        "data/raw/dry-bean/metadata.json",
        "data/raw/dry-bean/variables.csv",
    }

    loaded = pd.read_csv(acquisition.require_one_file("dataset.csv"))
    pd.testing.assert_frame_equal(loaded, original)

    metadata = json.loads(
        acquisition.require_one_file("metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["uci_id"] == 602
    assert metadata["dataset_doi"] == "10.24432/C50S4B"


def test_acquire_uci_dataset_reuses_complete_local_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "study"
    destination = project / "data" / "raw" / "dry-bean"
    destination.mkdir(parents=True)
    (destination / "dataset.csv").write_text(
        "Area,Class\n28395,SEKER\n",
        encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        '{"uci_id": 602}\n',
        encoding="utf-8",
    )
    (destination / "variables.csv").write_text(
        "name,role\nArea,Feature\nClass,Target\n",
        encoding="utf-8",
    )

    monkeypatch.delitem(sys.modules, "ucimlrepo", raising=False)

    acquisition = acquire_uci_dataset(
        dataset_id=602,
        destination="data/raw/dry-bean",
        project_root=project,
    )

    assert acquisition.require_one_file("dataset.csv").is_file()
    assert len(acquisition.files) == 3


def test_acquire_uci_dataset_rejects_partial_existing_materialization(
    tmp_path: Path,
) -> None:
    project = tmp_path / "study"
    destination = project / "data" / "raw" / "dry-bean"
    destination.mkdir(parents=True)
    (destination / "dataset.csv").write_text(
        "Area,Class\n28395,SEKER\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetDownloadError, match="incomplete prior"):
        acquire_uci_dataset(
            dataset_id=602,
            destination="data/raw/dry-bean",
            project_root=project,
        )
