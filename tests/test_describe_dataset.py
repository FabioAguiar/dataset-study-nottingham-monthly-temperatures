import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.describe_dataset import (
    DatasetDescriptionError,
    describe_dataset_structure,
)
from scripts.download_data import DatasetAcquisition


def _uci_acquisition(
    project: Path,
    *,
    metadata: dict[str, object],
) -> DatasetAcquisition:
    destination = project / "data" / "raw" / "dry-bean"
    destination.mkdir(parents=True)

    dataset_file = destination / "dataset.csv"
    dataset_file.write_text("Area,Class\n1,SEKER\n", encoding="utf-8")

    metadata_file = destination / "metadata.json"
    metadata_file.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    variables_file = destination / "variables.csv"
    variables_file.write_text(
        "name,role\nArea,Feature\nClass,Target\n",
        encoding="utf-8",
    )

    return DatasetAcquisition(
        source_kind="uci",
        source_reference="UCI ML Repository dataset 602",
        destination=destination,
        resolved_path=dataset_file,
        files=(dataset_file, metadata_file, variables_file),
        project_root=project,
    )


def test_describe_dataset_structure_reports_source_and_column_order(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": [1, 2],
            "Perimeter": [3.0, 4.0],
            "Class": ["SEKER", "SIRA"],
        }
    )
    acquisition = _uci_acquisition(
        tmp_path,
        metadata={
            "uci_id": 602,
            "name": "Dry Bean",
            "dataset_doi": "10.24432/C50S4B",
            "num_instances": 2,
        },
    )

    report = describe_dataset_structure(
        dataframe,
        acquisition=acquisition,
        expected_source_id=602,
    )

    assert report.row_count == 2
    assert report.column_count == 3
    assert report.columns == ("Area", "Perimeter", "Class")
    assert report.source_dataset_id == 602
    assert report.dataset_name == "Dry Bean"
    assert report.dataset_doi == "10.24432/C50S4B"
    assert report.data_file == "data/raw/dry-bean/dataset.csv"

    assert list(report.columns_frame()["Column"]) == [
        "Area",
        "Perimeter",
        "Class",
    ]


def test_describe_dataset_structure_rejects_wrong_uci_identity(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame({"Area": [1], "Class": ["SEKER"]})
    acquisition = _uci_acquisition(
        tmp_path,
        metadata={
            "uci_id": 999,
            "name": "Different Dataset",
            "num_instances": 1,
        },
    )

    with pytest.raises(
        DatasetDescriptionError,
        match="expected 602, observed 999",
    ):
        describe_dataset_structure(
            dataframe,
            acquisition=acquisition,
            expected_source_id=602,
        )


def test_describe_dataset_structure_rejects_source_row_count_mismatch(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {"Area": [1, 2], "Class": ["SEKER", "SIRA"]}
    )
    acquisition = _uci_acquisition(
        tmp_path,
        metadata={
            "uci_id": 602,
            "name": "Dry Bean",
            "num_instances": 13611,
        },
    )

    with pytest.raises(
        DatasetDescriptionError,
        match="source declares 13611, dataframe contains 2",
    ):
        describe_dataset_structure(
            dataframe,
            acquisition=acquisition,
            expected_source_id=602,
        )
