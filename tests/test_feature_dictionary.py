"""Tests for source-backed feature roles and data dictionaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.feature_dictionary import (
    FeatureDictionaryError,
    define_feature_roles_and_dictionary,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": [100, 200],
            "Roundness": [0.8, 0.9],
            "Class": ["SEKER", "SIRA"],
        }
    )


def _write_variables(path: Path) -> Path:
    pd.DataFrame(
        {
            "name": ["Area", "Roundness", "Class"],
            "role": ["Feature", "Feature", "Target"],
            "type": ["Integer", "Continuous", "Categorical"],
            "description": [
                "Bean area.",
                "Bean roundness.",
                "Bean variety.",
            ],
            "units": ["pixels", None, None],
            "missing_values": ["no", "no", "no"],
        }
    ).to_csv(path, index=False)
    return path


def test_defines_candidate_features_and_source_dictionary(tmp_path: Path) -> None:
    variables_file = _write_variables(tmp_path / "variables.csv")

    report = define_feature_roles_and_dictionary(
        _frame(),
        source_variables_file=variables_file,
        target="Class",
        identifiers=None,
    )

    assert report.feature_columns == ("Area", "Roundness")
    assert report.identifier_columns == ()
    assert report.target_column == "Class"
    assert report.column_frame()["Column"].tolist() == [
        "Area",
        "Roundness",
        "Class",
    ]
    assert report.column_frame()["Analytical role"].tolist() == [
        "Candidate feature",
        "Candidate feature",
        "Target",
    ]
    assert report.summary_frame().set_index("Metric").loc[
        "Candidate features", "Value"
    ] == 2


def test_accepts_source_identifier_role(tmp_path: Path) -> None:
    dataframe = _frame().assign(bean_id=["a", "b"])[
        ["bean_id", "Area", "Roundness", "Class"]
    ]
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["bean_id", "Area", "Roundness", "Class"],
            "role": ["ID", "Feature", "Feature", "Target"],
            "description": ["Identifier", "Area", "Roundness", "Class"],
            "units": [None, "pixels", None, None],
        }
    ).to_csv(variables_file, index=False)

    report = define_feature_roles_and_dictionary(
        dataframe,
        source_variables_file=variables_file,
        target="Class",
        identifiers="bean_id",
    )

    assert report.identifier_columns == ("bean_id",)
    assert report.feature_columns == ("Area", "Roundness")


def test_rejects_wrong_source_role(tmp_path: Path) -> None:
    variables_file = _write_variables(tmp_path / "variables.csv")
    variables = pd.read_csv(variables_file)
    variables.loc[variables["name"] == "Area", "role"] = "Target"
    variables.to_csv(variables_file, index=False)

    with pytest.raises(
        FeatureDictionaryError,
        match="Analytical/source role validation failed",
    ):
        define_feature_roles_and_dictionary(
            _frame(),
            source_variables_file=variables_file,
            target="Class",
        )


def test_rejects_missing_source_metadata_row(tmp_path: Path) -> None:
    variables_file = _write_variables(tmp_path / "variables.csv")
    variables = pd.read_csv(variables_file)
    variables.loc[variables["name"] != "Roundness"].to_csv(
        variables_file,
        index=False,
    )

    with pytest.raises(
        FeatureDictionaryError,
        match="missing from source variable metadata: Roundness",
    ):
        define_feature_roles_and_dictionary(
            _frame(),
            source_variables_file=variables_file,
            target="Class",
        )


def test_rejects_extra_source_metadata_row(tmp_path: Path) -> None:
    variables_file = _write_variables(tmp_path / "variables.csv")
    variables = pd.read_csv(variables_file)
    variables.loc[len(variables)] = {
        "name": "GhostColumn",
        "role": "Feature",
        "type": "Continuous",
        "description": "Not present.",
        "units": None,
        "missing_values": "no",
    }
    variables.to_csv(variables_file, index=False)

    with pytest.raises(
        FeatureDictionaryError,
        match="contains columns absent from dataset: GhostColumn",
    ):
        define_feature_roles_and_dictionary(
            _frame(),
            source_variables_file=variables_file,
            target="Class",
        )


def test_missing_description_and_unit_are_rendered_without_invention(
    tmp_path: Path,
) -> None:
    variables_file = _write_variables(tmp_path / "variables.csv")
    variables = pd.read_csv(variables_file)
    variables.loc[variables["name"] == "Roundness", "description"] = None
    variables.to_csv(variables_file, index=False)

    report = define_feature_roles_and_dictionary(
        _frame(),
        source_variables_file=variables_file,
        target="Class",
    )

    row = report.column_frame().set_index("Column").loc["Roundness"]
    assert row["Description"] == "Not documented by source"
    assert row["Unit"] == "Not specified"


def test_requires_data_dictionary_columns(tmp_path: Path) -> None:
    variables_file = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Roundness", "Class"],
            "role": ["Feature", "Feature", "Target"],
        }
    ).to_csv(variables_file, index=False)

    with pytest.raises(
        FeatureDictionaryError,
        match="missing required column",
    ):
        define_feature_roles_and_dictionary(
            _frame(),
            source_variables_file=variables_file,
            target="Class",
        )
