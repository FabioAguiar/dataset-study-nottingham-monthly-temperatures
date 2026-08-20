"""Tests for source-backed type, range, and domain validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_domains import (
    DomainValidationError,
    analyze_types_ranges_and_domains,
)


def _variables_file(tmp_path: Path) -> Path:
    path = tmp_path / "variables.csv"
    pd.DataFrame(
        {
            "name": ["Area", "Ratio", "Class"],
            "role": ["Feature", "Feature", "Target"],
            "type": ["Integer", "Continuous", "Categorical"],
        }
    ).to_csv(path, index=False)
    return path


def test_report_derives_types_and_observed_ranges_from_source_metadata(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": pd.Series([10, 20, 30], dtype="int64"),
            "Ratio": [0.2, 0.5, 0.9],
            "Class": ["A", "B", "A"],
        }
    )

    report = analyze_types_ranges_and_domains(
        dataframe,
        source_variables_file=_variables_file(tmp_path),
        domain_rules={"positive": ("Area",), "unit_interval": ("Ratio",)},
    )

    columns = report.column_frame().set_index("Column")
    assert columns.loc["Area", "Expected type"] == "integer"
    assert columns.loc["Ratio", "Expected type"] == "numeric"
    assert columns.loc["Class", "Expected type"] == "string"
    assert columns.loc["Area", "Observed minimum"] == 10.0
    assert columns.loc["Area", "Observed maximum"] == 30.0
    assert columns.loc["Class", "Observed minimum"] == "Not applicable"
    assert not report.has_type_mismatches
    assert not report.has_domain_violations


def test_report_detects_domain_violations_and_relations(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": pd.Series([10, -1, 30], dtype="int64"),
            "Ratio": [0.2, 1.4, 0.9],
            "Class": ["A", "B", "A"],
        }
    )

    report = analyze_types_ranges_and_domains(
        dataframe,
        source_variables_file=_variables_file(tmp_path),
        domain_rules={
            "positive": ("Area",),
            "unit_interval": ("Ratio",),
            "relations": (("Area", ">=", "Ratio"),),
        },
    )

    columns = report.column_frame().set_index("Column")
    assert columns.loc["Area", "Violation count"] == 1
    assert columns.loc["Ratio", "Violation count"] == 1
    assert report.domain_violation_columns == ("Area", "Ratio")
    assert report.violated_relations == ("Area >= Ratio",)


def test_missing_values_are_excluded_from_domain_assessment(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": pd.Series([10, None, 30], dtype="Int64"),
            "Ratio": [0.2, None, 0.9],
            "Class": ["A", None, "A"],
        }
    )

    report = analyze_types_ranges_and_domains(
        dataframe,
        source_variables_file=_variables_file(tmp_path),
        domain_rules={"positive": ("Area",), "unit_interval": ("Ratio",)},
    )

    assert not report.has_domain_violations
    assert report.column_frame()["Violation count"].sum() == 0


def test_type_mismatch_is_reported_without_mutating_dataframe(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "Area": [10.0, 20.0],
            "Ratio": [0.2, 0.3],
            "Class": ["A", "B"],
        }
    )
    original = dataframe.copy(deep=True)

    report = analyze_types_ranges_and_domains(
        dataframe,
        source_variables_file=_variables_file(tmp_path),
    )

    assert report.type_mismatch_columns == ("Area",)
    pd.testing.assert_frame_equal(dataframe, original)


def test_source_metadata_must_cover_dataset_exactly(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"Area": [1], "Class": ["A"]})

    with pytest.raises(DomainValidationError, match="source metadata columns absent"):
        analyze_types_ranges_and_domains(
            dataframe,
            source_variables_file=_variables_file(tmp_path),
        )


def test_unsupported_source_type_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "variables.csv"
    pd.DataFrame({"name": ["A"], "type": ["Mystery"]}).to_csv(path, index=False)

    with pytest.raises(DomainValidationError, match="Unsupported source type"):
        analyze_types_ranges_and_domains(
            pd.DataFrame({"A": [1]}),
            source_variables_file=path,
        )
