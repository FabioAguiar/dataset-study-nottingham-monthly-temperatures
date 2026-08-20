"""Reusable dataset structure and source-identity descriptions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.download_data import DatasetAcquisition


class DatasetDescriptionError(ValueError):
    """Raised when acquired source identity contradicts the loaded dataset."""


@dataclass(frozen=True, slots=True)
class DatasetStructureReport:
    """Describe physical dataset structure and acquisition-source identity."""

    row_count: int
    column_count: int
    columns: tuple[str, ...]
    source_kind: str
    source_name: str
    source_dataset_id: str | int | None
    dataset_name: str
    dataset_doi: str | None
    data_file: str
    source_declared_instances: int | None

    def summary_frame(self) -> pd.DataFrame:
        """Return a notebook-friendly structure and source summary."""
        source_label = {
            "uci": "UCI Machine Learning Repository",
            "kaggle": "Kaggle",
            "url": "Direct URL",
        }.get(self.source_kind, self.source_name)

        rows: list[tuple[str, object]] = [
            ("Dataset", self.dataset_name),
            ("Source", source_label),
            ("Source dataset ID", self.source_dataset_id or "Not provided"),
            ("DOI", self.dataset_doi or "Not provided"),
            ("Rows", self.row_count),
            ("Columns", self.column_count),
            ("Materialized data file", self.data_file),
        ]

        if self.source_declared_instances is not None:
            rows.insert(
                6,
                ("Source-declared instances", self.source_declared_instances),
            )

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def columns_frame(self) -> pd.DataFrame:
        """Return column order without anticipating type or role analysis."""
        return pd.DataFrame(
            {
                "Position": range(1, self.column_count + 1),
                "Column": self.columns,
            }
        )


def describe_dataset_structure(
    dataframe: pd.DataFrame,
    *,
    acquisition: DatasetAcquisition,
    expected_source_id: str | int | None = None,
) -> DatasetStructureReport:
    """Describe structure and validate identity of an acquired dataset.

    For UCI acquisitions, the materialized ``metadata.json`` is treated as
    source identity evidence. When available, its dataset ID and declared
    instance count are checked against the caller's expectation and the loaded
    dataframe. Data types, analytical roles, target semantics, and domain
    validity are intentionally outside this helper.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    metadata = _load_source_metadata(acquisition)

    dataset_name = _non_empty_text(
        metadata.get("name")
    ) or acquisition.source_reference
    dataset_doi = _non_empty_text(metadata.get("dataset_doi"))

    source_dataset_id: str | int | None = None
    source_declared_instances: int | None = None

    if acquisition.source_kind == "uci":
        source_dataset_id = _normalize_source_id(metadata.get("uci_id"))
        source_declared_instances = _normalize_optional_count(
            metadata.get("num_instances"),
            field="num_instances",
        )

        if source_dataset_id is None:
            raise DatasetDescriptionError(
                "UCI source metadata does not provide 'uci_id'."
            )

        if expected_source_id is not None:
            expected = _normalize_source_id(expected_source_id)
            if source_dataset_id != expected:
                raise DatasetDescriptionError(
                    "Loaded source identity does not match the expected UCI "
                    f"dataset ID: expected {expected!r}, observed "
                    f"{source_dataset_id!r}."
                )

        if (
            source_declared_instances is not None
            and source_declared_instances != len(dataframe)
        ):
            raise DatasetDescriptionError(
                "Loaded row count does not match the source metadata: "
                f"source declares {source_declared_instances}, dataframe "
                f"contains {len(dataframe)}."
            )

    data_file = _display_data_file(acquisition)

    return DatasetStructureReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(str(column) for column in dataframe.columns),
        source_kind=acquisition.source_kind,
        source_name=acquisition.source_reference,
        source_dataset_id=source_dataset_id,
        dataset_name=dataset_name,
        dataset_doi=dataset_doi,
        data_file=data_file,
        source_declared_instances=source_declared_instances,
    )


def _load_source_metadata(acquisition: DatasetAcquisition) -> dict[str, Any]:
    """Load materialized source metadata when the acquisition provides it."""
    if acquisition.source_kind != "uci":
        return {}

    metadata_path = acquisition.require_one_file("metadata.json")

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetDescriptionError(
            "Could not read the materialized UCI metadata.json."
        ) from exc

    if not isinstance(raw, dict):
        raise DatasetDescriptionError(
            "Materialized UCI metadata must be a JSON object."
        )

    return raw


def _display_data_file(acquisition: DatasetAcquisition) -> str:
    """Return the primary acquired file as a project-relative path."""
    path = acquisition.resolved_path.resolve()

    try:
        return path.relative_to(acquisition.project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _normalize_source_id(value: object) -> str | int | None:
    """Normalize source IDs while preserving ordinary integer identities."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        return int(text)

    return text


def _normalize_optional_count(value: object, *, field: str) -> int | None:
    """Normalize optional non-negative source metadata counts."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise DatasetDescriptionError(
            f"Source metadata field '{field}' must be an integer count."
        )

    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetDescriptionError(
            f"Source metadata field '{field}' must be an integer count."
        ) from exc

    if count < 0:
        raise DatasetDescriptionError(
            f"Source metadata field '{field}' cannot be negative."
        )

    return count


def _non_empty_text(value: object) -> str | None:
    """Return stripped text, or ``None`` for absent/empty values."""
    if value is None:
        return None

    text = str(value).strip()
    return text or None
