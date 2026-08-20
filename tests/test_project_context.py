from pathlib import Path

import pytest

from scripts.project_context import ProjectContext, find_project_root


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    project = tmp_path / "study"
    nested = project / "notebooks" / "nested"
    (project / "scripts").mkdir(parents=True)
    nested.mkdir(parents=True)
    (project / "scripts" / "download_data.py").write_text("", encoding="utf-8")

    assert find_project_root(nested) == project.resolve()


def test_display_hides_absolute_parent_directories(tmp_path: Path) -> None:
    project = tmp_path / "private-user" / "study"
    project.mkdir(parents=True)
    context = ProjectContext(root=project.resolve())
    dataset_file = project / "data" / "raw" / "sample.csv"

    assert context.display(dataset_file) == "data/raw/sample.csv"
    assert "private-user" not in context.display(dataset_file)


def test_path_rejects_escape_from_project_root(tmp_path: Path) -> None:
    project = tmp_path / "study"
    project.mkdir()
    context = ProjectContext(root=project.resolve())

    with pytest.raises(ValueError, match="escapes the project root"):
        context.path("..", "outside.csv")
