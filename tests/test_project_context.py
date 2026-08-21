import json
from pathlib import Path

import pytest

import scripts.project_context as project_context_module
from scripts.project_context import (
    PROJECT_MARKERS,
    ProjectContext,
    ProjectContextError,
    find_project_root,
)


def _make_checkout(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "scripts" / "download_data.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return root.resolve()


def test_find_project_root_from_root(tmp_path: Path) -> None:
    project = _make_checkout(tmp_path / "study")

    assert find_project_root(project) == project


@pytest.mark.parametrize("relative_start", ["notebooks", "notebooks/nested"])
def test_find_project_root_from_nested_directory(
    tmp_path: Path,
    relative_start: str,
) -> None:
    project = _make_checkout(tmp_path / "study")
    start = project / relative_start
    start.mkdir(parents=True, exist_ok=True)

    assert find_project_root(start) == project


def test_project_root_requires_all_strong_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "study"
    (project / "scripts").mkdir(parents=True)
    (project / "notebooks").mkdir()
    (project / "scripts" / "download_data.py").write_text("", encoding="utf-8")

    assert {marker.as_posix() for marker in PROJECT_MARKERS} == {
        "scripts/download_data.py",
        "pyproject.toml",
        "notebooks",
    }
    monkeypatch.delenv("DATASET_STUDY_ROOT", raising=False)
    monkeypatch.setattr(
        project_context_module,
        "__file__",
        str(tmp_path / "python" / "site-packages" / "scripts" / "project_context.py"),
    )
    with pytest.raises(ProjectContextError, match="authenticated checkout"):
        find_project_root(project)


def test_fake_site_packages_with_download_helper_is_not_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    (site_packages / "scripts").mkdir(parents=True)
    (site_packages / "scripts" / "download_data.py").write_text(
        "", encoding="utf-8"
    )
    monkeypatch.delenv("DATASET_STUDY_ROOT", raising=False)
    monkeypatch.setattr(
        project_context_module,
        "__file__",
        str(site_packages / "scripts" / "project_context.py"),
    )
    monkeypatch.chdir(tmp_path / "site-packages")

    with pytest.raises(ProjectContextError, match="authenticated checkout"):
        find_project_root(site_packages)


def test_real_start_wins_when_module_location_is_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _make_checkout(tmp_path / "study")
    site_packages = tmp_path / "python" / "site-packages"
    (site_packages / "scripts").mkdir(parents=True)
    (site_packages / "scripts" / "download_data.py").write_text(
        "", encoding="utf-8"
    )
    monkeypatch.delenv("DATASET_STUDY_ROOT", raising=False)
    monkeypatch.setattr(
        project_context_module,
        "__file__",
        str(site_packages / "scripts" / "project_context.py"),
    )

    assert find_project_root(project / "notebooks") == project


def test_no_valid_seed_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = tmp_path / "unrelated"
    module_tree = tmp_path / "python" / "site-packages" / "scripts"
    unrelated.mkdir()
    module_tree.mkdir(parents=True)
    monkeypatch.delenv("DATASET_STUDY_ROOT", raising=False)
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(
        project_context_module,
        "__file__",
        str(module_tree / "project_context.py"),
    )

    with pytest.raises(ProjectContextError, match="pyproject.toml"):
        find_project_root()


def test_display_hides_absolute_parent_directories(tmp_path: Path) -> None:
    project = tmp_path / "private-user" / "study"
    project.mkdir(parents=True)
    context = ProjectContext(root=project.resolve())
    dataset_file = project / "data" / "raw" / "sample.csv"

    assert context.display(dataset_file) == "data/raw/sample.csv"
    assert "private-user" not in context.display(dataset_file)
    assert context.display(tmp_path / "outside" / "secret.csv") == "secret.csv"


def test_path_rejects_escape_from_project_root(tmp_path: Path) -> None:
    project = tmp_path / "study"
    project.mkdir()
    context = ProjectContext(root=project.resolve())

    with pytest.raises(ValueError, match="escapes the project root"):
        context.path("..", "outside.csv")


def test_notebook_01_uses_one_authenticated_context_for_runtime_paths() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "01_data_understanding_and_exploration.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    def code_after_heading(heading: str) -> str:
        heading_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith(heading)
        )
        return "".join(notebook["cells"][heading_index + 1]["source"])

    section_2 = code_after_heading("## 2. Dataset Source and Python Acquisition")
    section_22 = code_after_heading("## 22. Exploration Handoff and Next Steps")

    context_position = section_2.index("PROJECT = get_project_context()")
    acquisition_position = section_2.index("acquisition = acquire_rdataset(")
    assert context_position < acquisition_position
    assert 'RAW_DATA_DIR = PROJECT.path("data", "raw", DATASET_NAME)' in section_2
    assert "project_root=PROJECT.root" in section_2
    assert "for acquired_path in (" in section_2
    assert "data_path," in section_2
    assert "metadata_path," in section_2
    assert "documentation_path," in section_2
    assert "relative_to(PROJECT.root)" in section_2

    assert "get_project_context(data_path)" not in section_22
    assert "exploration_handoff_path = PROJECT.path(" in section_22
    assert "project_root=PROJECT.root" in section_22
    assert "relative_to(PROJECT.root)" in section_22
    assert "exploration_handoff_report.write(" in section_22
    assert "load_and_validate_forecasting_exploration_handoff(" in section_22
