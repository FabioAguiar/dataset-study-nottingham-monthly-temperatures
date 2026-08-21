"""Cross-platform project context utilities for dataset study notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator


PROJECT_ROOT_ENV: Final[str] = "DATASET_STUDY_ROOT"
PROJECT_MARKER: Final[Path] = Path("scripts") / "download_data.py"
PROJECT_MARKERS: Final[tuple[Path, ...]] = (
    PROJECT_MARKER,
    Path("pyproject.toml"),
    Path("notebooks"),
)


class ProjectContextError(RuntimeError):
    """Raised when the dataset study project cannot be located."""


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Resolved and validated context for a dataset study project."""

    root: Path

    @property
    def name(self) -> str:
        """Return only the project directory name."""
        return self.root.name

    def path(self, *parts: str | Path) -> Path:
        """Build and validate an absolute path inside the project root."""
        candidate = self.root.joinpath(*parts).expanduser().resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            rendered = Path(*parts).as_posix() if parts else "."
            raise ValueError(
                f"Project path escapes the project root: {rendered}"
            ) from exc

        return candidate

    def display(self, path: str | Path | None = None) -> str:
        """Return a safe project-relative path for notebook output.

        Absolute parent directories such as ``/home/user`` or
        ``C:/Users/user`` are never displayed.
        """
        if path is None:
            return self.name

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        candidate = candidate.resolve()

        if candidate == self.root:
            return self.name

        try:
            relative_path = candidate.relative_to(self.root)
        except ValueError:
            # Never expose an absolute path outside the project.
            return candidate.name

        # Normalize presentation while filesystem operations remain
        # operating-system aware through pathlib.
        return relative_path.as_posix()

    def require_file(self, *parts: str | Path) -> Path:
        """Return a required project file or raise a concise error."""
        file_path = self.path(*parts)

        if not file_path.is_file():
            relative_path = Path(*parts).as_posix()
            raise FileNotFoundError(
                f"Required project file not found: {relative_path}"
            )

        return file_path

    def require_directory(self, *parts: str | Path) -> Path:
        """Return a required project directory or raise a concise error."""
        directory_path = self.path(*parts)

        if not directory_path.is_dir():
            relative_path = Path(*parts).as_posix()
            raise FileNotFoundError(
                f"Required project directory not found: {relative_path}"
            )

        return directory_path


def _candidate_roots(start: str | Path | None = None) -> Iterator[Path]:
    """Yield possible project roots without repeating candidates."""
    seeds: list[Path] = []

    configured_root = os.getenv(PROJECT_ROOT_ENV)
    if configured_root:
        seeds.append(Path(configured_root).expanduser())

    # Prefer the caller's working location when it belongs to a study project.
    start_path = Path(start).expanduser() if start else Path.cwd()
    seeds.append(start_path)

    # Editable installation maps this module back to project/scripts/ and acts
    # as a fallback when Jupyter starts outside the repository.
    module_location = Path(__file__).resolve()
    seeds.append(module_location.parents[1])

    observed: set[Path] = set()

    for seed in seeds:
        resolved_seed = seed.resolve()

        for candidate in (resolved_seed, *resolved_seed.parents):
            if candidate not in observed:
                observed.add(candidate)
                yield candidate


def _is_project_root(candidate: Path) -> bool:
    """Return whether a candidate has the complete study-checkout shape."""
    return (
        (candidate / PROJECT_MARKER).is_file()
        and (candidate / "pyproject.toml").is_file()
        and (candidate / "notebooks").is_dir()
    )


def find_project_root(start: str | Path | None = None) -> Path:
    """Find a project root authenticated by the complete marker set."""
    for candidate in _candidate_roots(start):
        if _is_project_root(candidate):
            return candidate

    raise ProjectContextError(
        "Dataset study project root not found. "
        "Expected an authenticated checkout containing: "
        f"{', '.join(marker.as_posix() for marker in PROJECT_MARKERS)}. "
        f"Install the project with 'python -m pip install -e .' or define "
        f"the {PROJECT_ROOT_ENV} environment variable."
    )


def get_project_context(
    start: str | Path | None = None,
) -> ProjectContext:
    """Resolve, validate, and return the current project context."""
    context = ProjectContext(root=find_project_root(start))
    context.require_file(PROJECT_MARKER)
    context.require_file("pyproject.toml")
    context.require_directory("notebooks")
    return context
