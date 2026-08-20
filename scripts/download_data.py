"""Reusable dataset acquisition helpers for dataset-study repositories.

Study-specific choices such as source identifiers and destination paths are
intentionally passed by callers. The module supports UCI ML Repository
datasets, Kaggle datasets, and direct HTTP, HTTPS, or FTP file downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
SUPPORTED_URL_SCHEMES: Final[frozenset[str]] = frozenset(
    {"http", "https", "ftp"}
)
DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS: Final[int] = 120
DEFAULT_UCI_DATA_FILENAME: Final[str] = "dataset.csv"
DEFAULT_UCI_METADATA_FILENAME: Final[str] = "metadata.json"
DEFAULT_UCI_VARIABLES_FILENAME: Final[str] = "variables.csv"

SourceKind = Literal["uci", "kaggle", "url"]


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or validated."""


@dataclass(frozen=True, slots=True)
class DatasetAcquisition:
    """Describe the materialized result of one dataset acquisition."""

    source_kind: SourceKind
    source_reference: str
    destination: Path
    resolved_path: Path
    files: tuple[Path, ...]
    project_root: Path

    @property
    def display_destination(self) -> str:
        """Return a project-relative destination for safe presentation."""
        try:
            return self.destination.relative_to(self.project_root).as_posix()
        except ValueError:
            return self.destination.name

    @property
    def relative_files(self) -> tuple[str, ...]:
        """Return safe project-relative POSIX file paths."""
        rendered: list[str] = []

        for file_path in self.files:
            try:
                display_path = file_path.relative_to(self.project_root)
            except ValueError:
                rendered.append(file_path.name)
            else:
                rendered.append(display_path.as_posix())

        return tuple(rendered)

    def require_files(self, pattern: str) -> tuple[Path, ...]:
        """Return acquired files matching a filename or glob pattern."""
        normalized_pattern = pattern.strip()
        if not normalized_pattern:
            raise ValueError("File selection pattern cannot be empty.")

        matches = tuple(
            file_path
            for file_path in self.files
            if file_path.match(normalized_pattern)
            or file_path.name == normalized_pattern
        )

        if not matches:
            raise FileNotFoundError(
                f"No acquired file matches '{normalized_pattern}' in "
                f"{self.display_destination}. Available files: "
                f"{list(self.relative_files)}"
            )

        return matches

    def require_one_file(self, pattern: str) -> Path:
        """Return exactly one matching file or raise a clear error."""
        matches = self.require_files(pattern)

        if len(matches) != 1:
            rendered = [path.name for path in matches]
            raise RuntimeError(
                f"Expected exactly one file matching '{pattern}', "
                f"found {len(matches)}: {rendered}"
            )

        return matches[0]


def resolve_project_path(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve a path relative to the supplied project root."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = root / candidate

    candidate = candidate.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Dataset destination must remain inside the project root."
        ) from exc

    return candidate


def discover_dataset_files(
    directory: str | Path,
    *,
    include_hidden: bool = False,
) -> tuple[Path, ...]:
    """Return deterministic recursive file paths from a dataset directory."""
    root = Path(directory).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root.name}"
        )

    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts
        if not include_hidden and any(
            part.startswith(".") for part in relative_parts
        ):
            continue

        files.append(path.resolve())

    return tuple(sorted(files, key=lambda item: item.as_posix()))


@contextmanager
def _suppress_console_output(enabled: bool):
    """Temporarily suppress third-party output, warnings, and logging.

    The implementation is cross-platform because ``os.devnull`` resolves to
    the operating system's null device (for example, ``/dev/null`` on POSIX
    systems and ``NUL`` on Windows).
    """
    if not enabled:
        yield
        return

    previous_logging_disable_level = logging.root.manager.disable

    with (
        open(os.devnull, "w", encoding="utf-8") as output_sink,
        warnings.catch_warnings(),
    ):
        # kagglehub imports tqdm.auto, which can emit this warning when the
        # optional Jupyter widget integration is not installed.
        warnings.filterwarnings(
            "ignore",
            message=r"IProgress not found.*",
        )

        logging.disable(logging.CRITICAL)

        try:
            with (
                redirect_stdout(output_sink),
                redirect_stderr(output_sink),
            ):
                yield
        finally:
            logging.disable(previous_logging_disable_level)


def _validate_filename(filename: str) -> str:
    """Reject empty or path-like filenames for direct downloads."""
    candidate = filename.strip()

    if not candidate or candidate in {".", ".."}:
        raise ValueError("The destination filename cannot be empty.")

    if Path(candidate).name != candidate:
        raise ValueError(
            "The destination filename must not contain directories."
        )

    return candidate


def calculate_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest for a local file."""
    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(DEFAULT_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_sha256(file_path: Path, expected_sha256: str | None) -> None:
    """Validate a file checksum when an expected digest was supplied."""
    if expected_sha256 is None:
        return

    expected = expected_sha256.strip().lower()

    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "sha256 must contain exactly 64 hexadecimal characters."
        )

    observed = calculate_sha256(file_path)

    if observed != expected:
        raise DatasetDownloadError(
            f"SHA-256 mismatch for {file_path.name}: "
            f"expected {expected}, observed {observed}."
        )



def _json_compatible(value):
    """Convert common repository metadata values to JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_compatible(item_method())
        except (TypeError, ValueError):
            pass

    return str(value)


def _materialize_uci_dataset(
    dataset,
    destination: Path,
) -> tuple[Path, Path, Path]:
    """Persist the UCI tabular import and its descriptive source metadata."""
    data_path = destination / DEFAULT_UCI_DATA_FILENAME
    metadata_path = destination / DEFAULT_UCI_METADATA_FILENAME
    variables_path = destination / DEFAULT_UCI_VARIABLES_FILENAME

    original = getattr(getattr(dataset, "data", None), "original", None)
    variables = getattr(dataset, "variables", None)
    metadata = getattr(dataset, "metadata", None)

    if original is None or not hasattr(original, "to_csv"):
        raise DatasetDownloadError(
            "ucimlrepo did not return a tabular dataset in data.original."
        )

    if variables is None or not hasattr(variables, "to_csv"):
        raise DatasetDownloadError(
            "ucimlrepo did not return the expected variable metadata table."
        )

    if metadata is None:
        raise DatasetDownloadError(
            "ucimlrepo did not return dataset metadata."
        )

    temporary_paths = {
        data_path: data_path.with_name(f".{data_path.name}.part"),
        metadata_path: metadata_path.with_name(
            f".{metadata_path.name}.part"
        ),
        variables_path: variables_path.with_name(
            f".{variables_path.name}.part"
        ),
    }

    for temporary_path in temporary_paths.values():
        temporary_path.unlink(missing_ok=True)

    try:
        original.to_csv(temporary_paths[data_path], index=False)
        variables.to_csv(temporary_paths[variables_path], index=False)
        temporary_paths[metadata_path].write_text(
            json.dumps(
                _json_compatible(metadata),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        for final_path, temporary_path in temporary_paths.items():
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise

    return data_path, metadata_path, variables_path


def acquire_uci_dataset(
    dataset_id: int,
    destination: str | Path,
    *,
    force: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Fetch a UCI dataset with ``ucimlrepo`` and materialize it locally.

    The official UCI Python client returns pandas dataframes rather than a
    project-local raw file. This helper persists three immutable acquisition
    artifacts under the requested raw-data directory: ``dataset.csv``,
    ``metadata.json``, and ``variables.csv``.
    """
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int):
        raise TypeError("UCI dataset_id must be an integer.")

    if dataset_id <= 0:
        raise ValueError("UCI dataset_id must be greater than zero.")

    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_files = (
        output_dir / DEFAULT_UCI_DATA_FILENAME,
        output_dir / DEFAULT_UCI_METADATA_FILENAME,
        output_dir / DEFAULT_UCI_VARIABLES_FILENAME,
    )
    existing = tuple(path.exists() for path in expected_files)

    if all(existing) and not force:
        return DatasetAcquisition(
            source_kind="uci",
            source_reference=f"UCI ML Repository dataset {dataset_id}",
            destination=output_dir,
            resolved_path=expected_files[0].resolve(),
            files=discover_dataset_files(output_dir),
            project_root=root,
        )

    if any(existing) and not force:
        raise DatasetDownloadError(
            "The UCI raw-data directory contains an incomplete prior "
            "materialization. Re-run with force=True to replace it."
        )

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise DatasetDownloadError(
            "ucimlrepo is not installed in the active Python environment. "
            "Install the project dependencies with: "
            "python -m pip install -e ."
        ) from exc

    try:
        dataset = fetch_ucirepo(id=dataset_id)
        data_path, _, _ = _materialize_uci_dataset(
            dataset,
            output_dir,
        )
    except DatasetDownloadError:
        raise
    except Exception as exc:
        raise DatasetDownloadError(
            f"UCI acquisition failed for dataset {dataset_id}: {exc}"
        ) from exc

    files = discover_dataset_files(output_dir)

    if not files:
        raise DatasetDownloadError(
            "The UCI acquisition completed but no visible files "
            "were found in the destination directory."
        )

    return DatasetAcquisition(
        source_kind="uci",
        source_reference=f"UCI ML Repository dataset {dataset_id}",
        destination=output_dir,
        resolved_path=data_path.resolve(),
        files=files,
        project_root=root,
    )


def download_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    show_progress: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Download a Kaggle dataset or one file into a project directory.

    Parameters
    ----------
    show_progress:
        When ``True``, preserve ``kagglehub`` progress and diagnostic output.
        The default suppresses third-party console output so absolute local
        paths are not exposed in notebook results.
    """
    normalized_handle = handle.strip()

    if not normalized_handle or "/" not in normalized_handle:
        raise ValueError(
            "Kaggle handle must use the 'owner/dataset' format."
        )

    output_dir = resolve_project_path(
        destination,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # The import itself is kept inside the suppression context because
        # kagglehub imports tqdm.auto, which may emit notebook widget warnings.
        with _suppress_console_output(enabled=not show_progress):
            import kagglehub

            resolved_path = kagglehub.dataset_download(
                normalized_handle,
                path=dataset_file,
                output_dir=str(output_dir),
                force_download=force,
            )
    except ImportError as exc:
        raise DatasetDownloadError(
            "kagglehub is not installed in the active Python environment. "
            "Install kagglehub before using Kaggle acquisition."
        ) from exc
    except TypeError as exc:
        raise DatasetDownloadError(
            "The installed kagglehub version does not support output_dir. "
            "Upgrade the project dependencies."
        ) from exc
    except Exception as exc:
        raise DatasetDownloadError(
            f"Kaggle download failed for '{normalized_handle}': {exc}"
        ) from exc

    result = Path(resolved_path).expanduser().resolve()

    if not result.exists():
        raise DatasetDownloadError(
            "kagglehub returned a path that does not exist."
        )

    return result


def acquire_kaggle_dataset(
    handle: str,
    destination: str | Path,
    *,
    dataset_file: str | None = None,
    force: bool = False,
    show_progress: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire a Kaggle dataset and return a notebook-friendly result.

    Third-party progress output is hidden by default to avoid exposing
    absolute paths in notebook output. Set ``show_progress=True`` when
    interactive download diagnostics are needed.
    """
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)

    resolved_path = download_kaggle_dataset(
        handle=handle,
        destination=output_dir,
        dataset_file=dataset_file,
        force=force,
        show_progress=show_progress,
        project_root=root,
    )

    files = discover_dataset_files(output_dir)

    if not files:
        raise DatasetDownloadError(
            "The Kaggle acquisition completed but no visible files "
            "were found in the destination directory."
        )

    return DatasetAcquisition(
        source_kind="kaggle",
        source_reference=handle.strip(),
        destination=output_dir,
        resolved_path=resolved_path,
        files=files,
        project_root=root,
    )


def download_url_file(
    url: str,
    destination: str | Path,
    *,
    filename: str | None = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sha256: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Download one HTTP, HTTPS, or FTP file atomically."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme not in SUPPORTED_URL_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_URL_SCHEMES))
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. "
            f"Supported schemes: {supported}."
        )

    resolved_filename = filename or unquote(Path(parsed.path).name)
    resolved_filename = _validate_filename(resolved_filename)

    output_dir = resolve_project_path(
        destination,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / resolved_filename

    if output_path.exists() and not force:
        _verify_sha256(output_path, sha256)
        return output_path.resolve()

    temporary_path = output_path.with_name(
        f".{output_path.name}.part"
    )
    temporary_path.unlink(missing_ok=True)

    request: str | Request

    if scheme in {"http", "https"}:
        request = Request(
            url,
            headers={"User-Agent": "dataset-study-downloader/1.0"},
        )
    else:
        request = url

    try:
        with (
            urlopen(request, timeout=timeout) as response,
            temporary_path.open("wb") as target,
        ):
            shutil.copyfileobj(
                response,
                target,
                length=DEFAULT_CHUNK_SIZE,
            )

        _verify_sha256(temporary_path, sha256)
        os.replace(temporary_path, output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"Download failed for '{url}': {exc}"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path.resolve()


def acquire_url_file(
    url: str,
    destination: str | Path,
    *,
    filename: str | None = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sha256: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> DatasetAcquisition:
    """Acquire one URL file and return a notebook-friendly result."""
    root = Path(project_root).expanduser().resolve()
    output_dir = resolve_project_path(destination, project_root=root)

    resolved_path = download_url_file(
        url=url,
        destination=output_dir,
        filename=filename,
        force=force,
        timeout=timeout,
        sha256=sha256,
        project_root=root,
    )

    return DatasetAcquisition(
        source_kind="url",
        source_reference=url,
        destination=output_dir,
        resolved_path=resolved_path,
        files=discover_dataset_files(output_dir),
        project_root=root,
    )


def _print_acquisition(acquisition: DatasetAcquisition) -> None:
    """Render a compact deterministic acquisition summary."""
    print(f"Source type: {acquisition.source_kind}")
    print(f"Source: {acquisition.source_reference}")
    print(f"Destination: {acquisition.display_destination}")
    print("Files:")

    for file_path in acquisition.relative_files:
        print(f"- {file_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire source datasets into a study repository.",
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    uci_parser = subparsers.add_parser(
        "uci",
        help="Acquire a dataset through the official ucimlrepo client.",
    )
    uci_parser.add_argument(
        "dataset_id",
        type=int,
        help="Numeric UCI Machine Learning Repository dataset ID.",
    )
    uci_parser.add_argument(
        "--destination",
        required=True,
        help="Project-relative output directory.",
    )
    uci_parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch again and replace the existing UCI materialization.",
    )

    kaggle_parser = subparsers.add_parser(
        "kaggle",
        help="Acquire a dataset through kagglehub.",
    )
    kaggle_parser.add_argument(
        "handle",
        help="Kaggle dataset handle in owner/dataset form.",
    )
    kaggle_parser.add_argument(
        "--destination",
        required=True,
        help="Project-relative output directory.",
    )
    kaggle_parser.add_argument(
        "--dataset-file",
        help="Optional path to one file inside the Kaggle dataset.",
    )
    kaggle_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again instead of reusing an existing result.",
    )
    kaggle_parser.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Show kagglehub progress and diagnostic output. "
            "Disabled by default to avoid exposing absolute local paths."
        ),
    )

    url_parser = subparsers.add_parser(
        "url",
        help="Acquire one file from an HTTP, HTTPS, or FTP URL.",
    )
    url_parser.add_argument(
        "url",
        help="Source HTTP, HTTPS, or FTP URL.",
    )
    url_parser.add_argument(
        "--destination",
        required=True,
        help="Project-relative output directory.",
    )
    url_parser.add_argument(
        "--filename",
        help="Local filename; defaults to the URL path component.",
    )
    url_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and replace an existing local file.",
    )
    url_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Connection/read timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS})."
        ),
    )
    url_parser.add_argument(
        "--sha256",
        help="Optional expected SHA-256 checksum.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.source == "uci":
            acquisition = acquire_uci_dataset(
                dataset_id=args.dataset_id,
                destination=args.destination,
                force=args.force,
            )
        elif args.source == "kaggle":
            acquisition = acquire_kaggle_dataset(
                handle=args.handle,
                destination=args.destination,
                dataset_file=args.dataset_file,
                force=args.force,
                show_progress=args.show_progress,
            )
        else:
            acquisition = acquire_url_file(
                url=args.url,
                destination=args.destination,
                filename=args.filename,
                force=args.force,
                timeout=args.timeout,
                sha256=args.sha256,
            )
    except (DatasetDownloadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_acquisition(acquisition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
