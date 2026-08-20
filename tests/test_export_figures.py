from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts import export_figures


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_export_figure_creates_missing_directory_and_png(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    destination = tmp_path / "missing" / "figure.png"

    result = export_figures.export_figure(figure, destination)

    assert result == destination
    assert destination.read_bytes().startswith(PNG_SIGNATURE)
    plt.close(figure)


def test_export_figure_overwrite_policy_is_explicit(tmp_path):
    destination = tmp_path / "figure.png"
    first, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    export_figures.export_figure(first, destination)
    original = destination.read_bytes()
    plt.close(first)

    second, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    with pytest.raises(FileExistsError):
        export_figures.export_figure(second, destination, overwrite=False)

    assert destination.read_bytes() == original
    export_figures.export_figure(second, destination, overwrite=True)
    assert destination.read_bytes().startswith(PNG_SIGNATURE)
    plt.close(second)


def test_export_figure_rejects_invalid_inputs(tmp_path):
    with pytest.raises(TypeError):
        export_figures.export_figure(object(), tmp_path / "figure.png")

    figure, _ = plt.subplots()
    with pytest.raises(ValueError):
        export_figures.export_figure(figure, tmp_path / "figure.jpg")
    plt.close(figure)


def test_export_figures_module_has_no_absolute_hardcoded_paths():
    source = Path(export_figures.__file__).read_text()
    assert "/home/" not in source
    assert "C:\\" not in source
