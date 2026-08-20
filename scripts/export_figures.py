"""Small mechanical helpers for exporting matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_figure(
    figure: Any,
    destination: str | Path,
    *,
    dpi: int = 160,
    overwrite: bool = True,
) -> Path:
    """Export a matplotlib ``Figure`` as a PNG and return its path."""

    if not hasattr(figure, "savefig"):
        raise TypeError("figure must be a matplotlib Figure with a savefig method.")
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    path = Path(destination)
    if path.suffix.lower() != ".png":
        raise ValueError("destination must use a .png suffix.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    return path
