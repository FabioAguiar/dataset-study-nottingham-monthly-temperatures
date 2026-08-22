from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import matplotlib.image as mpimg
import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _markdown_image_destinations(readme: Path) -> list[str]:
    destinations = []
    for match in MARKDOWN_IMAGE.finditer(readme.read_text(encoding="utf-8")):
        destination = match.group(1).strip()
        if destination.startswith("<") and destination.endswith(">"):
            destination = destination[1:-1].strip()
        else:
            destination = destination.split(maxsplit=1)[0]
        destinations.append(unquote(destination))
    return destinations


def test_readme_local_png_assets_are_safe_valid_and_nontrivial() -> None:
    root = Path(__file__).resolve().parents[1]
    destinations = _markdown_image_destinations(root / "README.md")
    assert destinations, "README.md must catalog at least one documentation image."

    local_pngs: list[Path] = []
    for destination in destinations:
        parsed = urlsplit(destination)
        lowered = destination.lower()
        assert not lowered.startswith("file://")
        assert not lowered.startswith("data:")
        assert not Path(destination).is_absolute()

        if parsed.scheme or parsed.netloc:
            continue

        relative = Path(parsed.path)
        assert ".." not in relative.parts
        resolved = (root / relative).resolve()
        resolved.relative_to(root.resolve())

        if relative.suffix.lower() == ".png":
            local_pngs.append(resolved)

    assert local_pngs, "README.md must reference at least one local PNG asset."
    for image_path in local_pngs:
        assert image_path.exists()
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        assert image_path.read_bytes().startswith(PNG_SIGNATURE)

        image = np.asarray(mpimg.imread(image_path))
        assert image.ndim in (2, 3)
        height, width = image.shape[:2]
        assert width >= 600
        assert height >= 300
        assert np.isfinite(image).all()
        visible = image[..., :3] if image.ndim == 3 else image
        assert float(np.ptp(visible)) > 0.01
        assert float(np.std(visible)) > 0.001
