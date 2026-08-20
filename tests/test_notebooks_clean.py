from __future__ import annotations

import json
from pathlib import Path

import pytest


OFFICIAL_NOTEBOOKS = (
    "01_data_understanding_and_exploration.ipynb",
    "02_data_preparation.ipynb",
    "03_model_selection_and_evaluation.ipynb",
    "04_final_model_and_bundle.ipynb",
    "05_inference_demo.ipynb",
)


@pytest.mark.parametrize("notebook_name", OFFICIAL_NOTEBOOKS)
def test_official_notebook_code_cells_are_clean(notebook_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / notebook_name).read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
