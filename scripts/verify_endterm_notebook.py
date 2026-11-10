"""Validate the notebook and execute it when the official data prerequisite exists."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "astana_housing_endterm.ipynb"
DATA = ROOT / "astana_real_estate_dataset.csv"

notebook = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(notebook)
print(f"Notebook schema valid: {len(notebook.cells)} cells")
if not DATA.exists():
    print("Execution not attempted: official CSV is missing (expected prerequisite blocker).")
    raise SystemExit(0)

client = NotebookClient(notebook, timeout=1800, kernel_name="python3")
client.execute(cwd=str(ROOT))
nbformat.write(notebook, NOTEBOOK)
print("Clean notebook execution completed successfully.")
