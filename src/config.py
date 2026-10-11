"""Rutas y parámetros compartidos por todas las etapas."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

for _d in (PROCESSED, REPORTS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)
