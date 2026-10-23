"""Rutas y parámetros compartidos por todas las etapas."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))

DATA_RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RUN_MODELS = MODELS / "runs"
EVIDENCIAS = ROOT / "evidencias"
CONFIGS = ROOT / "configs"

for _d in (PROCESSED, MODELS, RUN_MODELS, EVIDENCIAS):
    _d.mkdir(parents=True, exist_ok=True)


def load_yaml(name: str) -> dict:
    """Lee un YAML de configs/ (experimentos, versiones, equipo, conclusiones)."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
