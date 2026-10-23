"""Guardar y cargar una versión completa (preprocesamiento + modelo + umbral) en formato skops.

skops evita ejecutar código arbitrario al cargar: solo se aceptan los tipos declarados como confiables en config.json
(aquí, únicamente tipos internos de scikit-learn que el propio entrenamiento produjo).
"""
import json
from pathlib import Path

import pandas as pd
import skops.io as sio

from src.data import sha256_file


def load_release(release_dir) -> tuple:
    """Devuelve (pipeline, config). Comprueba el SHA-256 del modelo y que los tipos no confiables estén declarados."""
    d = Path(release_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    path = d / cfg["modelo_archivo"]
    got = sha256_file(path)
    if got != cfg["modelo_sha256"]:
        raise ValueError(f"SHA-256 de {path.name} distinto del declarado en config.json: {got}")
    permitidos = set(cfg["tipos_confiables_skops"])
    extra = set(sio.get_untrusted_types(file=path)) - permitidos
    if extra:
        raise ValueError(f"El modelo contiene tipos no declarados como confiables: {sorted(extra)}")
    return sio.load(path, trusted=sorted(permitidos)), cfg


def predict_scores(release_dir, df: pd.DataFrame) -> pd.DataFrame:
    """score_attack = probabilidad de ataque; y_pred = 1 si score_attack >= umbral."""
    pipe, cfg = load_release(release_dir)
    score = pipe.predict_proba(df[cfg["columnas_entrada"]])[:, 1]
    return pd.DataFrame({"score_attack": score, "y_pred": (score >= cfg["umbral"]).astype(int)}, index=df.index)
