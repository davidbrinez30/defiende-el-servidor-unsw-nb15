"""Predice con una versión guardada del modelo (preprocesamiento + modelo + umbral).

  # desde una carpeta local (después de `dvc pull`)
  python -m src.predict --release models/releases/v2 --input evidencias/recuperacion/entradas_validacion.csv --output salida.csv
  # desde Hugging Face, por SHA completo de la revisión
  python -m src.predict --hf-revision <SHA_COMPLETO> --input datos.csv --output salida.csv

La entrada debe traer las 42 columnas predictoras (puede traer además row_id, id, label, attack_cat, y_true: se ignoran).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import load_yaml
from src.modelio import predict_scores


def descargar(revision: str, destino: str | None = None) -> Path:
    from huggingface_hub import snapshot_download
    hf = load_yaml("equipo.yaml")["huggingface"]
    return Path(snapshot_download(f"{hf['usuario']}/{hf['repo_modelos']}", revision=revision, local_dir=destino))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--release", help="carpeta local de una versión (models/releases/v1 o v2)")
    g.add_argument("--hf-revision", help="SHA completo (o tag) de la revisión en Hugging Face")
    ap.add_argument("--hf-dir", default=None, help="carpeta donde guardar la descarga de Hugging Face")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    carpeta = Path(a.release) if a.release else descargar(a.hf_revision, a.hf_dir)
    df = pd.read_csv(a.input, encoding="utf-8-sig")
    out = predict_scores(carpeta, df)
    if "row_id" in df.columns:
        out.insert(0, "row_id", df["row_id"])
    (out.to_csv(a.output, index=False) if a.output else out.to_csv(sys.stdout, index=False))


if __name__ == "__main__":
    main()
