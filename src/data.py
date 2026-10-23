"""Carga verificada de los CSV y partición entrenamiento/validación sin fuga por entradas repetidas."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PARAMS, PROCESSED, ROOT

D = PARAMS["data"]
TARGET = D["target"]
ID_COL = D["id_col"]
DROP = D["drop_cols"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(which: str) -> Path:
    return ROOT / D[f"{which}_csv"]


def load_raw(which: str) -> pd.DataFrame:
    """Lee 'train' o 'test'. Falla con un mensaje claro si filas o SHA-256 no coinciden con lo esperado."""
    path = raw_path(which)
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecuta `dvc pull` (ver README).")
    got = sha256_file(path)
    if got != D[f"expected_{which}_sha256"]:
        raise ValueError(f"{path.name}: SHA-256 {got} distinto del esperado. ¿Archivo equivocado o intercambiado?")
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = D[f"expected_{which}_rows"]
    if len(df) != expected:
        raise ValueError(f"{path.name}: {len(df)} filas, se esperaban {expected}.")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Las 42 variables predictoras: todo menos id, attack_cat y label."""
    return [c for c in df.columns if c not in DROP + [TARGET]]


def pattern_hash(df: pd.DataFrame) -> pd.Series:
    """Huella del vector de entrada (sin id ni respuestas): filas con las mismas entradas -> mismo grupo."""
    return pd.util.hash_pandas_object(df[feature_columns(df)], index=False)


def prepare() -> dict:
    """Separa entrenamiento y validación agrupando por patrón de entradas. El test no se toca."""
    df = load_raw("train")
    groups = pattern_hash(df)
    sgkf = StratifiedGroupKFold(n_splits=PARAMS["split"]["n_splits"], shuffle=True, random_state=PARAMS["seed"])
    train_idx, val_idx = next(sgkf.split(df, df[TARGET], groups))

    shared = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
    if shared:
        raise RuntimeError(f"Fuga: {len(shared)} patrones de entrada están en entrenamiento y validación.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.iloc[train_idx].to_parquet(PROCESSED / "train.parquet", index=False)
    df.iloc[val_idx].to_parquet(PROCESSED / "val.parquet", index=False)

    # Asignación de filas reproducible: qué fila del archivo original fue a qué partición.
    asignacion = pd.DataFrame({
        "fila_archivo": range(len(df)),                       # posición 0-based dentro del CSV
        "id": df[ID_COL].to_numpy(),
        "particion": "entrenamiento",
        "grupo_patron": groups.map(lambda v: f"{int(v):016x}").to_numpy(),
    })
    asignacion.loc[val_idx, "particion"] = "validacion"
    asignacion.to_csv(PROCESSED / "split_assignment.csv", index=False)
    return {"train_rows": len(train_idx), "val_rows": len(val_idx), "patrones_compartidos": len(shared)}


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}: ejecuta `dvc repro prepare` o `dvc pull`.")
    return pd.read_parquet(path)
