"""Carga verificada de los CSV y partición entrenamiento/validación sin fuga por duplicados."""
import hashlib
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PARAMS, PROCESSED, REPORTS, ROOT

D = PARAMS["data"]
TARGET = D["target"]
DROP = D["drop_cols"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(which: str) -> pd.DataFrame:
    """Lee 'train' o 'test' y falla con un mensaje claro si no coincide con lo esperado."""
    path = ROOT / D[f"{which}_csv"]
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecuta `dvc pull` o descarga los CSV (ver README).")
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = D[f"expected_{which}_rows"]
    if len(df) != expected:
        raise ValueError(
            f"{path.name}: {len(df)} filas, se esperaban {expected}. "
            "¿Están intercambiados training/testing o es otra versión del archivo?"
        )
    if which == "train":
        got = sha256(path)
        if got != D["expected_train_sha256"]:
            raise ValueError(f"SHA-256 de entrenamiento distinto: {got}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP + [TARGET]]


def pattern_hash(df: pd.DataFrame) -> pd.Series:
    """Identifica el vector de entrada (sin id ni respuestas): filas iguales -> mismo grupo."""
    return pd.util.hash_pandas_object(df[feature_columns(df)], index=False)


def prepare() -> dict:
    """Separa entrenamiento y validación agrupando por patrón de entradas. El test no se toca."""
    df = load_raw("train")
    groups = pattern_hash(df)
    sgkf = StratifiedGroupKFold(
        n_splits=PARAMS["split"]["n_splits"], shuffle=True, random_state=PARAMS["seed"]
    )
    train_idx, val_idx = next(sgkf.split(df, df[TARGET], groups))
    train, val = df.iloc[train_idx], df.iloc[val_idx]

    overlap = len(set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]))
    if overlap:
        raise RuntimeError(f"Fuga: {overlap} patrones de entrada están en train y val.")

    train.to_parquet(PROCESSED / "train.parquet", index=False)
    val.to_parquet(PROCESSED / "val.parquet", index=False)

    # Ambigüedad de etiquetas: mismo vector de entrada con label distinto
    lab_per_group = df.groupby(groups.values)[TARGET].nunique()
    report = {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_attack_rate": round(float(train[TARGET].mean()), 4),
        "val_attack_rate": round(float(val[TARGET].mean()), 4),
        "patrones_distintos_total": int(groups.nunique()),
        "filas_repetidas_adicionales_total": int(len(df) - groups.nunique()),
        "grupos_con_label_normal_y_ataque": int((lab_per_group > 1).sum()),
        "patrones_compartidos_train_val": overlap,
        "metodo": "StratifiedGroupKFold (1 de n pliegues como validación), grupo = hash de las 42 entradas",
        "excluidas": DROP,
        "seed": PARAMS["seed"],
        "n_features": len(feature_columns(df)),
    }
    (REPORTS / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}: corre primero la etapa `prepare`.")
    return pd.read_parquet(path)
