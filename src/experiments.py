"""Corre los experimentos de configs/experimentos.yaml con las MISMAS particiones y los registra en MLflow.

Cada ejecución guarda: hipótesis, cambio, parámetros, semilla, commit de Git, referencias de datos (DVC), métricas de
validación (al umbral de la regla y a 0,5) y el modelo COMPLETO (preprocesamiento + estimador) como artefacto.
Se parte de un almacén MLflow vacío para que haya exactamente una ejecución por experimento.
"""
import shutil
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODELS, PARAMS, PROCESSED, ROOT, RUN_MODELS, load_yaml
from src.data import TARGET, feature_columns, load_split, sha256_file
from src.features import build_preprocessor
from src.metrics import POLITICA_DENOMINADOR_CERO, at_threshold, pick_threshold, threshold_free
from src.provenance import data_refs, git_info

M = PARAMS["mlflow"]
ESTIMADORES = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def make_pipeline(spec: dict, numeric_cols: list[str]) -> Pipeline:
    est = ESTIMADORES[spec["estimador"]["clase"]](**spec["estimador"]["params"])
    prep = build_preprocessor(numeric_cols, **spec["preprocesamiento"])
    return Pipeline([("prep", prep), ("model", est)])


def split_xy(df: pd.DataFrame):
    return df[feature_columns(df)], df[TARGET].to_numpy()


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in PARAMS["data"]["categorical"]]


def _reset_mlflow_store():
    """Almacén limpio: una sola ejecución por experimento (dvc repro vuelve a crear todo)."""
    for p in (ROOT / "mlflow.db", ROOT / "mlruns"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def run_all(solo: str | None = None) -> pd.DataFrame:
    """Corre todos los experimentos (etapa de DVC) o, con `solo`, uno de ellos en un almacén de prueba aparte:
    no toca mlflow.db, models/runs ni data/processed/val_proba, así que sirve para reproducir un experimento sin invalidar nada."""
    specs = load_yaml("experimentos.yaml")["experimentos"]
    if solo:
        specs = [s for s in specs if s["id"] == solo]
        if not specs:
            raise SystemExit(f"Experimento desconocido: {solo}. Opciones: {[s['id'] for s in load_yaml('experimentos.yaml')['experimentos']]}")
        uri, destino, proba_dir = "sqlite:///mlflow_prueba.db", MODELS / "pruebas", MODELS / "pruebas" / "val_proba"
        for p in (ROOT / "mlflow_prueba.db",):
            p.unlink(missing_ok=True)
    else:
        _reset_mlflow_store()
        uri, destino, proba_dir = M["tracking_uri"], RUN_MODELS, PROCESSED / "val_proba"
    train, val = load_split("train"), load_split("val")
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    num = numeric_columns(Xtr)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(M["experiment"])
    git, refs = git_info(), data_refs()
    if git["codigo_sin_confirmar"]:
        print("AVISO: hay cambios sin confirmar en el código; el commit registrado no describe exactamente esta ejecución.")
    proba_dir.mkdir(parents=True, exist_ok=True)
    destino.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in specs:
        Xf, yf = Xtr, ytr
        if spec["fraccion_entrenamiento"] < 1.0:  # misma semilla -> mismo subconjunto siempre
            idx = np.random.RandomState(PARAMS["seed"]).permutation(len(Xtr))[: int(len(Xtr) * spec["fraccion_entrenamiento"])]
            Xf, yf = Xtr.iloc[idx], ytr[idx]

        with mlflow.start_run(run_name=spec["id"]) as run:
            mlflow.set_tags({
                "etapa": "experimento", "hipotesis": " ".join(spec["hipotesis"].split()), "cambio": spec["cambio"],
                "referencia": str(spec["referencia"]), "familia": spec["familia"],
                "git_commit": git["git_commit"], "codigo_sin_confirmar": str(git["codigo_sin_confirmar"]),
                **{k: str(v) for k, v in refs.items()},
            })
            mlflow.log_params({
                "estimador": spec["estimador"]["clase"], "semilla": PARAMS["seed"],
                "log1p": spec["preprocesamiento"]["log1p"], "escalado": spec["preprocesamiento"]["escalado"],
                "fraccion_entrenamiento": spec["fraccion_entrenamiento"], "filas_entrenamiento": len(Xf),
                "filas_validacion": len(Xva), "regla_umbral_max_fpr": PARAMS["selection"]["max_fpr"],
                **{f"estimador__{k}": v for k, v in spec["estimador"]["params"].items()},
            })
            t0 = time.time()
            pipe = make_pipeline(spec, num).fit(Xf, yf)
            fit_s = time.time() - t0
            proba = pipe.predict_proba(Xva)[:, 1]
            np.save(proba_dir / f"{spec['id']}.npy", proba)

            thr = pick_threshold(yva, proba, PARAMS["selection"]["max_fpr"])
            m_regla, m_05 = at_threshold(yva, proba, thr), at_threshold(yva, proba, 0.5)
            libres = threshold_free(yva, proba) if np.unique(proba).size > 1 else {"roc_auc": 0.5, "pr_auc": float(yva.mean())}
            mlflow.log_metrics({
                **{f"val_{k}": v for k, v in m_regla.items()},                      # al umbral de la regla
                **{f"val_t05_{k}": v for k, v in m_05.items() if k != "umbral"},    # a 0,5
                **{f"val_{k}": v for k, v in libres.items()}, "fit_seconds": fit_s,
            })

            # Modelo completo: preprocesamiento + estimador (comprimido para que quepa en GitHub/DVC).
            modelo = destino / f"{spec['id']}.joblib"
            joblib.dump(pipe, modelo, compress=3)
            mlflow.log_artifact(str(modelo), artifact_path="modelo")
            mlflow.log_artifact(str(proba_dir / f"{spec['id']}.npy"), artifact_path="validacion")
            mlflow.log_dict({"politica_denominador_cero": POLITICA_DENOMINADOR_CERO, "umbral": thr,
                             "validacion_al_umbral": m_regla, "validacion_a_0.5": m_05, **libres},
                            "validacion/metricas.json")
            rows.append({
                "experimento": spec["id"], "familia": spec["familia"], "referencia": spec["referencia"],
                "run_id": run.info.run_id, "git_commit": git["git_commit"], "umbral": thr, **{k: m_regla[k] for k in
                ("recall", "precision", "fpr", "macro_f1", "accuracy", "tp", "fp", "fn", "tn")}, **libres,
                "macro_f1_t05": m_05["macro_f1"], "recall_t05": m_05["recall"], "fpr_t05": m_05["fpr"],
                "precision_t05": m_05["precision"], "fit_seconds": round(fit_s, 1),
                "modelo_sha256": sha256_file(modelo), "modelo_archivo": str(modelo.relative_to(ROOT)),
            })
            print(f"{spec['id']:32s} umbral={thr:.4f} macroF1={m_regla['macro_f1']:.4f} recall={m_regla['recall']:.4f} "
                  f"FPR={m_regla['fpr']:.4f} AUC={libres['roc_auc']:.4f} ({fit_s:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    if not solo:
        out.to_csv(PROCESSED / "val_proba" / "resumen_ejecuciones.csv", index=False)
    return out
