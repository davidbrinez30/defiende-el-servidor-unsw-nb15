"""Exporta el seguimiento de MLflow a archivos portátiles y arma la tabla de comparación.

evidencias/experimentos.csv   una fila por configuración (hipótesis, parámetros, semilla, commit, datos, run ID, métricas, conclusión)
evidencias/mlflow/            experimento.json, runs.csv y runs/<experimento>/ con parámetros, métricas, etiquetas y artefactos
                              (con SHA-256) de cada run; no depende del servidor ni del computador donde se entrenó.
"""
import json
import shutil

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import EVIDENCIAS, PARAMS, PROCESSED, load_yaml
from src.data import sha256_file
from src.hipotesis import evaluar

SALIDA = EVIDENCIAS / "mlflow"
TABLA = EVIDENCIAS / "experimentos.csv"


def _artefactos(client: MlflowClient, run_id: str, carpeta: str = "") -> list[dict]:
    salida = []
    for a in client.list_artifacts(run_id, carpeta or None):
        if a.is_dir:
            salida += _artefactos(client, run_id, a.path)
        else:
            local = client.download_artifacts(run_id, a.path)
            salida.append({"ruta": a.path, "bytes": a.file_size, "sha256": sha256_file(local)})
    return salida


def exportar() -> pd.DataFrame:
    specs = load_yaml("experimentos.yaml")["experimentos"]
    tabla = pd.read_csv(PROCESSED / "val_proba" / "resumen_ejecuciones.csv")

    mlflow.set_tracking_uri(PARAMS["mlflow"]["tracking_uri"])
    client = MlflowClient()
    exp = client.get_experiment_by_name(PARAMS["mlflow"]["experiment"])
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    por_nombre = {r.info.run_name: r for r in runs}
    if len(runs) != len(specs) or set(por_nombre) != {s["id"] for s in specs}:
        raise RuntimeError("El almacén de MLflow no tiene exactamente una ejecución por experimento.")

    if SALIDA.exists():
        shutil.rmtree(SALIDA)
    (SALIDA / "runs").mkdir(parents=True)
    planas = []
    for s in specs:
        r = por_nombre[s["id"]]
        arts = _artefactos(client, r.info.run_id)
        meta = {
            "experimento": s["id"], "run_id": r.info.run_id, "estado": r.info.status,
            "inicio_utc_ms": r.info.start_time, "fin_utc_ms": r.info.end_time,
            "etiquetas": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
            "parametros": r.data.params, "metricas": r.data.metrics,
            "artefactos": arts,
            "modelo_completo_en": f"models/runs/{s['id']}.joblib (versionado con DVC; SHA-256 en artefactos)",
        }
        d = SALIDA / "runs" / s["id"]
        d.mkdir()
        (d / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy(client.download_artifacts(r.info.run_id, "validacion/metricas.json"), d / "validacion_metricas.json")
        planas.append({"experimento": s["id"], "run_id": r.info.run_id, **{f"param.{k}": v for k, v in r.data.params.items()},
                       **{f"metric.{k}": v for k, v in r.data.metrics.items()},
                       **{f"tag.{k}": v for k, v in meta["etiquetas"].items() if k != "hipotesis"}})
    pd.DataFrame(planas).to_csv(SALIDA / "runs.csv", index=False)
    (SALIDA / "experimento.json").write_text(json.dumps({
        "nombre": exp.name, "experiment_id": exp.experiment_id, "ejecuciones": len(runs),
        "tracking_uri_original": PARAMS["mlflow"]["tracking_uri"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SALIDA / "LEEME.md").write_text(_leeme(), encoding="utf-8")

    # Tabla de comparación (una fila por configuración).
    t = tabla.set_index("experimento")
    filas = []
    for s in specs:
        conf, texto = evaluar(s, tabla)
        r = t.loc[s["id"]]
        filas.append({
            "id": s["id"], "hipotesis": " ".join(s["hipotesis"].split()), "cambio": s["cambio"], "familia": s["familia"],
            "referencia": s["referencia"] or "",
            "parametros": json.dumps({**s["estimador"]["params"], "preprocesamiento": s["preprocesamiento"],
                                      "fraccion_entrenamiento": s["fraccion_entrenamiento"]}, ensure_ascii=False, sort_keys=True),
            "estimador": s["estimador"]["clase"], "semilla": PARAMS["seed"],
            "commit_git": r["git_commit"], "referencia_dvc_original_train_md5": by_tag(por_nombre[s["id"]], "dvc_original_train_md5"),
            "referencia_dvc_preparado_train_parquet_sha256": by_tag(por_nombre[s["id"]], "preparado_train.parquet_sha256"),
            "referencia_dvc_preparado_val_parquet_sha256": by_tag(por_nombre[s["id"]], "preparado_val.parquet_sha256"),
            "run_id": r["run_id"], "umbral": r["umbral"], "recall": r["recall"], "precision": r["precision"], "fpr": r["fpr"],
            "macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
            "macro_f1_umbral_0.5": r["macro_f1_t05"], "recall_umbral_0.5": r["recall_t05"], "fpr_umbral_0.5": r["fpr_t05"],
            "precision_umbral_0.5": r["precision_t05"],
            "segundos_entrenamiento": r["fit_seconds"], "modelo_sha256": r["modelo_sha256"],
            "hipotesis_confirmada": "" if conf is None else ("si" if conf else "no"),
            "conclusion_validacion": texto,
        })
    out = pd.DataFrame(filas)
    out.to_csv(TABLA, index=False)
    return out


def by_tag(run, nombre: str) -> str:
    return run.data.tags.get(nombre, "")


def _leeme() -> str:
    return f"""# Exportación de MLflow

Contenido (sin depender del servidor de quien entrenó):

- `experimento.json`: nombre e identificador del experimento `{PARAMS['mlflow']['experiment']}`.
- `runs.csv`: una fila por ejecución con parámetros (`param.*`), métricas (`metric.*`) y etiquetas (`tag.*`).
- `runs/<experimento>/run.json`: run ID, estado, etiquetas (hipótesis, cambio, commit de Git, referencias DVC),
  parámetros, métricas y lista de artefactos con tamaño y SHA-256.
- `runs/<experimento>/validacion_metricas.json`: métricas de validación al umbral de la regla y a 0,5.
- Los modelos completos no se copian aquí (pesan decenas de MB): están en `models/runs/<experimento>.joblib`, versionados con DVC,
  y su SHA-256 coincide con el del artefacto de MLflow y con `evidencias/experimentos.csv`.

Cruce con la tabla: `run_id` y `experimento` son las mismas columnas en `evidencias/experimentos.csv`.

## Volver a abrir el seguimiento en la interfaz de MLflow

    dvc pull
    dvc repro --force experiments   # recrea mlflow.db y mlruns con exactamente una ejecución por experimento (unos minutos)
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
