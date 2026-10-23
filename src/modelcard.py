"""Genera las model cards (docs/model_cards/<vN>.md) a partir de los archivos de evidencia: ninguna cifra se escribe a mano."""
import json

from src.config import ROOT, load_yaml
from src.release import RELEASES

CARDS = ROOT / "docs" / "model_cards"
NOMBRES = {"v1": "regresión logística con log(1+x)", "v2": None}


def f(x, d=4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def pct(x, d=2) -> str:
    return f"{100 * x:.{d}f} %".replace(".", ",")


def _tabla_metricas(m: dict) -> str:
    fila = lambda n, v: f"| {n} | {v} |"
    return "\n".join([
        "| Métrica (clase positiva = ataque) | Validación |", "|---|---|",
        fila("Recall de ataque", pct(m["recall"])), fila("Precision de ataque", pct(m["precision"])),
        fila("Tasa de falsos positivos (FPR)", pct(m["fpr"])), fila("Macro-F1", f(m["macro_f1"])),
        fila("Accuracy (complementaria)", pct(m["accuracy"])),
        fila("Matriz de confusión (VP / FP / FN / VN)", f"{miles(m['tp'])} / {miles(m['fp'])} / {miles(m['fn'])} / {miles(m['tn'])}"),
    ])


def render(version: str, test: dict | None = None, hf: dict | None = None) -> str:
    cfg = json.loads((RELEASES / version / "config.json").read_text(encoding="utf-8"))
    eq = load_yaml("equipo.yaml")
    hfu = eq["huggingface"]
    repo_hf = f"{hfu['usuario']}/{hfu['repo_modelos']}"
    url_git = cfg["git"]["repositorio"]
    mv = cfg["metricas_validacion"]
    m, m05 = mv["al_umbral"], mv["a_0.5_referencia"]
    dvc = cfg["dvc"]
    titulo = {"v1": "v1 — referencia funcional", "v2": "v2 — revisión basada en los experimentos"}[version]

    partes = [f"""---
language: es
license: other
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, skops, mlops, education]
---

# Defiende el servidor · {titulo}

Clasificador binario de conexiones de red (**0 = normal, 1 = ataque**) entrenado con las particiones publicadas de
UNSW-NB15. Es un producto del taller de MLOps de la Universidad Externado de Colombia (2026-II); su objetivo es
aprender a **versionar y recuperar** código, datos, experimentos y modelos, no desplegar un detector real.

- **Modelo de esta versión:** {cfg['descripcion']} (experimento `{cfg['experimento']}`, familia `{cfg['familia']}`).
- **Umbral:** `{cfg['umbral']:.6f}`. Regla: `{cfg['regla_umbral']}`.
- **Revisión:** esta card describe la revisión etiquetada `{version}` de `{repo_hf}`; su SHA completo está en
  `evidencias/hf_publicacion.json` del repositorio de código.

## Qué contiene esta revisión

| Archivo | Para qué sirve |
|---|---|
| `modelo.skops` | Preprocesamiento + estimador completos en formato skops (no ejecuta código arbitrario al cargarse). |
| `config.json` | Umbral, columnas de entrada, hashes, run de MLflow, commit de Git, referencias DVC y métricas. |
| `requirements.txt` | Versiones exactas de las dependencias para cargarlo y usarlo. |
| `README.md` | Esta card. |

## Propósito y usos

- **Uso previsto:** estudiar el ciclo de vida de un modelo (datos → experimento → versión → recuperación) y comparar v1 con v2.
- **Fuera de alcance:** proteger una red real. UNSW-NB15 es un benchmark histórico de laboratorio (2015) y no demuestra
  eficacia sobre tráfico actual. No debe usarse como único control de seguridad.

## Datos

- UNSW-NB15, particiones publicadas `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332 filas),
  copia fijada de un mirror de terceros (los SHA-256 coinciden con los del enunciado, no están autenticados contra UNSW).
  Fuente oficial y condiciones de uso: https://research.unsw.edu.au/projects/unsw-nb15-dataset (uso académico; citar a Moustafa y Slay, 2015).
- **Entradas:** 42 variables (39 numéricas y 3 categóricas: `proto`, `service`, `state`). Se excluyen `label` (respuesta),
  `attack_cat` (revela la respuesta) e `id` (identifica filas; separa las clases solo por cómo viene ordenado el archivo).
- **Entrenamiento y validación:** salen solo del archivo de entrenamiento, con `StratifiedGroupKFold` (semilla {cfg['semilla']}); las
  filas con entradas idénticas siempre quedan juntas. El test no se usó para decidir nada de esta versión.

## Modelo, preprocesamiento y umbral

- Todo el preprocesamiento (log(1+x), escalado, codificación de categorías) está dentro del mismo objeto `Pipeline` y
  se ajustó **solo con entrenamiento**.
- **Umbral:** se eligió en validación con una regla declarada antes de ver resultados: *mayor recall con tasa de falsos positivos <= 5 %*.
- Denominadores cero: {cfg['politica_denominador_cero']}

## Métricas de validación ({miles(mv['filas'])} filas)""" + f"""

Al umbral `{cfg['umbral']:.6f}` (el que usa esta versión):

{_tabla_metricas(m)}

ROC-AUC {f(mv['roc_auc'])} · PR-AUC {f(mv['pr_auc'])}. Como referencia, con el umbral 0,5 la FPR sería {pct(m05['fpr'])} y el recall {pct(m05['recall'])}.
"""]

    if version == "v2" and "comparacion_con_v1" in cfg:
        etq = {"recall": "Recall", "precision": "Precision", "fpr": "FPR", "macro_f1": "Macro-F1", "accuracy": "Accuracy"}
        filas = "\n".join(
            f"| {etq[k]} | {f(v['v1'])} | {f(v['v2'])} | {'+' if v['diferencia'] >= 0 else ''}{f(v['diferencia'])} | {v['efecto']} |"
            for k, v in cfg["comparacion_con_v1"].items())
        partes.append(f"""## Qué cambió respecto a v1

Cambio único: el tipo de modelo (regresión logística → {cfg['familia'].replace('_', ' ')}); preprocesamiento, particiones y regla de umbral son los mismos. v2 es el experimento con mayor macro-F1 en validación entre los de modelo real (`evidencias/seleccion_validacion.json`).

| Métrica (validación) | v1 | v2 | Diferencia | Efecto |
|---|---|---|---|---|
{filas}

Lo que se acepta: un modelo más pesado y más lento de entrenar, a cambio del mejor equilibrio entre detectar ataques y no dar falsas alarmas en validación.
""")

    partes.append(f"""## Trazabilidad

| Pieza | Identificador |
|---|---|
| Código | repositorio {url_git}, tag `{version}`; commit que entrenó el modelo `{cfg['git']['commit_codigo_que_entreno']}` |
| Datos originales (DVC, md5) | entrenamiento `{dvc['dvc_original_train_md5']}` · test `{dvc['dvc_original_test_md5']}` |
| Datos preparados (sha256) | `train.parquet` `{dvc['preparado_train.parquet_sha256']}` · `val.parquet` `{dvc['preparado_val.parquet_sha256']}` · `split_assignment.csv` `{dvc['preparado_split_assignment.csv_sha256']}` |
| Experimento MLflow | `{cfg['mlflow']['experimento']}` · run ID `{cfg['mlflow']['run_id']}` |
| Modelo (`modelo.skops`) | SHA-256 `{cfg['modelo_sha256']}` |
| Modelo del run (joblib) | SHA-256 `{cfg['modelo_joblib_del_run_sha256']}` (la conversión a skops reproduce sus puntuaciones; diferencia máxima {cfg['verificacion_conversion']['diferencia_maxima_score']}) |

## Cómo usarlo

```python
import json
import pandas as pd
import skops.io as sio
from huggingface_hub import snapshot_download

carpeta = snapshot_download("{repo_hf}", revision="{version}")   # o el SHA completo de la revisión
cfg = json.load(open(f"{{carpeta}}/config.json", encoding="utf-8"))
modelo = sio.load(f"{{carpeta}}/modelo.skops", trusted=cfg["tipos_confiables_skops"])

df = pd.read_csv("conexiones.csv")                                  # mismas 42 columnas de entrada
score_attack = modelo.predict_proba(df[cfg["columnas_entrada"]])[:, 1]
y_pred = (score_attack >= cfg["umbral"]).astype(int)                # 1 = ataque
```

`trusted` lista únicamente tipos internos de scikit-learn ({', '.join('`' + t + '`' for t in cfg['tipos_confiables_skops']) or 'ninguno'}); revísalos antes de cargar.

## Limitaciones

- Benchmark histórico y sintético: no generaliza por sí mismo a una red actual.
- La validación (partición aleatoria agrupada del mismo archivo) es más favorable que el test publicado: allí las entradas
  son distintas a las de entrenamiento, así que se espera una tasa de falsos positivos mayor que la de validación.
- Hay filas con entradas idénticas y etiquetas contradictorias; ningún modelo con estas variables puede acertarlas todas.
- La FPR objetivo (5 %) se fijó en validación; en producción cada falsa alarma tiene un costo operativo que aquí no se modela.
- Los SHA-256 coinciden con los del enunciado, pero el mirror de origen no está autenticado contra el archivo oficial de UNSW.
""")

    if version == "v2":
        if test:
            c = test["matriz_confusion"]
            tm, ic = test["metricas"], test["ic95_bootstrap"]
            filas = "\n".join([
                "| Métrica (clase positiva = ataque) | Test final | IC 95 % (bootstrap) |", "|---|---|---|",
                f"| Recall de ataque | {pct(tm['recall'])} | {pct(ic['recall'][0])} – {pct(ic['recall'][1])} |",
                f"| Precision de ataque | {pct(tm['precision'])} | {pct(ic['precision'][0])} – {pct(ic['precision'][1])} |",
                f"| FPR | {pct(tm['fpr'])} | {pct(ic['fpr'][0])} – {pct(ic['fpr'][1])} |",
                f"| Macro-F1 | {f(tm['macro_f1'])} | {f(ic['macro_f1'][0])} – {f(ic['macro_f1'][1])} |",
                f"| Accuracy (complementaria) | {pct(tm['accuracy'])} | |",
                f"| Matriz de confusión (VP / FP / FN / VN) | {miles(c['tp'])} / {miles(c['fp'])} / {miles(c['fn'])} / {miles(c['tn'])} | |",
            ])
            hfinfo = ""
            if hf:
                hfinfo = (f"Revisión evaluada: `{hf['v2']['commit_sha']}` · SHA-256 de `modelo.skops`: `{hf['v2']['sha256']['modelo.skops']}`. "
                          "Esta card se actualizó después de congelar; el modelo no cambió.\n\n")
            partes.append(f"""## Evaluación final en test (una sola vez, solución congelada)

{hfinfo}Archivo `UNSW_NB15_testing-set.csv`, {miles(test['dataset']['filas_evaluadas'])} filas (toda la partición publicada), umbral `{test['umbral']:.6f}` fijado en validación.

{filas}

{test['lectura']}
""")
        else:
            partes.append("## Evaluación final en test\n\nSe añade aquí después de congelar la selección, evaluando una sola vez; "
                          "actualizar esta card no cambia el modelo (el hash de `modelo.skops` se conserva).\n")

    partes.append("""## Citas

- Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS.
- Moustafa, N. y Slay, J. (2016). The evaluation of network anomaly detection systems. Information Security Journal.
""")
    return "\n".join(partes)


def write(version: str, test: dict | None = None, hf: dict | None = None):
    CARDS.mkdir(parents=True, exist_ok=True)
    (CARDS / f"{version}.md").write_text(render(version, test, hf), encoding="utf-8")
