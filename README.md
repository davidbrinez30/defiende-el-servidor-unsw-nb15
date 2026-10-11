# Defiende el servidor — detección de tráfico de ataque en UNSW-NB15

Solución de ML para distinguir tráfico **normal** de **ataque** (`label`) con trazabilidad completa:
Git (código y decisiones), DVC (datos, modelo y pipeline), MLflow (experimentos y registro de modelos), Prefect (orquestación).
EDA: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Decisión final y resultado en test: [`DECISION.md`](DECISION.md). Comparación de experimentos: [`docs/experimentos.md`](docs/experimentos.md).

## Datos y versión
Copias públicas fijadas al commit `44bbd02b…` de `Nir-J/ML-Projects` (detalle, SHA-256 y advertencias en `docs/sources.json`).
Origen oficial y condiciones de uso: https://research.unsw.edu.au/projects/unsw-nb15-dataset (uso académico; citar los trabajos listados allí).
El pipeline **verifica** 175.341 filas de entrenamiento, 82.332 de prueba y el SHA-256 del entrenamiento; si no coinciden, se detiene con un mensaje claro.

## Estructura
```
params.yaml            parámetros y regla de umbral (DVC los rastrea)
dvc.yaml / dvc.lock    pipeline reproducible: prepare -> experiments -> select -> evaluate
src/                   data.py (carga y partición), features.py, experiment_configs.py (HIPÓTESIS),
                       experiments.py, selection.py, evaluate.py, predict.py, cli.py
flows/pipeline.py      flow de Prefect (mismas etapas, con estado y logs por tarea)
reports/               métricas y decisiones (JSON/CSV versionados en Git)
notebooks/01_eda.ipynb EDA propio (solo entrenamiento) con 5 conclusiones y sus decisiones
docs/                  fuentes y comparación de experimentos
models/final_model.joblib   pipeline completo + umbral (versionado con DVC)
mlflow.db, mlruns/     historial de MLflow (versionado con DVC)
```

## Reproducir desde cero
```bash
git clone https://github.com/davidbrinez30/defiende-el-servidor-unsw-nb15.git && cd defiende-el-servidor-unsw-nb15
pip install -r requirements.txt          # Python 3.13; ver nota de DVC abajo
# Opción A: datos y modelo desde el remote de DVC (configurar el remote, ver "Si pierdes tu entorno")
dvc pull
# Opción B: descargar los CSV (los comandos curl del enunciado) en data/raw/ y reproducir todo
dvc repro                                # prepara, entrena los 7 experimentos, elige y evalúa
```
Nota: `dvc` requiere `antlr4-python3-runtime==4.9.*`; en Python 3.13 su compilación puede fallar con setuptools antiguo.
Si ocurre, instalarlo antes con una versión de `setuptools` reciente o desde una rueda precompilada.

## Demostración
```bash
# 1) Pipeline orquestado (estados por tarea; el test se evalúa una sola vez y se conserva el resultado)
python -m flows.pipeline
# 2) Falla visible: oculta el CSV de entrenamiento y la tarea 1 queda en Failed con el motivo
# 3) Predicciones desde una versión guardada
python -m src.predict --input datos.csv --source file                  # models/final_model.joblib (DVC)
python -m src.predict --input datos.csv --source mlflow --version 3    # Model Registry de MLflow
# 4) Interfaz de MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 5) Recuperación con DVC
rm -rf data/raw/*.csv models/final_model.joblib && dvc pull
```

## Cómo responde a las preguntas de la guía
| Herramienta | Pregunta | Respuesta |
|---|---|---|
| Git | ¿Qué cambio produjo esta versión? | Las hipótesis están en un commit anterior a los resultados; `reports/selection.json` guarda el commit de la selección. |
| DVC | ¿Puedo recuperar los mismos datos y modelo? | `data/raw/*.dvc`, `dvc.lock`, `models/final_model.joblib` y `mlflow.db`/`mlruns` en el remote. |
| MLflow | ¿Qué ejecución produjo el modelo elegido? | `reports/selection.json` → `mlflow_run_id` y `mlflow_model_version`; el umbral va como etiqueta de la versión. |
| Prefect | ¿Qué ocurrió cuando ejecuté el pipeline? | Cada etapa es una tarea con estado `Completed`/`Failed` y su error. |

## Decisiones de método (resumen)
- **Sin fuga:** se excluyen `id` y `attack_cat`; el hash de las 42 entradas define grupos y la validación se separa con `StratifiedGroupKFold` (0 patrones compartidos). Transformaciones y codificadores se ajustan solo con entrenamiento (viven dentro del `Pipeline`).
- **Test reservado:** solo lo toca `src/evaluate.py`, una vez, con modelo y umbral congelados; el código se niega a reevaluar (salvo dentro de `dvc repro`, donde reproduce el mismo resultado congelado).
- **Umbral:** regla declarada de antemano en `params.yaml` (mayor recall con FPR ≤ 5 % en validación).

## Honestidad sobre el registro de MLflow
- `dvc repro` y el flow de Prefect vuelven a entrenar, así que MLflow acumula **ejecuciones repetidas** (mismas métricas) y **3 versiones** del modelo registrado, idénticas en comportamiento. La versión de referencia y su run están en `reports/selection.json`.
- Hay una ejecución `FINAL_…` fallida al inicio: MLflow/skops bloquearon por seguridad un tipo interno de HistGradientBoosting; se declaró de confianza solo ese tipo (`TRUSTED_TYPES` en `src/selection.py`).

## Si pierdes tu entorno local
Necesitas: (1) el repositorio Git (código, `dvc.lock`, reportes); (2) un **remote de DVC fuera de tu máquina** — en este trabajo se usó un remote local
(`~/dvc-remote-defiende`) para la demostración; para que otra persona reconstruya, súbelo a un remote compartido
(`dvc remote add -d storage s3://… | gdrive://…` y `dvc push`); (3) `pip install -r requirements.txt`. Los CSV también se pueden descargar de nuevo con los `curl` del enunciado: el pipeline comprueba filas y SHA-256.
