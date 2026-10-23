# Defiende el servidor — normal vs. ataque con UNSW-NB15

Taller de MLOps (Universidad Externado de Colombia, 2026-II). Clasifica conexiones de red como **normal (0)** o **ataque (1)** y,
sobre todo, demuestra que cada versión del modelo se puede **identificar, comparar y recuperar**: de qué código, datos y
experimento viene, y que predice igual después de recuperarla en un entorno nuevo.

> Índice de evidencias, identificadores (commits, run IDs, SHA de Hugging Face, hashes) y declaración de uso de IA: **[`entrega.md`](entrega.md)**.
> Informe: `informe.pdf` (en el ZIP de entrega). Procedencia de los datos: [`docs/sources.json`](docs/sources.json).

| Qué se versiona | Herramienta | Dónde |
|---|---|---|
| Código, configuraciones, documentación, referencias DVC | Git / GitHub | este repositorio; tags `v1`, `v2`, `congelado` |
| Datos originales y preparados, modelos de cada versión | DVC | `data/raw/*.dvc`, `dvc.yaml`, `dvc.lock`; almacenamiento HTTP público (ver más abajo) |
| Experimentos (hipótesis, parámetros, métricas, modelo completo) | MLflow | `evidencias/mlflow/` (exportación) y `mlflow.db` local recreable |
| Modelos publicados (v1, v2) con model card | Hugging Face Hub | https://huggingface.co/brinez/defiende-el-servidor-unsw-nb15 |

Cadena de trazabilidad: **commit de Git → referencia DVC → run ID de MLflow → revisión (SHA) de Hugging Face**.

## Requisitos

- **Python 3.12 o 3.13** (probado con 3.13). Git. ~2 GB libres. Internet para `pip`, DVC (GitHub) y Hugging Face.
- Dependencias fijadas en [`requirements.txt`](requirements.txt). Opcionales (EDA y Prefect): `requirements-extra.txt`.

## 1. Instalar

```bash
git clone https://github.com/davidbrinez30/defiende-el-servidor-unsw-nb15.git
cd defiende-el-servidor-unsw-nb15
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Recuperar los datos (DVC)

Los datos **no están en Git**: Git guarda solo punteros (`.dvc`, `dvc.lock`) y DVC los descarga del almacenamiento entregado,
un repositorio público de GitHub usado como remote HTTP de solo lectura
(`https://raw.githubusercontent.com/davidbrinez30/defiende-el-servidor-dvc-storage/main`, configurado en `.dvc/config`).

```bash
dvc pull data/raw/UNSW_NB15_training-set.csv.dvc data/raw/UNSW_NB15_testing-set.csv.dvc   # versión 1 de los datos: originales
dvc pull prepare                                                                           # versión 2: preparados y particiones
python -m src.cli verificar                                                                # compara SHA-256 de lo recuperado
```

- **Originales:** `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332), con los SHA-256 del enunciado.
- **Preparados:** `data/processed/train.parquet` (140.274 filas), `val.parquet` (35.067) y `split_assignment.csv` (fila → partición, semilla 42), producidos por la etapa `prepare`.
- Si el almacenamiento HTTP no estuviera disponible, el respaldo (ZIP aparte, ver `entrega.md`) se usa como remote local:
  descomprímelo en una carpeta, `dvc remote add -d respaldo <carpeta>` y `dvc pull`.

## 3. Recuperar y usar una versión del modelo

Cada versión (v1, v2) tiene su commit de código (tag), su referencia de datos y una **revisión inmutable en Hugging Face**
(SHA completo en `evidencias/hf_publicacion.json` y en `entrega.md`). Con DVC:

```bash
git checkout v2                       # o v1: el código y las referencias de esa versión
dvc pull release_v2                   # models/releases/v2/{modelo.skops, config.json, requirements.txt}
python -m src.predict --release models/releases/v2 --input evidencias/recuperacion/entradas_validacion.csv --output salida_v2.csv
```

Con Hugging Face, por SHA completo de la revisión (sin DVC):

```bash
python -m src.predict --hf-revision <SHA_COMPLETO_DE_V1_O_V2> --input evidencias/recuperacion/entradas_validacion.csv --output salida.csv
```

La salida trae `score_attack` (probabilidad de ataque) y `y_pred` (1 si `score_attack >= umbral`; el umbral va en `config.json`).
La entrada necesita las 42 columnas predictoras de UNSW-NB15 (se ignoran `id`, `label`, `attack_cat`).
`modelo.skops` se carga declarando como confiables solo tipos internos de scikit-learn; `src/modelio.py` lo comprueba y verifica el hash.

## 4. Reproducir un experimento

```bash
python -m src.cli experimento E2_logreg_log1p    # uno solo; usa un almacén MLflow de prueba y no toca el pipeline
python -m src.cli experimento E4_random_forest
mlflow ui --backend-store-uri sqlite:///mlflow_prueba.db
```

Requiere haber hecho `dvc pull prepare`. Los siete experimentos (E0–E6, con hipótesis y criterio medible) están en
[`configs/experimentos.yaml`](configs/experimentos.yaml); la comparación completa, en `evidencias/experimentos.csv`.
Todos usan **las mismas particiones**; el preprocesamiento (log(1+x), escalado, codificación) está dentro del `Pipeline`
y se ajusta solo con entrenamiento.

## 5. Reproducir todo el pipeline

```bash
dvc pull data/raw/UNSW_NB15_training-set.csv.dvc data/raw/UNSW_NB15_testing-set.csv.dvc
dvc repro prepare audit experiments release_v1 release_v2
```

| Etapa | Qué hace | Salidas |
|---|---|---|
| `prepare` | Partición entrenamiento/validación sin repartir entradas idénticas (`StratifiedGroupKFold`, semilla 42) | `data/processed/*` |
| `audit` | Fuentes, SHA-256, filas, clases, faltantes, duplicados, contradicciones, cruces con el test | `evidencias/auditoria.json` |
| `experiments` | Corre E0–E6 y los registra en MLflow; exporta la tabla y los registros | `models/runs/`, `evidencias/experimentos.csv`, `evidencias/mlflow/` |
| `release_v1` / `release_v2` | v1 fija (E2); v2 elegida con validación; empaquetado en skops con `config.json` | `models/releases/vN`, `evidencias/seleccion_validacion.json` |
| *(congelar)* | `python -m src.cli congelar` escribe `evidencias/seleccion.json`; se confirma con el tag `congelado` | `evidencias/seleccion.json` |
| `evaluate` | **Una sola vez**, v2 congelada sobre el test completo | `evidencias/metricas-test.json`, `predicciones-test.csv` |

Recalcular las métricas de test solo desde las predicciones: `python -m src.cli recalcular`.
Prefect es opcional (`requirements-extra.txt`, `python -m flows.pipeline`).

## 6. Prueba de recuperación en un entorno nuevo

```bash
python scripts/recuperar.py --revision main
```

Clona el repositorio en una carpeta nueva, crea un entorno virtual nuevo, instala las dependencias fijadas, recupera con DVC desde
el almacenamiento entregado (caché vacía), descarga v1 y v2 de Hugging Face por SHA completo, predice sobre 50 filas fijas de validación
y compara con las salidas guardadas antes. El registro queda en `evidencias/recuperacion/`.

## Estructura

```
configs/        experimentos.yaml (hipótesis), versiones.yaml (v1/v2), equipo.yaml (equipo y plataformas)
params.yaml     semilla, rutas, hashes esperados, regla de umbral (DVC lo rastrea)
src/            data, features, audit, experiments, hipotesis, exportar, release, modelcard, congelar, evaluate, modelio, predict, verificar, cli
scripts/        publicar_hf.py (publica v1/v2 en Hugging Face), recuperar.py (prueba de recuperación)
docs/           sources.json (procedencia), model_cards/ (cards de v1 y v2)
evidencias/     auditoria, experimentos, seleccion, metricas-test, predicciones-test, mlflow/, recuperacion/
notebooks/      01_eda.ipynb (EDA propio, solo con entrenamiento)
```

## Decisiones de método en una línea

- `id`, `attack_cat` y `label` **no** son entradas. Las filas con entradas idénticas nunca se reparten entre entrenamiento y validación.
- Umbral (regla escrita antes de ver resultados): **mayor recall con tasa de falsos positivos ≤ 5 % en validación**. v2 = mejor macro-F1 en validación con ese umbral.
- El test se audita (esquema, duplicados, cruces) pero no se usa para elegir nada; se evalúa una vez con la solución congelada.
- Denominadores cero: precision = 0 si no hay alertas; recall = 0 si no hay ataques reales; FPR = 0 si no hay normales reales.

## Seguridad

No hay tokens ni contraseñas en el repositorio. Para publicar en Hugging Face, el token (permiso *Write*) se define solo como
variable de entorno `HF_TOKEN` en la máquina de quien publica.

## Datos y licencia

UNSW-NB15 © UNSW Canberra (Moustafa y Slay, 2015); uso académico, citar los trabajos de la [página oficial](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
La copia usada proviene de un mirror de terceros fijado por commit (`docs/sources.json`); los SHA-256 coinciden con los del enunciado,
pero no están autenticados contra el archivo oficial. Es un benchmark histórico: no demuestra eficacia sobre una red actual.
