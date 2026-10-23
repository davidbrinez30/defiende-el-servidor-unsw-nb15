# Exportación de MLflow

Contenido (sin depender del servidor de quien entrenó):

- `experimento.json`: nombre e identificador del experimento `defiende-el-servidor`.
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
