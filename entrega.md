# Entrega — Defiende el servidor (MLOps 2026-II)

**Asunto / primera línea del mensaje:** `MLOps — Defiende el servidor — Grupo B — David Santiago Briñez Forero, Mattias Morales Serna`

## 1. Integrantes y grupo

- **David Santiago Briñez Forero** — Grupo B — Integrante del equipo.
- **Mattias Morales Serna** — Grupo B — Integrante del equipo.

## 2. Código y commit final

- Repositorio de código (público): https://github.com/davidbrinez30/defiende-el-servidor-unsw-nb15
- Commit con todas las evidencias: `848857d81759605e919ed71fb396993ae3c06634`. El **commit final de la entrega** es el que marca el tag `entrega` (`git rev-parse entrega`);
  solo agrega este archivo y el informe sobre el commit anterior. Su SHA completo está en el mensaje de entrega.
- Commit de la congelación de la selección (tag `congelado`, anterior a la evaluación en test): `9e1cd7b1456d024379522c784b7c7849b919fcdd`
- Repositorio de modelos en Hugging Face (público): https://huggingface.co/d4v1dsbf/defiende-el-servidor-unsw-nb15

## 3. Versiones v1 y v2

| | v1 | v2 |
|---|---|---|
| Descripción | Primera solución funcional: regresión logística con log(1+x) y escalado. | Revisión: modelo elegido por macro-F1 en validación entre todos los experimentos con modelo real. |
| Experimento | `E2_logreg_log1p` (regresion_logistica) | `E5_hist_gradient_boosting` (hist_gradient_boosting) |
| Commit del código que entrenó el modelo | `68d3332cb384b80686897994b97e0d2dc0b4a740` | `68d3332cb384b80686897994b97e0d2dc0b4a740` |
| Tag de Git y su commit completo | `v1` → `829e94b518f47ab19c807bda2dc6cc64059dc5d9` | `v2` → `008d971d9320b605f70627ab1973b18d6021b7b1` |
| Datos originales (DVC, md5) | entrenamiento `e55caabaa6cd4a8f1c06a227bcfababc`; test `e0beea40262e46168cdb81476dbc27b4` | entrenamiento `e55caabaa6cd4a8f1c06a227bcfababc`; test `e0beea40262e46168cdb81476dbc27b4` |
| Datos preparados (DVC, md5 del `dvc.lock`; sha256) | `train.parquet` `7486d22a233d74fae0f038e182cc7c35`, `val.parquet` `b1962c0a4bad5c513538ac486b042bfd`, `split_assignment.csv` `84620a6590a34976659c4b5838afd410`; sha256 en `config.json` | `train.parquet` `7486d22a233d74fae0f038e182cc7c35`, `val.parquet` `b1962c0a4bad5c513538ac486b042bfd`, `split_assignment.csv` `84620a6590a34976659c4b5838afd410`; sha256 en `config.json` |
| Run ID de MLflow | `9c11a62d87ab4109a4bff065f6e4b3e8` | `924822b52f714831bf054c763c5adab3` |
| Hugging Face | [d4v1dsbf/defiende-el-servidor-unsw-nb15](https://huggingface.co/d4v1dsbf/defiende-el-servidor-unsw-nb15) · tag `v1` · **SHA completo `3c9aabf11b57387f55d07261458c574f241be34c`** | [d4v1dsbf/defiende-el-servidor-unsw-nb15](https://huggingface.co/d4v1dsbf/defiende-el-servidor-unsw-nb15) · tag `v2` · **SHA completo `3639b0994cd1493723b8d8899f0e8311d9c798c3`** |
| SHA-256 de `modelo.skops` | `1b7fc9ed8ed2e86c7ab8ea47a6c16e6d1fdee37e5bbc5c489cbf6802b1c291d3` | `3e4ade35db97b10a53a0193361a824a700166db35b4854891fa04488ea6b36e8` |
| Umbral | `0.777403` (`y_pred = 1` si `score_attack >= umbral`) | `0.654376` (`y_pred = 1` si `score_attack >= umbral`) |
| Validación (recall / precision / FPR / macro-F1) | 91,21 % / 97,49 % / 5,00 % / 0,9157 | 95,42 % / 97,60 % / 5,00 % / 0,9464 |

Cambio entre versiones: Tipo de modelo (regresión logística -> modelo de árboles); misma regla de umbral y mismo preprocesamiento.
Cadena de trazabilidad: **commit de Git → referencia DVC → run ID de MLflow → revisión (SHA) de Hugging Face**.
El modelo que se describe (`config.json`), el que se descarga de Hugging Face y el que produce `predicciones-test.csv` es el mismo
archivo: `modelo.skops` con SHA-256 `3e4ade35db97b10a53a0193361a824a700166db35b4854891fa04488ea6b36e8` (v2).

## 4. Dónde está cada evidencia

| Evidencia | Ruta | Qué permite comprobar |
|---|---|---|
| Auditoría de datos | `evidencias/auditoria.json` | Fuentes y SHA-256, filas/columnas, clases, faltantes, duplicados, contradicciones, política de limpieza, variables usadas/excluidas, semilla, tamaños y huellas de splits, cruces con el test |
| Comparación de experimentos | `evidencias/experimentos.csv` | Una fila por configuración: hipótesis, familia, parámetros, semilla, commit, referencias DVC, run ID, métricas, umbral y conclusión |
| Registros de MLflow | `evidencias/mlflow/` | Parámetros, métricas, etiquetas y artefactos (con SHA-256) de cada run, sin depender de un servidor |
| Decisión de v2 (validación) | `evidencias/seleccion_validacion.json` | Ranking completo, intervalo bootstrap frente al segundo, costos frente a v1 |
| Selección congelada | `evidencias/seleccion.json` | Versión elegida, run ID, referencias Git/DVC/Hugging Face, hash, umbral, razón y registro de congelación (tag `congelado`) |
| Publicación en Hugging Face | `evidencias/hf_publicacion.json` | SHA de cada revisión, tags y SHA-256 de cada archivo, verificados por descarga |
| Métricas de test | `evidencias/metricas-test.json` | Versión y hash evaluados, hash del dataset, filas, umbral, matriz de confusión, métricas e IC |
| Predicciones de test | `evidencias/predicciones-test.csv` | `row_id` (`testing-set:<id>`), `y_true`, `score_attack`, `y_pred` para las 82.332 filas |
| Recuperación | `evidencias/recuperacion/` | 50 entradas fijas, salidas de v1/v2 antes y después, hashes, comandos, registro y comparación en entorno nuevo |
| Model cards | `docs/model_cards/v1.md`, `v2.md` | Las mismas que están en Hugging Face |
| Informe | `informe.pdf` (adjunto y en el repositorio) | Datos, experimentos, v1/v2, umbral, test y limitaciones |

## 5. Almacenamiento DVC y acceso

- **Remote por defecto (`almacen_http`)**: repositorio público de GitHub https://github.com/davidbrinez30/defiende-el-servidor-dvc-storage servido como HTTP de solo lectura en
  `https://raw.githubusercontent.com/davidbrinez30/defiende-el-servidor-dvc-storage/main`. No requiere credenciales: `dvc pull` funciona desde cualquier computador con internet; la caché local no interviene.
  Contiene los objetos de DVC: datos originales y preparados, modelos de los experimentos y paquetes de v1/v2.
- **Respaldo**: `dvc-storage.zip`, copia del mismo almacenamiento (se entrega aparte del ZIP de código).
  Uso: descomprimir en una carpeta, `dvc remote add -d respaldo <carpeta>` y `dvc pull`.
- Los CSV originales vienen de un mirror de terceros (ver `docs/sources.json`); los SHA-256 coinciden con los del enunciado.

## 6. Comandos y recursos

```bash
git clone https://github.com/davidbrinez30/defiende-el-servidor-unsw-nb15.git && cd defiende-el-servidor-unsw-nb15
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull data/raw/UNSW_NB15_training-set.csv.dvc data/raw/UNSW_NB15_testing-set.csv.dvc prepare release_v1 release_v2
python -m src.cli verificar                               # SHA-256 de datos y modelos
python -m src.predict --release models/releases/v2 --input evidencias/recuperacion/entradas_validacion.csv --output salida.csv
python -m src.predict --hf-revision 3639b0994cd1493723b8d8899f0e8311d9c798c3 --input evidencias/recuperacion/entradas_validacion.csv --output salida_hf.csv
python -m src.cli experimento E2_logreg_log1p             # reproduce un experimento
python scripts/recuperar.py --revision main               # prueba de recuperación en entorno nuevo
```

| Medición | Valor | Dónde se midió |
|---|---|---|
| Instalar dependencias en entorno nuevo | 208 s | Windows-11-10.0.26200-SP0, Python 3.13.15, 16 CPU |
| `dvc pull` (datos + modelos) | 9 s | ídem |
| Prueba de recuperación completa | 258 s | ídem |
| Entrenamiento de cada experimento | E0 0 s, E1 7 s, E2 7 s, E3 10 s, E4 18 s, E5 14 s, E6 4 s | Contenedor Linux en la nube (2 CPU, 7 GB de RAM, sin GPU), Python 3.13.16. Los tiempos de entrenamiento de evidencias/experimentos.csv se midieron ahí; los tiempos de la prueba de recuperación, en el equipo indicado en evidencias/recuperacion/comparacion.json. |

## 7. Resultados

- **Validación** (35.067 filas, umbral de cada versión con la regla *mayor recall con FPR ≤ 5 %*):
  v1 recall 91,21 %, macro-F1 0,9157; v2 recall 95,42 %, macro-F1 0,9464; ambas con FPR 5,00 % por construcción.
- **Test final de v2** (una sola vez, 82.332 filas): recall 96,93 % (IC95 96,76 %–97,10 %),
  precision 86,63 %, **FPR 18,33 %** (IC95 17,96 %–18,69 %), macro-F1 0,8979;
  matriz de confusión VP 43.942 / FP 6.781 / FN 1.390 / VN 30.219.
- Con el umbral fijado en validación, el test dio recall 96,93 %, FPR 18,33 % y macro-F1 0,8979; en validación fueron 95,42 %, 5,00 % y 0,9464. La FPR del test supera la meta de 5 % fijada en validación: el umbral elegido allí no controla las falsas alarmas en el test. El 89,6 % de las filas del test tiene entradas que no aparecen en entrenamiento ni validación, pero eso no basta para explicar la diferencia: la FPR fue 17,88 % en las entradas nuevas y 29,21 % en las ya vistas. La validación, partición aleatoria agrupada del mismo archivo, no reproduce el cambio entre las particiones publicadas por UNSW. El resultado se conserva tal cual; no se ajustó nada después de verlo.
- `python -m src.cli recalcular` recalcula estas métricas solo desde `predicciones-test.csv`.

## 8. Limitaciones conocidas

1. **El test empeora frente a la validación** (FPR 18,33 % frente a la meta de 5 %): el umbral fijado en validación no controla las falsas alarmas en el test publicado. Se conserva y se explica; no se ajustó nada.
2. **Solapamiento del test con entrenamiento/validación:** 10,4 % de las filas del test (8.541) tienen entradas idénticas a alguna fila de entrenamiento o validación; el resto (89,6 %) son entradas nuevas. Las conclusiones sobre test se limitan a estas particiones publicadas.
3. **Repeticiones y contradicciones:** 42,4 % de las filas de entrenamiento repiten las entradas de otra; hay 229 grupos de entradas iguales con etiquetas contradictorias (940 filas). Se conservan juntos en una misma partición y se reconocen como error irreducible con estas variables.
4. **Una sola partición de validación** (un pliegue de cinco); la ventaja de v2 sobre el segundo mejor (Random Forest) es pequeña: diferencia de macro-F1 0,0021 (IC95 0,0004 a 0,0037). Los intervalos de test tratan las filas como independientes y subestiman la incertidumbre.
5. **Datos:** mirror de terceros sin autenticar contra UNSW (los SHA-256 coinciden con el enunciado); benchmark histórico (2015) que no demuestra eficacia sobre una red actual.
6. **Historia del proyecto:** el repositorio conserva una primera iteración (commits `8e44467` a `06e3e86`) con un pipeline distinto que ya había evaluado el test con una configuración parecida. Se reconstruyó todo (auditoría, particiones, experimentos con una decisión cambiada a la vez, versiones, congelación y evaluación única) y las hipótesis de esta entrega se escribieron conociendo esos resultados previos, aunque antes de correr los experimentos definitivos (commit `68d3332`). Ningún modelo ni umbral se eligió con el test.
7. **Reproducibilidad de los paquetes:** skops guarda identificadores de objeto y fechas, así que reconstruir `models/releases/vN` produce otro hash de archivo con las mismas predicciones; el hash se fija al empaquetar y toda la recuperación se verifica contra él. Por eso las versiones se recuperan con `dvc pull`/Hugging Face y no con `dvc repro`.
8. Los repositorios son públicos; no se conoce el usuario de GitHub/Hugging Face del docente (se puede invitar si lo indica por el mismo canal).

## 9. Uso de IA, código externo y contribuciones

Se usó el asistente Claude (Anthropic) para escribir y depurar el código, redactar la documentación y preparar el informe, bajo mi dirección. Las cifras de los documentos se generan con scripts desde los archivos de evidencia, y el pipeline y la prueba de recuperación se ejecutaron y comprobaron de forma automática. Asumo el contenido y puedo explicar las decisiones. No se adaptó código externo con licencia propia: se usan scikit-learn, pandas, DVC, MLflow, skops y huggingface_hub como bibliotecas.

- **Verificaciones realizadas:** los SHA-256 de los datos coinciden con los del enunciado; las particiones no comparten entradas; v1 y v2 convertidas a skops reproducen las puntuaciones del run (diferencia máxima 0.0); lo descargado de Hugging Face por SHA coincide con lo recuperado con DVC; la prueba de recuperación en entorno nuevo dio **APROBADA** (v1: y_pred idénticas=True, diferencia máxima de score 3.9e-16, v2: y_pred idénticas=True, diferencia máxima de score 0.0e+00); las métricas de test se recalculan desde las predicciones.
- **Código externo:** ninguno adaptado; solo bibliotecas (scikit-learn, pandas, numpy, scipy, DVC, MLflow, skops, huggingface_hub, PyYAML, pyarrow, joblib) con sus licencias.
- **Contribuciones:** ver sección 1.
