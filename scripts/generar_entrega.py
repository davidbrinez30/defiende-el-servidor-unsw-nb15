"""Genera entrega.md (índice de la entrega) a partir de los archivos de evidencia y de Git: ningún identificador se escribe a mano.

    python scripts/generar_entrega.py

Lee configs/equipo.yaml, evidencias/*.json|csv, dvc.lock y los tags de Git. Debe correrse con todas las evidencias ya confirmadas
(el commit que se cita como «commit de las evidencias» es el HEAD actual; el commit final solo agrega entrega.md e informe).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidencias"


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=2):
    return f"{es(100 * x, d)} %"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def main():
    eq = yaml.safe_load((ROOT / "configs" / "equipo.yaml").read_text(encoding="utf-8"))
    pend = [i for i in eq["integrantes"] if "[COMPLETAR" in i["nombre"] or "[COMPLETAR" in i["grupo"]]
    if pend:
        sys.exit("Completa nombre completo y grupo en configs/equipo.yaml antes de generar entrega.md.")
    gh, hfc = eq["github"], eq["huggingface"]
    repo_url = f"https://github.com/{gh['usuario']}/{gh['repo_codigo']}"
    storage_url = f"https://github.com/{gh['usuario']}/{gh['repo_almacenamiento_dvc']}"
    raw_url = f"https://raw.githubusercontent.com/{gh['usuario']}/{gh['repo_almacenamiento_dvc']}/main"
    hf_repo = f"{hfc['usuario']}/{hfc['repo_modelos']}"

    sel, hf, test, rec = j(EV / "seleccion.json"), j(EV / "hf_publicacion.json"), j(EV / "metricas-test.json"), j(EV / "recuperacion" / "comparacion.json")
    aud, exp = j(EV / "auditoria.json"), pd.read_csv(EV / "experimentos.csv")
    cfg = {v: j(ROOT / "models" / "releases" / v / "config.json") for v in ("v1", "v2")}
    head = git("rev-parse", "HEAD")
    tag_commit = {t: git("rev-list", "-n", "1", t) for t in ("v1", "v2", "congelado")}

    integrantes = "\n".join(f"- **{i['nombre']}** — Grupo {i['grupo']} — {i['contribucion']}" for i in eq["integrantes"])
    val1, val2 = cfg["v1"]["metricas_validacion"]["al_umbral"], cfg["v2"]["metricas_validacion"]["al_umbral"]
    tm, ic, c = test["metricas"], test["ic95_bootstrap"], test["matriz_confusion"]
    t_fit = exp.set_index("id")["segundos_entrenamiento"]
    sub = test["por_subconjunto_de_entradas"]
    cru = aud["cruces_test"]
    dupl, cont = aud["archivos"]["entrenamiento"]["duplicados"], aud["archivos"]["entrenamiento"]["contradicciones"]
    tt = rec["tiempos_segundos"]

    def fila_version(v):
        k, h, d = cfg[v], hf[v], cfg[v]["dvc"]
        return [
            ("Descripción", k["descripcion"]),
            ("Experimento", f"`{k['experimento']}` ({k['familia']})"),
            ("Commit del código que entrenó el modelo", f"`{k['git']['commit_codigo_que_entreno']}`"),
            ("Tag de Git y su commit completo", f"`{v}` → `{tag_commit[v]}`"),
            ("Datos originales (DVC, md5)", f"entrenamiento `{d['dvc_original_train_md5']}`; test `{d['dvc_original_test_md5']}`"),
            ("Datos preparados (DVC, md5 del `dvc.lock`; sha256)",
             f"`train.parquet` `{d['prepare_outs_md5']['data/processed/train.parquet']}`, `val.parquet` `{d['prepare_outs_md5']['data/processed/val.parquet']}`, "
             f"`split_assignment.csv` `{d['prepare_outs_md5']['data/processed/split_assignment.csv']}`; sha256 en `config.json`"),
            ("Run ID de MLflow", f"`{k['mlflow']['run_id']}`"),
            ("Hugging Face", f"[{hf_repo}](https://huggingface.co/{hf_repo}) · tag `{v}` · **SHA completo `{h['commit_sha']}`**"),
            ("SHA-256 de `modelo.skops`", f"`{k['modelo_sha256']}`"),
            ("Umbral", f"`{k['umbral']:.6f}` (`y_pred = 1` si `score_attack >= umbral`)"),
            ("Validación (recall / precision / FPR / macro-F1)",
             "{} / {} / {} / {}".format(*[pct(cfg[v]["metricas_validacion"]["al_umbral"][m]) if m != "macro_f1" else es(cfg[v]["metricas_validacion"]["al_umbral"][m], 4)
                                        for m in ("recall", "precision", "fpr", "macro_f1")])),
        ]

    f1, f2 = fila_version("v1"), fila_version("v2")
    tabla_versiones = "| | v1 | v2 |\n|---|---|---|\n" + "\n".join(f"| {a[0]} | {a[1]} | {b[1]} |" for a, b in zip(f1, f2))

    txt = f"""# Entrega — Defiende el servidor (MLOps 2026-II)

**Asunto / primera línea del mensaje:** `MLOps — Defiende el servidor — Grupo {eq['integrantes'][0]['grupo']} — {', '.join(i['nombre'] for i in eq['integrantes'])}`

## 1. Integrantes y grupo

{integrantes}

## 2. Código y commit final

- Repositorio de código (público): {repo_url}
- Commit con todas las evidencias: `{head}`. El **commit final de la entrega** es el que marca el tag `entrega` (`git rev-parse entrega`);
  solo agrega este archivo y el informe sobre el commit anterior. Su SHA completo está en el mensaje de entrega.
- Commit de la congelación de la selección (tag `congelado`, anterior a la evaluación en test): `{tag_commit['congelado']}`
- Repositorio de modelos en Hugging Face (público): https://huggingface.co/{hf_repo}

## 3. Versiones v1 y v2

{tabla_versiones}

Cambio entre versiones: {sel['razon_de_seleccion']['cambio_respecto_a_v1']}
Cadena de trazabilidad: **commit de Git → referencia DVC → run ID de MLflow → revisión (SHA) de Hugging Face**.
El modelo que se describe (`config.json`), el que se descarga de Hugging Face y el que produce `predicciones-test.csv` es el mismo
archivo: `modelo.skops` con SHA-256 `{cfg['v2']['modelo_sha256']}` (v2).

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
| Predicciones de test | `evidencias/predicciones-test.csv` | `row_id` (`testing-set:<id>`), `y_true`, `score_attack`, `y_pred` para las {miles(test['dataset']['filas_evaluadas'])} filas |
| Recuperación | `evidencias/recuperacion/` | 50 entradas fijas, salidas de v1/v2 antes y después, hashes, comandos, registro y comparación en entorno nuevo |
| Model cards | `docs/model_cards/v1.md`, `v2.md` | Las mismas que están en Hugging Face |
| Informe | `informe.pdf` (adjunto y en el repositorio) | Datos, experimentos, v1/v2, umbral, test y limitaciones |

## 5. Almacenamiento DVC y acceso

- **Remote por defecto (`almacen_http`)**: repositorio público de GitHub {storage_url} servido como HTTP de solo lectura en
  `{raw_url}`. No requiere credenciales: `dvc pull` funciona desde cualquier computador con internet; la caché local no interviene.
  Contiene los objetos de DVC: datos originales y preparados, modelos de los experimentos y paquetes de v1/v2.
- **Respaldo**: `dvc-storage.zip`, copia del mismo almacenamiento{(' — ' + eq['respaldo_drive_url']) if eq.get('respaldo_drive_url') else ' (se entrega aparte del ZIP de código)'}.
  Uso: descomprimir en una carpeta, `dvc remote add -d respaldo <carpeta>` y `dvc pull`.
- Los CSV originales vienen de un mirror de terceros (ver `docs/sources.json`); los SHA-256 coinciden con los del enunciado.

## 6. Comandos y recursos

```bash
git clone {repo_url}.git && cd {gh['repo_codigo']}
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
dvc pull data/raw/UNSW_NB15_training-set.csv.dvc data/raw/UNSW_NB15_testing-set.csv.dvc prepare release_v1 release_v2
python -m src.cli verificar                               # SHA-256 de datos y modelos
python -m src.predict --release models/releases/v2 --input evidencias/recuperacion/entradas_validacion.csv --output salida.csv
python -m src.predict --hf-revision {hf['v2']['commit_sha']} --input evidencias/recuperacion/entradas_validacion.csv --output salida_hf.csv
python -m src.cli experimento E2_logreg_log1p             # reproduce un experimento
python scripts/recuperar.py --revision main               # prueba de recuperación en entorno nuevo
```

| Medición | Valor | Dónde se midió |
|---|---|---|
| Instalar dependencias en entorno nuevo | {es(tt['instalar_dependencias'], 0)} s | {rec['entorno_nuevo']['sistema']}, Python {rec['entorno_nuevo']['python']}, {rec['entorno_nuevo']['cpus']} CPU |
| `dvc pull` (datos + modelos) | {es(tt['dvc_pull'], 0)} s | ídem |
| Prueba de recuperación completa | {es(tt['total'], 0)} s | ídem |
| Entrenamiento de cada experimento | {', '.join(f"{k.split('_')[0]} {es(v, 0)} s" for k, v in t_fit.items())} | {eq['entorno_entrenamiento'].strip()} |

## 7. Resultados

- **Validación** ({miles(cfg['v2']['metricas_validacion']['filas'])} filas, umbral de cada versión con la regla *mayor recall con FPR ≤ 5 %*):
  v1 recall {pct(val1['recall'])}, macro-F1 {es(val1['macro_f1'], 4)}; v2 recall {pct(val2['recall'])}, macro-F1 {es(val2['macro_f1'], 4)}; ambas con FPR {pct(val2['fpr'])} por construcción.
- **Test final de v2** (una sola vez, {miles(test['dataset']['filas_evaluadas'])} filas): recall {pct(tm['recall'])} (IC95 {pct(ic['recall'][0])}–{pct(ic['recall'][1])}),
  precision {pct(tm['precision'])}, **FPR {pct(tm['fpr'])}** (IC95 {pct(ic['fpr'][0])}–{pct(ic['fpr'][1])}), macro-F1 {es(tm['macro_f1'], 4)};
  matriz de confusión VP {miles(c['tp'])} / FP {miles(c['fp'])} / FN {miles(c['fn'])} / VN {miles(c['tn'])}.
- {test['lectura']}
- `python -m src.cli recalcular` recalcula estas métricas solo desde `predicciones-test.csv`.

## 8. Limitaciones conocidas

1. **El test empeora frente a la validación** (FPR {pct(tm['fpr'])} frente a la meta de 5 %): el umbral fijado en validación no controla las falsas alarmas en el test publicado. Se conserva y se explica; no se ajustó nada.
2. **Solapamiento del test con entrenamiento/validación:** {pct(cru['porcentaje_test_con_entradas_vistas'] / 100, 1)} de las filas del test ({miles(cru['filas_test_con_entradas_vistas_en_entrenamiento_o_validacion'])}) tienen entradas idénticas a alguna fila de entrenamiento o validación; el resto ({pct(sub['entradas_nuevas']['porcentaje_del_test'] / 100, 1)}) son entradas nuevas. Las conclusiones sobre test se limitan a estas particiones publicadas.
3. **Repeticiones y contradicciones:** {pct(dupl['filas_repetidas_adicionales'] / dupl['filas'], 1)} de las filas de entrenamiento repiten las entradas de otra; hay {cont['grupos_con_etiquetas_contradictorias']} grupos de entradas iguales con etiquetas contradictorias ({miles(cont['filas_en_esos_grupos'])} filas). Se conservan juntos en una misma partición y se reconocen como error irreducible con estas variables.
4. **Una sola partición de validación** (un pliegue de cinco); la ventaja de v2 sobre el segundo mejor (Random Forest) es pequeña: diferencia de macro-F1 {es(sel['razon_de_seleccion']['diferencia_macro_f1_vs_segundo']['media_bootstrap'], 4)} (IC95 {es(sel['razon_de_seleccion']['diferencia_macro_f1_vs_segundo']['ic95'][0], 4)} a {es(sel['razon_de_seleccion']['diferencia_macro_f1_vs_segundo']['ic95'][1], 4)}). Los intervalos de test tratan las filas como independientes y subestiman la incertidumbre.
5. **Datos:** mirror de terceros sin autenticar contra UNSW (los SHA-256 coinciden con el enunciado); benchmark histórico (2015) que no demuestra eficacia sobre una red actual.
6. **Historia del proyecto:** el repositorio conserva una primera iteración (commits `8e44467` a `06e3e86`) con un pipeline distinto que ya había evaluado el test con una configuración parecida. Se reconstruyó todo (auditoría, particiones, experimentos con una decisión cambiada a la vez, versiones, congelación y evaluación única) y las hipótesis de esta entrega se escribieron conociendo esos resultados previos, aunque antes de correr los experimentos definitivos (commit `68d3332`). Ningún modelo ni umbral se eligió con el test.
7. **Reproducibilidad de los paquetes:** skops guarda identificadores de objeto y fechas, así que reconstruir `models/releases/vN` produce otro hash de archivo con las mismas predicciones; el hash se fija al empaquetar y toda la recuperación se verifica contra él. Por eso las versiones se recuperan con `dvc pull`/Hugging Face y no con `dvc repro`.
8. Los repositorios son públicos; no se conoce el usuario de GitHub/Hugging Face del docente (se puede invitar si lo indica por el mismo canal).

## 9. Uso de IA, código externo y contribuciones

{eq['declaracion_ia'].strip()}

- **Verificaciones realizadas:** los SHA-256 de los datos coinciden con los del enunciado; las particiones no comparten entradas; v1 y v2 convertidas a skops reproducen las puntuaciones del run (diferencia máxima {cfg['v2']['verificacion_conversion']['diferencia_maxima_score']}); lo descargado de Hugging Face por SHA coincide con lo recuperado con DVC; la prueba de recuperación en entorno nuevo dio **{rec['resultado']}** ({', '.join(f"{v}: y_pred idénticas={d['y_pred_identicas']}, diferencia máxima de score {d['diferencia_maxima_score']:.1e}" for v, d in rec['comparacion'].items())}); las métricas de test se recalculan desde las predicciones.
- **Código externo:** ninguno adaptado; solo bibliotecas (scikit-learn, pandas, numpy, scipy, DVC, MLflow, skops, huggingface_hub, PyYAML, pyarrow, joblib) con sus licencias.
- **Contribuciones:** ver sección 1.
"""
    (ROOT / "entrega.md").write_text(txt, encoding="utf-8")
    print(f"entrega.md escrito ({len(txt.splitlines())} líneas)")


if __name__ == "__main__":
    main()
