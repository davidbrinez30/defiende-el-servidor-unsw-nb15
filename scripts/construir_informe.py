"""Construye informe.pdf (máximo 5 páginas + referencias aparte) desde los archivos de evidencia.

    pip install matplotlib playwright && playwright install chromium     # solo para construir el informe
    python scripts/construir_informe.py

Todas las cifras salen de evidencias/*; no se escribe ninguna a mano.
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidencias"
AZUL, NARANJA, TINTA, TINTA2, GRIS = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d3"


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=2):
    return f"{es(100 * x, d)} %"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def png(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def estilo(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIS)
    ax.tick_params(colors=TINTA2, labelsize=7.5, length=0)
    ax.grid(axis="x", color=GRIS, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_experimentos(exp: pd.DataFrame, v2_id: str) -> str:
    d = exp[~exp["id"].str.startswith("E0")].sort_values("recall")
    fig, ax = plt.subplots(figsize=(4.3, 2.5))
    y = range(len(d))
    ax.hlines(y, 0.85, d["recall"], color=GRIS, linewidth=1.2)
    ax.scatter(d["recall"], y, s=40, color=AZUL, zorder=3, edgecolor="white", linewidth=1.2)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["recall"] + 0.003, yi, es(100 * r["recall"], 1) + " %", va="center", fontsize=7.5, color=TINTA)
    ax.set_yticks(list(y))
    ax.set_yticklabels([i.split("_")[0] + " " + i.split("_", 1)[1][:22] for i in d["id"]], fontsize=7.5, color=TINTA)
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("Recall de ataque en validación (FPR fijada en 5 %)", fontsize=7.5, color=TINTA2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    estilo(ax)
    ax.set_title("E0 (siempre ataque, recall 100 %, FPR 100 %) queda fuera de la escala", fontsize=7, color=TINTA2, loc="left")
    return png(fig)


def fig_val_test(v: dict, t: dict) -> str:
    import numpy as np
    etq = ["Recall", "Precision", "FPR", "Macro-F1"]
    val = [v["recall"], v["precision"], v["fpr"], v["macro_f1"]]
    tes = [t["recall"], t["precision"], t["fpr"], t["macro_f1"]]
    fig, ax = plt.subplots(figsize=(4.3, 2.5))
    x = np.arange(4)
    b1 = ax.bar(x - 0.2, val, 0.38, color=AZUL, label="Validación")
    b2 = ax.bar(x + 0.2, tes, 0.38, color=NARANJA, label="Test final")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, es(100 * r.get_height(), 1), ha="center", fontsize=7, color=TINTA)
    ax.set_xticks(x)
    ax.set_xticklabels(etq, fontsize=8, color=TINTA)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda s, _: f"{s * 100:.0f} %")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIS)
    ax.tick_params(length=0, labelsize=7.5, colors=TINTA2)
    ax.grid(axis="y", color=GRIS, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    return png(fig)


CSS = """
@page { size: A4; margin: 15mm 15mm 14mm 15mm; }
* { box-sizing: border-box; }
body { font-family: Inter, 'DejaVu Sans', sans-serif; font-size: 8.9pt; line-height: 1.38; color: #0b0b0b; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 2pt; letter-spacing: -0.2pt; }
.sub { color: #52514e; font-size: 8.5pt; margin-bottom: 7pt; }
h2 { break-after: avoid; font-size: 10.5pt; margin: 9pt 0 3pt; padding-bottom: 1.5pt; border-bottom: 1px solid #d9d8d3; }
p { margin: 0 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt; font-size: 7.7pt; }
th { text-align: left; font-weight: 600; color: #52514e; border-bottom: 1px solid #52514e; padding: 2pt 4pt; }
td { padding: 2pt 4pt; border-bottom: 1px solid #ecebe6; vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.2pt; background: #f1f0ec; padding: 0 2pt; border-radius: 2pt; }
.fig { display: flex; gap: 8pt; margin: 3pt 0; }
.fig figure { flex: 1; margin: 0; }
.fig img { width: 100%; }
figcaption { font-size: 7.5pt; color: #52514e; }
ul { margin: 2pt 0 4pt 12pt; padding: 0; }
li { margin-bottom: 1.5pt; }
.refs { page-break-before: always; font-size: 8.3pt; }
.aviso { background: #fbf1ea; border-left: 3px solid #eb6834; padding: 3pt 6pt; margin: 4pt 0; }
"""


def construir() -> str:
    eq = yaml.safe_load((ROOT / "configs" / "equipo.yaml").read_text(encoding="utf-8"))
    aud, exp = j(EV / "auditoria.json"), pd.read_csv(EV / "experimentos.csv")
    sel, test, rec, hf = j(EV / "seleccion.json"), j(EV / "metricas-test.json"), j(EV / "recuperacion" / "comparacion.json"), j(EV / "hf_publicacion.json")
    dec = j(EV / "seleccion_validacion.json")
    c1, c2 = (j(ROOT / "models" / "releases" / v / "config.json") for v in ("v1", "v2"))
    v1, v2 = c1["metricas_validacion"]["al_umbral"], c2["metricas_validacion"]["al_umbral"]
    tm, ic, cm = test["metricas"], test["ic95_bootstrap"], test["matriz_confusion"]
    tr, te = aud["archivos"]["entrenamiento"], aud["archivos"]["test"]
    part, cru = aud["particiones"], aud["cruces_test"]
    sub = test["por_subconjunto_de_entradas"]
    nombres = ", ".join(i["nombre"] for i in eq["integrantes"])
    grupo = eq["integrantes"][0]["grupo"]
    gh = eq["github"]
    dif = dec["diferencia_macro_f1_vs_segundo"]

    filas_exp = "\n".join(
        f"<tr><td>{r.id.split('_')[0]}</td><td>{html.escape(r.cambio)}</td><td>{r.familia.replace('_', ' ')}</td>"
        f"<td class=n>{es(r.recall, 4)}</td><td class=n>{es(r.precision, 4)}</td><td class=n>{es(r.macro_f1, 4)}</td>"
        f"<td class=n>{es(r.roc_auc, 4)}</td><td>{'sí' if r.hipotesis_confirmada == 'si' else 'no'}</td></tr>"
        for r in exp.itertuples())
    filas_cmp = "\n".join(
        f"<tr><td>{n}</td><td class=n>{a}</td><td class=n>{b}</td><td class=n>{c}</td></tr>"
        for n, a, b, c in [
            ("Recall", pct(v1["recall"]), pct(v2["recall"]), f"+{es(100 * (v2['recall'] - v1['recall']))} pts"),
            ("Precision", pct(v1["precision"]), pct(v2["precision"]), f"+{es(100 * (v2['precision'] - v1['precision']))} pts"),
            ("FPR", pct(v1["fpr"]), pct(v2["fpr"]), "igual (regla)"),
            ("Macro-F1", es(v1["macro_f1"], 4), es(v2["macro_f1"], 4), f"+{es(v2['macro_f1'] - v1['macro_f1'], 4)}"),
            ("Umbral", es(c1["umbral"], 4), es(c2["umbral"], 4), ""),
            ("Falsos negativos / falsos positivos", f"{miles(v1['fn'])} / {miles(v1['fp'])}", f"{miles(v2['fn'])} / {miles(v2['fp'])}", ""),
        ])
    c_ = c2["costos_frente_a_v1"]
    cat = test["por_familia_de_ataque"]
    peor_cat = min((k for k in cat if k != "Normal"), key=lambda k: cat[k]["recall"])
    n_conf = int((exp['hipotesis_confirmada'] == 'si').sum())
    resto = min(v['recall'] for k, v in cat.items() if k not in ('Normal', peor_cat))
    img1, img2 = fig_experimentos(exp, c2["experimento"]), fig_val_test(v2, tm)

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Defiende el servidor — informe</title><style>{CSS}</style></head><body>
<h1>Defiende el servidor: versionar y recuperar un clasificador normal/ataque</h1>
<div class="sub">MLOps 2026-II · Universidad Externado de Colombia · {html.escape(nombres)} · Grupo {grupo} ·
Código: github.com/{gh['usuario']}/{gh['repo_codigo']} · Modelos: huggingface.co/{eq['huggingface']['usuario']}/{eq['huggingface']['repo_modelos']}</div>

<p><b>Resumen.</b> Se construyó un clasificador de conexiones de red (0 normal, 1 ataque) con UNSW-NB15 y se versionaron código (Git), datos (DVC),
experimentos (MLflow) y modelos (Hugging Face Hub) de modo que cada versión se recupera por identificadores y hashes. v1 es una regresión logística;
v2, un boosting de árboles elegido con validación. Con la misma regla de umbral, v2 detecta {pct(v2['recall'])} de los ataques frente a {pct(v1['recall'])} de v1 en validación.
En el test publicado, v2 mantiene {pct(tm['recall'])} de recall pero la tasa de falsos positivos sube a <b>{pct(tm['fpr'])}</b> (meta de validación: 5 %); se conserva y se explica.
La prueba de recuperación en un entorno nuevo resultó <b>{rec['resultado'].lower()}</b> para ambas versiones.</p>

<h2>1. Datos y limpieza</h2>
<p>Se usaron las particiones publicadas: entrenamiento ({miles(tr['filas'])} filas, {tr['columnas']} columnas) y test ({miles(te['filas'])}), copia fijada de un mirror de terceros
(commit <code>44bbd02b</code>, descargada el 2026-10-11); los SHA-256 coinciden con los del enunciado y se verifican en cada carga (no autentican contra UNSW). No hay valores faltantes;
«-» en <code>service</code> es una categoría con significado. Entradas del modelo: 42 variables (39 numéricas, 3 categóricas); se excluyen <code>label</code> (respuesta),
<code>attack_cat</code> (revela la respuesta) e <code>id</code> (separa las clases con AUC {es(aud['variables']['analisis_id']['auc_id_vs_label'], 2)} solo por cómo viene ordenado el archivo).</p>
<ul>
<li><b>Duplicados por entradas:</b> {pct(tr['duplicados']['filas_repetidas_adicionales'] / tr['filas'], 1)} de las filas de entrenamiento repiten las entradas de otra ({miles(tr['duplicados']['patrones_distintos'])} patrones distintos); en test, {pct(te['duplicados']['filas_repetidas_adicionales'] / te['filas'], 1)}.</li>
<li><b>Contradicciones:</b> {tr['contradicciones']['grupos_con_etiquetas_contradictorias']} grupos de entradas iguales con etiquetas distintas ({miles(tr['contradicciones']['filas_en_esos_grupos'])} filas, {es(tr['contradicciones']['porcentaje_de_filas'], 2)} %). Política: no se eliminan, imputan ni recortan; se mantienen juntos en una misma partición y se reconocen como error irreducible.</li>
<li><b>Particiones sin fuga:</b> <code>StratifiedGroupKFold</code> (5 pliegues, semilla {part['semilla']}, grupo = huella de las 42 entradas); un pliegue es validación: {miles(part['entrenamiento']['filas'])} filas de entrenamiento y {miles(part['validacion']['filas'])} de validación, {part['patrones_compartidos_entrenamiento_validacion']} patrones compartidos. Codificación, escalado y log(1+x) viven dentro del <code>Pipeline</code> y se ajustan solo con entrenamiento.</li>
<li><b>Solapamiento con el test (auditado sin etiquetas):</b> {pct(cru['porcentaje_test_con_entradas_vistas'] / 100, 1)} de las filas del test ({miles(cru['filas_test_con_entradas_vistas_en_entrenamiento_o_validacion'])}) tienen entradas idénticas a alguna de entrenamiento o validación; el {pct(sub['entradas_nuevas']['porcentaje_del_test'] / 100, 1)} restante son entradas nuevas.</li>
</ul>
<p>DVC conserva dos versiones de datos: <b>originales</b> (<code>data/raw/*.dvc</code>, md5 <code>{c1['dvc']['dvc_original_train_md5'][:8]}…</code> y <code>{c1['dvc']['dvc_original_test_md5'][:8]}…</code>) y <b>preparadas</b>
(<code>train.parquet</code>, <code>val.parquet</code>, <code>split_assignment.csv</code>, en <code>dvc.lock</code>), ambas recuperables desde un remote HTTP público.</p>

<h2>2. Experimentos</h2>
<p>Siete configuraciones de cuatro familias (baseline, regresión logística, Random Forest, boosting) sobre las <b>mismas particiones</b>; cada una cambia una sola decisión respecto a su referencia,
con hipótesis y criterio medible escritos antes de entrenar (commit <code>{c1['git']['commit_codigo_que_entreno'][:7]}</code>). MLflow registra hipótesis, parámetros, semilla, commit, versión de datos, métricas y el modelo completo (un run por experimento, exportados a <code>evidencias/mlflow/</code>).
Métricas con ataque como clase positiva, en el umbral de cada experimento (mayor recall con FPR ≤ 5 % en validación).</p>
<table><tr><th>ID</th><th>Cambio</th><th>Familia</th><th class=n>Recall</th><th class=n>Precision</th><th class=n>Macro-F1</th><th class=n>ROC-AUC</th><th>¿Hipótesis?</th></tr>
{filas_exp}</table>
<div class="fig"><figure><img src="data:image/png;base64,{img1}"><figcaption>Fig. 1. Recall en validación con la FPR fijada en 5 % (todas las ejecuciones comparten partición).</figcaption></figure>
<div style="flex:1;font-size:8.3pt"><p><b>Lectura.</b> El preprocesamiento (log(1+x), E2) y los árboles (E4, E5) mejoran de forma clara y con costos distintos; el <i>class_weight</i> (E3) cambia el punto de operación a umbral 0,5 pero no el orden de las puntuaciones; con 25 % de los datos (E6) se pierde poco ({es(exp.set_index('id').loc['E4_random_forest','macro_f1'] - exp.set_index('id').loc['E6_random_forest_25pct_datos','macro_f1'], 4)} de macro-F1) y el entrenamiento es unas {es(exp.set_index('id').loc['E4_random_forest','segundos_entrenamiento'] / exp.set_index('id').loc['E6_random_forest_25pct_datos','segundos_entrenamiento'], 1)} veces más rápido.
{n_conf} de {len(exp)} hipótesis se cumplieron según su criterio (<code>experimentos.csv</code>){' — todas —' if n_conf == len(exp) else ''}; debe leerse con cautela: se escribieron conociendo una iteración previa del proyecto (§6).</p></div></div>

<h2>3. Versiones v1 y v2 y elección del umbral</h2>
<p><b>Umbral.</b> Regla declarada antes de ver resultados: <i>entre los umbrales con FPR ≤ 5 % en validación, el de mayor recall</i>; se predice ataque si <code>score_attack ≥ umbral</code>. Se prefirió a 0,5 porque 0,5 deja una FPR de {pct(c2['metricas_validacion']['a_0.5_referencia']['fpr'])} en v2 ({pct(c1['metricas_validacion']['a_0.5_referencia']['fpr'])} en v1) y una falsa alarma tiene costo operativo.
Denominadores cero: precision = 0 sin alertas, recall = 0 sin ataques reales, FPR = 0 sin normales reales (no ocurre en validación ni test).</p>
<p><b>v1</b> (fija de antemano): regresión logística con log(1+x) (E2). <b>v2</b> (elegida solo con validación, antes de tocar el test): el experimento de modelo real con mayor macro-F1, <code>{c2['experimento']}</code>;
cambio único = tipo de modelo. Frente al segundo (Random Forest) su ventaja es pequeña: {es(dif['media_bootstrap'], 4)} de macro-F1 (IC 95 % {es(dif['ic95'][0], 4)} a {es(dif['ic95'][1], 4)}, bootstrap de {dif['remuestreos']}).</p>
<table><tr><th>Validación ({miles(c2['metricas_validacion']['filas'])} filas)</th><th class=n>v1</th><th class=n>v2</th><th class=n>Cambio</th></tr>{filas_cmp}</table>
<p><b>Qué mejora:</b> se detectan más ataques con las mismas falsas alarmas. <b>Qué cuesta:</b> el paquete pasa de {miles(round(c_['tamano_modelo_bytes']['v1'] / 1024))} KB a {miles(round(c_['tamano_modelo_bytes']['v2'] / 1024))} KB,
el entrenamiento de {es(c_['segundos_entrenamiento']['v1'], 1)} s a {es(c_['segundos_entrenamiento']['v2'], 1)} s y se pierde la lectura de coeficientes. <b>Qué se acepta:</b> {miles(v2['fn'])} ataques no detectados y {miles(v2['fp'])} falsas alarmas en validación.</p>

<h2>4. Test final</h2>
<p>La selección se congeló (<code>evidencias/seleccion.json</code>, tag <code>congelado</code>) después de publicar v1/v2 y antes de abrir el test; la solución congelada se evaluó <b>una vez</b> sobre las {miles(test['dataset']['filas_evaluadas'])} filas del archivo (ninguna excluida).</p>
<div class="fig"><figure><img src="data:image/png;base64,{img2}"><figcaption>Fig. 2. v2 con el umbral fijado en validación.</figcaption></figure>
<div style="flex:1"><table><tr><th>Test final</th><th class=n>Valor</th><th class=n>IC 95 %</th></tr>
<tr><td>Recall</td><td class=n>{pct(tm['recall'])}</td><td class=n>{pct(ic['recall'][0])}–{pct(ic['recall'][1])}</td></tr>
<tr><td>Precision</td><td class=n>{pct(tm['precision'])}</td><td class=n>{pct(ic['precision'][0])}–{pct(ic['precision'][1])}</td></tr>
<tr><td>FPR</td><td class=n>{pct(tm['fpr'])}</td><td class=n>{pct(ic['fpr'][0])}–{pct(ic['fpr'][1])}</td></tr>
<tr><td>Macro-F1</td><td class=n>{es(tm['macro_f1'], 4)}</td><td class=n>{es(ic['macro_f1'][0], 4)}–{es(ic['macro_f1'][1], 4)}</td></tr>
<tr><td>VP / FP / FN / VN</td><td class=n colspan=2>{miles(cm['tp'])} / {miles(cm['fp'])} / {miles(cm['fn'])} / {miles(cm['tn'])}</td></tr></table></div></div>
<div class="aviso"><b>El resultado empeora y se conserva.</b> La FPR pasa de 5 % a {pct(tm['fpr'])}: el umbral de validación no controla las falsas alarmas en el test. No se debe solo a entradas desconocidas:
la FPR fue {pct(sub['entradas_nuevas']['fpr'])} en las entradas nuevas y {pct(sub['entradas_ya_vistas_en_entrenamiento_o_validacion']['fpr'])} en las ya vistas. La validación (partición aleatoria agrupada del mismo archivo) no reproduce el cambio entre las particiones publicadas.
Por familia de ataque, el menor recall es <i>{peor_cat}</i> ({pct(cat[peor_cat]['recall'])}); el resto supera {es(100 * resto, 0)} %. Las conclusiones se limitan a estas particiones de un benchmark histórico.</div>

<h2>5. Trazabilidad y recuperación</h2>
<p>Cadena: <b>commit de Git → referencia DVC → run ID de MLflow → revisión de Hugging Face</b>. v1: tag <code>v1</code>, run <code>{c1['mlflow']['run_id'][:8]}…</code>, HF <code>{hf['v1']['commit_sha'][:10]}…</code>, modelo <code>{c1['modelo_sha256'][:10]}…</code>.
v2: tag <code>v2</code>, run <code>{c2['mlflow']['run_id'][:8]}…</code>, HF <code>{hf['v2']['commit_sha'][:10]}…</code>, modelo <code>{c2['modelo_sha256'][:10]}…</code> (identificadores completos en <code>entrega.md</code>).
El mismo archivo <code>modelo.skops</code> se describe, se descarga y produce las predicciones.
<b>Prueba de recuperación</b> en carpeta, entorno virtual y cachés nuevos ({rec['entorno_nuevo']['sistema']}, Python {rec['entorno_nuevo']['python']}): clonar, instalar dependencias fijadas ({es(rec['tiempos_segundos']['instalar_dependencias'], 0)} s),
<code>dvc pull</code> desde el almacenamiento entregado ({es(rec['tiempos_segundos']['dvc_pull'], 0)} s) con huellas SHA-256 coincidentes, descarga de v1 y v2 de Hugging Face por SHA completo y predicción sobre {rec['entradas']['filas']} filas fijas de validación:
y_pred idénticas y diferencia máxima de puntuación {max(d['diferencia_maxima_score'] for d in rec['comparacion'].values()):.1e} (tolerancia {rec['comparacion']['v1']['tolerancia']:.0e}) en ambas versiones; {rec['v1_vs_v2']['filas_con_distinta_prediccion']} de {rec['entradas']['filas']} filas difieren entre v1 y v2.</p>

<h2>6. Limitaciones</h2>
<ul>
<li>Benchmark de 2015 en laboratorio; copia de un mirror sin autenticar contra UNSW. No demuestra eficacia sobre una red actual.</li>
<li>Una sola partición de validación; la ventaja de v2 sobre Random Forest es pequeña. Los IC de test tratan filas como independientes y subestiman la incertidumbre (hay entradas repetidas).</li>
<li>El test difiere de la validación y la FPR se multiplica por {es(tm['fpr'] / 0.05, 1)}: antes de usarlo habría que fijar el umbral con datos del dominio de despliegue o controlar la FPR con un conjunto de calibración independiente.</li>
<li>Historia: el repositorio conserva una primera iteración con un pipeline distinto que ya había evaluado el test con una configuración parecida; se reconstruyó todo y ningún modelo ni umbral se eligió con el test, pero las hipótesis se escribieron conociendo esos resultados previos.</li>
<li>Los paquetes skops no son reproducibles byte a byte al reconstruirse (identificadores de objeto, fechas): el hash se fija al empaquetar y las versiones se recuperan por DVC/Hugging Face.</li>
<li>Uso de IA: el asistente Claude escribió y depuró código y documentación bajo dirección del estudiante; las cifras salen de scripts y la recuperación se verificó de forma automática (ver <code>entrega.md</code>).</li>
</ul>

<div class="refs"><h2>Referencias</h2>
<ul>
<li>Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. <i>MilCIS</i>. https://research.unsw.edu.au/projects/unsw-nb15-dataset</li>
<li>Moustafa, N. y Slay, J. (2016). The evaluation of network anomaly detection systems. <i>Information Security Journal</i>, 25(1–3).</li>
<li>Pedregosa et al. (2011). Scikit-learn: machine learning in Python. <i>JMLR</i> 12. · Guía de errores comunes: https://scikit-learn.org/stable/common_pitfalls.html</li>
<li>DVC: https://dvc.org/doc · MLflow Tracking: https://mlflow.org/docs/latest/tracking.html · skops: https://skops.readthedocs.io</li>
<li>Hugging Face Hub, descarga por revisión: https://huggingface.co/docs/huggingface_hub/guides/download · Model cards: https://huggingface.co/docs/hub/model-cards</li>
</ul></div>
</body></html>"""


def main():
    from playwright.sync_api import sync_playwright
    salida = ROOT / "informe.pdf"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.set_content(construir(), wait_until="load")
        pg.pdf(path=str(salida), format="A4", print_background=True, prefer_css_page_size=True)
        b.close()
    from pypdf import PdfReader
    n = len(PdfReader(str(salida)).pages)
    print(f"{salida.name}: {n} páginas en total (el límite es 5 sin contar la página de referencias)")
    if n > 6:
        sys.exit("El informe excede el límite: recorta contenido.")


if __name__ == "__main__":
    main()
