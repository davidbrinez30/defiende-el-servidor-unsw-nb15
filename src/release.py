"""Empaqueta cada versión (v1, v2) como una carpeta autocontenida en formato skops.

models/releases/<vN>/
    modelo.skops        preprocesamiento + estimador completos (skops: no ejecuta código al cargar)
    config.json         umbral, columnas de entrada, hashes, run de MLflow, commit de Git, referencias DVC, métricas
    requirements.txt    dependencias mínimas para cargar y predecir

v1 es fija (configs/versiones.yaml). v2 se elige AUTOMÁTICAMENTE con la validación: mayor macro-F1 al umbral de la regla
entre los experimentos con modelo real; el test no interviene en ninguna parte de este módulo.
"""
import json
import platform
import shutil
from importlib.metadata import version as pkg_version

import joblib
import numpy as np
import pandas as pd
import skops.io as sio

from src.config import EVIDENCIAS, MODELS, PARAMS, PROCESSED, ROOT, load_yaml
from src.data import ID_COL, TARGET, feature_columns, load_split, sha256_file
from src.modelio import predict_scores
from src.metrics import POLITICA_DENOMINADOR_CERO, at_threshold, macro_f1_diff_bootstrap
from src.provenance import data_refs, dvc_lock_outs

RELEASES = MODELS / "releases"
RESUMEN = PROCESSED / "val_proba" / "resumen_ejecuciones.csv"
MUESTRA = EVIDENCIAS / "recuperacion" / "entradas_validacion.csv"
DEPENDENCIAS_INFERENCIA = ["scikit-learn", "numpy", "pandas", "scipy", "skops", "huggingface_hub"]
METRICAS = ["recall", "precision", "fpr", "macro_f1", "accuracy"]


def resumen() -> pd.DataFrame:
    if not RESUMEN.exists():
        raise FileNotFoundError(f"Falta {RESUMEN}: ejecuta `dvc repro experiments` (o `dvc pull`).")
    return pd.read_csv(RESUMEN)


def muestra_fija(n_por_clase: int = 25) -> pd.DataFrame:
    """Muestra fija de validación para la prueba de recuperación: 25 normales + 25 ataques (semilla del proyecto)."""
    val = load_split("val")
    rng = np.random.RandomState(PARAMS["seed"])
    partes = []
    for clase in (0, 1):
        idx = np.where(val[TARGET].to_numpy() == clase)[0]
        partes.append(np.sort(rng.choice(idx, n_por_clase, replace=False)))
    sel = val.iloc[np.sort(np.concatenate(partes))]
    out = pd.DataFrame({"row_id": [f"training-set:{i}" for i in sel[ID_COL]]})
    out = pd.concat([out, sel[feature_columns(val)].reset_index(drop=True)], axis=1)
    out["y_true"] = sel[TARGET].to_numpy()
    MUESTRA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MUESTRA, index=False)
    return out


def _versiones_instaladas() -> dict:
    return {"python": platform.python_version(), **{p: pkg_version(p) for p in DEPENDENCIAS_INFERENCIA}}


def _empaquetar(version: str, fila: pd.Series, descripcion: str, extra: dict) -> dict:
    """Convierte el modelo de un run (joblib) a skops, comprueba que predice idéntico y escribe la carpeta de la versión."""
    destino = RELEASES / version
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    run_joblib = ROOT / fila["modelo_archivo"]
    if sha256_file(run_joblib) != fila["modelo_sha256"]:
        raise ValueError(f"{run_joblib.name} no coincide con el hash registrado en el run {fila['run_id']}.")
    pipe = joblib.load(run_joblib)

    skops_path = destino / "modelo.skops"
    sio.dump(pipe, skops_path)
    tipos = sorted(sio.get_untrusted_types(file=skops_path))
    cargado = sio.load(skops_path, trusted=tipos)

    # El modelo convertido debe producir exactamente las puntuaciones de validación registradas en el run.
    val = load_split("val")
    cols = feature_columns(val)
    guardadas = np.load(PROCESSED / "val_proba" / f"{fila['experimento']}.npy")
    nuevas = cargado.predict_proba(val[cols])[:, 1]
    dif = float(np.max(np.abs(nuevas - guardadas)))
    if dif > 1e-12:
        raise RuntimeError(f"Conversión a skops cambió las predicciones (diferencia máxima {dif}).")

    y = val[TARGET].to_numpy()
    umbral = float(fila["umbral"])
    refs = data_refs()
    cfg = {
        "version": version,
        "descripcion": descripcion,
        "experimento": fila["experimento"],
        "familia": fila["familia"],
        "mlflow": {"experimento": PARAMS["mlflow"]["experiment"], "run_id": fila["run_id"]},
        "git": {"commit_codigo_que_entreno": fila["git_commit"],
                "repositorio": f"https://github.com/{load_yaml('equipo.yaml')['github']['usuario']}/"
                               f"{load_yaml('equipo.yaml')['github']['repo_codigo']}",
                "tag": version},
        "dvc": {**refs, "prepare_outs_md5": dvc_lock_outs("prepare")},
        "modelo_archivo": "modelo.skops",
        "modelo_sha256": sha256_file(skops_path),
        "modelo_joblib_del_run_sha256": fila["modelo_sha256"],
        "tipos_confiables_skops": tipos,
        "columnas_entrada": cols,
        "excluidas_de_la_entrada": ["id", "attack_cat", "label"],
        "clase_positiva": "ataque (label = 1)",
        "regla_umbral": "y_pred = 1 (ataque) si score_attack >= umbral; score_attack = probabilidad de ataque",
        "umbral": umbral,
        "semilla": PARAMS["seed"],
        "metricas_validacion": {
            "filas": int(len(val)),
            "al_umbral": at_threshold(y, guardadas, umbral),
            "a_0.5_referencia": at_threshold(y, guardadas, 0.5),
            "roc_auc": float(fila["roc_auc"]), "pr_auc": float(fila["pr_auc"]),
        },
        "politica_denominador_cero": POLITICA_DENOMINADOR_CERO,
        "verificacion_conversion": {"filas": int(len(val)), "diferencia_maxima_score": dif,
                                    "nota": "skops reproduce las puntuaciones del run de MLflow"},
        "entorno": _versiones_instaladas(),
        **extra,
    }
    (destino / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reqs = [f"{p}=={pkg_version(p)}" for p in DEPENDENCIAS_INFERENCIA]
    (destino / "requirements.txt").write_text(
        f"# Dependencias para cargar y usar esta versión (Python {platform.python_version()}, probado)\n" + "\n".join(reqs) + "\n",
        encoding="utf-8")
    # Salida guardada ANTES de cualquier recuperación: la prueba de recuperación debe reproducirla.
    if not MUESTRA.exists():
        muestra_fija()
    entradas = pd.read_csv(MUESTRA)
    salida = predict_scores(destino, entradas)
    salida.insert(0, "row_id", entradas["row_id"])
    salida.to_csv(MUESTRA.parent / f"salida_antes_{version}.csv", index=False)
    print(f"{version}: {fila['experimento']} umbral={umbral:.4f} macroF1={cfg['metricas_validacion']['al_umbral']['macro_f1']:.4f} "
          f"modelo.skops sha256={cfg['modelo_sha256'][:12]}… tipos_no_confiables={tipos}", flush=True)
    return cfg


def build_v1() -> dict:
    v = load_yaml("versiones.yaml")["versiones"]["v1"]
    tabla = resumen()
    fila = tabla[tabla["experimento"] == v["experimento"]].iloc[0]
    muestra_fija()
    return _empaquetar("v1", fila, v["descripcion"], {"seleccion": "fija, declarada en configs/versiones.yaml antes de los resultados"})


def build_v2() -> dict:
    """Elige v2 con la validación, guarda la decisión (evidencias/seleccion_validacion.json) y empaqueta."""
    vcfg = load_yaml("versiones.yaml")
    v1, v2 = vcfg["versiones"]["v1"], vcfg["versiones"]["v2"]
    tabla = resumen()
    excl = set(v2.get("candidatos_excluidos", []))
    cand = tabla[~tabla["experimento"].isin(excl)].sort_values(
        ["macro_f1", "roc_auc", "experimento"], ascending=[False, False, True]).reset_index(drop=True)
    elegido, segundo = cand.iloc[0], cand.iloc[1]
    if elegido["experimento"] == v1["experimento"]:
        raise RuntimeError("El mejor modelo en validación es el de v1: v2 no se distinguiría de v1. Revisa configs/versiones.yaml.")

    val = load_split("val")
    y = val[TARGET].to_numpy()
    pe = np.load(PROCESSED / "val_proba" / f"{elegido['experimento']}.npy")
    ps = np.load(PROCESSED / "val_proba" / f"{segundo['experimento']}.npy")
    n_boot = PARAMS["selection"]["bootstrap_resamples"]
    d_media, d_ic = macro_f1_diff_bootstrap(y, pe, ps, elegido["umbral"], segundo["umbral"], n=n_boot, seed=PARAMS["seed"])

    f1 = tabla[tabla["experimento"] == v1["experimento"]].iloc[0]
    cambios = {}
    for m in METRICAS:
        d = float(elegido[m] - f1[m])
        mejor_si_sube = m != "fpr"
        cambios[m] = {"v1": float(f1[m]), "v2": float(elegido[m]), "diferencia": d,
                      "efecto": "sin cambio" if abs(d) < 1e-12 else
                                ("mejora" if (d > 0) == mejor_si_sube else "empeora")}
    cols = ["experimento", "familia", "run_id", "umbral", *METRICAS, "roc_auc", "pr_auc", "fit_seconds"]
    decision = {
        "criterio": ("Mayor macro-F1 en VALIDACIÓN al umbral de la regla (mayor recall con FPR <= "
                     f"{100 * PARAMS['selection']['max_fpr']:.0f} %). Desempate: mayor ROC-AUC, luego orden alfabético. "
                     "El test no interviene."),
        "candidatos_excluidos": sorted(excl),
        "ranking_validacion": cand[cols].to_dict("records"),
        "excluidos": tabla[tabla["experimento"].isin(excl)][cols].to_dict("records"),
        "elegido": {c: elegido[c] if not isinstance(elegido[c], (np.integer, np.floating)) else elegido[c].item() for c in cols},
        "segundo_mejor": {"experimento": segundo["experimento"], "macro_f1": float(segundo["macro_f1"])},
        "diferencia_macro_f1_vs_segundo": {
            "media_bootstrap": d_media, "ic95": d_ic, "remuestreos": n_boot,
            "lectura": ("el intervalo no incluye 0: la ventaja del elegido sobre el segundo es distinguible en validación"
                        if d_ic[0] > 0 or d_ic[1] < 0 else
                        "el intervalo incluye 0: elegido y segundo no se distinguen con esta validación; se conserva la regla acordada"),
        },
        "cambio_respecto_a_v1": {"descripcion": v2["cambio_respecto_a_v1"], "v1_experimento": v1["experimento"],
                                 "metricas": cambios},
        "test": "No usado en esta decisión. Se evalúa una sola vez después de congelar.",
    }
    EVIDENCIAS.mkdir(exist_ok=True)
    (EVIDENCIAS / "seleccion_validacion.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    cfg_v1 = json.loads((RELEASES / "v1" / "config.json").read_text(encoding="utf-8"))
    costos = {
        "tamano_modelo_bytes": {"v1": (RELEASES / "v1" / cfg_v1["modelo_archivo"]).stat().st_size, "v2": None},
        "segundos_entrenamiento": {"v1": float(f1["fit_seconds"]), "v2": float(elegido["fit_seconds"])},
    }
    cfg = _empaquetar("v2", elegido, v2["descripcion"], {
        "seleccion": "automática por macro-F1 en validación (ver evidencias/seleccion_validacion.json)",
        "comparacion_con_v1": cambios,
        "costos_frente_a_v1": costos,
    })
    # El tamaño de v2 se conoce una vez escrito el paquete.
    costos["tamano_modelo_bytes"]["v2"] = (RELEASES / "v2" / cfg["modelo_archivo"]).stat().st_size
    cfg["costos_frente_a_v1"] = costos
    (RELEASES / "v2" / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg
