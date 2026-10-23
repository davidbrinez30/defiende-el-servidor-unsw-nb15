"""Evaluación FINAL en test de la solución congelada (v2). Se ejecuta una sola vez.

Antes de tocar el test comprueba que existe la selección congelada (evidencias/seleccion.json) y que el modelo,
el umbral y su hash coinciden con ella. Nada de lo que se calcula aquí vuelve a usarse para elegir modelo o umbral.
"""
import json
import subprocess

import numpy as np
import pandas as pd

from src.config import EVIDENCIAS, PARAMS, PROCESSED, ROOT
from src.data import ID_COL, TARGET, load_raw, pattern_hash, raw_path, sha256_file
from src.metrics import POLITICA_DENOMINADOR_CERO, at_threshold, confusion, from_confusion, threshold_free
from src.modelio import predict_scores
from src.provenance import git_info
from src.release import RELEASES

PRED = EVIDENCIAS / "predicciones-test.csv"
METR = EVIDENCIAS / "metricas-test.json"
SEL = EVIDENCIAS / "seleccion.json"
METRICAS = ["recall", "precision", "fpr", "macro_f1", "accuracy"]


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def verificar_congelacion() -> dict:
    """La selección debe existir y coincidir con el modelo que se va a evaluar."""
    if not SEL.exists():
        raise SystemExit("No hay evidencias/seleccion.json: primero se congela la selección (python -m src.cli congelar).")
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    cfg = json.loads((RELEASES / sel["version_elegida"] / "config.json").read_text(encoding="utf-8"))
    modelo = RELEASES / sel["version_elegida"] / cfg["modelo_archivo"]
    if sha256_file(modelo) != sel["modelo_sha256"] or cfg["modelo_sha256"] != sel["modelo_sha256"]:
        raise SystemExit("El modelo a evaluar no coincide con el hash congelado en seleccion.json.")
    if abs(cfg["umbral"] - sel["umbral"]) > 0:
        raise SystemExit("El umbral de config.json no coincide con el congelado en seleccion.json.")
    # La congelación debe estar en un commit anterior a esta evaluación y no haber cambiado desde entonces.
    tag = sel["congelacion"]["tag_git"]
    commit_tag = _git("rev-list", "-n", "1", tag)
    prueba = {"tag": tag, "commit_del_tag": commit_tag}
    if commit_tag:
        prueba["es_ancestro_de_HEAD"] = _git("merge-base", "--is-ancestor", commit_tag, "HEAD") is not None
        en_tag = _git("show", f"{tag}:evidencias/seleccion.json")
        actual = SEL.read_text(encoding="utf-8").strip()
        prueba["seleccion_sin_cambios_desde_el_tag"] = (en_tag or "").strip() == actual
        if not prueba["seleccion_sin_cambios_desde_el_tag"]:
            raise SystemExit("seleccion.json cambió después de congelar: la evaluación se invalida.")
    else:
        prueba["aviso"] = "El tag de congelación no existe en este clon; se verificó solo el hash del modelo."
    return {"sel": sel, "cfg": cfg, "prueba": prueba}


def _ic(y, pred, n, seed):
    rng = np.random.RandomState(seed)
    acc = {m: [] for m in METRICAS}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        r = from_confusion(confusion(y[i], pred[i]))
        for m in METRICAS:
            acc[m].append(r[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in acc.items()}


def evaluar_test() -> dict:
    v = verificar_congelacion()
    sel, cfg = v["sel"], v["cfg"]
    te = load_raw("test")                       # verifica SHA-256 y filas del archivo completo
    y = te[TARGET].to_numpy()
    rel = RELEASES / sel["version_elegida"]
    out = predict_scores(rel, te)
    score, pred = out["score_attack"].to_numpy(), out["y_pred"].to_numpy()

    pr = pd.DataFrame({"row_id": [f"testing-set:{i}" for i in te[ID_COL]], "y_true": y, "score_attack": score, "y_pred": pred})
    assert pr["row_id"].is_unique and len(pr) == PARAMS["data"]["expected_test_rows"]
    pr.to_csv(PRED, index=False)

    umbral = cfg["umbral"]
    m = at_threshold(y, score, umbral)
    conf = {k: m[k] for k in ("tp", "fp", "fn", "tn")}
    met = {k: m[k] for k in METRICAS}
    val = cfg["metricas_validacion"]["al_umbral"]

    # Diagnóstico posterior a la congelación (no decide nada): entradas ya vistas vs nuevas, y por familia de ataque.
    asign = pd.read_csv(PROCESSED / "split_assignment.csv")
    vistas_set = set(asign["grupo_patron"])
    hex_te = pattern_hash(te).map(lambda x: f"{int(x):016x}")
    vistas = hex_te.isin(vistas_set).to_numpy()
    sub = {}
    for nombre, mask in (("entradas_ya_vistas_en_entrenamiento_o_validacion", vistas), ("entradas_nuevas", ~vistas)):
        c = confusion(y[mask], pred[mask])
        sub[nombre] = {"filas": int(mask.sum()), "porcentaje_del_test": round(100 * float(mask.mean()), 3), **c, **from_confusion(c)}
    por_cat = {}
    for cat, g in te.groupby("attack_cat"):
        idx = g.index.to_numpy()
        if cat == "Normal":
            por_cat[cat] = {"filas": int(len(idx)), "falsas_alarmas": int(pred[idx].sum()), "fpr": float(pred[idx].mean())}
        else:
            por_cat[cat] = {"filas": int(len(idx)), "detectados": int(pred[idx].sum()), "recall": float(pred[idx].mean())}

    peor = met["macro_f1"] < val["macro_f1"]
    lectura = (
        f"Con el umbral fijado en validación, el test dio recall {100 * met['recall']:.2f} %, FPR {100 * met['fpr']:.2f} % y macro-F1 "
        f"{met['macro_f1']:.4f}; en validación fueron {100 * val['recall']:.2f} %, {100 * val['fpr']:.2f} % y {val['macro_f1']:.4f}. "
        + (f"La FPR del test supera la meta de {100 * PARAMS['selection']['max_fpr']:.0f} % fijada en validación. "
           if met["fpr"] > PARAMS["selection"]["max_fpr"] else
           f"La FPR del test se mantiene dentro de la meta de {100 * PARAMS['selection']['max_fpr']:.0f} %. ")
        + f"El {sub['entradas_nuevas']['porcentaje_del_test']:.1f} % de las filas del test tiene entradas que no aparecen en "
          "entrenamiento ni validación; esa diferencia de distribución explica por qué la validación "
        + ("fue más optimista que el test." if peor else "no fue más optimista que el test.")
        + " El resultado se conserva tal cual; no se ajustó nada después de verlo."
    )
    git = git_info()
    res = {
        "version_evaluada": sel["version_elegida"],
        "modelo": {"archivo": f"models/releases/{sel['version_elegida']}/{cfg['modelo_archivo']}", "sha256": cfg["modelo_sha256"],
                   "experimento": cfg["experimento"], "mlflow_run_id": cfg["mlflow"]["run_id"],
                   "huggingface": sel["referencias"]["huggingface"]},
        "dataset": {"archivo": raw_path("test").name, "sha256": sha256_file(raw_path("test")),
                    "filas_publicadas": PARAMS["data"]["expected_test_rows"], "filas_evaluadas": int(len(te)),
                    "filas_excluidas": 0, "etiqueta": TARGET},
        "umbral": umbral,
        "clase_positiva": "ataque (label = 1)",
        "regla_umbral": cfg["regla_umbral"],
        "matriz_confusion": {**conf, "disposicion": "tp: ataque detectado; fp: normal marcado como ataque; fn: ataque no detectado; tn: normal aceptado"},
        "metricas": met,
        "ic95_bootstrap": {**_ic(y, pred, PARAMS["selection"]["bootstrap_resamples"], PARAMS["seed"]),
                           "nota": f"{PARAMS['selection']['bootstrap_resamples']} remuestreos de filas, umbral fijo"},
        "metricas_ranking": threshold_free(y, score),
        "referencia_umbral_0.5_no_usado": at_threshold(y, score, 0.5),
        "validacion_al_mismo_umbral": val,
        "por_subconjunto_de_entradas": sub,
        "por_familia_de_ataque": por_cat,
        "politica_denominador_cero": POLITICA_DENOMINADOR_CERO,
        "archivo_predicciones": {"ruta": "evidencias/predicciones-test.csv", "sha256": sha256_file(PRED), "filas": int(len(pr)),
                                 "columnas": "row_id (testing-set:<id>), y_true, score_attack (probabilidad de ataque), y_pred"},
        "recalcular": "python -m src.cli recalcular   (usa solo predicciones-test.csv y el umbral de este archivo)",
        "congelacion_verificada": v["prueba"],
        "git": {"commit_de_la_evaluacion": git["git_commit"], "codigo_sin_confirmar": git["codigo_sin_confirmar"]},
        "lectura": lectura,
    }
    METR.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(lectura)
    return res


def recalcular() -> dict:
    """Recalcula las métricas solo con predicciones-test.csv y las compara con metricas-test.json."""
    res = json.loads(METR.read_text(encoding="utf-8"))
    pr = pd.read_csv(PRED)
    umbral = res["umbral"]
    y_pred = (pr["score_attack"].to_numpy() >= umbral).astype(int)
    coincide_pred = bool((y_pred == pr["y_pred"].to_numpy()).all())
    m = at_threshold(pr["y_true"].to_numpy(), pr["score_attack"].to_numpy(), umbral)
    difs = {k: abs(m[k] - res["metricas"][k]) for k in METRICAS}
    difs.update({k: abs(m[k] - res["matriz_confusion"][k]) for k in ("tp", "fp", "fn", "tn")})
    ok = coincide_pred and max(difs.values()) < 1e-9 and len(pr) == res["dataset"]["filas_evaluadas"] and pr["row_id"].is_unique \
        and sha256_file(PRED) == res["archivo_predicciones"]["sha256"]
    print(f"filas={len(pr)} y_pred coincide con score>=umbral: {coincide_pred}; diferencia máxima en métricas: {max(difs.values()):.2e}; "
          f"hash de predicciones coincide: {sha256_file(PRED) == res['archivo_predicciones']['sha256']} -> {'OK' if ok else 'FALLA'}")
    if not ok:
        raise SystemExit(1)
    return {"ok": ok, "diferencias": difs}
