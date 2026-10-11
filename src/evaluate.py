"""Evaluación ÚNICA en test con el modelo y el umbral congelados. Se conserva el resultado aunque empeore."""
import json

import joblib
import mlflow
import pandas as pd

from src.config import MODELS, PARAMS, REPORTS
from src.data import TARGET, feature_columns, load_raw, pattern_hash
from src.experiments import git_commit
from src.metrics import at_threshold, threshold_free

OUT = REPORTS / "test_metrics.json"


def evaluate_test_once(force: bool = False) -> dict:
    if OUT.exists() and not force:
        raise RuntimeError(
            f"{OUT.name} ya existe: el test se evalúa UNA sola vez y el resultado se conserva. "
            "Usa force=True solo si cambió la decisión congelada (y documéntalo)."
        )
    frozen = json.loads((REPORTS / "selection.json").read_text())
    bundle = joblib.load(MODELS / "final_model.joblib")
    if bundle["experiment"] != frozen["selected_experiment"] or abs(bundle["threshold"] - frozen["threshold"]) > 1e-12:
        raise RuntimeError("El modelo guardado no corresponde a la decisión congelada en selection.json.")

    test = load_raw("test")
    X, y = test[bundle["feature_columns"]], test[TARGET].to_numpy()
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    thr = bundle["threshold"]
    m = at_threshold(y, proba, thr)
    m.update(threshold_free(y, proba))

    # Análisis de errores (post hoc, NO se usa para elegir nada): recall por familia de ataque
    test = test.assign(pred=(proba >= thr).astype(int))
    atk = test[test[TARGET] == 1].groupby("attack_cat")["pred"].agg(["mean", "size"])
    por_familia = {k: {"recall": round(float(r["mean"]), 4), "n": int(r["size"])} for k, r in atk.iterrows()}
    fp_por_servicio = (test[(test[TARGET] == 0) & (test.pred == 1)]["service"].value_counts().head(5)).to_dict()

    # Limitación: ¿cuántas filas de test repiten un patrón de entradas visto en entrenamiento?
    train = load_raw("train")
    seen = set(pattern_hash(train))
    share_seen = float(pattern_hash(test).isin(seen).mean())

    result = {
        "model": frozen["selected_experiment"], "threshold": thr,
        "test_rows": len(test), "metrics": {k: (float(v) if not isinstance(v, int) else v) for k, v in m.items()},
        "recall_por_familia": por_familia, "falsas_alarmas_por_servicio_top5": fp_por_servicio,
        "porcentaje_test_con_patron_visto_en_train": round(share_seen, 4),
        "git_commit": git_commit(), "mlflow_run_id": frozen["mlflow_run_id"],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    mlflow.set_tracking_uri(PARAMS["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=frozen["mlflow_run_id"]):
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m.items() if k != "threshold"})
    return result
