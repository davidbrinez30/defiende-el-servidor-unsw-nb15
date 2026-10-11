"""Elige modelo y umbral SOLO con validación, congela la decisión y registra el modelo en MLflow."""
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import MODELS, PARAMS, PROCESSED, REPORTS
from src.data import load_split
from src.experiment_configs import EXPERIMENTS
from src.experiments import git_commit, make_pipeline, numeric_columns, split_xy
from src.metrics import at_threshold, pick_threshold, threshold_free

M = PARAMS["mlflow"]
MAX_FPR = PARAMS["selection"]["max_fpr"]
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def paired_bootstrap(y, pa, pb, thr_a, thr_b, n=500, seed=0):
    """Diferencia de macro-F1 (A-B) con IC95% remuestreando filas de validación."""
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        fa = f1_score(y[i], pa[i] >= thr_a, average="macro")
        fb = f1_score(y[i], pb[i] >= thr_b, average="macro")
        diffs.append(fa - fb)
    return float(np.mean(diffs)), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def select_and_freeze() -> dict:
    val = load_split("val")
    _, yva = split_xy(val)

    rows = []
    for cfg in EXPERIMENTS:
        if cfg["model_family"] == "dummy":
            continue
        proba = np.load(PROCESSED / f"val_proba_{cfg['id']}.npy")
        thr = pick_threshold(yva, proba, MAX_FPR)
        m = at_threshold(yva, proba, thr)
        rows.append({"experiment": cfg["id"], **m, **threshold_free(yva, proba)})
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(REPORTS / "validation_at_chosen_threshold.csv", index=False)

    best = table.iloc[0]
    second = table.iloc[1]
    pa = np.load(PROCESSED / f"val_proba_{best['experiment']}.npy")
    pb = np.load(PROCESSED / f"val_proba_{second['experiment']}.npy")
    diff, ci = paired_bootstrap(yva, pa, pb, best["threshold"], second["threshold"])

    # Curva umbral -> (recall, FPR) del modelo elegido, para justificar el equilibrio
    tradeoff = [at_threshold(yva, pa, t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    pd.DataFrame(tradeoff).to_csv(REPORTS / "threshold_tradeoff_val.csv", index=False)

    cfg = next(c for c in EXPERIMENTS if c["id"] == best["experiment"])

    # Reentrena la configuración ganadora SOLO con entrenamiento y comprueba que reproduce la validación.
    train = load_split("train")
    Xtr, ytr = split_xy(train)
    Xva, _ = split_xy(val)
    num = numeric_columns(Xtr)
    pipe = make_pipeline(cfg, num).fit(Xtr, ytr)
    reproduced = np.allclose(pipe.predict_proba(Xva)[:, 1], pa, atol=1e-9)
    if not reproduced:
        raise RuntimeError("El reentrenamiento no reproduce las probabilidades de validación del experimento.")

    thr = float(best["threshold"])
    mlflow.set_tracking_uri(M["tracking_uri"])
    mlflow.set_experiment(M["experiment"])
    with mlflow.start_run(run_name="FINAL_" + cfg["id"]) as run:
        mlflow.set_tags({"stage": "final", "git_commit": git_commit(), "selected_from": cfg["id"]})
        mlflow.log_params({"threshold": thr, "max_fpr_rule": MAX_FPR, "model_family": cfg["model_family"]})
        mlflow.log_metrics({f"val_{k}": float(best[k]) for k in
                            ["recall", "precision", "fpr", "macro_f1", "accuracy", "roc_auc", "pr_auc"]})
        mlflow.log_artifact(str(REPORTS / "validation_at_chosen_threshold.csv"))
        mlflow.log_artifact(str(REPORTS / "threshold_tradeoff_val.csv"))
        info = mlflow.sklearn.log_model(
            pipe, name="model", registered_model_name=M["registered_model"],
            # skops bloquea por seguridad tipos que no revisó. Este modelo lo entrenamos nosotros, así que
            # confiamos únicamente en ese tipo concreto (no en "todo lo que reporte get_untrusted_types").
            skops_trusted_types=TRUSTED_TYPES,
            signature=infer_signature(Xva.head(50), pipe.predict_proba(Xva.head(50))[:, 1]),
        )
        version = info.registered_model_version
        MlflowClient().set_model_version_tag(M["registered_model"], str(version), "threshold", str(thr))
        MlflowClient().set_model_version_tag(M["registered_model"], str(version), "selected_from", cfg["id"])
        run_id = run.info.run_id

    # Pipeline completo (preprocesamiento + modelo) + umbral, versionado con DVC
    joblib.dump({"pipeline": pipe, "threshold": thr, "feature_columns": list(Xtr.columns),
                 "experiment": cfg["id"], "mlflow_run_id": run_id}, MODELS / "final_model.joblib")

    decision = {
        "selected_experiment": cfg["id"], "model_family": cfg["model_family"], "threshold": thr,
        "rule": f"mayor recall con FPR<={MAX_FPR} en validación; modelo por macro-F1 con ese umbral",
        "validation_at_threshold": {k: (float(best[k]) if not isinstance(best[k], str) else best[k])
                                    for k in ["recall", "precision", "fpr", "macro_f1", "accuracy", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]},
        "runner_up": second["experiment"],
        "macro_f1_diff_vs_runner_up": {"mean": diff, "ci95": ci},
        "mlflow_run_id": run_id, "mlflow_model_version": str(version),
        "reproduced_validation_probabilities": bool(reproduced),
        "git_commit_at_selection": git_commit(),
    }
    (REPORTS / "selection.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    print(table[["experiment", "threshold", "recall", "precision", "fpr", "macro_f1"]].round(4).to_string(index=False))
    print("ELEGIDO:", cfg["id"], "umbral", round(thr, 4), "| dif vs 2º:", round(diff, 4), [round(c, 4) for c in ci])
    return decision
