"""Corre todos los experimentos con las MISMAS particiones y los registra en MLflow."""
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import PARAMS, PROCESSED, REPORTS
from src.data import TARGET, feature_columns, load_split
from src.experiment_configs import EXPERIMENTS
from src.features import build_preprocessor
from src.metrics import at_threshold, threshold_free

M = PARAMS["mlflow"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "sin-git"


def make_pipeline(cfg: dict, numeric_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(numeric_cols, log=cfg["log"], scale=cfg["scale"])),
        ("model", cfg["build"]()),
    ])


def split_xy(df: pd.DataFrame):
    X = df[feature_columns(df)]
    return X, df[TARGET].to_numpy()


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in PARAMS["data"]["categorical"]]


def run_all() -> pd.DataFrame:
    train, val = load_split("train"), load_split("val")
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    num = numeric_columns(Xtr)

    mlflow.set_tracking_uri(M["tracking_uri"])
    mlflow.set_experiment(M["experiment"])
    commit = git_commit()
    rows = []

    for cfg in EXPERIMENTS:
        Xf, yf = Xtr, ytr
        if cfg["train_fraction"] < 1.0:  # misma semilla -> mismo subconjunto siempre
            idx = np.random.RandomState(PARAMS["seed"]).permutation(len(Xtr))[: int(len(Xtr) * cfg["train_fraction"])]
            Xf, yf = Xtr.iloc[idx], ytr[idx]

        with mlflow.start_run(run_name=cfg["id"]):
            mlflow.set_tags({
                "hypothesis": cfg["hypothesis"], "compared_against": str(cfg["vs"]),
                "git_commit": commit, "train_sha256": PARAMS["data"]["expected_train_sha256"][:12],
            })
            mlflow.log_params({
                "model_family": cfg["model_family"], "log1p": cfg["log"], "scale": cfg["scale"],
                "train_fraction": cfg["train_fraction"], "n_train_rows": len(Xf), "seed": PARAMS["seed"],
                **{f"model__{k}": v for k, v in cfg["build"]().get_params().items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
            })
            t0 = time.time()
            pipe = make_pipeline(cfg, num).fit(Xf, yf)
            fit_s = time.time() - t0
            proba = pipe.predict_proba(Xva)[:, 1]
            np.save(PROCESSED / f"val_proba_{cfg['id']}.npy", proba)

            at05 = at_threshold(yva, proba, 0.5)
            tf = threshold_free(yva, proba) if len(np.unique(proba)) > 1 else {"roc_auc": 0.5, "pr_auc": float(yva.mean())}
            metrics = {**{f"val_{k}": v for k, v in at05.items() if k != "threshold"}, **{f"val_{k}": v for k, v in tf.items()}}
            mlflow.log_metrics({**metrics, "fit_seconds": fit_s})
            rows.append({"experiment": cfg["id"], "vs": cfg["vs"], "fit_seconds": round(fit_s, 1),
                         "run_id": mlflow.active_run().info.run_id, **{k.replace("val_", ""): v for k, v in metrics.items()}})
            print(f"{cfg['id']:34s} macroF1={at05['macro_f1']:.4f} recall={at05['recall']:.4f} "
                  f"FPR={at05['fpr']:.4f} AUC={tf['roc_auc']:.4f} ({fit_s:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "experiments.csv", index=False)
    return out
