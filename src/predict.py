"""Carga una versión guardada del modelo y produce predicciones.

  python -m src.predict --input datos.csv --source file      # models/final_model.joblib (versionado con DVC)
  python -m src.predict --input datos.csv --source mlflow    # Model Registry de MLflow (versión indicada)
"""
import argparse

import joblib
import pandas as pd

from src.config import MODELS, PARAMS

TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def load_bundle(source: str = "file", version: str | None = None) -> dict:
    if source == "file":
        return joblib.load(MODELS / "final_model.joblib")
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(PARAMS["mlflow"]["tracking_uri"])
    name = PARAMS["mlflow"]["registered_model"]
    client = MlflowClient()
    version = version or max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)).version
    mv = client.get_model_version(name, str(version))
    thr = float(mv.tags["threshold"])
    uri = f"models:/{name}/{version}"
    try:
        pipe = mlflow.sklearn.load_model(uri)
    except Exception:
        # skops exige declarar de confianza el tipo interno de HistGradientBoosting (lo entrenamos nosotros).
        from mlflow.artifacts import download_artifacts
        from skops.io import load
        path = download_artifacts(artifact_uri=uri)
        import glob
        f = glob.glob(f"{path}/*.skops")[0]
        pipe = load(f, trusted=TRUSTED)
    return {"pipeline": pipe, "threshold": thr, "experiment": mv.tags.get("selected_from"), "version": version}


def predict(df: pd.DataFrame, source: str = "file", version: str | None = None) -> pd.DataFrame:
    b = load_bundle(source, version)
    cols = b.get("feature_columns") or list(df.columns)
    proba = b["pipeline"].predict_proba(df[cols])[:, 1]
    return pd.DataFrame({"proba_ataque": proba.round(4), "alerta": (proba >= b["threshold"]).astype(int)}, index=df.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV con las columnas predictoras de UNSW-NB15")
    ap.add_argument("--source", choices=["file", "mlflow"], default="file")
    ap.add_argument("--version", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    out = predict(pd.read_csv(a.input, encoding="utf-8-sig"), a.source, a.version)
    print(out.head(10).to_string()) if not a.output else out.to_csv(a.output)
