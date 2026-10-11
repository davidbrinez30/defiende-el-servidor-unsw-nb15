"""Transformaciones de entrada. Todo se ajusta SOLO con entrenamiento porque vive dentro del Pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import PARAMS

CATEGORICAL = PARAMS["data"]["categorical"]


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(numeric_cols: list[str], log: bool, scale: bool) -> ColumnTransformer:
    steps = []
    if log:
        steps.append(("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    num = Pipeline(steps) if steps else "passthrough"
    # Categorías raras (<100 filas) y desconocidas en test se agrupan en "infrecuente" en vez de fallar.
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    return ColumnTransformer(
        [("num", num, numeric_cols), ("cat", cat, CATEGORICAL)], verbose_feature_names_out=False
    )
