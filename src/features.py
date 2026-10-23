"""Preprocesamiento. Todo se ajusta SOLO con entrenamiento porque vive dentro del Pipeline.

Se usan únicamente objetos de scikit-learn y numpy (sin funciones propias) para que el modelo guardado
pueda cargarse en cualquier entorno con las dependencias de requirements.txt, sin este repositorio.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import PARAMS

CATEGORICAL = PARAMS["data"]["categorical"]


def build_preprocessor(numeric_cols: list[str], log1p: bool, escalado: bool) -> ColumnTransformer:
    """Numéricas: log(1+x) opcional (todas son >= 0 en entrenamiento) y escalado opcional.
    Categóricas: one-hot; categorías con menos de 100 filas y categorías desconocidas -> «infrecuente»."""
    steps = []
    if log1p:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if escalado:
        steps.append(("escalado", StandardScaler()))
    num = Pipeline(steps) if steps else "passthrough"
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)
    return ColumnTransformer([("num", num, numeric_cols), ("cat", cat, CATEGORICAL)], verbose_feature_names_out=False)
