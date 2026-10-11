"""Definición de experimentos. Las HIPÓTESIS se escriben (y se versionan en Git) antes de entrenar.

Reglas de comparación justa: mismas particiones train/val para todos; se cambia UNA decisión
respecto a su experimento de referencia (columna `vs`).
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42

EXPERIMENTS = [
    dict(
        id="E0_baseline_siempre_ataque", vs=None, model_family="dummy",
        hypothesis="Predecir siempre la clase mayoritaria (ataque) da ~68% de accuracy pero 100% de falsas alarmas "
                   "y macro-F1 bajo. Sirve de piso: cualquier modelo útil debe superarlo en FPR y macro-F1.",
        build=lambda: DummyClassifier(strategy="constant", constant=1),
        log=False, scale=False, train_fraction=1.0,
    ),
    dict(
        id="E1_logreg_cruda", vs="E0", model_family="logistic_regression",
        hypothesis="Un modelo lineal sobre variables escaladas pero SIN transformar (colas largas) detecta bastante "
                   "mejor que el baseline, aunque se verá limitado por las escalas de bytes/tasas.",
        build=lambda: LogisticRegression(max_iter=2000, solver="lbfgs"),
        log=False, scale=True, train_fraction=1.0,
    ),
    dict(
        id="E2_logreg_log1p", vs="E1", model_family="logistic_regression",
        hypothesis="Aplicar log(1+x) antes de escalar mejora la regresión logística (menos influencia de valores "
                   "extremos): espero subir el macro-F1 y reducir la FPR frente a E1.",
        build=lambda: LogisticRegression(max_iter=2000, solver="lbfgs"),
        log=True, scale=True, train_fraction=1.0,
    ),
    dict(
        id="E3_logreg_log1p_balanceada", vs="E2", model_family="logistic_regression",
        hypothesis="Con class_weight='balanced' el modelo atiende más a la clase normal (minoritaria, 32%): espero "
                   "MENOS falsas alarmas a costa de algo de recall frente a E2.",
        build=lambda: LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"),
        log=True, scale=True, train_fraction=1.0,
    ),
    dict(
        id="E4_random_forest", vs="E2", model_family="random_forest",
        hypothesis="Un modelo de árboles captura interacciones entre atributos (la EDA mostró que ninguna variable "
                   "separa sola). Espero superar claramente a la mejor regresión logística en macro-F1.",
        build=lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        log=False, scale=False, train_fraction=1.0,
    ),
    dict(
        id="E5_hist_gradient_boosting", vs="E4", model_family="hist_gradient_boosting",
        hypothesis="Gradient boosting (más complejidad secuencial) mejora a Random Forest en AUC; pero con duplicados "
                   "y etiquetas ambiguas el techo es bajo, así que la ganancia será pequeña (<1 punto).",
        build=lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED),
        log=False, scale=False, train_fraction=1.0,
    ),
    dict(
        id="E6_random_forest_25pct_datos", vs="E4", model_family="random_forest",
        hypothesis="Usar solo 25% de los datos de entrenamiento perderá poco frente a E4 porque hay muchas filas "
                   "repetidas: la pérdida de macro-F1 será menor a 1 punto y el costo en tiempo baja ~4x.",
        build=lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        log=False, scale=False, train_fraction=0.25,
    ),
]
