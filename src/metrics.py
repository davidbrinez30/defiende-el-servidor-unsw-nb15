"""Métricas de detección. Clase positiva = ataque (1). Política de denominadores cero declarada abajo."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

POLITICA_DENOMINADOR_CERO = (
    "precision = 0 si no se emite ninguna alerta (TP+FP = 0); recall = 0 si no hay ataques reales (TP+FN = 0); "
    "FPR = 0 si no hay conexiones normales reales (FP+TN = 0). En validación y test ninguno de los denominadores es cero."
)


def confusion(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()), "tn": int(((y == 0) & (pred == 0)).sum()),
    }


def from_confusion(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    div = lambda a, b: a / b if b else 0.0
    precision, recall, fpr = div(tp, tp + fp), div(tp, tp + fn), div(fp, fp + tn)
    f1_atk = div(2 * precision * recall, precision + recall)
    prec_n, rec_n = div(tn, tn + fn), div(tn, tn + fp)
    f1_nor = div(2 * prec_n * rec_n, prec_n + rec_n)
    return {
        "recall": recall, "precision": precision, "fpr": fpr, "macro_f1": (f1_atk + f1_nor) / 2,
        "f1_ataque": f1_atk, "accuracy": div(tp + tn, tp + fp + fn + tn),
    }


def at_threshold(y, proba, thr: float) -> dict:
    """Todas las métricas con la regla `alerta si score >= umbral`."""
    pred = (np.asarray(proba) >= thr).astype(int)
    c = confusion(y, pred)
    return {"umbral": float(thr), **from_confusion(c), **c}


def threshold_free(y, proba) -> dict:
    return {"roc_auc": float(roc_auc_score(y, proba)), "pr_auc": float(average_precision_score(y, proba))}


def pick_threshold(y, proba, max_fpr: float) -> float:
    """Mayor recall con FPR <= max_fpr. Si la puntuación es constante no hay umbral que ordenar: se usa 0,5."""
    proba = np.asarray(proba)
    if np.unique(proba).size == 1:
        return 0.5
    fpr, tpr, thr = roc_curve(y, proba)
    ok = np.where(fpr <= max_fpr)[0]
    return float(thr[ok[np.argmax(tpr[ok])]])


def macro_f1_diff_bootstrap(y, pa, pb, thr_a, thr_b, n=500, seed=0):
    """Diferencia de macro-F1 (A-B) remuestreando filas de validación: media e IC 95 %."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        d.append(f1_score(y[i], pa[i] >= thr_a, average="macro") - f1_score(y[i], pb[i] >= thr_b, average="macro"))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
