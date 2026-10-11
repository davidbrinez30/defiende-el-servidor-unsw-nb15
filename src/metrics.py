"""Métricas que importan para detección: recall, precisión, tasa de falsos positivos y macro-F1."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def at_threshold(y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "accuracy": float((pred == y).mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_free(y, proba) -> dict:
    return {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba)}


def pick_threshold(y, proba, max_fpr: float) -> float:
    """Mayor recall con FPR <= max_fpr (regla declarada en params.yaml)."""
    fpr, tpr, thr = roc_curve(y, proba)
    ok = np.where(fpr <= max_fpr)[0]
    # Entre los que cumplen el tope de FPR, el primero con recall máximo (menor FPR en empates).
    best = ok[np.argmax(tpr[ok])]
    return float(thr[best])
