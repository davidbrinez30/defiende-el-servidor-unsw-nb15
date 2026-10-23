"""Evalúa el criterio medible de cada hipótesis (configs/experimentos.yaml) con las métricas de VALIDACIÓN."""
import pandas as pd

NOMBRES = {"recall": "recall", "precision": "precision", "fpr": "FPR", "macro_f1": "macro-F1", "accuracy": "accuracy",
           "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "fit_seconds": "segundos de entrenamiento"}


def _col(metrica: str, umbral: str) -> str:
    if umbral == "0.5":
        return {"recall": "recall_t05", "fpr": "fpr_t05", "precision": "precision_t05", "macro_f1": "macro_f1_t05"}[metrica]
    return metrica


def _num(v, metrica):
    return f"{v:.1f}" if metrica == "fit_seconds" else f"{v:.4f}".replace(".", ",")


def evaluar(spec: dict, tabla: pd.DataFrame) -> tuple[bool | None, str]:
    """Devuelve (confirmada, texto). Sin criterio -> (None, '')."""
    criterios = spec.get("criterio") or []
    if not criterios:
        return None, ""
    fila = tabla.set_index("experimento")
    partes, ok = [], True
    for c in criterios:
        col = _col(c["metrica"], str(c.get("umbral", "regla")))
        a, b = float(fila.loc[spec["id"], col]), float(fila.loc[c["vs"], col])
        m = float(c.get("margen", 0.0))
        op = c["op"]
        if op == "mayor":
            cumple, rel = a > b + m, f">{' ' + _num(m, c['metrica']) + ' sobre' if m else ''}"
        elif op == "menor":
            cumple, rel = a < b - m, f"<{' ' + _num(m, c['metrica']) + ' bajo' if m else ''}"
        elif op == "no_cae_mas_de":
            cumple, rel = a >= b - m, f"no cae más de {_num(m, c['metrica'])} respecto a"
        elif op == "dif_abs_menor":
            cumple, rel = abs(a - b) < m, f"difiere menos de {_num(m, c['metrica'])} de"
        elif op == "razon_mayor":
            cumple, rel = a > 0 and b / a > m, f"se reduce más de {m:g} veces respecto a"
        else:
            raise ValueError(f"Operador desconocido: {op}")
        ok &= bool(cumple)
        sufijo = " a umbral 0,5" if str(c.get("umbral")) == "0.5" else ""
        partes.append(f"{NOMBRES[c['metrica']]}{sufijo}: {_num(a, c['metrica'])} {rel} {c['vs'].split('_')[0]} "
                      f"({_num(b, c['metrica'])}) -> {'cumple' if cumple else 'no cumple'}")
    return ok, ("Hipótesis confirmada. " if ok else "Hipótesis NO confirmada. ") + "; ".join(partes) + "."
