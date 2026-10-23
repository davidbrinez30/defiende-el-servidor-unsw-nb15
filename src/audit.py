"""Auditoría de datos -> evidencias/auditoria.json.

Describe fuentes, huellas, filas/columnas, clases, faltantes, duplicados por entradas, contradicciones, política de
limpieza, variables usadas/excluidas, particiones y cruces. El test solo se AUDITA (esquema, duplicados, cruces):
ninguna cifra de este archivo se usa para elegir modelos ni umbrales.
"""
import hashlib
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import EVIDENCIAS, PARAMS, PROCESSED, ROOT
from src.data import D, ID_COL, TARGET, feature_columns, load_raw, pattern_hash, raw_path, sha256_file


def _archivo(which: str, df: pd.DataFrame) -> dict:
    path = raw_path(which)
    clases = df[TARGET].value_counts().sort_index()
    faltantes = df.isna().sum()
    return {
        "archivo": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "nombres_columnas": list(df.columns),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "clases_label": {"0_normal": int(clases.get(0, 0)), "1_ataque": int(clases.get(1, 0))},
        "proporcion_ataque": round(float(df[TARGET].mean()), 6),
        "attack_cat": {k: int(v) for k, v in df["attack_cat"].value_counts().items()},
        "faltantes_total": int(faltantes.sum()),
        "faltantes_por_columna": {c: int(v) for c, v in faltantes.items() if v},
        "texto_guion_en_service": int((df["service"] == "-").sum()),
        "nota_faltantes": "No hay celdas vacías (NaN). El valor «-» de `service` es una categoría con significado "
                          "(servicio no identificado), no un dato ausente.",
        "id_unico": bool(df[ID_COL].is_unique), "id_min": int(df[ID_COL].min()), "id_max": int(df[ID_COL].max()),
    }


def _duplicados(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    h = pattern_hash(df)
    conteo = h.map(h.value_counts())
    por_grupo = df.groupby(h.values)[TARGET].agg(["nunique", "size", "mean"])
    contra = por_grupo[por_grupo["nunique"] > 1]
    rep_extra = int(len(df) - h.nunique())
    dup = {
        "filas": int(len(df)),
        "patrones_distintos": int(h.nunique()),
        "filas_repetidas_adicionales": rep_extra,
        "porcentaje_repetidas_adicionales": round(100 * rep_extra / len(df), 2),
        "filas_en_grupos_con_repeticion": int((conteo > 1).sum()),
        "grupos_con_repeticion": int((por_grupo["size"] > 1).sum()),
        "tamano_maximo_de_grupo": int(por_grupo["size"].max()),
    }
    cont = {
        "grupos_con_etiquetas_contradictorias": int(len(contra)),
        "filas_en_esos_grupos": int(contra["size"].sum()),
        "porcentaje_de_filas": round(100 * contra["size"].sum() / len(df), 3),
    }
    return {"duplicados": dup, "contradicciones": cont}, h


def build_audit() -> dict:
    tr, te = load_raw("train"), load_raw("test")
    a_tr, a_te = _archivo("train", tr), _archivo("test", te)
    d_tr, h_tr = _duplicados(tr)
    d_te, h_te = _duplicados(te)

    asign = pd.read_csv(PROCESSED / "split_assignment.csv")
    val_mask = asign["particion"].to_numpy() == "validacion"
    h_val, h_ent = set(h_tr[val_mask]), set(h_tr[~val_mask])
    ids_hash = lambda ids: hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()

    # id: ¿significa algo para el problema o solo refleja cómo se organizó el archivo?
    auc_id = roc_auc_score(tr[TARGET], tr[ID_COL])
    deciles = tr.groupby(pd.qcut(tr[ID_COL], 10, labels=range(1, 11)), observed=True)[TARGET].mean()

    set_tr, set_te = set(h_tr), set(h_te)
    test_en_train = h_te.isin(h_ent)
    test_en_val = h_te.isin(h_val)
    test_en_cualquiera = test_en_train | test_en_val

    fuentes = json.loads((ROOT / "docs" / "sources.json").read_text(encoding="utf-8"))
    out = {
        "fuentes": fuentes,
        "archivos": {"entrenamiento": {**a_tr, **d_tr}, "test": {**a_te, **d_te}},
        "verificacion_de_particiones": {
            "entrenamiento_esperado": {"filas": D["expected_train_rows"], "sha256": D["expected_train_sha256"]},
            "test_esperado": {"filas": D["expected_test_rows"], "sha256": D["expected_test_sha256"]},
            "coincide": bool(a_tr["filas"] == D["expected_train_rows"] and a_te["filas"] == D["expected_test_rows"]
                             and a_tr["sha256"] == D["expected_train_sha256"] and a_te["sha256"] == D["expected_test_sha256"]),
            "nota": "El nombre del archivo no garantiza la partición correcta (hay copias con training/testing intercambiados): "
                    "se comprueban filas y SHA-256. Los SHA-256 coinciden con los del anexo del enunciado; el origen es un mirror "
                    "de terceros, no está autenticado contra el archivo oficial de UNSW.",
        },
        "variables": {
            "usadas_como_entrada": feature_columns(tr),
            "cantidad_usadas": len(feature_columns(tr)),
            "categoricas": D["categorical"],
            "numericas": [c for c in feature_columns(tr) if c not in D["categorical"]],
            "excluidas": {
                "label": "Respuesta que se quiere predecir (0 = normal, 1 = ataque).",
                "attack_cat": "Familia del ataque: revela la respuesta (Normal = label 0), sería fuga de información.",
                "id": "Identifica filas, no describe la conexión. Evidencia: ver `analisis_id`.",
            },
            "analisis_id": {
                "auc_id_vs_label": round(float(max(auc_id, 1 - auc_id)), 4),
                "tasa_ataque_por_decil_de_id_pct": {int(k): round(100 * float(v), 1) for k, v in deciles.items()},
                "conclusion": "El archivo viene ordenado y `id` separa las clases casi perfecto (AUC ~0,95) sin describir tráfico: "
                              "es un artefacto de cómo se organizó el archivo. Se excluye de las entradas y se conserva solo "
                              "para auditoría y para relacionar predicciones (row_id).",
            },
        },
        "politica_de_limpieza": {
            "descripcion": "No se eliminan filas, no se imputan valores ni se recortan outliers. Las filas repetidas se conservan "
                           "pero se mantienen juntas en una misma partición (grupo = vector de entrada idéntico). Los grupos con "
                           "etiquetas contradictorias se conservan y se reconocen como error irreducible con estas variables. "
                           "El test no se filtra ni se modifica.",
            "efecto_en_la_poblacion": "La población estudiada es la publicada por UNSW en sus dos particiones; no se cambia.",
            "categorias_raras": "Dentro del Pipeline, las categorías con menos de 100 filas de entrenamiento y las categorías "
                                "desconocidas se agrupan en «infrecuente» (OneHotEncoder, ajustado solo con entrenamiento).",
        },
        "particiones": {
            "metodo": "StratifiedGroupKFold (n_splits=5, shuffle=True), se toma el primer pliegue como validación",
            "semilla": PARAMS["seed"],
            "grupo": "huella (hash) del vector de las 42 variables de entrada",
            "entrenamiento": {"filas": int((~val_mask).sum()), "tasa_ataque": round(float(tr.loc[~val_mask, TARGET].mean()), 4)},
            "validacion": {"filas": int(val_mask.sum()), "tasa_ataque": round(float(tr.loc[val_mask, TARGET].mean()), 4)},
            "patrones_compartidos_entrenamiento_validacion": int(len(h_val & h_ent)),
            "huellas": {
                "split_assignment_csv_sha256": sha256_file(PROCESSED / "split_assignment.csv"),
                "ids_entrenamiento_sha256": ids_hash(asign.loc[~val_mask, "id"]),
                "ids_validacion_sha256": ids_hash(asign.loc[val_mask, "id"]),
                "train_parquet_sha256": sha256_file(PROCESSED / "train.parquet"),
                "val_parquet_sha256": sha256_file(PROCESSED / "val.parquet"),
            },
            "archivo_de_asignacion": "data/processed/split_assignment.csv (fila_archivo, id, particion, grupo_patron)",
        },
        "cruces_test": {
            "uso": "Solo auditoría de entradas (no se usan etiquetas ni métricas del test para decidir nada).",
            "filas_test": int(len(te)),
            "filas_test_con_entradas_iguales_a_entrenamiento": int(test_en_train.sum()),
            "filas_test_con_entradas_iguales_a_validacion": int(test_en_val.sum()),
            "filas_test_con_entradas_vistas_en_entrenamiento_o_validacion": int(test_en_cualquiera.sum()),
            "porcentaje_test_con_entradas_vistas": round(100 * float(test_en_cualquiera.mean()), 3),
            "patrones_distintos_test": int(len(set_te)),
            "patrones_test_tambien_en_entrenamiento": int(len(set_te & set_tr)),
            "ids_en_comun_entrenamiento_y_test": int(len(set(tr[ID_COL]) & set(te[ID_COL]))),
            "nota_ids": "Los ids de ambos archivos empiezan en 1: coinciden numéricamente pero no identifican la misma conexión; "
                        "por eso row_id incluye el nombre del archivo.",
            "implicacion": f"El {100 - 100 * float(test_en_cualquiera.mean()):.1f} % de las filas del test tiene entradas que no aparecen en "
                           "entrenamiento ni validación; la validación (partición aleatoria agrupada del mismo archivo) no reproduce "
                           "ese cambio, así que puede resultar más optimista que el test.",
        },
    }
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    (EVIDENCIAS / "auditoria.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
