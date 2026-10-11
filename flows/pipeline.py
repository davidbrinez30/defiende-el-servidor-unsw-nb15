"""Pipeline coordinado con Prefect: preparación -> experimentos -> selección -> evaluación en test.

Cada etapa es una tarea con estado propio (Completed / Failed) y reintentos desactivados a propósito:
si algo falla (CSV ausente, filas que no coinciden, fuga entre particiones...) el flow lo muestra con su motivo.

  python -m flows.pipeline
"""
import json

from prefect import flow, get_run_logger, task

from src import data, evaluate, experiments, selection
from src.config import REPORTS


@task(name="1-validar-datos-crudos")
def validar_datos():
    log = get_run_logger()
    tr, te = data.load_raw("train"), data.load_raw("test")  # lanza error claro si algo no cuadra
    log.info(f"train={len(tr)} filas, test={len(te)} filas; SHA-256 de train verificado")
    return {"train": len(tr), "test": len(te)}


@task(name="2-preparar-particiones")
def preparar():
    rep = data.prepare()
    get_run_logger().info(f"train={rep['train_rows']} val={rep['val_rows']} patrones compartidos={rep['patrones_compartidos_train_val']}")
    return rep


@task(name="3-experimentos-mlflow")
def experimentos():
    tabla = experiments.run_all()
    get_run_logger().info("\n" + tabla[["experiment", "macro_f1", "recall", "fpr", "roc_auc"]].round(4).to_string(index=False))
    return len(tabla)


@task(name="4-elegir-modelo-y-umbral")
def elegir():
    d = selection.select_and_freeze()
    get_run_logger().info(f"Elegido {d['selected_experiment']} con umbral {d['threshold']:.4f} (versión MLflow {d['mlflow_model_version']})")
    return d


@task(name="5-evaluar-test-una-vez")
def evaluar(evaluar_test: bool):
    log = get_run_logger()
    out = REPORTS / "test_metrics.json"
    if out.exists():
        log.warning("test_metrics.json ya existe: el test se evalúa una sola vez; se conserva ese resultado.")
        return json.loads(out.read_text())
    if not evaluar_test:
        log.info("Evaluación en test omitida (evaluar_test=False).")
        return None
    r = evaluate.evaluate_test_once()
    log.info(f"TEST macro-F1={r['metrics']['macro_f1']:.4f} recall={r['metrics']['recall']:.4f} FPR={r['metrics']['fpr']:.4f}")
    return r


@flow(name="defiende-el-servidor", log_prints=True)
def pipeline(evaluar_test: bool = True):
    v = validar_datos()
    p = preparar(wait_for=[v])
    e = experimentos(wait_for=[p])
    s = elegir(wait_for=[e])
    return evaluar(evaluar_test, wait_for=[s])


if __name__ == "__main__":
    pipeline()
