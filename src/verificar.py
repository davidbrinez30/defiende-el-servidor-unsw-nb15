"""Comprueba las huellas (SHA-256) de lo recuperado: datos originales, datos preparados y modelos de v1/v2."""
import json

from src.config import EVIDENCIAS, PARAMS, PROCESSED, ROOT
from src.data import sha256_file
from src.release import RELEASES


def _estado(ok: bool | None) -> str:
    return "OK     " if ok else ("FALTA  " if ok is None else "DIFIERE")


def verificar() -> bool:
    """Imprime una línea por archivo. Devuelve True si todo lo presente coincide y no falta nada."""
    todo_ok, filas = True, []

    def chequear(nombre, ruta, esperado):
        nonlocal todo_ok
        if not ruta.exists():
            filas.append((None, nombre, "no existe (¿falta `dvc pull`?)"))
            todo_ok = False
            return
        got = sha256_file(ruta)
        ok = got == esperado
        todo_ok &= ok
        filas.append((ok, nombre, got if ok else f"{got} (esperado {esperado})"))

    for k, nombre in (("train", "UNSW_NB15_training-set.csv"), ("test", "UNSW_NB15_testing-set.csv")):
        chequear(f"original/{nombre}", ROOT / PARAMS["data"][f"{k}_csv"], PARAMS["data"][f"expected_{k}_sha256"])

    aud = EVIDENCIAS / "auditoria.json"
    if aud.exists():
        h = json.loads(aud.read_text(encoding="utf-8"))["particiones"]["huellas"]
        for nombre, clave in (("train.parquet", "train_parquet_sha256"), ("val.parquet", "val_parquet_sha256"),
                              ("split_assignment.csv", "split_assignment_csv_sha256")):
            chequear(f"preparado/{nombre}", PROCESSED / nombre, h[clave])
    for v in ("v1", "v2"):
        cfgp = RELEASES / v / "config.json"
        if cfgp.exists():
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            chequear(f"modelo/{v}/{cfg['modelo_archivo']}", RELEASES / v / cfg["modelo_archivo"], cfg["modelo_sha256"])
        else:
            filas.append((None, f"modelo/{v}/config.json", "no existe (¿falta `dvc pull`?)"))
            todo_ok = False
    for ok, nombre, detalle in filas:
        print(f"{_estado(ok)} {nombre:55s} {detalle}")
    print("RESULTADO:", "todas las huellas coinciden" if todo_ok else "hay diferencias o archivos faltantes")
    return todo_ok
