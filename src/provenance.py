"""Trazabilidad: commit de Git, referencias DVC y huellas de los archivos usados."""
import re
import subprocess
from pathlib import Path

from src.config import PROCESSED, ROOT
from src.data import sha256_file

CODE_PATHS = ["src", "configs", "params.yaml", "dvc.yaml", "scripts"]


def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def git_info() -> dict:
    """Commit completo del código y si hay cambios sin confirmar en el código/configuración (no en resultados)."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--", *CODE_PATHS))
        return {"git_commit": commit, "git_commit_corto": commit[:7], "codigo_sin_confirmar": dirty}
    except Exception:
        return {"git_commit": "sin-git", "git_commit_corto": "sin-git", "codigo_sin_confirmar": True}


def dvc_file_md5(dvc_path: Path) -> str | None:
    """md5 que DVC guarda en un archivo .dvc de seguimiento."""
    if not dvc_path.exists():
        return None
    m = re.search(r"md5:\s*([0-9a-f]{32})", dvc_path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def data_refs() -> dict:
    """Versiones de datos en uso: originales (md5 de DVC) y preparados (sha256 calculado ahora)."""
    raw = ROOT / "data" / "raw"
    refs = {
        "dvc_original_train_md5": dvc_file_md5(raw / "UNSW_NB15_training-set.csv.dvc"),
        "dvc_original_test_md5": dvc_file_md5(raw / "UNSW_NB15_testing-set.csv.dvc"),
    }
    for name in ("train.parquet", "val.parquet", "split_assignment.csv"):
        p = PROCESSED / name
        refs[f"preparado_{name}_sha256"] = sha256_file(p) if p.exists() else None
    return refs


def dvc_lock_outs(stage: str) -> dict:
    """md5 de las salidas de una etapa según dvc.lock (existe tras haberla ejecutado)."""
    import yaml
    lock = ROOT / "dvc.lock"
    if not lock.exists():
        return {}
    st = yaml.safe_load(lock.read_text(encoding="utf-8")).get("stages", {}).get(stage, {})
    return {o["path"]: o.get("md5") for o in st.get("outs", [])}
