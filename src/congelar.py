"""Congela la selección de v2 ANTES de evaluar el test: escribe evidencias/seleccion.json.

Reúne la decisión tomada con validación (evidencias/seleccion_validacion.json), las referencias cruzadas
Git / DVC / MLflow / Hugging Face y el registro fechado de la congelación. Después se confirma en Git con el tag
`congelado`; evaluate.py comprueba que ese archivo no cambió desde el tag.
"""
import json
import subprocess
from datetime import datetime, timezone

from src.config import EVIDENCIAS, ROOT, load_yaml
from src.data import sha256_file
from src.release import RELEASES

SEL = EVIDENCIAS / "seleccion.json"
HF = EVIDENCIAS / "hf_publicacion.json"


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def congelar(exigir_hf: bool = True) -> dict:
    dec = json.loads((EVIDENCIAS / "seleccion_validacion.json").read_text(encoding="utf-8"))
    cfg = {v: json.loads((RELEASES / v / "config.json").read_text(encoding="utf-8")) for v in ("v1", "v2")}
    eq = load_yaml("equipo.yaml")
    hf_cfg = eq["huggingface"]
    repo_id = f"{hf_cfg['usuario']}/{hf_cfg['repo_modelos']}"

    hf = None
    if HF.exists():
        hf = json.loads(HF.read_text(encoding="utf-8"))
        for v in ("v1", "v2"):
            if hf[v]["sha256"]["modelo.skops"] != cfg[v]["modelo_sha256"]:
                raise SystemExit(f"El modelo publicado en Hugging Face para {v} no coincide con el local.")
            if hf[v]["sha256"]["config.json"] != sha256_file(RELEASES / v / "config.json"):
                raise SystemExit(f"config.json publicado para {v} no coincide con el local.")
    elif exigir_hf:
        raise SystemExit("Falta evidencias/hf_publicacion.json: publica v1/v2 en Hugging Face antes de congelar.")

    c2 = cfg["v2"]
    el = dec["elegido"]
    sel = {
        "version_elegida": "v2",
        "experimento": c2["experimento"],
        "mlflow_run_id": c2["mlflow"]["run_id"],
        "umbral": c2["umbral"],
        "regla_umbral": c2["regla_umbral"],
        "modelo_sha256": c2["modelo_sha256"],
        "modelo_joblib_del_run_sha256": c2["modelo_joblib_del_run_sha256"],
        "razon_de_seleccion": {
            "criterio": dec["criterio"],
            "macro_f1_validacion": el["macro_f1"], "recall_validacion": el["recall"], "fpr_validacion": el["fpr"],
            "segundo_mejor": dec["segundo_mejor"],
            "diferencia_macro_f1_vs_segundo": dec["diferencia_macro_f1_vs_segundo"],
            "cambio_respecto_a_v1": dec["cambio_respecto_a_v1"]["descripcion"],
        },
        "decision_completa": {"archivo": "evidencias/seleccion_validacion.json",
                              "sha256": sha256_file(EVIDENCIAS / "seleccion_validacion.json")},
        "referencias": {
            "git": {"repositorio": c2["git"]["repositorio"], "tag_v1": "v1", "tag_v2": "v2",
                    "commit_tag_v1": _git("rev-list", "-n", "1", "v1"), "commit_tag_v2": _git("rev-list", "-n", "1", "v2"),
                    "commit_codigo_que_entreno": c2["git"]["commit_codigo_que_entreno"]},
            "dvc": {**c2["dvc"], "almacenamiento": "ver entrega.md (remote HTTP de solo lectura en GitHub)"},
            "mlflow": {"experimento": c2["mlflow"]["experimento"], "run_id": c2["mlflow"]["run_id"],
                       "export": "evidencias/mlflow/runs/" + c2["experimento"] + "/run.json"},
            "huggingface": {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                            "tag": "v2", "commit_sha": hf["v2"]["commit_sha"] if hf else None,
                            "sha256_archivos": hf["v2"]["sha256"] if hf else None,
                            "v1": {"tag": "v1", "commit_sha": hf["v1"]["commit_sha"] if hf else None,
                                   "modelo_sha256": cfg["v1"]["modelo_sha256"], "umbral": cfg["v1"]["umbral"],
                                   "experimento": cfg["v1"]["experimento"], "mlflow_run_id": cfg["v1"]["mlflow"]["run_id"]}},
        },
        "congelacion": {
            "fecha_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tag_git": "congelado",
            "test_evaluado_antes_de_congelar": False,
            "como_se_verifica": "El commit con tag `congelado` contiene este archivo; `git log --follow evidencias/seleccion.json` "
                                "y la fecha de `evidencias/predicciones-test.csv` (commit posterior) lo ordenan. "
                                "evaluate.py rechaza correr si este archivo cambió desde el tag.",
            "regla": "No se ajusta modelo ni umbral después de ver el test; si el resultado empeora, se explica y se conserva.",
        },
    }
    SEL.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selección congelada: {sel['version_elegida']} = {sel['experimento']} (umbral {sel['umbral']:.6f}) "
          f"modelo {sel['modelo_sha256'][:12]}…")
    return sel
