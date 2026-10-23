"""Prueba de recuperación en una carpeta y un entorno NUEVOS (sin reutilizar cachés de la ejecución original).

    python scripts/recuperar.py [--revision main] [--destino CARPETA_NUEVA]

Hace, y registra con los comandos reales, tiempos y salida:
  1. git clone del repositorio de código + checkout de la revisión elegida (y anotación del commit completo);
  2. entorno virtual nuevo e instalación de requirements.txt (dependencias fijadas);
  3. dvc pull desde el almacenamiento entregado (remote HTTP) con caché vacía: datos originales, preparados y modelos de v1/v2;
  4. verificación de huellas SHA-256 (python -m src.cli verificar);
  5. descarga desde Hugging Face de v1 y v2 por su SHA COMPLETO (evidencias/hf_publicacion.json) y predicción sobre las
     50 entradas fijas de validación (evidencias/recuperacion/entradas_validacion.csv);
  6. comparación con las salidas guardadas antes de recuperar (salida_antes_v1/v2.csv): y_pred idénticas y diferencia
     máxima de score <= tolerancia (1e-9), y coincidencia de hashes entre lo descargado de Hugging Face y lo recuperado con DVC.
El resultado se copia a evidencias/recuperacion/ del repositorio desde el que corres este script.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
TOLERANCIA = 1e-9
REG: list[dict] = []


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def limpio(texto: str) -> str:
    """Quita el nombre de usuario del sistema de las rutas que quedan en el registro."""
    return texto.replace(str(Path.home()), "~")


def paso(nombre: str, cmd: list[str], cwd: Path, env: dict, log: Path, mostrar: int = 12) -> str:
    texto = limpio(" ".join(str(c) for c in cmd))
    print(f"\n▶ {nombre}\n  $ {texto}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    salida = limpio((r.stdout or "") + (r.stderr or ""))
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {nombre} ===\n$ {texto}\n(cwd: {cwd.name}; {dt:.1f} s; código de salida {r.returncode})\n{salida}\n")
    REG.append({"paso": nombre, "comando": texto, "segundos": round(dt, 1), "codigo_salida": r.returncode})
    print("\n".join("  " + l for l in salida.strip().splitlines()[-mostrar:]), flush=True)
    if r.returncode != 0:
        sys.exit(f"FALLÓ el paso «{nombre}» (código {r.returncode}). Revisa {log}.")
    return salida


def main():
    for flujo in (sys.stdout, sys.stderr):  # consolas de Windows con codificación antigua
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default="main", help="rama, tag o commit del repositorio de código a recuperar")
    ap.add_argument("--destino", default=None, help="carpeta NUEVA donde hacer la prueba (no debe existir)")
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--dvc-remote-url", default=None, help="solo pruebas: reemplaza la URL del remote HTTP")
    ap.add_argument("--hf-simulado", default=None, help="solo pruebas sin red: carpeta con v1/ y v2/ en lugar de Hugging Face")
    ap.add_argument("--no-copiar", action="store_true")
    a = ap.parse_args()

    if sys.version_info < (3, 12):
        sys.exit(f"Se necesita Python 3.12 o 3.13 (tienes {platform.python_version()}).")
    eq = yaml.safe_load((ROOT / "configs" / "equipo.yaml").read_text(encoding="utf-8"))
    url = a.repo_url or f"https://github.com/{eq['github']['usuario']}/{eq['github']['repo_codigo']}.git"
    destino = Path(a.destino or Path.home() / f"recuperacion-defiende-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if destino.exists():
        sys.exit(f"{destino} ya existe: la prueba exige una carpeta nueva.")
    destino.mkdir(parents=True)
    log = destino / "registro.log"
    t_total = time.time()

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update({"DVC_NO_ANALYTICS": "1", "HF_HOME": str(destino / "hf-cache"), "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"Prueba de recuperación · {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n"
                f"Sistema: {platform.platform()} · Python {platform.python_version()} · CPU {os.cpu_count()}\nDestino: {limpio(str(destino))}\n")

    # 1) código
    paso("1. Clonar el repositorio", ["git", "clone", url, "proyecto"], destino, env, log)
    proy = destino / "proyecto"
    paso("1b. Obtener la revisión elegida", ["git", "checkout", a.revision], proy, env, log)
    commit = paso("1c. Commit completo recuperado", ["git", "rev-parse", "HEAD"], proy, env, log).strip().splitlines()[-1]

    # 2) entorno nuevo
    paso("2. Crear entorno virtual nuevo", [sys.executable, "-m", "venv", str(destino / "venv")], proy, env, log)
    py = destino / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    t_pip = time.time()
    paso("2b. Instalar dependencias fijadas", [str(py), "-m", "pip", "install", "-r", "requirements.txt"], proy, env, log, mostrar=3)
    seg_pip = round(time.time() - t_pip, 1)
    paso("2c. Versiones instaladas", [str(py), "-m", "pip", "freeze"], proy, env, log, mostrar=0)

    # 3) datos y modelos con DVC (caché vacía: el clon es nuevo)
    if a.dvc_remote_url:
        paso("3a. (prueba) URL alternativa del remote", [str(py), "-m", "dvc", "remote", "modify", "--local", "almacen_http", "url", a.dvc_remote_url], proy, env, log)
    t_dvc = time.time()
    paso("3. Recuperar con DVC datos originales, preparados y modelos",
         [str(py), "-m", "dvc", "pull", "data/raw/UNSW_NB15_training-set.csv.dvc", "data/raw/UNSW_NB15_testing-set.csv.dvc",
          "prepare", "release_v1", "release_v2"], proy, env, log, mostrar=4)
    seg_dvc = round(time.time() - t_dvc, 1)
    verif = paso("4. Verificar huellas SHA-256", [str(py), "-m", "src.cli", "verificar"], proy, env, log, mostrar=20)

    # 5) Hugging Face por SHA completo
    pub = json.loads((proy / "evidencias" / "hf_publicacion.json").read_text(encoding="utf-8"))
    rec = proy / "evidencias" / "recuperacion"
    entradas = rec / "entradas_validacion.csv"
    comparacion, hf_sha = {}, {}
    for v in ("v1", "v2"):
        sha = pub[v]["commit_sha"]
        assert len(sha) == 40, f"{v}: se requiere el SHA completo"
        carpeta = destino / "hf" / v
        if a.hf_simulado:
            shutil.copytree(Path(a.hf_simulado) / v, carpeta)
            print(f"\n▶ 5. ({v}) Hugging Face SIMULADO desde {a.hf_simulado}")
        else:
            paso(f"5. Descargar {v} de Hugging Face por SHA {sha}",
                 [str(py), "-c", "from src.predict import descargar; print(descargar(%r, %r))" % (sha, str(carpeta))], proy, env, log, mostrar=3)
        paso(f"5b. Predecir con {v} sobre las entradas fijas",
             [str(py), "-m", "src.predict", "--release", str(carpeta), "--input", str(entradas),
              "--output", str(rec / f"salida_despues_{v}.csv")], proy, env, log, mostrar=1)

        # 6) comparación con lo guardado antes y coherencia de hashes
        antes = pd.read_csv(rec / f"salida_antes_{v}.csv")
        despues = pd.read_csv(rec / f"salida_despues_{v}.csv")
        misma_fila = bool((antes["row_id"] == despues["row_id"]).all())
        dif = float((antes["score_attack"] - despues["score_attack"]).abs().max())
        cfg_hf = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        cfg_dvc = json.loads((proy / "models" / "releases" / v / "config.json").read_text(encoding="utf-8"))
        hf_modelo = sha256(carpeta / "modelo.skops")
        dvc_modelo = sha256(proy / "models" / "releases" / v / "modelo.skops")
        hf_sha[v] = hf_modelo
        comparacion[v] = {
            "revision_hf_sha": sha, "filas": int(len(antes)), "mismas_row_id_en_mismo_orden": misma_fila,
            "y_pred_identicas": bool((antes["y_pred"] == despues["y_pred"]).all()),
            "diferencia_maxima_score": dif, "tolerancia": TOLERANCIA, "dentro_de_tolerancia": dif <= TOLERANCIA,
            "sha256_modelo_hf": hf_modelo, "sha256_modelo_dvc": dvc_modelo, "sha256_modelo_config": cfg_hf["modelo_sha256"],
            "hashes_coinciden": hf_modelo == dvc_modelo == cfg_hf["modelo_sha256"] == cfg_dvc["modelo_sha256"],
            "umbral": cfg_hf["umbral"], "alertas_predichas": int(despues["y_pred"].sum()),
        }
    v1, v2 = (pd.read_csv(rec / f"salida_despues_{v}.csv") for v in ("v1", "v2"))
    ok = all(c["mismas_row_id_en_mismo_orden"] and c["y_pred_identicas"] and c["dentro_de_tolerancia"] and c["hashes_coinciden"]
             for c in comparacion.values()) and "RESULTADO: todas las huellas coinciden" in verif

    resultado = {
        "fecha_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resultado": "APROBADA" if ok else "FALLÓ",
        "repositorio": url, "revision_pedida": a.revision, "commit_recuperado": commit,
        "entorno_nuevo": {"carpeta": "carpeta nueva creada para la prueba", "venv_nuevo": True, "cache_dvc_vacia": True,
                          "hf_cache_nueva": True, "python": platform.python_version(), "sistema": platform.platform(),
                          "cpus": os.cpu_count()},
        "tiempos_segundos": {"instalar_dependencias": seg_pip, "dvc_pull": seg_dvc, "total": round(time.time() - t_total, 1)},
        "huellas_verificadas": "RESULTADO: todas las huellas coinciden" in verif,
        "comparacion": comparacion,
        "v1_vs_v2": {"filas_con_distinta_prediccion": int((v1["y_pred"] != v2["y_pred"]).sum()),
                     "nota": "No es obligatorio que difieran; se informa para mostrar que son versiones distintas."},
        "entradas": {"archivo": "evidencias/recuperacion/entradas_validacion.csv", "filas": int(len(v1)), "sha256": sha256(entradas)},
        "pasos": REG,
        "hf_simulado": bool(a.hf_simulado),
    }
    (rec / "comparacion.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy(log, rec / "registro.log")
    archivos = sorted(p for p in rec.iterdir() if p.is_file() and p.name != "hashes.json")
    (rec / "hashes.json").write_text(json.dumps({p.name: sha256(p) for p in archivos}, indent=2) + "\n", encoding="utf-8")
    (rec / "comandos.md").write_text("# Comandos ejecutados en la prueba de recuperación\n\n```\n" + "\n".join(
        f"[{r['segundos']:>7} s] {r['comando']}" for r in REG) + "\n```\n", encoding="utf-8")

    if not a.no_copiar:
        destino_ev = ROOT / "evidencias" / "recuperacion"
        destino_ev.mkdir(parents=True, exist_ok=True)
        for p in rec.iterdir():
            if p.is_file():
                shutil.copy(p, destino_ev / p.name)
        print(f"\nEvidencia copiada a {destino_ev}")
    print(f"\n{'✔' if ok else '✘'} PRUEBA DE RECUPERACIÓN {resultado['resultado']} — commit {commit}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
