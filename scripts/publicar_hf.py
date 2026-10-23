"""Publica v1 y v2 en Hugging Face Hub como DOS revisiones (commits) con tags v1 y v2, y comprueba la descarga por SHA.

Antes (una sola vez), crea un token de tipo *Write* en https://huggingface.co/settings/tokens y guárdalo SOLO en tu equipo:

    PowerShell:   $env:HF_TOKEN = Read-Host "Token de Hugging Face"      # se escribe en la terminal, no en un archivo
    Linux/macOS:  read -s HF_TOKEN && export HF_TOKEN

Uso (desde la raíz del repositorio, con las dependencias de requirements.txt instaladas y `dvc pull` hecho):

    python scripts/publicar_hf.py            # v1 y v2: dos commits + tags; escribe evidencias/hf_publicacion.json
    python scripts/publicar_hf.py --final    # después de evaluar en test: actualiza SOLO la model card de main

El token nunca se escribe en ningún archivo ni en la salida. Es idempotente: si los tags ya existen, los verifica y no vuelve a subir.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

EVID = ROOT / "evidencias" / "hf_publicacion.json"
ARCHIVOS_MODELO = ["modelo.skops", "config.json", "requirements.txt"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def obtener_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if not token:
        sys.exit("No hay token de Hugging Face. Define la variable de entorno HF_TOKEN (ver el encabezado de este archivo).")
    return token


def archivos_de(version: str) -> dict[str, Path]:
    rel = ROOT / "models" / "releases" / version
    f = {n: rel / n for n in ARCHIVOS_MODELO}
    f["README.md"] = ROOT / "docs" / "model_cards" / f"{version}.md"
    faltan = [str(p) for p in f.values() if not p.exists()]
    if faltan:
        sys.exit(f"Faltan archivos de {version}: {faltan}. Ejecuta `dvc pull` y `python -m src.cli modelcards {version}`.")
    return f


def verificar_descarga(api: HfApi, repo_id: str, revision: str, locales: dict[str, Path], token: str) -> dict:
    """Descarga cada archivo por la revisión indicada y compara su SHA-256 con el local."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, ruta in locales.items():
            bajado = hf_hub_download(repo_id, nombre, revision=revision, token=token, local_dir=tmp, force_download=True)
            got, esperado = sha256(bajado), sha256(ruta)
            if got != esperado:
                sys.exit(f"Descarga de {nombre} en la revisión {revision} no coincide con el archivo local.")
            out[nombre] = got
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="actualizar la model card de v2 en main (tras evaluar en test)")
    a = ap.parse_args()

    eq = yaml.safe_load((ROOT / "configs" / "equipo.yaml").read_text(encoding="utf-8"))
    repo_id = f"{eq['huggingface']['usuario']}/{eq['huggingface']['repo_modelos']}"
    token = obtener_token()
    api = HfApi(token=token)
    quien = api.whoami()["name"]
    if quien != eq["huggingface"]["usuario"]:
        sys.exit(f"El token pertenece a «{quien}», pero configs/equipo.yaml espera «{eq['huggingface']['usuario']}».")
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"Repositorio: https://huggingface.co/{repo_id}")

    pub = json.loads(EVID.read_text(encoding="utf-8")) if EVID.exists() else {}
    pub.update({"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"})

    if not a.final:
        refs = {t.name: t.target_commit for t in api.list_repo_refs(repo_id).tags}
        padre = None
        for v in ("v1", "v2"):
            locales = archivos_de(v)
            cfg = json.loads(locales["config.json"].read_text(encoding="utf-8"))
            if v in refs:
                sha_commit = refs[v]
                print(f"{v}: el tag ya existe ({sha_commit}); se verifica sin volver a subir.")
            else:
                ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in locales.items()]
                info = api.create_commit(
                    repo_id, operations=ops, parent_commit=padre,
                    commit_message=(f"{v}: {cfg['descripcion']} Experimento {cfg['experimento']}, run MLflow {cfg['mlflow']['run_id']}, "
                                    f"umbral {cfg['umbral']:.6f}, modelo sha256 {cfg['modelo_sha256'][:12]}"))
                sha_commit = info.oid
                api.create_tag(repo_id, tag=v, revision=sha_commit, tag_message=f"Versión {v} del modelo")
                print(f"{v}: commit {sha_commit}, tag {v} creado.")
            padre = sha_commit
            hashes = verificar_descarga(api, repo_id, sha_commit, locales, token)
            if hashes["modelo.skops"] != cfg["modelo_sha256"]:
                sys.exit(f"{v}: el modelo publicado no coincide con el hash de config.json.")
            pub[v] = {"commit_sha": sha_commit, "tag": v, "experimento": cfg["experimento"], "mlflow_run_id": cfg["mlflow"]["run_id"],
                      "umbral": cfg["umbral"], "sha256": hashes,
                      "descarga_verificada": "cada archivo descargado por el SHA de la revisión coincide con el local"}
        EVID.write_text(json.dumps(pub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Escrito {EVID.relative_to(ROOT)}. Súbelo al repositorio de código (git add/commit/push).")
    else:
        if "v2" not in pub:
            sys.exit("Primero publica v1 y v2 (sin --final).")
        locales = archivos_de("v2")
        actual = api.list_repo_refs(repo_id)
        main_antes = next(b.target_commit for b in actual.branches if b.name == "main")
        info = api.create_commit(
            repo_id, operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(locales["README.md"]))],
            parent_commit=main_antes,
            commit_message="Model card de v2: agrega la evaluación final en test. El modelo no cambia (mismo sha256 de modelo.skops).")
        hashes = verificar_descarga(api, repo_id, info.oid, locales, token)
        if hashes["modelo.skops"] != pub["v2"]["sha256"]["modelo.skops"]:
            sys.exit("El modelo de main no coincide con la revisión v2 evaluada.")
        pub["main_con_evaluacion_final"] = {"commit_sha": info.oid, "revision_evaluada_v2": pub["v2"]["commit_sha"],
                                           "sha256": hashes, "nota": "Solo cambió README.md; modelo.skops, config.json y requirements.txt son idénticos a v2."}
        EVID.write_text(json.dumps(pub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"main actualizado: {info.oid}")


if __name__ == "__main__":
    main()
