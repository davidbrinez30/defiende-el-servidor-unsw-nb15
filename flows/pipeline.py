"""Flow de Prefect OPCIONAL (no suma puntos ni hace falta para reproducir): orquesta las mismas etapas que dvc.yaml.

  pip install -r requirements-extra.txt
  python -m flows.pipeline
"""
import subprocess
import sys

from prefect import flow, get_run_logger, task


def _stage(*args):
    get_run_logger().info("python -m src.cli " + " ".join(args))
    subprocess.run([sys.executable, "-m", "src.cli", *args], check=True)


@task(name="1-preparar-particiones")
def preparar():
    _stage("prepare")


@task(name="2-auditar-datos")
def auditar():
    _stage("audit")


@task(name="3-experimentos-mlflow")
def experimentos():
    _stage("experiments")


@task(name="4-empaquetar-v1")
def v1():
    _stage("release", "v1")


@task(name="5-elegir-y-empaquetar-v2")
def v2():
    _stage("release", "v2")


@flow(name="defiende-el-servidor", log_prints=True)
def pipeline():
    preparar()
    auditar()
    experimentos()
    v1()
    v2()


if __name__ == "__main__":
    pipeline()
