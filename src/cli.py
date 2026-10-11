"""Punto de entrada por etapa (lo usan dvc.yaml y el flow de Prefect): python -m src.cli <etapa>"""
import json
import sys

from src import data, evaluate, experiments, selection


def main(stage: str):
    if stage == "prepare":
        print(json.dumps(data.prepare(), indent=1, ensure_ascii=False))
    elif stage == "experiments":
        experiments.run_all()
    elif stage == "select":
        selection.select_and_freeze()
    elif stage == "evaluate":
        # Dentro de `dvc repro` los outputs se borran antes de correr la etapa: reproduce el MISMO resultado
        # congelado. Fuera de DVC, el guard de evaluate.py impide una segunda evaluación accidental.
        evaluate.evaluate_test_once(force=True)
    else:
        raise SystemExit(f"Etapa desconocida: {stage}")


if __name__ == "__main__":
    main(sys.argv[1])
