"""Punto de entrada por etapa (lo usan dvc.yaml y el flow de Prefect): python -m src.cli <etapa> [argumentos]

  prepare | audit | experiments | experimento <id> | verificar | release v1|v2 | congelar | evaluate | recalcular | modelcards [v1|v2]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["prepare", "audit", "experiments", "experimento", "release", "congelar", "evaluate",
                                      "recalcular", "modelcards", "verificar"])
    ap.add_argument("version", nargs="?", help="v1|v2 (release, modelcards) o el id del experimento (experimento)")
    ap.add_argument("--sin-hf", action="store_true", help="congelar sin evidencias/hf_publicacion.json (solo pruebas locales)")
    a = ap.parse_args()

    if a.etapa == "prepare":
        from src import data
        print(json.dumps(data.prepare(), indent=1, ensure_ascii=False))
    elif a.etapa == "audit":
        from src import audit
        audit.build_audit()
        print("evidencias/auditoria.json escrito")
    elif a.etapa == "experiments":
        from src import experiments, exportar
        experiments.run_all()
        exportar.exportar()
        print("evidencias/experimentos.csv y evidencias/mlflow/ escritos")
    elif a.etapa == "experimento":
        from src import experiments
        if not a.version:
            raise SystemExit("Uso: python -m src.cli experimento E2_logreg_log1p")
        print(experiments.run_all(solo=a.version).T.to_string())
    elif a.etapa == "verificar":
        from src import verificar
        raise SystemExit(0 if verificar.verificar() else 1)
    elif a.etapa == "release":
        from src import release
        if a.version == "v1":
            release.build_v1()
        elif a.version == "v2":
            release.build_v2()
        else:
            raise SystemExit("Uso: python -m src.cli release v1|v2")
    elif a.etapa == "congelar":
        from src import congelar
        congelar.congelar(exigir_hf=not a.sin_hf)
    elif a.etapa == "evaluate":
        from src import evaluate
        evaluate.evaluar_test()
    elif a.etapa == "recalcular":
        from src import evaluate
        evaluate.recalcular()
    elif a.etapa == "modelcards":
        from src import modelcard
        from src.config import EVIDENCIAS
        m, h = EVIDENCIAS / "metricas-test.json", EVIDENCIAS / "hf_publicacion.json"
        test = json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
        hfd = json.loads(h.read_text(encoding="utf-8")) if h.exists() else None
        for v in ([a.version] if a.version else ["v1", "v2"]):
            modelcard.write(v, test if v == "v2" else None, hfd if v == "v2" else None)
            print(f"docs/model_cards/{v}.md escrito")


if __name__ == "__main__":
    main()
