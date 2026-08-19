"""Treina o melhor modelo encontrado pelo GA (hoje: Regressão Logística) na
base de treino completa e persiste os artefatos que a API precisa em runtime:

- api/model/modelo.joblib          o modelo treinado
- api/model/scaler.joblib          o StandardScaler ajustado no treino
- api/model/feature_names.json     ordem das 30 features esperadas no /predict

Roda uma vez, localmente, antes de construir a imagem Docker — a API não
re-treina nada, só carrega esses 3 arquivos na inicialização.
"""
import json
import sys
from pathlib import Path

import joblib

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ga_utils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "experiments" / "results"
MODEL_DIR = REPO_ROOT / "api" / "model"


def main():
    melhores_hiperparametros = json.loads(
        (RESULTADOS_DIR / "best_hyperparams_regressao_logistica.json").read_text()
    )["melhor_individuo"]

    X_train, X_test, y_train, y_test, feature_names, scaler = ga_utils.carregar_dados()

    modelo = ga_utils.construir_log(melhores_hiperparametros)
    modelo.fit(X_train, y_train)

    metricas = ga_utils.calcular_metricas(y_test, modelo.predict(X_test))
    print("Métricas no holdout de teste:", metricas)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_DIR / "modelo.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"Artefatos salvos em {MODEL_DIR}")


if __name__ == "__main__":
    main()
