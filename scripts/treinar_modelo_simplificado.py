"""Reconstroi o exame simplificado a partir dos hiperparâmetros salvos pelo notebook.

O caminho principal é o notebook 03: depois do GA ele chama
ga_utils.persistir_exame_simplificado e grava feature_importances.json.
Este script só serve se você não quiser reabrir o Jupyter — usa os JSON em
experiments/results/best_hyperparams_*.json, escolhe o mesmo vencedor
(maior recall, desempate por F1) e exporta de novo.
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ga_utils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "experiments" / "results"

ARQUIVOS = {
    "regressao_linear": "best_hyperparams_regressao_linear.json",
    "regressao_logistica": "best_hyperparams_regressao_logistica.json",
    "random_forest": "best_hyperparams_random_forest.json",
}


def main():
    X_train, X_test, y_train, y_test, feature_names, _scaler = ga_utils.carregar_dados()
    candidatos = {}
    for algoritmo, arquivo in ARQUIVOS.items():
        hiper = json.loads((RESULTADOS_DIR / arquivo).read_text(encoding="utf-8"))["melhor_individuo"]
        modelo = ga_utils.construir_modelo(algoritmo, hiper)
        modelo.fit(X_train, y_train)
        y_pred = ga_utils.prever_modelo(algoritmo, modelo, X_test, hiper)
        metricas = ga_utils.calcular_metricas(y_test, y_pred)
        candidatos[algoritmo] = (modelo, hiper, metricas)
        print(algoritmo, metricas)

    melhor = max(
        candidatos,
        key=lambda nome: (candidatos[nome][2]["recall"], candidatos[nome][2]["f1"]),
    )
    modelo, hiper, _metricas = candidatos[melhor]
    print(f"\nVencedor (mesmo critério do notebook): {melhor}")

    payload = ga_utils.persistir_exame_simplificado(
        algoritmo=melhor,
        modelo=modelo,
        hiperparametros=hiper,
        feature_names=feature_names,
    )
    print(f"\n{payload['n']} medidas do exame simplificado:")
    for i, item in enumerate(payload["features"], start=1):
        print(f"  {i:2}. {item['feature']:24} {item['importancia']:.4f}")
    print("JSON:", RESULTADOS_DIR / "feature_importances.json")


if __name__ == "__main__":
    main()
