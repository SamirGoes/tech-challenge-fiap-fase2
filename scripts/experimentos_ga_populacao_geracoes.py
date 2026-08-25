"""Experimentos de sensibilidade do GA a tamanho de população e número de
gerações, nos 3 algoritmos (Fase 2, item 1: "realizar ao menos 3 experimentos
com diferentes configurações do algoritmo genético — tamanho da população,
taxas de mutação, etc.").

A execução "oficial", registrada, desses experimentos vive no notebook
`notebooks/03_ga_llm_breast_cancer.ipynb` (seção "Testes de sensibilidade a
população/gerações"). Este script é um jeito de regenerar os mesmos JSONs
sem precisar abrir o Jupyter — útil para automação ou para o pygame pedir
dados novos rapidamente. É idempotente: se o JSON de uma combinação já
existe, pula (não recalcula).

Roda o GA de cada um dos 3 algoritmos 3 vezes, variando uma configuração por
vez para isolar o efeito de cada uma:

- pop_baixa_ger_baixa: baseline (população e gerações pequenas)
- pop_alta_ger_baixa: só aumenta a população, mesmas gerações
- pop_baixa_ger_alta: só aumenta as gerações, mesma população

Cada combinação (algoritmo x configuração) grava um JSON em
experiments/results/ga_history_<algoritmo>_<config>.json, no formato
consumido pela visualização em pygame.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ga_utils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

ALGORITMOS = {
    "random_forest": {
        "modelo_base": "RandomForest",
        "criar_individuo": ga_utils.criar_individuo_rf,
        "mutar": ga_utils.mutar_rf,
        "fitness_fn": ga_utils.fitness_rf,
    },
    "regressao_logistica": {
        "modelo_base": "RegressaoLogistica",
        "criar_individuo": ga_utils.criar_individuo_log,
        "mutar": ga_utils.mutar_log,
        "fitness_fn": ga_utils.fitness_log,
    },
    "regressao_linear": {
        "modelo_base": "RegressaoLinear",
        "criar_individuo": ga_utils.criar_individuo_linear,
        "mutar": ga_utils.mutar_linear,
        "fitness_fn": ga_utils.fitness_linear,
    },
}

CONFIGS = [
    {
        "nome": "pop_baixa_ger_baixa",
        "tamanho_populacao": 10,
        "n_geracoes": 10,
        "taxa_mutacao": 0.2,
    },
    {
        "nome": "pop_alta_ger_baixa",
        "tamanho_populacao": 30,
        "n_geracoes": 10,
        "taxa_mutacao": 0.2,
    },
    {
        "nome": "pop_baixa_ger_alta",
        "tamanho_populacao": 10,
        "n_geracoes": 30,
        "taxa_mutacao": 0.2,
    },
]


def main():
    X_train, _, y_train, _, _, _ = ga_utils.carregar_dados()

    for nome_algoritmo, cfg_algoritmo in ALGORITMOS.items():
        for cfg in CONFIGS:
            caminho = RESULTS_DIR / f"ga_history_{nome_algoritmo}_{cfg['nome']}.json"
            if caminho.exists():
                print(f"{caminho.name}: já existe, pulando")
                continue

            print(f"Rodando {nome_algoritmo} / {cfg['nome']} ...")
            historico = ga_utils.executar_ga(
                criar_individuo=cfg_algoritmo["criar_individuo"],
                mutar=cfg_algoritmo["mutar"],
                fitness_fn=cfg_algoritmo["fitness_fn"],
                X=X_train,
                y=y_train,
                tamanho_populacao=cfg["tamanho_populacao"],
                n_geracoes=cfg["n_geracoes"],
                taxa_mutacao=cfg["taxa_mutacao"],
                seed=42,
            )
            ga_utils.exportar_historico_json(
                historico,
                algoritmo="genetico_hiperparametros",
                modelo_base=cfg_algoritmo["modelo_base"],
                caminho=caminho,
            )
            print(
                f"  -> {caminho.name}: melhor_fitness final = "
                f"{historico[-1]['melhor_fitness']:.4f}"
            )


if __name__ == "__main__":
    main()
