import json
import random

import numpy as np
import ga_utils


def test_crossover_filhos_recombinam_genes_dos_pais():
    random.seed(1)
    pai1 = {"a": 1, "b": "x"}
    pai2 = {"a": 2, "b": "y"}
    filho1, filho2 = ga_utils.crossover(pai1, pai2)
    assert {filho1["a"], filho2["a"]} == {1, 2}
    assert {filho1["b"], filho2["b"]} == {"x", "y"}


def test_selecao_torneio_escolhe_o_melhor_quando_k_cobre_populacao():
    populacao = [{"n": i} for i in range(5)]
    fitnesses = [0.1, 0.9, 0.3, 0.2, 0.05]
    random.seed(42)
    vencedores = {ga_utils.selecao_torneio(populacao, fitnesses, k=5)["n"] for _ in range(20)}
    assert vencedores == {1}


def test_criar_individuo_rf_respeita_limites():
    random.seed(0)
    for _ in range(30):
        ind = ga_utils.criar_individuo_rf()
        assert 50 <= ind["n_estimators"] <= 300
        assert ind["max_depth"] in (None, 3, 5, 8, 12, 16, 20)
        assert 2 <= ind["min_samples_split"] <= 20
        assert 1 <= ind["min_samples_leaf"] <= 10
        assert ind["max_features"] in ("sqrt", "log2", None)
        assert ind["criterion"] in ("gini", "entropy")


def test_criar_individuo_log_respeita_limites():
    random.seed(0)
    for _ in range(30):
        ind = ga_utils.criar_individuo_log()
        assert 0.001 <= ind["C"] <= 100
        assert ind["penalty"] in ("l1", "l2")
        assert ind["class_weight"] in (None, "balanced")


def test_criar_individuo_linear_respeita_limites():
    random.seed(0)
    for _ in range(30):
        ind = ga_utils.criar_individuo_linear()
        assert ind["fit_intercept"] in (True, False)
        assert ind["positive"] in (True, False)
        assert 0.3 <= ind["threshold"] <= 0.7


def _dataset_sintetico():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


def test_fitness_rf_retorna_valor_entre_0_e_1():
    X, y = _dataset_sintetico()
    fitness = ga_utils.fitness_rf(ga_utils.RF_BASELINE, X, y)
    assert 0.0 <= fitness <= 1.0


def test_fitness_log_retorna_valor_entre_0_e_1():
    X, y = _dataset_sintetico()
    fitness = ga_utils.fitness_log(ga_utils.LOG_BASELINE, X, y)
    assert 0.0 <= fitness <= 1.0


def test_fitness_linear_threshold_baixo_favorece_recall():
    X, y = _dataset_sintetico()
    baixo = dict(ga_utils.LINEAR_BASELINE, threshold=0.3)
    alto = dict(ga_utils.LINEAR_BASELINE, threshold=0.9)
    fitness_baixo = ga_utils.fitness_linear(baixo, X, y)
    fitness_alto = ga_utils.fitness_linear(alto, X, y)
    # limiar baixo prevê mais casos como positivos -> mais recall -> fitness (que pesa recall) mais alto
    assert fitness_baixo >= fitness_alto


def test_executar_ga_devolve_uma_entrada_por_geracao_com_melhor_individuo():
    X, y = _dataset_sintetico()
    historico = ga_utils.executar_ga(
        criar_individuo=ga_utils.criar_individuo_rf,
        mutar=ga_utils.mutar_rf,
        fitness_fn=ga_utils.fitness_rf,
        X=X,
        y=y,
        tamanho_populacao=5,
        n_geracoes=3,
        taxa_mutacao=0.2,
        seed=0,
    )
    assert [g["geracao"] for g in historico] == [1, 2, 3]
    for geracao in historico:
        assert 0.0 <= geracao["melhor_fitness"] <= 1.0
        assert set(geracao["melhores_params"]) == {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "criterion",
        }


def test_executar_ga_com_elitismo_nao_piora_o_melhor_fitness_entre_geracoes():
    X, y = _dataset_sintetico()
    historico = ga_utils.executar_ga(
        criar_individuo=ga_utils.criar_individuo_rf,
        mutar=ga_utils.mutar_rf,
        fitness_fn=ga_utils.fitness_rf,
        X=X,
        y=y,
        tamanho_populacao=6,
        n_geracoes=4,
        taxa_mutacao=0.3,
        seed=7,
    )
    melhores = [g["melhor_fitness"] for g in historico]
    assert all(atual >= anterior for anterior, atual in zip(melhores, melhores[1:]))


def test_executar_ga_e_deterministico_com_mesma_seed():
    X, y = _dataset_sintetico()
    kwargs = dict(
        criar_individuo=ga_utils.criar_individuo_log,
        mutar=ga_utils.mutar_log,
        fitness_fn=ga_utils.fitness_log,
        X=X,
        y=y,
        tamanho_populacao=5,
        n_geracoes=3,
        taxa_mutacao=0.3,
        seed=1,
    )
    historico_1 = ga_utils.executar_ga(**kwargs)
    historico_2 = ga_utils.executar_ga(**kwargs)
    assert historico_1 == historico_2


def test_exportar_historico_json_grava_no_formato_esperado_para_o_pygame(tmp_path):
    historico = [
        {"geracao": 1, "melhor_fitness": 0.85, "melhores_params": {"n_estimators": 100}},
        {"geracao": 2, "melhor_fitness": 0.89, "melhores_params": {"n_estimators": 120}},
    ]
    caminho = tmp_path / "subdir" / "ga_history.json"

    payload = ga_utils.exportar_historico_json(
        historico, algoritmo="genetico_hiperparametros", modelo_base="RandomForest", caminho=caminho
    )

    assert payload == {
        "algoritmo": "genetico_hiperparametros",
        "modelo_base": "RandomForest",
        "geracoes": historico,
    }
    assert json.loads(caminho.read_text(encoding="utf-8")) == payload
