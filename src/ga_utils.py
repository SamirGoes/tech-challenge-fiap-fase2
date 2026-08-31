import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Inconsistent values:.*penalty is deprecated.*",
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"


def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["id", "Unnamed: 32", "diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        list(X.columns),
        scaler,
    )


def calcular_metricas(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def _fitness_cv(construir_modelo, individuo, X_train, y_train, n_splits=5, random_state=42):
    """Fitness via StratifiedKFold só no X_train — o X_test nunca entra na busca do GA."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    accs, recs, f1s = [], [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx_treino, idx_val in skf.split(X_train, y_train):
        modelo = construir_modelo(individuo)
        modelo.fit(X_train[idx_treino], y_train[idx_treino])
        y_pred = modelo.predict(X_train[idx_val])

        accs.append(accuracy_score(y_train[idx_val], y_pred))
        recs.append(recall_score(y_train[idx_val], y_pred, zero_division=0))
        f1s.append(f1_score(y_train[idx_val], y_pred, zero_division=0))

    return 0.5 * np.mean(recs) + 0.3 * np.mean(f1s) + 0.2 * np.mean(accs)


# --------------------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------------------

RF_BASELINE = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "gini",
}


def criar_individuo_rf():
    return {
        "n_estimators": random.randint(50, 300),
        "max_depth": random.choice([None, 3, 5, 8, 12, 16, 20]),
        "min_samples_split": random.randint(2, 20),
        "min_samples_leaf": random.randint(1, 10),
        "max_features": random.choice(["sqrt", "log2", None]),
        "criterion": random.choice(["gini", "entropy"]),
    }


def mutar_rf(individuo, taxa=0.2):
    novo = dict(individuo)
    if random.random() < taxa:
        novo["n_estimators"] = random.randint(50, 300)
    if random.random() < taxa:
        novo["max_depth"] = random.choice([None, 3, 5, 8, 12, 16, 20])
    if random.random() < taxa:
        novo["min_samples_split"] = random.randint(2, 20)
    if random.random() < taxa:
        novo["min_samples_leaf"] = random.randint(1, 10)
    if random.random() < taxa:
        novo["max_features"] = random.choice(["sqrt", "log2", None])
    if random.random() < taxa:
        novo["criterion"] = random.choice(["gini", "entropy"])
    return novo


def construir_rf(individuo):
    return RandomForestClassifier(
        n_estimators=individuo["n_estimators"],
        max_depth=individuo["max_depth"],
        min_samples_split=individuo["min_samples_split"],
        min_samples_leaf=individuo["min_samples_leaf"],
        max_features=individuo["max_features"],
        criterion=individuo["criterion"],
        random_state=42,
    )


def fitness_rf(individuo, X_train, y_train):
    return _fitness_cv(construir_rf, individuo, X_train, y_train)


# --------------------------------------------------------------------------
# Regressão Logística
# --------------------------------------------------------------------------

LOG_BASELINE = {"C": 1.0, "penalty": "l2", "class_weight": None}


def criar_individuo_log():
    return {
        "C": random.uniform(0.01, 10),
        "penalty": random.choice(["l1", "l2"]),
        "class_weight": random.choice([None, "balanced"]),
    }


def mutar_log(individuo, taxa=0.3):
    novo = dict(individuo)
    if random.random() < taxa:
        novo["C"] = random.uniform(0.01, 10)
    if random.random() < taxa:
        novo["penalty"] = random.choice(["l1", "l2"])
    if random.random() < taxa:
        novo["class_weight"] = random.choice([None, "balanced"])
    return novo


def construir_log(individuo):
    return LogisticRegression(
        C=individuo["C"],
        penalty=individuo["penalty"],
        class_weight=individuo["class_weight"],
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )


def fitness_log(individuo, X_train, y_train):
    return _fitness_cv(construir_log, individuo, X_train, y_train)


# --------------------------------------------------------------------------
# Regressão Linear
# --------------------------------------------------------------------------

LINEAR_BASELINE = {"fit_intercept": True, "positive": False, "threshold": 0.5}


def criar_individuo_linear():
    return {
        "fit_intercept": random.choice([True, False]),
        "positive": random.choice([True, False]),
        "threshold": random.uniform(0.3, 0.7),
    }


def mutar_linear(individuo, taxa=0.1):
    novo = dict(individuo)
    if random.random() < taxa:
        novo["fit_intercept"] = random.choice([True, False])
    if random.random() < taxa:
        novo["positive"] = random.choice([True, False])
    if random.random() < taxa:
        novo["threshold"] = random.uniform(0.3, 0.7)
    return novo


def construir_linear(individuo):
    return LinearRegression(
        fit_intercept=individuo["fit_intercept"], positive=individuo["positive"]
    )


def prever_linear(modelo, X, individuo):
    y_pred_continuo = modelo.predict(X)
    return (y_pred_continuo >= individuo["threshold"]).astype(int)

def fitness_linear(individuo, X_train, Y_train, n_splits=5, random_state=42):
    # adicionado posteriormente, pois quando usávamos o x_test ele convergia muito rápido
    # pois quando via o fitness já usava o holdout, então o modelo já sabia a resposta e não precisava generalizar
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    accs, recs, f1s = [], [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx_treino, idx_val in skf.split(X_train, Y_train):
        modelo = construir_linear(individuo)
        modelo.fit(X_train[idx_treino], Y_train[idx_treino])
        y_pred = prever_linear(modelo, X_train[idx_val], individuo)

        accs.append(accuracy_score(Y_train[idx_val], y_pred))
        recs.append(recall_score(Y_train[idx_val], y_pred, zero_division=0))
        f1s.append(f1_score(Y_train[idx_val], y_pred, zero_division=0))

    return 0.5 * np.mean(recs) + 0.3 * np.mean(f1s) + 0.2 * np.mean(accs)


# --------------------------------------------------------------------------
# Compartilhado
# --------------------------------------------------------------------------


def crossover(pai1, pai2):
    filho1, filho2 = {}, {}
    for chave in pai1:
        if random.random() < 0.5:
            filho1[chave], filho2[chave] = pai1[chave], pai2[chave]
        else:
            filho1[chave], filho2[chave] = pai2[chave], pai1[chave]
    return filho1, filho2


def selecao_torneio(populacao, fitnesses, k=3):
    k = min(k, len(populacao))
    indices = random.sample(range(len(populacao)), k)
    melhor_idx = max(indices, key=lambda i: fitnesses[i])
    return dict(populacao[melhor_idx])


def selecao_roleta(populacao, fitnesses):
    minimo = min(fitnesses)
    pesos = [f - minimo + 1e-6 for f in fitnesses]
    total = sum(pesos)
    corte = random.uniform(0, total)
    acumulado = 0.0
    for individuo, peso in zip(populacao, pesos):
        acumulado += peso
        if acumulado >= corte:
            return dict(individuo)
    return dict(populacao[-1])


CRIAR_INDIVIDUO = {
    "random_forest": criar_individuo_rf,
    "regressao_logistica": criar_individuo_log,
    "regressao_linear": criar_individuo_linear,
}
FITNESS = {
    "random_forest": fitness_rf,
    "regressao_logistica": fitness_log,
    "regressao_linear": fitness_linear,
}
MUTAR = {
    "random_forest": mutar_rf,
    "regressao_logistica": mutar_log,
    "regressao_linear": mutar_linear,
}


def rodar_ga(algoritmo, X_train, y_train, X_test, y_test, tamanho_populacao, geracoes,
             mutacao, selecao_func=selecao_torneio, mutacao_final=None, seed=None):
    # X_test/y_test ficam só na assinatura por compatibilidade com quem chama —
    # o fitness (abaixo) usa StratifiedKFold só no X_train para não vazar o holdout.
    if seed is not None:
        random.seed(seed)

    criar_individuo = CRIAR_INDIVIDUO[algoritmo]
    fitness_func = FITNESS[algoritmo]
    mutar_func = MUTAR[algoritmo]

    populacao = [criar_individuo() for _ in range(tamanho_populacao)]
    historico = []

    for geracao in range(geracoes):
        taxa = mutacao if mutacao_final is None else (
            mutacao + (mutacao_final - mutacao) * (geracao / max(geracoes - 1, 1))
        )
        fitnesses = [fitness_func(ind, X_train, y_train) for ind in populacao]
        historico.append({
            "geracao": geracao,
            "melhor": max(fitnesses),
            "media": float(np.mean(fitnesses)),
            "taxa_mutacao": taxa,
        })
        print(f"Geração {geracao}: melhor={max(fitnesses):.4f}  média={np.mean(fitnesses):.4f}  mutação={taxa:.3f}")

        ranking = sorted(range(len(populacao)), key=lambda i: fitnesses[i], reverse=True)
        nova_populacao = [dict(populacao[i]) for i in ranking[:2]]
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_func(populacao, fitnesses)
            pai2 = selecao_func(populacao, fitnesses)
            filho1, filho2 = crossover(pai1, pai2)
            nova_populacao.append(mutar_func(filho1, taxa))
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(mutar_func(filho2, taxa))
        populacao = nova_populacao

    melhor = populacao[int(np.argmax(fitnesses))]
    return populacao, historico, fitnesses, melhor


# --------------------------------------------------------------------------
# Feature importances → exame simplificado (API + Angular leem o JSON)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
N_FEATURES_SIMPLIFICADO = 10


def construir_modelo(algoritmo, individuo):
    if algoritmo == "random_forest":
        return construir_rf(individuo)
    if algoritmo == "regressao_logistica":
        return construir_log(individuo)
    if algoritmo == "regressao_linear":
        return construir_linear(individuo)
    raise ValueError(f"Algoritmo desconhecido: {algoritmo}")


def prever_modelo(algoritmo, modelo, X, individuo=None):
    if algoritmo == "regressao_linear":
        return prever_linear(modelo, X, individuo)
    return modelo.predict(X)


def importancias_do_modelo(modelo, feature_names):
    if hasattr(modelo, "feature_importances_"):
        pesos = np.asarray(modelo.feature_importances_, dtype=float).ravel()
        metodo = "feature_importances_"
    elif hasattr(modelo, "coef_"):
        pesos = np.abs(np.asarray(modelo.coef_, dtype=float)).ravel()
        metodo = "abs(coef_)"
    else:
        pesos = np.ones(len(feature_names), dtype=float)
        metodo = "uniforme"

    ranking = sorted(
        (
            {"feature": nome, "importancia": float(peso)}
            for nome, peso in zip(feature_names, pesos)
        ),
        key=lambda item: item["importancia"],
        reverse=True,
    )
    return metodo, ranking


def _dados_brutos():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["id", "Unnamed: 32", "diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1})
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def persistir_exame_simplificado(
    algoritmo,
    modelo,
    hiperparametros,
    feature_names,
    n=N_FEATURES_SIMPLIFICADO,
    resultados_dir=None,
    model_dir=None,
):
    """Grava experiments/results/feature_importances.json, api/model/feature_importances.json,
    api/model/modelo_simplificado.joblib e api/model/scaler_simplificado.joblib."""
    resultados_dir = Path(resultados_dir) if resultados_dir else REPO_ROOT / "experiments" / "results"
    model_dir = Path(model_dir) if model_dir else REPO_ROOT / "api" / "model"
    resultados_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metodo, ranking = importancias_do_modelo(modelo, feature_names)
    escolhidas = ranking[:n]
    nomes = [item["feature"] for item in escolhidas]

    X_train, X_test, y_train, y_test = _dados_brutos()
    scaler_simplificado = StandardScaler()
    X_train_n = scaler_simplificado.fit_transform(X_train[nomes].to_numpy())
    X_test_n = scaler_simplificado.transform(X_test[nomes].to_numpy())

    modelo_simplificado = construir_modelo(algoritmo, hiperparametros)
    modelo_simplificado.fit(X_train_n, y_train)
    y_pred = prever_modelo(algoritmo, modelo_simplificado, X_test_n, hiperparametros)
    metricas = calcular_metricas(y_test, y_pred)

    payload = {
        "algoritmo": algoritmo,
        "metodo": metodo,
        "n": n,
        "hiperparametros": hiperparametros,
        "metricas_holdout": metricas,
        "features": escolhidas,
        "ranking_completo": ranking,
    }
    texto = json.dumps(payload, indent=2, default=str)
    (resultados_dir / "feature_importances.json").write_text(texto, encoding="utf-8")
    (model_dir / "feature_importances.json").write_text(texto, encoding="utf-8")
    joblib.dump(modelo_simplificado, model_dir / "modelo_simplificado.joblib")
    joblib.dump(scaler_simplificado, model_dir / "scaler_simplificado.joblib")
    return payload
