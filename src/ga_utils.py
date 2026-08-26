"""Funções do Algoritmo Genético usadas para otimizar os 3 modelos de
diagnóstico (Regressão Linear, Regressão Logística, Random Forest) treinados
em notebooks/01_analise_e_modelagem.ipynb.

Cada algoritmo tem seu próprio criar_individuo_*/mutar_*/fitness_* porque os
hiperparâmetros que cada um aceita são diferentes. Seleção e crossover são
compartilhados entre os três porque só mexem no dict do indivíduo, sem
precisar saber o que cada chave significa.
"""
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

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"


def carregar_dados():
    """Mesmo pré-processamento do notebook 01: dropa colunas não-preditivas,
    faz split 80/20 estratificado (random_state=42) e ajusta o StandardScaler
    só no treino."""
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


def _fitness_cv(construir_modelo, individuo, X, y, n_splits=5):
    """Treina o modelo em validação cruzada estratificada e devolve o fitness
    ponderado (0.5*recall + 0.3*F1 + 0.2*accuracy). Usado por fitness_rf e
    fitness_log — fitness_linear é separada porque precisa aplicar o
    threshold na predição contínua."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = np.asarray(X), np.asarray(y)

    accs, recs, f1s = [], [], []
    for treino_idx, val_idx in skf.split(X, y):
        modelo = construir_modelo(individuo)
        modelo.fit(X[treino_idx], y[treino_idx])
        y_pred = modelo.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], y_pred))
        recs.append(recall_score(y[val_idx], y_pred, zero_division=0))
        f1s.append(f1_score(y[val_idx], y_pred, zero_division=0))

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


def fitness_rf(individuo, X, y):
    return _fitness_cv(construir_rf, individuo, X, y)


# --------------------------------------------------------------------------
# Regressão Logística
# --------------------------------------------------------------------------

LOG_BASELINE = {"C": 1.0, "penalty": "l2", "class_weight": None}


def criar_individuo_log():
    ''''
    10 ** random.uniform(-3, 2) merece destaque: random.uniform(-3, 2) sorteia um expoente entre -3 e 2 (uniformemente), e 10 ** eleva. 
    Isso dá chance igual pra C cair perto de 0.001, perto de 1, ou perto de 100 — se fosse random.uniform(0.001, 100) direto, 
    quase todo sorteio cairia acima de 50 (porque a maior parte do intervalo numérico está lá), e valores pequenos de C quase nunca apareceriam.
    MUITO CARA DE IA
    '''
    return {
        "C": 10 ** random.uniform(-3, 2),
        "penalty": random.choice(["l1", "l2"]),
        "class_weight": random.choice([None, "balanced"]),
    }


def mutar_log(individuo, taxa=0.3):
    novo = dict(individuo)
    if random.random() < taxa:
        novo["C"] = 10 ** random.uniform(-3, 2)
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


def fitness_log(individuo, X, y):
    return _fitness_cv(construir_log, individuo, X, y)


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


def fitness_linear(individuo, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = np.asarray(X), np.asarray(y)

    accs, recs, f1s = [], [], []
    for treino_idx, val_idx in skf.split(X, y):
        modelo = construir_linear(individuo)
        modelo.fit(X[treino_idx], y[treino_idx])
        y_pred = prever_linear(modelo, X[val_idx], individuo)
        accs.append(accuracy_score(y[val_idx], y_pred))
        recs.append(recall_score(y[val_idx], y_pred, zero_division=0))
        f1s.append(f1_score(y[val_idx], y_pred, zero_division=0))

    return 0.5 * np.mean(recs) + 0.3 * np.mean(f1s) + 0.2 * np.mean(accs)


# --------------------------------------------------------------------------
# Compartilhado (seleção e crossover não precisam saber o que cada chave
# do dict significa, então são os mesmos para os 3 algoritmos)
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
    """Ranking das medidas segundo o modelo já treinado pelo GA.

    Random Forest: feature_importances_.
    Modelos lineares (logística / linear): |coef_|.
    """
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
    """Lê as importâncias do modelo otimizado pelo GA, treina de novo só com
    as N medidas mais importantes e grava o JSON que a API/o Angular consomem.

    Arquivos:
    - experiments/results/feature_importances.json
    - api/model/feature_importances.json
    - api/model/modelo_simplificado.joblib
    - api/model/scaler_simplificado.joblib
    """
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
