"""API que expõe o modelo de diagnóstico otimizado pelo GA + explicação via LLM.

Roda tanto localmente (uvicorn, para testar) quanto em produção como container
Lambda (via o adaptador Mangum, que traduz eventos do Lambda em requisições ASGI).
O modelo é carregado uma vez na inicialização — não re-treina nada por requisição.

Há dois pacotes de modelo:
- Exame completo (30 medidas): POST /predict e POST /predict/lote
- Exame simplificado (10 medidas mais importantes): POST /predict/simplificado e POST /predict/lote/simplificado
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import llm_utils  # noqa: E402
import lote_utils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "model"
SSM_PARAMETER_NAME = os.environ.get("ANTHROPIC_API_KEY_PARAM", "/fase2/anthropic-api-key")

# Métricas do modelo no holdout de teste (ver scripts/treinar_modelo_final.py).
METRICAS_MODELO = {"accuracy": 0.9912, "recall": 0.9762, "f1": 0.9880}

# Resultado do GA que treinou este modelo (experiments/results/best_hyperparams_regressao_logistica.json).
CONTEXTO_GA = {
    "nome_modelo": "regressao_logistica",
    "hiperparametros": {"C": 0.0256, "penalty": "l2", "class_weight": "balanced"},
}


@dataclass
class PacoteModelo:
    modelo: object
    scaler: object
    feature_names: list[str]
    metricas: dict
    contexto_ga: str
    ranking: list[dict] | None = None
    algoritmo: str = "regressao_logistica"
    metodo: str = ""
    hiperparametros: dict | None = None


app = FastAPI(title="Diagnóstico de Câncer de Mama — GA + LLM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = joblib.load(MODEL_DIR / "modelo.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())

PACOTE_COMPLETO = PacoteModelo(
    modelo=modelo,
    scaler=scaler,
    feature_names=feature_names,
    metricas=METRICAS_MODELO,
    contexto_ga=llm_utils.formatar_contexto_ga(
        CONTEXTO_GA["nome_modelo"], CONTEXTO_GA["hiperparametros"]
    ),
)


def _carregar_pacote_simplificado() -> PacoteModelo | None:
    """Lê o JSON exportado pelo notebook (feature importances do melhor modelo do GA)."""
    ranking_path = MODEL_DIR / "feature_importances.json"
    if not ranking_path.is_file():
        ranking_path = REPO_ROOT / "experiments" / "results" / "feature_importances.json"
    modelo_path = MODEL_DIR / "modelo_simplificado.joblib"
    scaler_path = MODEL_DIR / "scaler_simplificado.joblib"
    if not ranking_path.is_file() or not modelo_path.is_file() or not scaler_path.is_file():
        return None

    ranking_json = json.loads(ranking_path.read_text(encoding="utf-8"))
    nomes = [item["feature"] for item in ranking_json.get("features") or []]
    if not nomes:
        return None

    algoritmo = ranking_json.get("algoritmo") or "regressao_logistica"
    hiperparams = ranking_json.get("hiperparametros") or CONTEXTO_GA["hiperparametros"]
    metodo = ranking_json.get("metodo") or "feature_importances_"
    contexto = (
        llm_utils.formatar_contexto_ga(algoritmo, hiperparams)
        + " O modelo do exame simplificado usa só as medidas mais importantes "
        "depois do treino otimizado pelo Algoritmo Genético."
    )
    return PacoteModelo(
        modelo=joblib.load(modelo_path),
        scaler=joblib.load(scaler_path),
        feature_names=nomes,
        metricas=ranking_json.get("metricas_holdout") or METRICAS_MODELO,
        contexto_ga=contexto,
        ranking=ranking_json.get("features") or [],
        algoritmo=algoritmo,
        metodo=metodo,
        hiperparametros=hiperparams,
    )


PACOTE_SIMPLIFICADO = _carregar_pacote_simplificado()


def obter_chave_anthropic() -> str | None:
    """Localmente lê o .env; em produção (Lambda) cai para o SSM Parameter Store."""
    chave_local = llm_utils.carregar_chave_do_env()
    if chave_local:
        return chave_local

    try:
        import boto3

        cliente = boto3.client("ssm")
        resposta = cliente.get_parameter(Name=SSM_PARAMETER_NAME, WithDecryption=True)
        return resposta["Parameter"]["Value"]
    except Exception:
        return None


# Primeira linha do data.csv (diagnóstico real: maligno) — só para o exemplo do /docs.
EXEMPLO_FEATURES = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189,
}

EXEMPLO_FEATURES_SIMPLIFICADO = {
    nome: EXEMPLO_FEATURES[nome]
    for nome in (PACOTE_SIMPLIFICADO.feature_names if PACOTE_SIMPLIFICADO else [])
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"features": EXEMPLO_FEATURES}})

    features: dict[str, float]


class PredictRequestSimplificado(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"features": EXEMPLO_FEATURES_SIMPLIFICADO}})

    features: dict[str, float]


class PredictResponse(BaseModel):
    predicao: str
    resultado: str
    tem_doenca: bool
    chance_doenca: float
    probabilidade: float
    explicacao: str | None


class ResultadoLoteItem(BaseModel):
    linha: int
    id: str | None = None
    predicao: str
    resultado: str
    tem_doenca: bool
    chance_doenca: float
    probabilidade: float
    diagnostico_real: str | None = None
    acertou: bool | None = None
    explicacao: str | None = None


class PredictLoteResponse(BaseModel):
    total: int
    positivos: int
    negativos: int
    acertos: int | None = None
    resultados: list[ResultadoLoteItem]
    erros: list[dict] = []


class FeatureSimplificado(BaseModel):
    chave: str
    rotulo: str
    importancia: float


class FeaturesSimplificadoResponse(BaseModel):
    algoritmo: str
    metodo: str
    n: int
    metricas_holdout: dict
    features: list[FeatureSimplificado]


def exigir_simplificado() -> PacoteModelo:
    if PACOTE_SIMPLIFICADO is None:
        raise HTTPException(
            status_code=503,
            detail="Exame simplificado ainda sem ranking. Rode a seção de feature importances no notebook 03_ga_llm_breast_cancer.ipynb",
        )
    return PACOTE_SIMPLIFICADO


def classificar_paciente(features: dict, pacote: PacoteModelo) -> dict:
    """Roda só o modelo (sem Claude). Usado no /predict e no lote."""
    faltando = set(pacote.feature_names) - set(features)
    if faltando:
        raise HTTPException(status_code=422, detail=f"Features faltando: {sorted(faltando)}")

    valores_ordenados = [[features[nome] for nome in pacote.feature_names]]
    valores_escalados = pacote.scaler.transform(valores_ordenados)

    if hasattr(pacote.modelo, "predict_proba"):
        classe_predita = int(pacote.modelo.predict(valores_escalados)[0])
        probas = pacote.modelo.predict_proba(valores_escalados)[0]
        tem_doenca = classe_predita == 1
        chance_doenca = float(probas[1])
        probabilidade = float(probas[classe_predita])
    else:
        continuo = float(pacote.modelo.predict(valores_escalados)[0])
        limiar = float((pacote.hiperparametros or {}).get("threshold", 0.5))
        tem_doenca = continuo >= limiar
        classe_predita = 1 if tem_doenca else 0
        chance_doenca = float(np.clip(continuo, 0.0, 1.0))
        probabilidade = chance_doenca if tem_doenca else float(1.0 - chance_doenca)

    return {
        "valores_escalados": valores_escalados,
        "predicao": "Maligno" if tem_doenca else "Benigno",
        "resultado": "positivo" if tem_doenca else "negativo",
        "tem_doenca": tem_doenca,
        "chance_doenca": chance_doenca,
        "probabilidade": probabilidade,
    }


def gerar_laudo_individual(classificacao: dict, pacote: PacoteModelo, chave=None) -> str:
    chave = chave if chave is not None else obter_chave_anthropic()
    if not chave:
        return "Claude não foi chamado: chave não encontrada no arquivo .env."
    valores_originais = pacote.scaler.inverse_transform(classificacao["valores_escalados"])[0]
    features_importantes = llm_utils.obter_features_importantes(
        pacote.modelo, pacote.feature_names, valores_originais
    )
    prompt = llm_utils.montar_prompt(
        predicao=classificacao["predicao"],
        chance_doenca=classificacao["chance_doenca"],
        features_importantes=features_importantes,
        accuracy=pacote.metricas["accuracy"],
        recall=pacote.metricas["recall"],
        f1=pacote.metricas["f1"],
        contexto_ga=pacote.contexto_ga,
    )
    try:
        return llm_utils.gerar_explicacao(prompt, api_key=chave)
    except Exception as erro:
        return f"Não foi possível gerar a explicação: {erro}"


def montar_resposta_predict(features: dict, pacote: PacoteModelo) -> PredictResponse:
    classificacao = classificar_paciente(features, pacote)
    return PredictResponse(
        predicao=classificacao["predicao"],
        resultado=classificacao["resultado"],
        tem_doenca=classificacao["tem_doenca"],
        chance_doenca=classificacao["chance_doenca"],
        probabilidade=classificacao["probabilidade"],
        explicacao=gerar_laudo_individual(classificacao, pacote),
    )


async def montar_resposta_lote(arquivo: UploadFile, pacote: PacoteModelo) -> PredictLoteResponse:
    nome = (arquivo.filename or "").lower()
    if nome and not nome.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    conteudo = (await arquivo.read()).decode("utf-8-sig")
    pacientes, erros = lote_utils.ler_pacientes_csv(conteudo, pacote.feature_names)
    if not pacientes:
        raise HTTPException(
            status_code=422,
            detail=erros or "Nenhuma linha válida no CSV.",
        )

    chave = obter_chave_anthropic()
    resultados = []
    acertos = 0
    comparados = 0
    for paciente in pacientes:
        classificacao = classificar_paciente(paciente["features"], pacote)
        diagnostico_real = paciente["diagnosis"]
        acertou = None
        if diagnostico_real:
            comparados += 1
            acertou = diagnostico_real == classificacao["predicao"]
            if acertou:
                acertos += 1
        resultados.append(
            ResultadoLoteItem(
                linha=paciente["linha"],
                id=paciente["id"],
                predicao=classificacao["predicao"],
                resultado=classificacao["resultado"],
                tem_doenca=classificacao["tem_doenca"],
                chance_doenca=classificacao["chance_doenca"],
                probabilidade=classificacao["probabilidade"],
                diagnostico_real=diagnostico_real,
                acertou=acertou,
                explicacao=gerar_laudo_individual(classificacao, pacote, chave=chave),
            )
        )

    positivos = sum(1 for item in resultados if item.tem_doenca)
    negativos = len(resultados) - positivos

    return PredictLoteResponse(
        total=len(resultados),
        positivos=positivos,
        negativos=negativos,
        acertos=acertos if comparados else None,
        resultados=resultados,
        erros=erros,
    )


@app.get("/health")
def health():
    return {"status": "ok", "modelo_simplificado": PACOTE_SIMPLIFICADO is not None}


@app.get("/status")
def status():
    return {"status": "tech-challenge-fase-2"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    return montar_resposta_predict(payload.features, PACOTE_COMPLETO)


@app.post("/predict/lote", response_model=PredictLoteResponse)
async def predict_lote(arquivo: UploadFile = File(..., description="CSV no formato do data.csv")):
    """Importa um CSV e classifica cada linha com o mesmo laudo do /predict."""
    return await montar_resposta_lote(arquivo, PACOTE_COMPLETO)


@app.get(
    "/predict/simplificado/features",
    response_model=FeaturesSimplificadoResponse,
    summary="Medidas do exame simplificado",
)
def listar_features_simplificado():
    """Lista as 10 medidas do exame simplificado (para montar o formulário)."""
    pacote = exigir_simplificado()
    por_nome = {item["feature"]: item["importancia"] for item in (pacote.ranking or [])}
    return FeaturesSimplificadoResponse(
        algoritmo=pacote.algoritmo,
        metodo=pacote.metodo,
        n=len(pacote.feature_names),
        metricas_holdout=pacote.metricas,
        features=[
            FeatureSimplificado(
                chave=nome,
                rotulo=llm_utils.nome_amigavel(nome),
                importancia=float(por_nome.get(nome, 0.0)),
            )
            for nome in pacote.feature_names
        ],
    )


@app.post("/predict/simplificado", response_model=PredictResponse, summary="Exame simplificado")
def predict_simplificado(payload: PredictRequestSimplificado):
    return montar_resposta_predict(payload.features, exigir_simplificado())


@app.post(
    "/predict/lote/simplificado",
    response_model=PredictLoteResponse,
    summary="Lote — exame simplificado",
)
async def predict_lote_simplificado(
    arquivo: UploadFile = File(..., description="CSV com as 10 medidas do exame simplificado (ou o data.csv completo)"),
):
    """Mesmo CSV do lote completo: só as 10 colunas mais importantes são usadas."""
    return await montar_resposta_lote(arquivo, exigir_simplificado())


# Adaptador para rodar como AWS Lambda (container image) — não interfere no uso local via uvicorn.
try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
