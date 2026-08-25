"""API que expõe o modelo de diagnóstico otimizado pelo GA + explicação via LLM.

Roda tanto localmente (uvicorn, para testar) quanto em produção como container
Lambda (via o adaptador Mangum, que traduz eventos do Lambda em requisições ASGI).
O modelo é carregado uma vez na inicialização — não re-treina nada por requisição.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import llm_utils  # noqa: E402
import lote_utils  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"
SSM_PARAMETER_NAME = os.environ.get("ANTHROPIC_API_KEY_PARAM", "/fase2/anthropic-api-key")

# Métricas do modelo no holdout de teste (ver scripts/treinar_modelo_final.py).
METRICAS_MODELO = {"accuracy": 0.9912, "recall": 0.9762, "f1": 0.9880}

# Resultado do GA que treinou este modelo (experiments/results/best_hyperparams_regressao_logistica.json).
CONTEXTO_GA = {
    "nome_modelo": "regressao_logistica",
    "hiperparametros": {"C": 0.0256, "penalty": "l2", "class_weight": "balanced"},
}

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


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"features": EXEMPLO_FEATURES}})

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


def classificar_paciente(features: dict) -> dict:
    """Roda só o modelo (sem Claude). Usado no /predict e no lote."""
    faltando = set(feature_names) - set(features)
    if faltando:
        raise HTTPException(status_code=422, detail=f"Features faltando: {sorted(faltando)}")

    valores_ordenados = [[features[nome] for nome in feature_names]]
    valores_escalados = scaler.transform(valores_ordenados)
    classe_predita = int(modelo.predict(valores_escalados)[0])
    probas = modelo.predict_proba(valores_escalados)[0]
    tem_doenca = classe_predita == 1
    return {
        "valores_escalados": valores_escalados,
        "predicao": "Maligno" if tem_doenca else "Benigno",
        "resultado": "positivo" if tem_doenca else "negativo",
        "tem_doenca": tem_doenca,
        "chance_doenca": float(probas[1]),
        "probabilidade": float(probas[classe_predita]),
    }


def gerar_laudo_individual(classificacao: dict, chave=None) -> str:
    chave = chave if chave is not None else obter_chave_anthropic()
    if not chave:
        return "Claude não foi chamado: chave não encontrada no arquivo .env."
    valores_originais = scaler.inverse_transform(classificacao["valores_escalados"])[0]
    features_importantes = llm_utils.obter_features_importantes(
        modelo, feature_names, valores_originais
    )
    prompt = llm_utils.montar_prompt(
        predicao=classificacao["predicao"],
        chance_doenca=classificacao["chance_doenca"],
        features_importantes=features_importantes,
        accuracy=METRICAS_MODELO["accuracy"],
        recall=METRICAS_MODELO["recall"],
        f1=METRICAS_MODELO["f1"],
        contexto_ga=llm_utils.formatar_contexto_ga(
            CONTEXTO_GA["nome_modelo"], CONTEXTO_GA["hiperparametros"]
        ),
    )
    try:
        return llm_utils.gerar_explicacao(prompt, api_key=chave)
    except Exception as erro:
        return f"Não foi possível gerar a explicação: {erro}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    classificacao = classificar_paciente(payload.features)
    return PredictResponse(
        predicao=classificacao["predicao"],
        resultado=classificacao["resultado"],
        tem_doenca=classificacao["tem_doenca"],
        chance_doenca=classificacao["chance_doenca"],
        probabilidade=classificacao["probabilidade"],
        explicacao=gerar_laudo_individual(classificacao),
    )


@app.post("/predict/lote", response_model=PredictLoteResponse)
async def predict_lote(arquivo: UploadFile = File(..., description="CSV no formato do data.csv")):
    """Importa um CSV e classifica cada linha com o mesmo laudo do /predict."""
    nome = (arquivo.filename or "").lower()
    if nome and not nome.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    conteudo = (await arquivo.read()).decode("utf-8-sig")
    pacientes, erros = lote_utils.ler_pacientes_csv(conteudo, feature_names)
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
        classificacao = classificar_paciente(paciente["features"])
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
                explicacao=gerar_laudo_individual(classificacao, chave=chave),
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


# Adaptador para rodar como AWS Lambda (container image) — não interfere no uso local via uvicorn.
try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
