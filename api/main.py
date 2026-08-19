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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import llm_utils  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"
SSM_PARAMETER_NAME = os.environ.get("ANTHROPIC_API_KEY_PARAM", "/fase2/anthropic-api-key")

# Métricas do modelo no holdout de teste (ver scripts/treinar_modelo_final.py),
# usadas só para dar contexto de desempenho na explicação da LLM.
METRICAS_MODELO = {"accuracy": 0.9912, "recall": 0.9762, "f1": 0.9880}

app = FastAPI(title="Diagnóstico de Câncer de Mama — GA + LLM")

modelo = joblib.load(MODEL_DIR / "modelo.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())


def obter_chave_anthropic() -> str | None:
    """Em produção lê do SSM Parameter Store; localmente cai para a variável
    de ambiente ANTHROPIC_API_KEY, pra facilitar testar sem precisar de AWS."""
    chave_local = os.environ.get("ANTHROPIC_API_KEY")
    if chave_local:
        return chave_local

    try:
        import boto3

        cliente = boto3.client("ssm")
        resposta = cliente.get_parameter(Name=SSM_PARAMETER_NAME, WithDecryption=True)
        return resposta["Parameter"]["Value"]
    except Exception:
        return None


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    predicao: str
    probabilidade: float
    explicacao: str | None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    faltando = set(feature_names) - set(payload.features)
    if faltando:
        raise HTTPException(status_code=422, detail=f"Features faltando: {sorted(faltando)}")

    valores_ordenados = [[payload.features[nome] for nome in feature_names]]
    valores_escalados = scaler.transform(valores_ordenados)

    classe_predita = int(modelo.predict(valores_escalados)[0])
    probabilidade = float(modelo.predict_proba(valores_escalados)[0][classe_predita])
    predicao_texto = "Maligno" if classe_predita == 1 else "Benigno"

    chave = obter_chave_anthropic()
    explicacao = None
    if chave:
        features_importantes = llm_utils.obter_features_importantes(
            modelo, feature_names, valores_escalados[0]
        )
        prompt = llm_utils.montar_prompt(
            predicao=predicao_texto,
            probabilidade=probabilidade,
            features_importantes=features_importantes,
            accuracy=METRICAS_MODELO["accuracy"],
            recall=METRICAS_MODELO["recall"],
            f1=METRICAS_MODELO["f1"],
        )
        try:
            explicacao = llm_utils.gerar_explicacao(prompt, api_key=chave)
        except Exception as erro:
            explicacao = f"Não foi possível gerar a explicação: {erro}"

    return PredictResponse(predicao=predicao_texto, probabilidade=probabilidade, explicacao=explicacao)


# Adaptador para rodar como AWS Lambda (container image) — não interfere no uso local via uvicorn.
try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
