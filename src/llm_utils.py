"""Funções para gerar explicações em linguagem natural das predições do
modelo, usando a API da Anthropic (Claude)."""
import os

import numpy as np

SYSTEM_CONTEXT = (
    "Você é um assistente que ajuda profissionais de saúde a interpretar resultados "
    "de um modelo de machine learning para triagem de câncer de mama. "
    "Este modelo é uma ferramenta de apoio à decisão, NÃO substitui diagnóstico médico."
)


def obter_features_importantes(modelo, feature_names, valores_paciente, top_n=3):
    """Pega as features que mais pesaram na predição: feature_importances_ no
    Random Forest, coef_ nos modelos lineares."""
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_).ravel()
    else:
        importancias = np.ones(len(feature_names))

    idx_top = np.argsort(importancias)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "valor": float(valores_paciente[i]), "importancia": float(importancias[i])}
        for i in idx_top
    ]


def montar_prompt(predicao, probabilidade, features_importantes, accuracy, recall, f1):
    features_txt = "\n".join(
        f"- {f['feature']}: {f['valor']:.3f} (importância: {f['importancia']:.3f})"
        for f in features_importantes
    )
    return (
        f"{SYSTEM_CONTEXT}\n\n"
        f"O modelo classificou este caso como: {predicao.upper()} "
        f"(confiança: {probabilidade:.0%})\n\n"
        "As características que mais influenciaram essa classificação foram:\n"
        f"{features_txt}\n\n"
        f"Desempenho geral do modelo: acurácia {accuracy:.1%}, "
        f"recall {recall:.1%}, F1 {f1:.2f}.\n\n"
        "Escreva uma explicação curta (3 a 5 frases), em português, para a equipe "
        "médica, sem jargão de machine learning, dizendo por que o modelo chegou "
        "nessa classificação e reforçando que é uma triagem, não um diagnóstico "
        "definitivo."
    )


def gerar_explicacao(prompt, api_key=None, modelo="claude-haiku-4-5-20251001", max_tokens=400):
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. Defina a variável de ambiente "
            "antes de chamar gerar_explicacao()."
        )

    client = anthropic.Anthropic(api_key=api_key)
    resposta = client.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(bloco.text for bloco in resposta.content if bloco.type == "text").strip()
