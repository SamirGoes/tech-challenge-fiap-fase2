"""NLP com LLM: transforma o resultado numérico do modelo (otimizado pelo GA)
em um laudo em linguagem natural, no tom de uma pessoa explicando o exame.

Usa a API da Anthropic (Claude). O system prompt fica separado da mensagem
com os dados do paciente — isso é a parte de prompt engineering.
A chave fica só no arquivo .env na raiz do projeto.
"""
from pathlib import Path

import numpy as np

# Nomes mais fáceis de ler do que radius_worst, area_mean, etc.
_MEDIDA = {
    "radius": "tamanho (raio)",
    "texture": "textura",
    "perimeter": "perímetro (contorno)",
    "area": "área",
    "smoothness": "suavidade da superfície",
    "compactness": "compactação",
    "concavity": "concavidade",
    "concave points": "pontos côncavos no contorno",
    "symmetry": "simetria",
    "fractal_dimension": "irregularidade do formato",
}
_ESTATISTICA = {
    "mean": "média",
    "se": "variação",
    "worst": "extremo",
}

NOMES_MODELO = {
    "regressao_linear": "Regressão Linear",
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
}

# System prompt do laudo de um paciente (usado na API, no front e no notebook).
SYSTEM_PROMPT = """Você é um profissional de saúde explicando o resultado de uma triagem automática de câncer de mama, como se estivesse conversando com um colega.

Escreva um laudo curto em português (4 a 7 frases), em tom humano — não robótico.

Regras:
- Comece dizendo se o resultado é POSITIVO (indica doença / tumor maligno) ou NEGATIVO (não indica doença / tumor benigno).
- Se for POSITIVO, informe a chance de o tumor ser maligno. Se for NEGATIVO, informe a chance de o tumor ser benigno. Use exatamente o número indicado para essa classe — não inverta e não fale da classe oposta.
- Explique o porquê com as características do tumor (tamanho, textura, formato). Fale como uma pessoa: "nossa detecção automática deu X% principalmente por causa do tamanho do tumor...".
- Use só os números e fatos que estão na mensagem. Não invente exames, sintomas nem valores.
- Deixe claro no final que isso é uma triagem automática e não substitui o diagnóstico médico.
- Não use jargão de machine learning (não fale em modelo, features, hiperparâmetros, algoritmo genético, fitness).
"""

# Relatório dos experimentos do GA (só o notebook). Tom de equipe, não de laudo individual.
SYSTEM_PROMPT_RELATORIO = """Você explica para a equipe médica o resultado de um experimento de triagem automática de câncer de mama.

Escreva um relatório curto em português (5 a 8 frases), claro, sem jargão de programação.

Regras:
- Diga qual abordagem ficou melhor depois da otimização e cite acurácia, recall e F1 exatamente como vieram.
- Compare com a versão original em linguagem simples (melhorou, ficou igual ou piorou).
- Cite as medidas do tumor que mais pesaram, usando os nomes amigáveis.
- Não invente números nem exames. Use só o que está na mensagem.
- Não faça diagnóstico de um paciente. Isto é um relatório do experimento.
"""


def nome_amigavel(feature):
    """Converte radius_worst -> 'tamanho (raio) (extremo)'."""
    if feature.endswith("_mean"):
        base, stat = feature[: -len("_mean")], "mean"
    elif feature.endswith("_se"):
        base, stat = feature[: -len("_se")], "se"
    elif feature.endswith("_worst"):
        base, stat = feature[: -len("_worst")], "worst"
    else:
        return feature
    medida = _MEDIDA.get(base, base.replace("_", " "))
    return f"{medida} ({_ESTATISTICA[stat]})"


def carregar_chave_do_env(caminho=None):
    """Lê a chave da Anthropic só do arquivo .env (nada de variável de ambiente)."""
    arquivo = Path(caminho) if caminho else Path(__file__).resolve().parents[1] / ".env"
    if not arquivo.is_file():
        return None
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        if linha.startswith("ANTHROPIC_API_KEY="):
            valor = linha.split("=", 1)[1].strip().strip('"').strip("'")
            if valor and "cole-sua-chave" not in valor:
                return valor
            return None
    return None


def obter_features_importantes(modelo, feature_names, valores_paciente, top_n=3):
    """Algoritmo simples de explicabilidade: pega as features que mais pesaram.

    Random Forest usa feature_importances_; modelos lineares usam o |coef_|.
    Os valores do paciente são só para mostrar no laudo (de preferência na
    escala original, não a padronizada).
    """
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_).ravel()
    else:
        importancias = np.ones(len(feature_names))

    idx_top = np.argsort(importancias)[::-1][:top_n]
    return [
        {
            "feature": feature_names[i],
            "nome": nome_amigavel(feature_names[i]),
            "valor": float(valores_paciente[i]),
            "importancia": float(importancias[i]),
        }
        for i in idx_top
    ]


def _formatar_gene(valor):
    if isinstance(valor, float):
        return f"{valor:.4g}"
    return str(valor)


def formatar_contexto_ga(nome_algoritmo, hiperparametros=None):
    """Texto curto com o resultado do GA, para injetar no prompt."""
    nome = NOMES_MODELO.get(nome_algoritmo, nome_algoritmo)
    if not hiperparametros:
        return (
            f"Este resultado veio do modelo {nome}, "
            "cujos hiperparâmetros foram escolhidos por um Algoritmo Genético."
        )
    genes = ", ".join(f"{k}={_formatar_gene(v)}" for k, v in hiperparametros.items())
    return (
        f"Este resultado veio do modelo {nome}, "
        f"cujos hiperparâmetros foram otimizados por um Algoritmo Genético: {genes}."
    )


def montar_prompt(
    predicao,
    chance_doenca,
    features_importantes,
    accuracy,
    recall,
    f1,
    contexto_ga=None,
):
    """Monta a mensagem do usuário com os dados deste paciente.

    predicao: "Maligno" ou "Benigno"
    chance_doenca: probabilidade de o tumor ser maligno (0 a 1)
    """
    tem_doenca = predicao.strip().lower().startswith("malign")
    chance_benigno = 1.0 - chance_doenca
    if tem_doenca:
        resultado = "POSITIVO"
        situacao = "indica doença — tumor classificado como MALIGNO"
        chance_laudo = chance_doenca
        classe_laudo = "maligno"
        exemplo = (
            '"Nossa detecção automática deu 80% de chance de o tumor ser maligno, '
            'principalmente por causa do tamanho do tumor e da irregularidade do contorno."'
        )
    else:
        resultado = "NEGATIVO"
        situacao = "não indica doença — tumor classificado como BENIGNO"
        chance_laudo = chance_benigno
        classe_laudo = "benigno"
        exemplo = (
            '"Nossa detecção automática deu 80% de chance de o tumor ser benigno, '
            'principalmente por causa do tamanho menor do tumor e do contorno mais regular."'
        )

    features_txt = "\n".join(
        f"- {f.get('nome', nome_amigavel(f['feature']))}: {f['valor']:.3f} "
        f"(peso nesta decisão: {f['importancia']:.3f})"
        for f in features_importantes
    )

    bloco_ga = contexto_ga or "Modelo de diagnóstico otimizado por Algoritmo Genético."

    return (
        "Dados deste paciente (use só o que está aqui, não invente):\n\n"
        f"- Resultado da triagem automática: {resultado} ({situacao})\n"
        f"- Chance que deve aparecer no laudo: {chance_laudo:.0%} de o tumor ser {classe_laudo}\n\n"
        "Características que mais pesaram nesta conclusão "
        "(explicabilidade do modelo):\n"
        f"{features_txt}\n\n"
        f"{bloco_ga}\n"
        f"Desempenho desse modelo nos testes: acurácia {accuracy:.1%}, "
        f"recall {recall:.1%}, F1 {f1:.2f}.\n\n"
        f"No texto, use {chance_laudo:.0%} de chance de o tumor ser {classe_laudo} "
        "(não inverta e não destaque a classe oposta).\n"
        f"Escreva o laudo em tom humano, como no exemplo: {exemplo}"
    )


def montar_prompt_relatorio(comparacao, melhor_algoritmo, hiperparametros, metricas, features_top):
    """Prompt do relatório dos experimentos (tabela GA + medidas mais importantes)."""
    if hasattr(comparacao, "to_dict"):
        linhas = comparacao.to_dict("records")
    else:
        linhas = comparacao

    tabela = "\n".join(
        f"- {linha['algoritmo']} ({linha['versao']}): "
        f"acurácia {linha['accuracy']:.1%}, recall {linha['recall']:.1%}, F1 {linha['f1']:.2f}"
        for linha in linhas
    )
    medidas = "\n".join(
        f"- {item.get('nome', nome_amigavel(item['feature']))}: peso {item['importancia']:.3f}"
        for item in features_top
    )
    nome = NOMES_MODELO.get(melhor_algoritmo, melhor_algoritmo)
    genes = ", ".join(f"{k}={_formatar_gene(v)}" for k, v in (hiperparametros or {}).items())
    return (
        "Resultados do experimento (use só o que está aqui):\n\n"
        f"Comparação original vs. otimizado:\n{tabela}\n\n"
        f"Vencedor: {nome}\n"
        f"Configuração escolhida: {genes or 'não informada'}\n"
        f"Holdout do vencedor: acurácia {metricas['accuracy']:.1%}, "
        f"recall {metricas['recall']:.1%}, F1 {metricas['f1']:.2f}.\n\n"
        f"Medidas do tumor que mais pesaram no vencedor:\n{medidas}\n"
    )


def gerar_explicacao(
    prompt,
    api_key=None,
    modelo="claude-haiku-4-5-20251001",
    max_tokens=500,
    system_prompt=SYSTEM_PROMPT,
):
    api_key = api_key or carregar_chave_do_env()
    if not api_key:
        raise RuntimeError(
            "Chave da Anthropic não encontrada. Preencha ANTHROPIC_API_KEY no arquivo "
            ".env na raiz do projeto (veja .env.example)."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resposta = client.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(bloco.text for bloco in resposta.content if bloco.type == "text").strip()
