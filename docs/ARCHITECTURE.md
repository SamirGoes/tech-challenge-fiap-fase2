# Arquitetura — Fase 2 (GA + LLM)

Este documento descreve como as peças da Fase 2 se encaixam: o dataset e os 3 modelos de diagnóstico (herdados da Fase 1), o Algoritmo Genético que otimiza os hiperparâmetros de cada um, e a integração com a LLM (Claude) que traduz predições em explicações para profissionais de saúde.

## Visão geral

```mermaid
flowchart LR
    A[data/data.csv] --> B[ga_utils.carregar_dados]
    B --> C1[GA — Regressão Linear]
    B --> C2[GA — Regressão Logística]
    B --> C3[GA — Random Forest]

    C1 --> D[Melhor indivíduo\nde cada experimento]
    C2 --> D
    C3 --> D

    D --> E[Reavaliação no holdout\nde teste]
    E --> F[experiments/results/\ncomparison_table.csv]

    E --> G[llm_utils.montar_prompt]
    G --> H[llm_utils.gerar_explicacao\nAnthropic API]
    H --> I[Explicação em\nlinguagem natural]
```

## Fluxo do Algoritmo Genético

O notebook `03_ga_llm_breast_cancer.ipynb` roda esse fluxo **uma vez por algoritmo** (3 experimentos, cada um com sua própria população/mutação/nº de gerações). O loop de gerações fica escrito direto na célula do notebook — não existe um "motor" único escondido num módulo separado, só as funções de apoio (`criar_individuo_*`, `mutar_*`, `fitness_*`, `crossover`, `selecao_torneio`) vêm de `src/ga_utils.py`.

```mermaid
flowchart TD
    Start([Início]) --> Init[Gerar população inicial\nN indivíduos aleatórios]
    Init --> Eval[Avaliar fitness de cada indivíduo\nStratifiedKFold 5-fold\n0.5·recall + 0.3·F1 + 0.2·accuracy]
    Eval --> Log[Guardar melhor/média fitness\nda geração]
    Log --> Check{Atingiu o nº\nde gerações?}
    Check -- não --> Elite[Elitismo: preserva\nos 2 melhores]
    Elite --> Select[Seleção por torneio\nk=3]
    Select --> Cross[Crossover uniforme\npor gene]
    Cross --> Mut[Mutação\npor gene]
    Mut --> NextGen[Nova geração]
    NextGen --> Eval
    Check -- sim --> Best[Melhor indivíduo\njá visto]
    Best --> End([Fim])
```

## Sequência da interpretação via LLM

```mermaid
sequenceDiagram
    participant NB as Notebook 03
    participant M as Modelo otimizado
    participant U as llm_utils.py
    participant API as API Anthropic (Claude)

    NB->>M: predict(caso de teste)
    M-->>NB: predição + probabilidade
    NB->>U: obter_features_importantes(modelo, ...)
    U-->>NB: top features do caso
    NB->>U: montar_prompt(contexto)
    U-->>NB: prompt estruturado
    NB->>U: gerar_explicacao(prompt)
    U->>API: messages.create(...)
    API-->>U: resposta em texto
    U-->>NB: explicação em português
```

## Por que cada algoritmo tem seu próprio código de GA

`ga_utils.py` **não** tem uma abstração única de "espaço de genes" compartilhada entre os 3 algoritmos — cada um tem sua função `criar_individuo_*` e `mutar_*` própria, porque os hiperparâmetros de Random Forest, Regressão Logística e Regressão Linear são bem diferentes entre si (categóricos vs. contínuos, ranges diferentes, e no caso da Regressão Linear um gene extra que nem é hiperparâmetro do sklearn — o `threshold`). Só o que é genuinamente igual entre os três — `crossover` (troca chaves de um dict) e `selecao_torneio` (compara uma lista de fitness) — é compartilhado, porque essas duas funções não precisam saber o que cada gene significa.

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `src/ga_utils.py` | `carregar_dados` (mesmo pré-processamento do notebook 01); `criar_individuo_*`/`mutar_*`/`fitness_*`/`construir_*` para cada um dos 3 algoritmos; `crossover` e `selecao_torneio` compartilhados; `calcular_metricas` |
| `src/llm_utils.py` | `obter_features_importantes` (feature_importances_ ou coef_); `montar_prompt`; `gerar_explicacao` (chama a API Anthropic) |
| `notebooks/03_ga_llm_breast_cancer.ipynb` | Orquestra tudo: carrega os dados, roda os 3 experimentos (loop de gerações visível na célula), compara com o baseline, gera as explicações via LLM |
