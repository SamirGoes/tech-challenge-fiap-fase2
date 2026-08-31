# Arquitetura — Fase 2 (GA + LLM)

Este documento descreve como as peças da Fase 2 se encaixam: o dataset e o modelo de diagnóstico (Regressão Logística, herdado da Fase 1), o Algoritmo Genético que otimiza seus hiperparâmetros com 3 configurações de operadores diferentes, e a integração com a LLM (Claude) que traduz predições em explicações para profissionais de saúde.

## Visão geral

```mermaid
flowchart LR
    A[data/data.csv] --> B[ga_utils.carregar_dados]
    B --> C1[Experimento 1\nTorneio + mutação fixa]
    B --> C2[Experimento 2\nTorneio + mutação adaptativa]
    B --> C3[Experimento 3\nRoleta + mutação adaptativa]

    C1 --> D[Melhor indivíduo\nde cada experimento]
    C2 --> D
    C3 --> D

    D --> E[Reavaliação no holdout\nde teste — nunca visto pelo GA]
    E --> F[experiments/results/\ncomparison_table.csv]

    E --> G[llm_utils.montar_prompt]
    G --> H[llm_utils.gerar_explicacao\nAnthropic API]
    H --> I[Explicação em\nlinguagem natural]
```

## Fluxo do Algoritmo Genético

O notebook `GA_HyperparametersOptimization.ipynb` roda esse fluxo **3 vezes**, sempre na Regressão Logística, cada vez com uma configuração diferente de operadores: Experimento 1 (seleção por torneio + mutação fixa), Experimento 2 (torneio + mutação adaptativa) e Experimento 3 (roleta + mutação adaptativa). O loop de gerações é um único "motor" — `ga_utils.rodar_ga(algoritmo, ...)` — que cada célula de experimento chama passando o algoritmo e os parâmetros do GA; internamente ele despacha para as funções de apoio (`criar_individuo_*`, `mutar_*`, `fitness_*`) do algoritmo escolhido.

```mermaid
flowchart TD
    Start([Início]) --> Init[Gerar população inicial\nN indivíduos aleatórios]
    Init --> Eval["Avaliar fitness de cada indivíduo\nStratifiedKFold 5-fold, só no X_train\n0.5·recall + 0.3·F1 + 0.2·accuracy"]
    Eval --> Log[Guardar melhor/média fitness\nda geração]
    Log --> Check{Atingiu o nº\nde gerações?}
    Check -- não --> Elite[Elitismo: preserva\nos 2 melhores]
    Elite --> Select[Seleção\ntorneio k=3 ou roleta]
    Select --> Cross[Crossover uniforme\npor gene]
    Cross --> Mut[Mutação\npor gene, fixa ou adaptativa]
    Mut --> NextGen[Nova geração]
    NextGen --> Eval
    Check -- sim --> Best[Melhor indivíduo\njá visto]
    Best --> End([Fim])
```

O `X_test` (holdout) **não aparece nesse fluxo** — ele só é usado uma vez, depois do GA terminar, para avaliar o indivíduo vencedor (ver `RELATORIO_TECNICO.md`, seção 2.3, sobre por que isso importa).

## Sequência da interpretação via LLM

```mermaid
sequenceDiagram
    participant NB as Notebook
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

`ga_utils.py` **não** tem uma abstração única de "espaço de genes" compartilhada entre os 3 algoritmos — cada um tem sua função `criar_individuo_*` e `mutar_*` própria, porque os hiperparâmetros de Random Forest, Regressão Logística e Regressão Linear são bem diferentes entre si (categóricos vs. contínuos, ranges diferentes, e no caso da Regressão Linear um gene extra que nem é hiperparâmetro do sklearn — o `threshold`). O que é genuinamente igual entre os três — `crossover`, `selecao_torneio`, `selecao_roleta` — é compartilhado, porque essas funções não precisam saber o que cada gene significa. `rodar_ga` amarra tudo: recebe o nome do algoritmo e despacha para o `criar_individuo_*`/`fitness_*`/`mutar_*` certo via os dicts `CRIAR_INDIVIDUO`/`FITNESS`/`MUTAR`.

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `src/ga_utils.py` | `carregar_dados` (mesmo pré-processamento do notebook 01); `criar_individuo_*`/`mutar_*`/`fitness_*`/`construir_*` para cada um dos 3 algoritmos; `crossover`, `selecao_torneio`, `selecao_roleta` compartilhados; `rodar_ga` (motor do GA, genérico por algoritmo); `calcular_metricas` |
| `src/llm_utils.py` | `SYSTEM_PROMPT`; `obter_features_importantes` (explicabilidade); `formatar_contexto_ga`; `montar_prompt`; `gerar_explicacao` (API Anthropic, com system prompt) |
| `notebooks/GA_HyperparametersOptimization.ipynb` | Orquestra tudo: carrega os dados, roda os 3 experimentos via `ga_utils.rodar_ga`, compara com o baseline, gera as explicações via LLM |

## Implementação em nuvem (opcional, item de pontuação extra)

O item opcional do enunciado ("escalabilidade automática para lidar com variações de demanda") foi implementado expondo o melhor modelo otimizado (Regressão Logística, ver `RELATORIO_TECNICO.md`) como uma API na AWS.

```mermaid
flowchart TB
    U[Cliente / vídeo de demo] -->|HTTPS| FU[Lambda Function URL]
    FU --> LAMBDA["AWS Lambda\n(container image, autoscaling 0→N)"]
    LAMBDA --> MODEL["modelo.joblib + scaler.joblib\n(api/model/, embutidos na imagem)"]
    LAMBDA -->|lê chave| SSM["SSM Parameter Store\nSecureString: /fase2/anthropic-api-key"]
    LAMBDA -->|chama| ANTHROPIC[API Anthropic]
    LAMBDA -->|logs| CW[CloudWatch Logs]
    ECR["ECR\n(imagem Docker)"] -->|deploy| LAMBDA
```

### Por que essa stack

| Componente | Papel | Por quê |
|---|---|---|
| **AWS Lambda** (container image) | Roda a API (FastAPI via adaptador Mangum), escala de 0 a N instâncias por carga | Nível gratuito **permanente** (1M requisições + 400.000 GB-segundos/mês); é o recurso que satisfaz "escalabilidade automática" do enunciado |
| **Lambda Function URL** | Endpoint HTTPS público | Não precisa de API Gateway — gratuito, direto no Lambda |
| **SSM Parameter Store** (`SecureString`) | Guarda `ANTHROPIC_API_KEY` fora do código/imagem | Gratuito, ao contrário do Secrets Manager (~US$0,40/segredo/mês) |
| **ECR** | Armazena a imagem Docker versionada | Pré-requisito do Lambda em modo container |
| **CloudWatch Logs** | Logs automáticos de cada requisição | Satisfaz "monitoramento e logging adequados"; 5GB/mês grátis |
| **Terraform** (`terraform/`) | Declara toda a infraestrutura acima como código | "Infraestrutura como código" pedido no enunciado |

### Fluxo de uma requisição

1. Cliente faz `POST /predict` com os dados do paciente (30 features).
2. A Lambda já está com o modelo carregado em memória (`api/model/*.joblib`) — não treina nada em runtime.
3. O modelo prevê benigno/maligno, a chance de ter a doença (`chance_doenca` = P(maligno)) e se o resultado é positivo ou negativo.
4. `llm_utils.montar_prompt` injeta esses dados + o contexto do GA; `llm_utils.gerar_explicacao` chama a API Anthropic com o system prompt (chave lida do `.env` local, ou do SSM na AWS).
5. Resposta: `{"predicao": ..., "resultado": "positivo"|"negativo", "tem_doenca": ..., "chance_doenca": ..., "probabilidade": ..., "explicacao": ...}`.

### Controle de custo

- `reserved_concurrent_executions = 5` (variável `concorrencia_maxima` no Terraform) — limita o teto de instâncias simultâneas, evitando custo surpresa.
- Modelo Claude Haiku (padrão em `llm_utils.py`) — o mais barato da linha Anthropic.
- Tudo dentro do nível gratuito da AWS para o volume esperado de uma demonstração acadêmica.

Passos de deploy detalhados: [`terraform/README.md`](../terraform/README.md).
