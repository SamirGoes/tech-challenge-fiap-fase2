# Relatório Técnico — Fase 2: Otimização via Algoritmo Genético + LLM

**Projeto 1** — Otimização de um modelo de diagnóstico com Algoritmos Genéticos, com integração de LLM para interpretação de resultados.

## 1. Contexto e objetivo

Na Fase 1, o notebook `notebooks/01_analise_e_modelagem.ipynb` treinou 3 modelos para classificar tumores de mama (maligno/benigno) a partir do dataset Wisconsin Breast Cancer: Regressão Linear, Regressão Logística e Random Forest, todos com hiperparâmetros escolhidos manualmente/por padrão do scikit-learn. A Regressão Logística foi o modelo com melhor resultado otimizado nessa fase.

A Fase 2 tem dois objetivos:
1. Otimizar os hiperparâmetros da **Regressão Logística** usando um **Algoritmo Genético (GA)**, implementado em `src/ga_utils.py`, comparando 3 configurações diferentes de operadores do GA (em vez de repetir a otimização rasa nos 3 algoritmos da Fase 1, o foco aqui é uma comparação mais profunda de seleção/mutação em cima do modelo que já havia se destacado).
2. Integrar uma **LLM (Claude, via API da Anthropic)** para traduzir predições individuais em explicações de linguagem natural para profissionais de saúde, implementado em `src/llm_utils.py`.

## 2. Implementação do Algoritmo Genético

### 2.1 Codificação (representação de genes)

Cada indivíduo é um cromossomo — um dicionário `{hiperparâmetro: valor}` — representando uma configuração completa da Regressão Logística:

| Gene | Espaço de busca |
|---|---|
| `C` (regularização, inverso da força) | contínuo, `[0.01, 10]` |
| `penalty` | `l1` ou `l2` |
| `class_weight` | `None` ou `balanced` |

`src/ga_utils.py` também mantém o código de Random Forest e Regressão Linear (`criar_individuo_rf`/`criar_individuo_linear` etc.), reaproveitado da Fase 1, mas o notebook desta fase só executa o caminho da Regressão Logística.

O loop de gerações do GA é uma função só — `ga_utils.rodar_ga(algoritmo, ...)` — chamada uma vez por experimento no notebook `GA_HyperparametersOptimization.ipynb`. Ela despacha internamente para as funções de apoio do algoritmo escolhido (`criar_individuo_log`, `mutar_log`, `fitness_log`) e recebe como parâmetros a seleção (`crossover`/`selecao_torneio`/`selecao_roleta`, compartilhadas entre os algoritmos porque operam sobre o dict genérico sem precisar saber o que cada gene significa) e a taxa de mutação.

### 2.2 Operadores genéticos

- **Seleção**: torneio, k=3 (padrão) ou roleta (fitness-proportionate, Experimento 3 — ver 2.4).
- **Crossover**: uniforme por gene — cada gene do filho vem aleatoriamente de um dos 2 pais.
- **Mutação**: por gene, com uma taxa configurável por experimento — reamostra o valor do gene dentro do seu espaço de busca. Pode ser fixa ou adaptativa (decai linearmente geração a geração).
- **Elitismo**: os 2 melhores indivíduos de cada geração sobrevivem intactos para a próxima.

Os 3 experimentos (seção 2.4) isolam o efeito de cada operador: o Experimento 1 é a configuração padrão (torneio + mutação fixa), o Experimento 2 troca a mutação para adaptativa, e o Experimento 3 soma a seleção por roleta em cima da mutação adaptativa.

### 2.3 Função de fitness

```
fitness = 0.5 · recall + 0.3 · F1 + 0.2 · accuracy
```

O recall pesa mais porque, em diagnóstico de câncer, um falso negativo (maligno classificado como benigno) é clinicamente mais grave que um falso positivo.

Calculada via **`StratifiedKFold` de 5 dobras sobre o `X_train`**: para cada indivíduo, o modelo é treinado em 4 dobras e avaliado na dobra restante, repetindo 5 vezes; o fitness final é a média das 5 avaliações. **O conjunto de teste (`X_test`, 114 amostras) nunca é usado durante a busca do GA** — ele só entra depois, uma única vez, para avaliar o indivíduo já escolhido (seção 3).

> **Nota de metodologia**: a primeira versão desta função calculava o fitness diretamente no `X_test` a cada geração (apesar do nome `_fitness_cv` sugerir cross-validation) — um vazamento de dado clássico, já que o GA estaria otimizando diretamente a métrica do próprio holdout de teste, inflando o ganho reportado. Identificamos isso e reescrevemos a função para usar `StratifiedKFold` de verdade só no treino, com os números atualizados nas seções 3 e 4.

### 2.4 Os 3 experimentos (configurações diferentes)

Um experimento por configuração de operadores, todos na Regressão Logística, população 20 e 15 gerações:

| Experimento | Seleção | Mutação | Fitness final (CV, treino) |
|---|---|---|---|
| 1 — Padrão | Torneio (k=3) | Fixa, 20% | 0.9734 |
| 2 — Mutação adaptativa | Torneio (k=3) | Adaptativa, 20%→5% | 0.9678 |
| 3 — Mutação adaptativa + roleta | Roleta | Adaptativa, 20%→5% | 0.9635 |

Hiperparâmetros encontrados pelo GA em cada experimento:

| Experimento | Melhores hiperparâmetros |
|---|---|
| 1 — Padrão | `C≈0.218`, `penalty=l2`, `class_weight=balanced` |
| 2 — Mutação adaptativa | `C≈0.320`, `penalty=l1`, `class_weight=balanced` |
| 3 — Mutação adaptativa + roleta | `C≈0.481`, `penalty=l1`, `class_weight=balanced` |

A curva de convergência de cada experimento (melhor/média fitness por geração) está registrada nos gráficos do notebook e nos arquivos `experiments/results/fitness_history_*.csv`.

## 3. Comparativo de desempenho: original vs. otimizado

Tabela gerada por `experiments/results/comparison_table.csv`, holdout de teste (20%, `random_state=42`, nunca visto pelo GA):

| Versão | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Original (hiperparâmetros padrão do sklearn) | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| **Experimento 1 — Padrão (GA)** | **0.9825** | 0.9762 | **0.9762** | **0.9762** |
| Experimento 2 — Mutação adaptativa | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| Experimento 3 — Mutação adaptativa + roleta | 0.9737 | 0.9756 | 0.9524 | 0.9639 |

**Leitura dos resultados:**
- **Experimento 1 (configuração padrão do GA)** foi o único a melhorar o baseline no holdout: recall subiu de 95,2% para 97,6% e accuracy de 97,4% para 98,2% — um ganho real e modesto, coerente com a fitness priorizar recall. É o modelo escolhido para a demonstração de interpretação com LLM (seção 5).
- **Experimentos 2 e 3** convergiram para hiperparâmetros diferentes do baseline (e diferentes entre si), mas **empataram exatamente com o Original** no holdout de teste. Com um holdout pequeno (114 amostras), pequenas diferenças de fitness em validação cruzada podem não se traduzir em diferença de métrica no teste — cada erro a mais/a menos já move a métrica por ~0,9 ponto percentual, então configurações "quase tão boas" no treino podem aterrissar no mesmo número final.
- Nenhuma configuração piorou o baseline — diferente do que acontecia com o Random Forest na versão anterior deste projeto (3 algoritmos), que não é mais o escopo desta fase.

## 4. Desafios enfrentados e soluções

- **Vazamento de dados no fitness (achado principal)**: como descrito na seção 2.3, a função de fitness usava o `X_test` diretamente em vez de validação cruzada no treino. Isso foi identificado durante a revisão do notebook (o próprio nome `_fitness_cv` não correspondia ao que o código fazia) e corrigido para `StratifiedKFold` de 5 dobras, sem tocar no holdout durante a busca. Os números das seções 3 e 4 já refletem a versão corrigida.
- **Dataset pequeno (569 amostras, holdout de 114)**: com poucas amostras de teste, cada predição pesa bastante na métrica final, o que explica por que 2 dos 3 experimentos empataram exatamente com o baseline (seção 3) mesmo com hiperparâmetros diferentes. O fitness em validação cruzada (5-fold, mais amostras por dobra que um holdout único) é uma estimativa mais estável do que o resultado no holdout, mas ainda assim é uma estimativa — não uma garantia de ranking idêntico no teste final.
- **Compatibilidade futura do scikit-learn**: a versão instalada (1.8) já emite aviso de depreciação para o parâmetro `penalty` da Regressão Logística (removido na 1.10). `requirements.txt` foi fixado em `scikit-learn>=1.3,<1.10` para evitar quebra futura sem precisar redesenhar o espaço de genes agora.

## 5. Integração com LLM

### 5.1 Abordagem

Para cada predição individual, injetamos no prompt os fatos do caso — se o resultado é **positivo** (tem doença / maligno) ou **negativo** (não tem / benigno), a **porcentagem de chance de o tumor ser maligno**, as características que mais pesaram naquela decisão (explicabilidade via `|coef_|`, com nomes em português e valores na escala original) e um resumo do que o Algoritmo Genético escolheu para aquele modelo. A LLM (Claude) faz o papel de NLP: transforma esses números num **laudo em tom humano**, no estilo "nossa detecção automática deu 80% principalmente por causa do tamanho do tumor…".

O notebook usa o modelo do **Experimento 1** (melhor recall no holdout, ver seção 3) para gerar os laudos de demonstração.

### 5.2 Prompt engineering

Em `src/llm_utils.py` o prompt está separado em duas peças:
- **System prompt** (`SYSTEM_PROMPT`): define o papel (profissional explicando um exame), o tom humano, a obrigação de usar só os números passados e o aviso de que não substitui diagnóstico médico.
- **User prompt** (`montar_prompt`): só os dados daquele paciente + o contexto do GA. Não é uma pergunta aberta — ancora a resposta em fatos concretos e reduz alucinação.

Ver `docs/GUIA_CONCEITOS.md` para o detalhamento de cada técnica.

### 5.3 Avaliação da qualidade das interpretações

Não existe métrica automática confiável para "qualidade de uma explicação médica em texto livre". Adotamos uma avaliação qualitativa manual com 3 critérios (nota 1-5): **clareza**, **correção clínica** e **utilidade acionável**, aplicada ao laudo de demonstração e ao relatório dos experimentos gerados pelo notebook.

## 6. Arquitetura da solução

Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para os diagramas completos (visão geral, fluxo do GA, sequência da chamada à LLM, e a arquitetura de nuvem descrita abaixo).

### 6.1 Implementação em nuvem (item opcional, pontuação extra)

O modelo otimizado no Experimento 1 (seção 3) foi exposto como uma API na AWS:

- **AWS Lambda** (container image) rodando uma API FastAPI (`api/main.py`), com **autoscaling automático de 0 a N instâncias** conforme a carga de requisições — satisfaz o requisito de "escalabilidade automática para lidar com variações de demanda".
- **Lambda Function URL** como endpoint HTTPS público, sem custo adicional de API Gateway.
- **SSM Parameter Store** (`SecureString`) para a `ANTHROPIC_API_KEY`, evitando o custo do Secrets Manager.
- **CloudWatch Logs**, automático, para monitoramento e logging.
- **Terraform** (`terraform/`) declarando toda essa infraestrutura como código.
- `scripts/treinar_modelo_final.py` treina o modelo uma única vez e persiste `modelo.joblib`/`scaler.joblib` — a API carrega esses artefatos na inicialização, sem re-treinar por requisição.

Detalhes completos (diagrama, tabela de componentes, passos de deploy) em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) e [terraform/README.md](terraform/README.md). O deploy real na conta AWS não foi executado como parte desta entrega — a infraestrutura está pronta e testada localmente (API validada com `uvicorn`, ver seção de verificação), faltando apenas `terraform apply` com credenciais AWS configuradas.

## 7. Escopo não incluído nesta fase

- GA aplicado à CNN de pneumonia ou mamografia — fora do escopo definido para este projeto (foco no modelo tabular do notebook 01).
- Otimização por GA de Regressão Linear e Random Forest — feita em uma versão anterior deste projeto; o escopo atual foca em uma comparação mais aprofundada de operadores do GA (seleção, mutação) em cima da Regressão Logística, o modelo que já havia se destacado na Fase 1.
