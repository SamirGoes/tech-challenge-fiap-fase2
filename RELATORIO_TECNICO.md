# Relatório Técnico — Fase 2: Otimização via Algoritmo Genético + LLM

**Projeto 1** — Otimização de modelos de diagnóstico com Algoritmos Genéticos, com integração de LLM para interpretação de resultados.

## 1. Contexto e objetivo

Na Fase 1, o notebook `notebooks/01_analise_e_modelagem.ipynb` treinou 3 modelos para classificar tumores de mama (maligno/benigno) a partir do dataset Wisconsin Breast Cancer: Regressão Linear, Regressão Logística e Random Forest, todos com hiperparâmetros escolhidos manualmente/por padrão do scikit-learn.

A Fase 2 tem dois objetivos:
1. Otimizar os hiperparâmetros dos 3 modelos usando um **Algoritmo Genético (GA)**, implementado em `src/ga_utils.py`.
2. Integrar uma **LLM (Claude, via API da Anthropic)** para traduzir predições individuais em explicações de linguagem natural para profissionais de saúde, implementado em `src/llm_utils.py`.

## 2. Implementação do Algoritmo Genético

### 2.1 Codificação (representação de genes)

Cada indivíduo é um cromossomo — um dicionário `{hiperparâmetro: valor}` — representando uma configuração completa de um modelo. Como os 3 algoritmos têm espaços de hiperparâmetros diferentes, cada um tem sua própria função `criar_individuo_*` em `src/ga_utils.py`:

| Algoritmo | Genes otimizados |
|---|---|
| Regressão Linear | `fit_intercept`, `positive`, `threshold` (limiar de classificação, ausente no modelo original — ver seção 4) |
| Regressão Logística | `C` (regularização, escala log), `penalty` (L1/L2), `class_weight` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion` |

O loop de gerações do GA está escrito diretamente nas células do notebook `03_ga_llm_breast_cancer.ipynb` (um bloco por algoritmo/experimento), chamando as funções de apoio de `ga_utils.py`: `criar_individuo_*`, `mutar_*`, `fitness_*` (específicas de cada algoritmo, porque os hiperparâmetros são diferentes) e `crossover`/`selecao_torneio` (compartilhadas entre os três, porque operam sobre o dict genérico sem precisar saber o que cada gene significa).

### 2.2 Operadores genéticos

- **Seleção**: torneio, k=3 — sorteia 3 indivíduos da população, o de maior fitness vence.
- **Crossover**: uniforme por gene — cada gene do filho vem aleatoriamente de um dos 2 pais.
- **Mutação**: por gene, com uma taxa configurável por experimento — reamostra o valor do gene dentro do seu espaço de busca.
- **Elitismo**: os 2 melhores indivíduos de cada geração sobrevivem intactos para a próxima.

### 2.3 Função de fitness

```
fitness = 0.5 · recall + 0.3 · F1 + 0.2 · accuracy
```

Calculada via `StratifiedKFold` (5 dobras) sobre o conjunto de treino. O recall pesa mais porque, em diagnóstico de câncer, um falso negativo (maligno classificado como benigno) é clinicamente mais grave que um falso positivo.

### 2.4 Os 3 experimentos (configurações diferentes)

Um experimento por algoritmo, cada um com uma configuração diferente do GA:

| Experimento | Algoritmo | População | Mutação | Gerações |
|---|---|---|---|---|
| 1 | Regressão Linear | 10 | 10% | 10 |
| 2 | Regressão Logística | 20 | 30% | 10 |
| 3 | Random Forest | 15 | 20% | 20 |

Hiperparâmetros encontrados pelo GA em cada experimento (execução registrada em `notebooks/03_ga_llm_breast_cancer.ipynb`, dados brutos em `experiments/results/`):

| Experimento | Melhores hiperparâmetros | Fitness (CV, treino) |
|---|---|---|
| 1 — Regressão Linear | `fit_intercept=True`, `positive=True`, `threshold≈0.337` | 0.9668 |
| 2 — Regressão Logística | `C≈0.0256`, `penalty=l2`, `class_weight=balanced` | 0.9648 |
| 3 — Random Forest | `n_estimators=147`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=2`, `max_features=sqrt`, `criterion=entropy` | 0.9635 |

A curva de convergência de cada experimento (melhor/média fitness por geração) está registrada nos gráficos do notebook 03 e nos arquivos `experiments/results/fitness_history_<algoritmo>.csv`.

## 3. Comparativo de desempenho: original vs. otimizado

Tabela gerada por `experiments/results/comparison_table.csv`, holdout de teste (20%, `random_state=42`), mesmo split usado no notebook 01:

| Modelo | Versão | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Regressão Linear | Original (limiar 0,5) | 0.9649 | 1.0000 | 0.9048 | 0.9500 |
| Regressão Linear | **Otimizado (GA)** | 0.9649 | 0.9318 | **0.9762** | 0.9535 |
| Regressão Logística | Original¹ | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| Regressão Logística | **Otimizado (GA)** | **0.9912** | **1.0000** | **0.9762** | **0.9880** |
| Random Forest | Original | 0.9737 | 1.0000 | 0.9286 | 0.9630 |
| Random Forest | Otimizado (GA) | 0.9649 | 1.0000 | 0.9048 | 0.9500 |

¹ *A Regressão Logística original recalculada aqui usa `solver='liblinear'` (necessário para o GA poder explorar tanto `penalty='l1'` quanto `'l2'`), diferente do `solver` padrão (`lbfgs`) usado no notebook 01 — por isso os números do baseline não batem exatamente com a tabela da Fase 1 (0.9649/0.9750/0.9286/0.9512). Ambos são o "mesmo modelo" com hiperparâmetros de otimização padrão, apenas com um solver numérico diferente.*

**Leitura dos resultados:**
- **Regressão Linear**: o GA trocou o limiar de 0,5 para ~0,34, sacrificando um pouco de precisão para aumentar bastante o recall (90,5% → 97,6%) — coerente com a fitness priorizar recall. Ganho real e explicável.
- **Regressão Logística**: melhora em todas as métricas simultaneamente (accuracy 97,4%→99,1%, recall 95,2%→97,6%, F1 0,964→0,988) — o melhor resultado entre os 3 experimentos, e por isso foi o modelo escolhido para a demonstração de interpretação com LLM no notebook.
- **Random Forest**: **o GA não superou o baseline neste holdout específico** — o indivíduo com melhor fitness na validação cruzada (147 árvores, `entropy`) teve accuracy/recall/F1 piores que o Random Forest original (100 árvores, parâmetros padrão) quando avaliado no conjunto de teste. Ver discussão na seção 4.

## 4. Desafios enfrentados e soluções

- **`LinearRegression` não tem hiperparâmetro de regularização**: diferente de Random Forest e Regressão Logística, o sklearn `LinearRegression` não expõe nada como `alpha`/`C` para o GA otimizar de forma significativa. Solução: adicionar o **limiar de classificação** (`threshold`, usado para converter a predição contínua em classe 0/1, fixo em 0,5 no notebook 01) como gene — é o parâmetro com maior impacto real no trade-off recall/precisão desse modelo.
- **3 algoritmos, 3 espaços de hiperparâmetros diferentes**: em vez de forçar uma abstração única para os três, cada algoritmo tem sua própria `criar_individuo_*`/`mutar_*`/`fitness_*` em `ga_utils.py` — só `crossover` e `selecao_torneio` são compartilhados, porque não dependem do significado de cada gene. Isso deixa o código um pouco repetitivo entre os três, mas cada bloco fica legível sozinho, sem indireção.
- **Dataset pequeno (569 amostras)**: risco de overfitting da busca do GA a um único split treino/validação. Solução: fitness calculado via validação cruzada estratificada (5-fold), não um único holdout.
- **Compatibilidade futura do scikit-learn**: a versão instalada (1.8) já emite aviso de depreciação para o parâmetro `penalty` da Regressão Logística (removido na 1.10). `requirements.txt` foi fixado em `scikit-learn>=1.3,<1.10` para evitar quebra futura sem precisar redesenhar o espaço de genes agora.
- **GA otimizando para a validação cruzada, não para o holdout final**: no experimento do Random Forest, o indivíduo com melhor fitness em validação cruzada (5-fold, treino) teve desempenho **pior** que o baseline original quando avaliado no holdout de teste (accuracy 96,5% vs. 97,4%, recall 90,5% vs. 92,9% — ver seção 3). É um lembrete real de que a fitness do GA é uma *estimativa* de generalização, não uma garantia — com um dataset pequeno (569 amostras, holdout de apenas 114), a variância entre CV e holdout pode ser grande o suficiente para inverter o ranking entre configurações próximas. Não "corrigimos" esse resultado artificialmente: reportamos como está, porque é um achado legítimo do experimento (o inverso ocorreu na Regressão Linear e na Regressão Logística, que melhoraram de fato).

## 5. Integração com LLM

### 5.1 Abordagem

Para cada predição individual (caso de teste), montamos um contexto estruturado — classe prevista, confiança, as features mais influentes daquele caso específico (via `feature_importances_` no Random Forest ou `coef_` nos modelos lineares, extraídas por `llm_utils.obter_features_importantes`) e as métricas gerais do modelo — e pedimos à LLM (Claude, `src/llm_utils.py`) uma explicação curta em português, sem jargão de ML, voltada a uma equipe médica.

O notebook seleciona automaticamente o **melhor modelo otimizado entre os 3** (maior recall no holdout, desempate por F1) para gerar as explicações de demonstração — na execução registrada, foi a **Regressão Logística otimizada** (recall 97,6%, F1 0,988, ver seção 3). As features mostradas ao usuário estão na escala padronizada (`StandardScaler`) usada para treinar o modelo, não na unidade de medida original — uma limitação conhecida a melhorar em uma próxima iteração (converter de volta à escala original antes de montar o prompt).

### 5.2 Prompt engineering

A função `montar_prompt` em `src/llm_utils.py`:
- Define um papel explícito para a LLM (assistente de interpretação clínica).
- Declara os limites da ferramenta ("NÃO substitui diagnóstico médico").
- Passa dados estruturados (não uma pergunta aberta) — ancora a resposta em fatos concretos do modelo.
- Especifica o formato de saída (3-5 frases, sem jargão técnico).

Ver `docs/GUIA_CONCEITOS.md` para o detalhamento de cada técnica.

### 5.3 Avaliação da qualidade das interpretações

Não existe métrica automática confiável para "qualidade de uma explicação médica em texto livre". Adotamos uma avaliação qualitativa manual com 3 critérios (nota 1-5): **clareza**, **correção clínica** e **utilidade acionável**, aplicada a um caso de acerto e um caso de erro do melhor modelo otimizado (rubric no notebook 03, seção de avaliação qualitativa).

Na execução registrada neste repositório, `ANTHROPIC_API_KEY` não estava configurada — o notebook montou e exibiu os dois prompts estruturados (caso de acerto e caso de erro) mas não chamou a API de verdade, apenas confirmando que a integração está pronta e funcional. _Ao rodar com a chave configurada, preencher aqui as notas atribuídas na tabela `avaliacao_qualitativa` do notebook e um breve comentário sobre a qualidade observada._

## 6. Arquitetura da solução

Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para os diagramas completos (visão geral, fluxo do GA, sequência da chamada à LLM).

## 7. Escopo não incluído nesta fase

- Implementação em nuvem / escalabilidade automática (item opcional do enunciado) — não implementado nesta entrega.
- GA aplicado à CNN de pneumonia ou mamografia — fora do escopo definido para este projeto (foco nos 3 modelos tabulares do notebook 01).
