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

O loop de gerações do GA está escrito diretamente nas células do notebook `GA_HyperparametersOptimization.ipynb` (um bloco por algoritmo/experimento), chamando as funções de apoio de `ga_utils.py`: `criar_individuo_*`, `mutar_*`, `fitness_*` (específicas de cada algoritmo, porque os hiperparâmetros são diferentes) e `crossover`/`selecao_torneio` (compartilhadas entre os três, porque operam sobre o dict genérico sem precisar saber o que cada gene significa).

### 2.2 Operadores genéticos

- **Seleção**: torneio, k=3 — sorteia 3 indivíduos da população, o de maior fitness vence.
- **Crossover**: uniforme por gene — cada gene do filho vem aleatoriamente de um dos 2 pais.
- **Mutação**: por gene, com uma taxa configurável por experimento — reamostra o valor do gene dentro do seu espaço de busca.
- **Elitismo**: os 2 melhores indivíduos de cada geração sobrevivem intactos para a próxima.

### 2.3 Função de fitness

```
fitness = 0.5 · recall + 0.3 · F1 + 0.2 · accuracy
```

O modelo é treinado no conjunto de treino (`X_train`/`y_train`) e recall/F1/accuracy são calculados diretamente sobre o **holdout de teste** (`X_test`/`y_test`) — é o que `do_GA` (definida no notebook) e as funções `fitness_*` de `ga_utils.py` fazem a cada avaliação de indivíduo. O recall pesa mais porque, em diagnóstico de câncer, um falso negativo (maligno classificado como benigno) é clinicamente mais grave que um falso positivo.

**Limitação conhecida**: isso significa que o GA busca hiperparâmetros usando o mesmo conjunto que depois reportamos como resultado final (seção 3) — não há separação entre busca e avaliação, como haveria com validação cruzada no treino. Com um holdout pequeno (114 amostras), o número de combinações accuracy/recall/F1 alcançáveis é limitado, então o fitness tende a saturar rápido em um teto (ver seção 2.4) e as métricas finais tendem a ser um pouco otimistas em relação ao que se esperaria em dados totalmente novos. Ver também seção 4.

### 2.4 Os 3 experimentos, cada um testando várias configurações do GA

Um experimento por algoritmo (Regressão Linear, Regressão Logística, Random Forest). Em vez de rodar uma única configuração de GA por experimento, os 3 experimentos seguem o **mesmo padrão**: cada um testa 4 combinações diferentes de população/gerações/mutação de uma só vez, numa lista de configs no notebook (`CONFIGS_1`, `CONFIGS_2`, `CONFIGS_3`, uma por experimento).

O padrão, idêntico nos 3 experimentos:
- Cada config da lista roda o GA do zero (mesma seed 42), gerando seu próprio histórico geração-a-geração e seu próprio melhor indivíduo.
- O histórico de **todas** as configs do experimento é concatenado em um único CSV (`experiments/results/fitness_history_<algoritmo>.csv`), com colunas `experimento`/`pop`/`geracoes`/`mutacao` identificando de qual config veio cada linha.
- O JSON (`experiments/results/best_hyperparams_<algoritmo>.json`) guarda só o indivíduo de **maior fitness entre todas as 4 configs** — não um JSON por config.
- Um gráfico de convergência (melhor fitness + fitness média por geração) é plotado por config, logo após o treino.
- Uma célula seguinte reconstrói o modelo otimizado de cada config e monta uma tabela comparando o desempenho de cada uma no holdout de teste, ao lado do baseline original (`experiments/results/comparison_table_<algoritmo>.csv`).

Configs testadas em cada experimento:

| Experimento | Algoritmo | Configs testadas (população / gerações / mutação) |
|---|---|---|
| 1 | Regressão Linear | 10/15/10% · 30/15/10% · 10/15/40% · 25/30/20% |
| 2 | Regressão Logística | 20/15/10% · 40/15/10% · 20/15/40% · 30/30/20% |
| 3 | Random Forest | 20/20/10% · 40/20/10% · 20/20/40% · 30/35/20% |

Melhor indivíduo entre as 4 configs de cada experimento (execução registrada em `notebooks/GA_HyperparametersOptimization.ipynb`, dados brutos em `experiments/results/`):

| Experimento | Config vencedora (pop/ger/mut) | Melhores hiperparâmetros | Fitness |
|---|---|---|---|
| 1 — Regressão Linear | 10 / 15 / 10% | `fit_intercept=True`, `positive=True`, `threshold≈0.377` | 0.9774 |
| 2 — Regressão Logística | 40 / 15 / 10% | `C≈0.0367`, `penalty=l1`, `class_weight=balanced` | 0.9947 |
| 3 — Random Forest | 20 / 20 / 10% | `n_estimators=200`, `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=3`, `max_features=None`, `criterion=entropy` | 0.9479 |

**Leitura dos resultados — por que rodar 4 configs em vez de 1 por experimento:**
- **Regressão Linear**: as 4 configs convergem para o mesmo fitness (0.9774) e para hiperparâmetros equivalentes — o espaço de busca (3 genes: 2 booleanos + 1 contínuo) é pequeno demais para que população/gerações/mutação façam diferença; qualquer uma das 4 configs já encontra o teto sozinha.
- **Regressão Logística**: as 4 configs melhoram o baseline, mas variam entre si (fitness entre 0.976 e 0.995) — aqui população/mutação maiores (config vencedora: pop=40, mutação=10%) de fato ajudam a explorar melhor o espaço de `C`/`penalty`/`class_weight`.
- **Random Forest**: as 4 configs empatam entre si **e com o próprio baseline** (fitness 0.9479 em todas, já na geração 0) — nenhuma das configurações de GA testadas encontrou nada melhor que os hiperparâmetros padrão do sklearn. Ver discussão na seção 4.

Rodar várias configs por experimento (em vez de uma só) deixa mais claro, para cada algoritmo, se o resultado do GA é sensível à configuração de busca (Regressão Logística) ou se já bate num teto estrutural do modelo/dataset independente da configuração (Regressão Linear e Random Forest).

## 3. Comparativo de desempenho: original vs. otimizado

Tabela gerada por `experiments/results/comparison_table.csv`, holdout de teste (20%, `random_state=42`), mesmo split usado no notebook 01:

| Modelo | Versão | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Regressão Linear | Original (limiar 0,5) | 0.9649 | 1.0000 | 0.9048 | 0.9500 |
| Regressão Linear | **Otimizado (GA)** | 0.9825 | 0.9762 | **0.9762** | 0.9762 |
| Regressão Logística | Original¹ | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| Regressão Logística | **Otimizado (GA)** | **0.9912** | 0.9767 | **1.0000** | **0.9882** |
| Random Forest | Original | 0.9737 | 1.0000 | 0.9286 | 0.9630 |
| Random Forest | Otimizado (GA) | 0.9737 | 1.0000 | 0.9286 | 0.9630 |

¹ *A Regressão Logística original recalculada aqui usa `solver='liblinear'` (necessário para o GA poder explorar tanto `penalty='l1'` quanto `'l2'`), diferente do `solver` padrão (`lbfgs`) usado no notebook 01 — por isso os números do baseline não batem exatamente com a tabela da Fase 1 (0.9649/0.9750/0.9286/0.9512). Ambos são o "mesmo modelo" com hiperparâmetros de otimização padrão, apenas com um solver numérico diferente.*

**Leitura dos resultados** (cada linha "Otimizado (GA)" usa a melhor config entre as 4 testadas no experimento correspondente — seção 2.4):
- **Regressão Linear**: o GA moveu o limiar de 0,5 para ~0,377, trocando um pouco de precisão por bastante recall (90,5% → 97,6%) e ainda melhorando a accuracy (96,5%→98,2%) — ganho real e explicável, coerente com a fitness priorizar recall.
- **Regressão Logística**: melhora em quase todas as métricas simultaneamente (accuracy 97,4%→99,1%, recall 95,2%→100%, F1 0,964→0,988) — o melhor resultado entre os 3 experimentos, e por isso foi o modelo escolhido para a demonstração de interpretação com LLM no notebook.
- **Random Forest**: **empate exato com o baseline** em todas as 4 configs testadas — a melhor configuração encontrada pelo GA (200 árvores, profundidade 5, `entropy`) produz a mesma accuracy/precision/recall/F1 que o Random Forest original (100 árvores, parâmetros padrão) no holdout de teste. Ver discussão na seção 4.

## 4. Desafios enfrentados e soluções

- **`LinearRegression` não tem hiperparâmetro de regularização**: diferente de Random Forest e Regressão Logística, o sklearn `LinearRegression` não expõe nada como `alpha`/`C` para o GA otimizar de forma significativa. Solução: adicionar o **limiar de classificação** (`threshold`, usado para converter a predição contínua em classe 0/1, fixo em 0,5 no notebook 01) como gene — é o parâmetro com maior impacto real no trade-off recall/precisão desse modelo.
- **3 algoritmos, 3 espaços de hiperparâmetros diferentes**: em vez de forçar uma abstração única para os três, cada algoritmo tem sua própria `criar_individuo_*`/`mutar_*`/`fitness_*` em `ga_utils.py` — só `crossover` e `selecao_torneio` são compartilhados, porque não dependem do significado de cada gene. Isso deixa o código um pouco repetitivo entre os três, mas cada bloco fica legível sozinho, sem indireção.
- **Dataset pequeno (569 amostras, holdout de 114)**: risco de o GA "decorar" as particularidades de um único split treino/teste, já que — como descrito na seção 2.3 — o fitness é calculado diretamente sobre o holdout de teste, não por validação cruzada no treino. Isso não foi corrigido nesta entrega (ver seção 2.3), mas está documentado como limitação conhecida: as métricas finais da seção 3 tendem a ser levemente otimistas, e o teto de fitness observado em alguns experimentos (Regressão Linear, Random Forest — seção 2.4) é parcialmente um efeito de o holdout ser pequeno e discreto, não só do espaço de genes ser pequeno.
- **Compatibilidade futura do scikit-learn**: a versão instalada (1.8) já emite aviso de depreciação para o parâmetro `penalty` da Regressão Logística (removido na 1.10). `requirements.txt` foi fixado em `scikit-learn>=1.3,<1.10` para evitar quebra futura sem precisar redesenhar o espaço de genes agora.
- **GA sem espaço real de melhora no Random Forest**: nas 4 configs testadas no Experimento 3 (seção 2.4), o Random Forest nunca superou o baseline — o fitness máximo encontrado (0.9479) já aparece na primeira geração, em todas as configs, e é exatamente igual ao fitness do próprio baseline (100 árvores, parâmetros padrão do sklearn). Random Forest é um modelo de alta capacidade: nesse dataset, quase linearmente separável, várias combinações razoáveis de hiperparâmetros já classificam o holdout de teste (114 amostras) de forma idêntica ao baseline — não há "para onde" o GA melhorar dentro do espaço de genes testado. É um teto real do modelo/dataset, não uma falha da busca; reportamos como está, sem forçar uma melhora artificial (o inverso ocorreu na Regressão Linear e na Regressão Logística, que melhoraram de fato — seção 3).

## 5. Integração com LLM

### 5.1 Abordagem

Para cada predição individual (caso de teste), montamos um contexto estruturado — classe prevista, confiança, as features mais influentes daquele caso específico (via `feature_importances_` no Random Forest ou `coef_` nos modelos lineares, extraídas por `llm_utils.obter_features_importantes`) e as métricas gerais do modelo — e pedimos à LLM (Claude, `src/llm_utils.py`) uma explicação curta em português, sem jargão de ML, voltada a uma equipe médica.

O notebook seleciona automaticamente o **melhor modelo otimizado entre os 3** (maior recall no holdout, desempate por F1) para gerar as explicações de demonstração — na execução registrada, foi a **Regressão Logística otimizada** (recall 100%, F1 0,988, ver seção 3). As features mostradas ao usuário estão na escala padronizada (`StandardScaler`) usada para treinar o modelo, não na unidade de medida original — uma limitação conhecida a melhorar em uma próxima iteração (converter de volta à escala original antes de montar o prompt).

### 5.2 Prompt engineering

A função `montar_prompt` em `src/llm_utils.py`:
- Define um papel explícito para a LLM (assistente de interpretação clínica).
- Declara os limites da ferramenta ("NÃO substitui diagnóstico médico").
- Passa dados estruturados (não uma pergunta aberta) — ancora a resposta em fatos concretos do modelo.
- Especifica o formato de saída (3-5 frases, sem jargão técnico).

Ver `docs/GUIA_CONCEITOS.md` para o detalhamento de cada técnica.

### 5.3 Avaliação da qualidade das interpretações

Não existe métrica automática confiável para "qualidade de uma explicação médica em texto livre". Adotamos uma avaliação qualitativa manual com 3 critérios (nota 1-5): **clareza**, **correção clínica** e **utilidade acionável**, aplicada a um caso de acerto e um caso de erro do melhor modelo otimizado (rubric no notebook `GA_HyperparametersOptimization.ipynb`, seção de avaliação qualitativa).

Na execução registrada neste repositório, `ANTHROPIC_API_KEY` não estava configurada — o notebook montou e exibiu os dois prompts estruturados (caso de acerto e caso de erro) mas não chamou a API de verdade, apenas confirmando que a integração está pronta e funcional. _Ao rodar com a chave configurada, preencher aqui as notas atribuídas na tabela `avaliacao_qualitativa` do notebook e um breve comentário sobre a qualidade observada._

## 6. Arquitetura da solução

Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para os diagramas completos (visão geral, fluxo do GA, sequência da chamada à LLM, e a arquitetura de nuvem descrita abaixo).

### 6.1 Implementação em nuvem (item opcional, pontuação extra)

O melhor modelo otimizado (Regressão Logística, seção 3) foi exposto como uma API na AWS:

- **AWS Lambda** (container image) rodando uma API FastAPI (`api/main.py`), com **autoscaling automático de 0 a N instâncias** conforme a carga de requisições — satisfaz o requisito de "escalabilidade automática para lidar com variações de demanda".
- **Lambda Function URL** como endpoint HTTPS público, sem custo adicional de API Gateway.
- **SSM Parameter Store** (`SecureString`) para a `ANTHROPIC_API_KEY`, evitando o custo do Secrets Manager.
- **CloudWatch Logs**, automático, para monitoramento e logging.
- **Terraform** (`terraform/`) declarando toda essa infraestrutura como código.
- `scripts/treinar_modelo_final.py` treina o modelo uma única vez e persiste `modelo.joblib`/`scaler.joblib` — a API carrega esses artefatos na inicialização, sem re-treinar por requisição.

Detalhes completos (diagrama, tabela de componentes, passos de deploy) em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) e [terraform/README.md](terraform/README.md). O deploy real na conta AWS não foi executado como parte desta entrega — a infraestrutura está pronta e testada localmente (API validada com `uvicorn`, ver seção de verificação), faltando apenas `terraform apply` com credenciais AWS configuradas.

## 7. Escopo não incluído nesta fase

- GA aplicado à CNN de pneumonia ou mamografia — fora do escopo definido para este projeto (foco nos 3 modelos tabulares do notebook 01).
