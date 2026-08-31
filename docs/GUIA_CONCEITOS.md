# Guia de Conceitos — Fase 2 (GA + LLM)

Cola de estudo para explicar o projeto (vídeo de defesa, perguntas da banca, dúvidas do grupo). Não é a documentação técnica em si — para isso, ver [ARCHITECTURE.md](ARCHITECTURE.md) e o notebook [`GA_HyperparametersOptimization.ipynb`](../notebooks/GA_HyperparametersOptimization.ipynb).

Para uma leitura linha a linha do código do GA (o que cada gene pode valer, exemplos concretos de crossover/mutação, e o passo a passo do Experimento 1), ver [GA_EXPLICADO.md](GA_EXPLICADO.md).

## Algoritmo Genético (GA)

**O que é, em uma frase?** Uma busca inspirada em evolução biológica: em vez de testar hiperparâmetros manualmente, mantemos várias configurações candidatas e as fazemos "evoluir" ao longo de gerações, mantendo as melhores e descartando as piores.

| Termo | Analogia biológica | O que é neste projeto |
|---|---|---|
| **População** | Grupo de indivíduos de uma espécie | Conjunto de configurações de hiperparâmetros testadas numa geração |
| **Indivíduo / Cromossomo** | Um organismo | Uma configuração completa de hiperparâmetros de um modelo (ex.: `{n_estimators: 180, max_depth: 8, ...}`) |
| **Gene** | Um trecho de DNA que controla uma característica | Um hiperparâmetro específico (ex.: `n_estimators`) |
| **Fitness** | Capacidade de sobreviver e se reproduzir | O quão bom aquele modelo é: `0.5·recall + 0.3·F1 + 0.2·accuracy` |
| **Seleção** | Sobrevivência dos mais aptos | Escolher quais configurações viram "pais" da próxima geração (aqui: torneio, k=3) |
| **Crossover** | Reprodução sexuada, filhos herdam genes dos 2 pais | Combinar hiperparâmetros de 2 configurações "pai" para gerar uma nova |
| **Mutação** | Mutação genética aleatória | Trocar aleatoriamente o valor de um hiperparâmetro, com baixa probabilidade |
| **Elitismo** | — (conceito só de GA, sem analogia direta) | Garantir que as melhores configurações da geração atual sobrevivem intactas para a próxima |
| **Geração** | Uma geração de uma espécie | Uma iteração completa do loop: avaliar → selecionar → cruzar → mutar |

### Por que essas escolhas de design?

- **Por que fitness pesa mais o recall (50%)?** Em diagnóstico de câncer, um falso negativo (tumor maligno classificado como benigno) é o erro mais grave — atrasa um diagnóstico real. Um falso positivo (benigno classificado como maligno) custa exames extras, mas é um erro "mais seguro". Por isso o recall (capacidade de não deixar passar casos malignos) pesa mais que a acurácia geral.
- **Por que seleção por torneio, não roleta?** Roleta dá chance de reprodução proporcional ao fitness — mas se todos os indivíduos têm fitness parecido (comum depois de algumas gerações), a roleta vira quase um sorteio uniforme e perde força seletiva. Torneio (sortear k indivíduos, o melhor vence) continua discriminando bem mesmo nesse cenário, e é simples de ajustar (k maior = mais pressão seletiva).
- **Por que crossover uniforme, não de um ponto?** Crossover de um ponto corta o cromossomo em uma posição fixa e troca os pedaços — faz sentido quando genes vizinhos têm relação entre si (como em DNA real, ou problemas de rota/sequência). Aqui os hiperparâmetros não têm essa vizinhança natural, então o crossover uniforme (decide gene a gene, independentemente) evita vieses artificiais de ordem.
- **Por que 5-fold CV para o fitness, e não só um split treino/validação?** Com poucos dados (569 amostras), uma única divisão treino/validação pode dar uma estimativa de desempenho ruidosa — o GA poderia "aprender" a explorar uma coincidência daquele split específico. Validação cruzada estratificada em 5 dobras dá uma estimativa mais estável do desempenho real de cada configuração.

### Por hiperparâmetro otimizado

**Random Forest**
- `n_estimators`: quantas árvores compõem a floresta — mais árvores geralmente reduz variância, mas com retorno decrescente e custo computacional maior.
- `max_depth`: profundidade máxima de cada árvore — controla overfitting (árvores muito profundas decoram o treino).
- `min_samples_split` / `min_samples_leaf`: o quão fácil é uma árvore criar uma nova divisão — valores maiores geram árvores mais simples/generalistas.
- `max_features`: quantas features são consideradas em cada split — menos features por split aumenta a diversidade entre árvores.
- `criterion`: `gini` ou `entropy` — duas formas diferentes de medir o quão "pura" fica uma divisão.

**Regressão Logística**
- `C`: inverso da força de regularização — `C` menor = modelo mais regularizado/simples (evita overfitting, mas pode underfitar).
- `penalty`: tipo de regularização — L1 pode zerar coeficientes pouco úteis (seleção implícita de features), L2 só os encolhe.
- `class_weight`: se as classes recebem peso igual no treino ou se o modelo compensa o leve desbalanceamento do dataset (357 benignos vs. 212 malignos).

**Regressão Linear**
- `fit_intercept` / `positive`: opções estruturais do ajuste linear.
- `threshold`: como `LinearRegression` não tem regularização para o GA otimizar, o limiar que converte a predição contínua em classe 0/1 (fixo em 0,5 no notebook 01) vira o gene mais importante — desloca o equilíbrio entre recall e precisão.

## Prompt Engineering

**O que é, em uma frase?** A prática de estruturar deliberadamente a entrada de uma LLM (em vez de perguntar de forma solta) para obter respostas mais consistentes, corretas e no formato que você precisa.

**Técnicas usadas em `src/llm_utils.py`:**
- **System prompt separado da mensagem** (`SYSTEM_PROMPT`) — a LLM assume o papel de alguém explicando o exame em voz alta; o aviso de que **não substitui diagnóstico médico** fica aqui, não misturado com os números.
- **User prompt só com fatos** (`montar_prompt`) — resultado POSITIVO/NEGATIVO (tem / não tem a doença), porcentagem de chance de o tumor ser maligno, top features da explicabilidade e o texto do GA. Não perguntamos "o que você acha?".
- **Exemplo de tom** ("Nossa detecção automática deu 80% por causa do tamanho do tumor...") — ancora o estilo da resposta, para sair um laudo humano e não um relatório técnico.
- **Nomes amigáveis das features** (`nome_amigavel`) — `area_worst` vira "área (tamanho do tumor)", para a LLM falar de tamanho/textura/formato em vez do nome cru da coluna.

## Como interpretar as curvas de convergência dos 3 experimentos

Cada experimento roda com uma configuração diferente do GA (ver tabela no notebook 03). Ao olhar as curvas de "melhor fitness" e "fitness média" por geração:

- **Curva subindo e depois estabilizando** = convergência — o GA encontrou uma boa região do espaço de busca e parou de melhorar muito.
- **Curva de "fitness média" muito abaixo da "melhor fitness"** = a população ainda tem bastante diversidade (bom sinal de exploração, mas indica que ainda não convergiu totalmente).
- **Mutação alta (ex. experimento da Regressão Logística, 30%)** tende a deixar a curva média mais "ruidosa" — a exploração é maior, mas a convergência pode ser mais lenta.
- **População maior** tende a convergir para um resultado melhor com menos gerações (mais candidatos testados por geração), ao custo de mais tempo de computação por geração.
