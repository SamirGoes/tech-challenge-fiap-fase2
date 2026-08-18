# GA na prática — do código à explicação, linha a linha

Este documento complementa o [`GUIA_CONCEITOS.md`](GUIA_CONCEITOS.md) (que é mais conceitual/glossário) com uma leitura detalhada do que o código de `src/ga_utils.py` faz de verdade — os valores possíveis de cada gene, o que cada operador (seleção, crossover, mutação) pode ou não produzir, e um passo a passo linha a linha do Experimento 1 (Regressão Linear) do notebook `03_ga_llm_breast_cancer.ipynb`.

## O que é um Algoritmo Genético (GA)

Um GA é uma técnica de otimização que busca uma boa solução simulando evolução biológica. Em vez de calcular a resposta diretamente, ele mantém um grupo de soluções candidatas ("população"), mede quão boa cada uma é ("fitness"), e usa 3 operadores — **seleção**, **crossover** e **mutação** — pra gerar novas gerações que tendem a ser melhores que as anteriores.

Aqui, cada "solução candidata" é uma configuração de hiperparâmetros de um modelo de ML, representada como um dict Python simples:

```python
{"fit_intercept": True, "positive": False, "threshold": 0.42}
```

Isso é o **indivíduo** (ou cromossomo). Cada chave do dict é um **gene**.

## 1. Representação — quais valores cada gene pode assumir

Definido nas funções `criar_individuo_*` — é aqui que fica o espaço de busca de cada algoritmo.

### Random Forest ([`ga_utils.py:92-100`](../src/ga_utils.py))
```python
def criar_individuo_rf():
    return {
        "n_estimators": random.randint(50, 300),
        "max_depth": random.choice([None, 3, 5, 8, 12, 16, 20]),
        "min_samples_split": random.randint(2, 20),
        "min_samples_leaf": random.randint(1, 10),
        "max_features": random.choice(["sqrt", "log2", None]),
        "criterion": random.choice(["gini", "entropy"]),
    }
```

| Gene | Tipo | Possibilidades |
|---|---|---|
| `n_estimators` | inteiro | qualquer valor de 50 a 300 |
| `max_depth` | categórico | `None`, `3`, `5`, `8`, `12`, `16` ou `20` (7 opções) |
| `min_samples_split` | inteiro | 2 a 20 |
| `min_samples_leaf` | inteiro | 1 a 10 |
| `max_features` | categórico | `"sqrt"`, `"log2"` ou `None` (3 opções) |
| `criterion` | categórico | `"gini"` ou `"entropy"` (2 opções) |

Espaço de busca total: 251 × 7 × 19 × 10 × 3 × 2 ≈ **2 milhões de combinações possíveis**.

### Regressão Logística ([`ga_utils.py:143-148`](../src/ga_utils.py))
```python
def criar_individuo_log():
    return {
        "C": 10 ** random.uniform(-3, 2),
        "penalty": random.choice(["l1", "l2"]),
        "class_weight": random.choice([None, "balanced"]),
    }
```

| Gene | Tipo | Possibilidades |
|---|---|---|
| `C` | contínuo (log-escala) | qualquer valor entre `10^-3 = 0.001` e `10^2 = 100` |
| `penalty` | categórico | `"l1"` ou `"l2"` (2 opções) |
| `class_weight` | categórico | `None` ou `"balanced"` (2 opções) |

`10 ** random.uniform(-3, 2)` merece destaque: sorteia o **expoente** entre -3 e 2 (uniformemente), depois eleva. Isso dá chance igual a `C` cair perto de 0.001, perto de 1, ou perto de 100 — se fosse `random.uniform(0.001, 100)` direto, quase todo sorteio cairia acima de 50 (é onde está a maior parte do intervalo numérico), e valores pequenos de `C` quase nunca apareceriam.

### Regressão Linear ([`ga_utils.py:184-189`](../src/ga_utils.py))
```python
def criar_individuo_linear():
    return {
        "fit_intercept": random.choice([True, False]),
        "positive": random.choice([True, False]),
        "threshold": random.uniform(0.3, 0.7),
    }
```

| Gene | Tipo | Possibilidades |
|---|---|---|
| `fit_intercept` | booleano | `True` ou `False` |
| `positive` | booleano | `True` ou `False` |
| `threshold` | contínuo | qualquer valor entre 0.3 e 0.7 |

## 2. Seleção — `selecao_torneio` ([`ga_utils.py:246-249`](../src/ga_utils.py))

```python
def selecao_torneio(populacao, fitnesses, k=3):
    k = min(k, len(populacao))
    indices = random.sample(range(len(populacao)), k)
    melhor_idx = max(indices, key=lambda i: fitnesses[i])
    return dict(populacao[melhor_idx])
```

Aqui as "possibilidades" não são valores de gene, é *quem pode virar pai*. Qualquer indivíduo tem chance de ser sorteado pro torneio (`random.sample` escolhe `k=3` índices ao acaso, sem reposição), mas só o de maior fitness entre os 3 sorteados vence. Um indivíduo com fitness baixo só vira pai se, por sorte, só entrar em torneios contra indivíduos ainda piores — raro, mas possível. É assim que o torneio dá **preferência** aos melhores sem ser 100% determinístico.

Exemplo com população de fitness `[0.6, 0.9, 0.3, 0.5, 0.8]`: se o torneio sortear os índices `[0, 2, 3]` (fitness 0.6, 0.3, 0.5), o vencedor é o índice `0` — mesmo sem ser o melhor da população inteira, porque os outros 2 concorrentes *desse torneio específico* eram piores.

## 3. Crossover — `crossover` ([`ga_utils.py:236-243`](../src/ga_utils.py))

```python
def crossover(pai1, pai2):
    filho1, filho2 = {}, {}
    for chave in pai1:
        if random.random() < 0.5:
            filho1[chave], filho2[chave] = pai1[chave], pai2[chave]
        else:
            filho1[chave], filho2[chave] = pai2[chave], pai1[chave]
    return filho1, filho2
```

Pra cada gene, o filho1 herda ou o valor do pai1 ou do pai2 (50/50), e o filho2 sempre fica com o valor "oposto" (os dois filhos juntos preservam todo o material genético dos pais, sem duplicar nem perder nada). **Nenhum valor novo é criado aqui** — só recombina o que já existe.

```python
pai1 = {"fit_intercept": True,  "positive": False, "threshold": 0.35}
pai2 = {"fit_intercept": False, "positive": True,  "threshold": 0.60}

# random.random() sorteia, por gene: 0.2, 0.8, 0.1
# 0.2 < 0.5 -> fit_intercept: filho1 pega do pai1
# 0.8 >= 0.5 -> positive: filho1 pega do pai2
# 0.1 < 0.5 -> threshold: filho1 pega do pai1

filho1 = {"fit_intercept": True,  "positive": True,  "threshold": 0.35}
filho2 = {"fit_intercept": False, "positive": False, "threshold": 0.60}
```

`filho1["threshold"]` só pode ser `0.35` ou `0.60` — nunca um valor intermediário como `0.47`. Isso é o que diferencia crossover de mutação.

## 4. Mutação — `mutar_*`

Aqui sim entram valores **novos**, fora do que os pais tinham. A lógica é igual nos 3 algoritmos, só muda o conjunto de genes. Exemplo com a Regressão Logística ([`ga_utils.py:151-159`](../src/ga_utils.py)):

```python
def mutar_log(individuo, taxa=0.3):
    novo = dict(individuo)
    if random.random() < taxa:
        novo["C"] = 10 ** random.uniform(-3, 2)
    if random.random() < taxa:
        novo["penalty"] = random.choice(["l1", "l2"])
    if random.random() < taxa:
        novo["class_weight"] = random.choice([None, "balanced"])
    return novo
```

As possibilidades de mutação são exatamente as mesmas de `criar_individuo_log` — quando um gene sofre mutação, ele é resorteado do **espaço de busca inteiro**, não ajustado a partir do valor atual. Um `C=0.05` pode virar `C=87.3` numa única mutação (salto grande), não um ajuste fino tipo `0.05 → 0.06`.

```python
filho = {"C": 0.05, "penalty": "l1", "class_weight": None}

# taxa=0.3, sorteia 3 números: 0.15, 0.82, 0.41
# 0.15 < 0.3 -> C sofre mutação: novo valor sorteado do zero, ex. 12.7
# 0.82 >= 0.3 -> penalty NÃO muda
# 0.41 >= 0.3 -> class_weight NÃO muda

resultado = {"C": 12.7, "penalty": "l1", "class_weight": None}
```

## 5. Fitness — a "nota" de cada indivíduo

```python
return 0.5 * np.mean(recs) + 0.3 * np.mean(f1s) + 0.2 * np.mean(accs)
```

Recall, F1 e accuracy calculados em 5-fold CV (ver [`_fitness_cv`](../src/ga_utils.py) pra RF/Logística, ou o loop equivalente dentro de `fitness_linear` que aplica o `threshold` antes de medir as métricas). Fitness não gera possibilidades novas — é a função de avaliação que diz quais indivíduos merecem ser pais (via seleção) e quais sobrevivem por elitismo.

## Quem realmente constrói a próxima geração

```python
nova_populacao = [dict(populacao[i]) for i in ranking[:2]]  # elitismo
while len(nova_populacao) < POP_1:
    pai1 = ga_utils.selecao_torneio(populacao, fitnesses)
    pai2 = ga_utils.selecao_torneio(populacao, fitnesses)
    filho1, filho2 = ga_utils.crossover(pai1, pai2)
    nova_populacao.append(ga_utils.mutar_linear(filho1, MUTACAO_1))
    nova_populacao.append(ga_utils.mutar_linear(filho2, MUTACAO_1))
```

A `nova_populacao` é 100% composta por 4 fontes: **elitismo** (2 vagas, cópia exata dos melhores) + **torneio → crossover → mutação** (8 vagas, repetido até completar). Fitness não cria nada sozinha — ela só dá a "nota" que os outros operadores usam para ter direção.

Cada peça isolada não seria suficiente:
- **Só fitness + elitismo**: a população nunca mudaria, ficaria travada nos mesmos indivíduos iniciais.
- **Só crossover, sem mutação**: o GA fica limitado a recombinar o que já existia na população inicial — nunca alcança um valor que nenhum indivíduo inicial tinha.
- **Só mutação, sem seleção**: vira busca aleatória pura, sem aproveitar o que já foi aprendido.
- **Só torneio, sem crossover/mutação**: a população encolhe pro mesmo indivíduo repetido, perdendo toda diversidade.

| Operador | O que faz sozinho | Por que precisa dos outros |
|---|---|---|
| **Fitness** | Mede quão bom é cada indivíduo | Não cria nada — só dá a "nota" que os outros usam |
| **Seleção (torneio)** | Decide quem tem mais chance de virar pai | Sem crossover/mutação, não geraria nada novo |
| **Crossover** | Recombina genes dos pais em filhos | Sem mutação, fica limitado ao que já existia na população |
| **Mutação** | Injeta valores genuinamente novos | Sem seleção, não teria direção — seria busca aleatória |
| **Elitismo** | Garante que o melhor não se perde | Não gera variação — só protege o que já foi achado |

## Passo a passo — Experimento 1 (Regressão Linear), linha por linha

Código exato da célula do notebook (`notebooks/03_ga_llm_breast_cancer.ipynb`):

```python
random.seed(42)
```
Fixa a semente do gerador aleatório — sem isso, cada execução do notebook sortearia coisas diferentes. Com a seed fixa, o resultado é **reprodutível**.

```python
POP_1, GERACOES_1, MUTACAO_1 = 10, 10, 0.1
```
População de 10 indivíduos, 10 gerações, taxa de mutação de 10%.

```python
populacao = [ga_utils.criar_individuo_linear() for _ in range(POP_1)]
```
Gera a população inicial: 10 dicts aleatórios.

```python
historico_linear = []
```
Lista que vai guardar um resumo (melhor/média fitness) de cada geração, pra plotar o gráfico de convergência e salvar em CSV depois.

```python
for geracao in range(GERACOES_1):
```
Início do loop principal — roda 10 vezes.

```python
    fitnesses = [ga_utils.fitness_linear(ind, X_train, y_train) for ind in populacao]
```
O passo mais caro: avalia a fitness de cada um dos 10 indivíduos (treina + valida em 5-fold CV cada um). Resultado: lista de 10 floats na mesma ordem da `populacao`.

```python
    historico_linear.append({"geracao": geracao, "melhor": max(fitnesses), "media": float(np.mean(fitnesses))})
    print(f"Geração {geracao}: melhor={max(fitnesses):.4f}  média={np.mean(fitnesses):.4f}")
```
Guarda e imprime o progresso dessa geração.

```python
    ranking = sorted(range(len(populacao)), key=lambda i: fitnesses[i], reverse=True)
```
Ordena os **índices** (não os indivíduos diretamente) por fitness decrescente — preserva a ligação entre indivíduo e seu fitness sem precisar zipar as duas listas.

```python
    nova_populacao = [dict(populacao[i]) for i in ranking[:2]]  # elitismo
```
Pega os 2 melhores índices e faz uma **cópia** (`dict(...)`) de cada um — cópia é importante pra não acidentalmente compartilhar o mesmo objeto entre gerações.

```python
    while len(nova_populacao) < POP_1:
        pai1 = ga_utils.selecao_torneio(populacao, fitnesses)
        pai2 = ga_utils.selecao_torneio(populacao, fitnesses)
        filho1, filho2 = ga_utils.crossover(pai1, pai2)
        nova_populacao.append(ga_utils.mutar_linear(filho1, MUTACAO_1))
        if len(nova_populacao) < POP_1:
            nova_populacao.append(ga_utils.mutar_linear(filho2, MUTACAO_1))
```
Preenche as 8 vagas restantes: escolhe 2 pais por torneio na população **atual** (a antiga), cruza, muta os dois filhos, adiciona. O `if` interno evita passar do tamanho `POP_1` caso sobre só 1 vaga.

```python
    populacao = nova_populacao
```
Fim do `while`: a população antiga é descartada, a nova vira a população da próxima iteração do `for`.

```python
melhor_linear = populacao[int(np.argmax(fitnesses))]
```
Fora do loop `for`, depois das 10 gerações: acha o índice do maior valor na última lista de fitness calculada, pegando o melhor indivíduo já avaliado.

```python
pd.DataFrame(historico_linear).to_csv(RESULTADOS_DIR / "fitness_history_regressao_linear.csv", index=False)
with open(RESULTADOS_DIR / "best_hyperparams_regressao_linear.json", "w") as f:
    json.dump({"algoritmo": "regressao_linear", "melhor_individuo": melhor_linear}, f, indent=2, default=str)
```
Salva o histórico de convergência em CSV e o melhor resultado em JSON, em `experiments/results/`.

```python
plt.figure(figsize=(8, 4))
plt.plot([h["geracao"] for h in historico_linear], [h["melhor"] for h in historico_linear], label="Melhor fitness", marker="o")
plt.plot([h["geracao"] for h in historico_linear], [h["media"] for h in historico_linear], label="Fitness média", marker="o")
plt.xlabel("Geração"); plt.ylabel("Fitness"); plt.title("Convergência do GA — Regressão Linear (pop=10, mutação=10%)")
plt.legend(); plt.tight_layout(); plt.show()
```
Gráfico de convergência: melhor e média fitness por geração, lado a lado — mostra visualmente se a população está convergindo (linhas se aproximando) ou ainda diversa (linhas afastadas).

### Célula seguinte — comparação com o baseline

```python
modelo_linear_original = ga_utils.construir_linear(ga_utils.LINEAR_BASELINE)
modelo_linear_original.fit(X_train, y_train)
y_pred_original = ga_utils.prever_linear(modelo_linear_original, X_test, ga_utils.LINEAR_BASELINE)
metricas_linear_original = ga_utils.calcular_metricas(y_test, y_pred_original)
```
Reconstrói o modelo **original** (config fixa do notebook 01, `threshold=0.5`), treina no treino inteiro (agora sem CV — é avaliação final), prevê no teste e calcula métricas contra `y_test`.

```python
modelo_linear_otimizado = ga_utils.construir_linear(melhor_linear)
modelo_linear_otimizado.fit(X_train, y_train)
y_pred_otimizado = ga_utils.prever_linear(modelo_linear_otimizado, X_test, melhor_linear)
metricas_linear_otimizado = ga_utils.calcular_metricas(y_test, y_pred_otimizado)
```
O mesmo processo, com `melhor_linear` (o dict que o GA encontrou) no lugar do baseline.

```python
pd.DataFrame([metricas_linear_original, metricas_linear_otimizado], index=["Original", "Otimizado (GA)"])
```
Junta as 2 métricas numa tabela de 2 linhas — mostra com números concretos se o GA melhorou o modelo (recall foi de 90,5% pra 97,6% nesse experimento).
