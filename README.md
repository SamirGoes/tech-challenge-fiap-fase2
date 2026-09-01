# Tech Challenge — Fase 2: Otimização via Algoritmo Genético + LLM

Projeto 1 do Tech Challenge Fase 2 (pós-graduação IADT/FIAP): otimização de modelos de diagnóstico de câncer de mama usando **Algoritmos Genéticos**, com integração de uma **LLM (Claude)** para interpretar os resultados em linguagem natural.

Este repositório dá continuidade ao trabalho da [Fase 1](https://github.com/Rafael-Evangelista/tech-challenge-fiap), que treinou 3 modelos (Regressão Linear, Regressão Logística, Random Forest) para classificar tumores de mama como malignos ou benignos, a partir do dataset Wisconsin Breast Cancer.

## O grupo é composto por:
- Ana Clara
- Diego Justino
- Rafael Evangelista
- Samir Góes

## O que este projeto faz

1. **Otimiza os hiperparâmetros da Regressão Logística** (o modelo que se destacou na Fase 1) usando um Algoritmo Genético implementado do zero (`src/ga_utils.py`).
2. **Roda 3 experimentos**, comparando operadores diferentes do GA: seleção por torneio com mutação fixa, seleção por torneio com mutação adaptativa, e seleção por roleta com mutação adaptativa.
3. **Compara** o modelo original com cada versão otimizada, no mesmo holdout de teste — nunca visto pelo GA durante a busca (fitness calculado via `StratifiedKFold` no treino).
4. **Integra a API da Anthropic (Claude)** para gerar um laudo em linguagem natural (NLP): resultado positivo/negativo, chance de ter a doença e uma explicação humana das características que mais pesaram (`src/llm_utils.py`).

## Estrutura

```
tech-challenge-fiap-fase2/
├── data/
│   └── data.csv                           # dataset Wisconsin Breast Cancer (569 amostras)
├── src/
│   ├── ga_utils.py                        # Algoritmo Genético: criar/mutar/fitness por algoritmo, crossover, seleção
│   └── llm_utils.py                       # Integração com a API Anthropic (Claude)
├── notebooks/
│   └── GA_HyperparametersOptimization.ipynb  # Orquestra os 3 experimentos, comparação e demonstração da LLM
├── tests/                                 # Testes automatizados (pytest)
├── experiments/results/                   # Saída dos experimentos: histórico de fitness, melhores hiperparâmetros, comparação final
├── docs/
│   ├── ARCHITECTURE.md                    # Diagramas de arquitetura (Mermaid), incluindo a nuvem
├── frontend/                              # Interface Angular 22 (form /predict + relatório /predict/lote)
├── api/                                   # [opcional] API do modelo otimizado (FastAPI + Lambda)
│   ├── main.py
│   └── model/                             # modelo/scaler treinados, gerados por scripts/treinar_modelo_final.py
├── scripts/
│   ├── treinar_modelo_final.py            # [opcional] treina e persiste o melhor modelo do GA
│   └── treinar_modelo_simplificado.py     # [opcional] reexporta feature_importances.json sem reabrir o notebook
├── terraform/                             # [opcional] infraestrutura como código (AWS)
├── Dockerfile                             # [opcional] imagem da API para deploy em nuvem
├── RELATORIO_TECNICO.md                   # Relatório técnico da Fase 2
└── requirements.txt
```

## Como rodar

```bash
python -m pip install -r requirements.txt

# Para a seção de LLM: preencha a chave no arquivo .env (veja .env.example).
# Sem o .env, o GA roda normalmente — só não chama o Claude.

jupyter notebook notebooks/GA_HyperparametersOptimization.ipynb
```

### Testes automatizados

```bash
python -m pytest tests/ -v
```

## Resultados

Baseline (Regressão Logística com hiperparâmetros padrão) vs. as 3 configurações do GA, no mesmo holdout de teste (20%, `random_state=42`, nunca visto pelo GA):

| Versão | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Original | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| **Experimento 1 — Padrão (torneio + mutação fixa)** | **0.9912** | **0.9767** | **1.0000** | **0.9882** |
| Experimento 2 — Mutação adaptativa | 0.9912 | 0.9767 | 1.0000 | 0.9882 |
| Experimento 3 — Mutação adaptativa + roleta | 0.9912 | 0.9767 | 1.0000 | 0.9882 |

Detalhes de cada experimento, exame simplificado, LLM, API/front e links dos entregáveis estão no [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md).

## Implementação em nuvem (opcional)

O melhor modelo otimizado (Regressão Logística) é exposto como uma API (FastAPI) rodando em AWS Lambda com autoscaling automático, atrás de uma Function URL — arquitetura completa em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#implementação-em-nuvem-opcional-item-de-pontuação-extra), passos de deploy em [`terraform/README.md`](terraform/README.md).

**Aplicação publicada:** https://d3u4nxzfcqh14d.cloudfront.net/

```bash
# 1. Treinar e persistir o modelo (gera api/model/*.joblib)
python scripts/treinar_modelo_final.py
# O exame simplificado sai do notebook (seção Feature importances).
# Sem reabrir o Jupyter, dá para reexportar com:
python scripts/treinar_modelo_simplificado.py

# 2. Testar a API localmente, sem AWS (a chave sai do .env)
python -m uvicorn api.main:app --reload

# 3. Interface gráfica (outro terminal) — precisa da API no ar
cd frontend
npm start
# Abre http://localhost:4200
# Exame completo chama /predict. A aba "Exame simplificado" lê GET /predict/simplificado/features
# (lista vem do JSON exportado pelo notebook) e envia POST /predict/simplificado.
```

## Entregáveis

- [x] Repositório Git com código, testes e documentação
- [x] Algoritmo Genético implementado (`src/ga_utils.py`) — 3 experimentos com operadores diferentes (seleção, mutação)
- [x] Integração com LLM (`src/llm_utils.py`)
- [x] Testes automatizados (`tests/`)
- [x] Relatório técnico (`RELATORIO_TECNICO.md`)
- [x] Notebook de demonstração (`notebooks/GA_HyperparametersOptimization.ipynb`)
- [x] **Opcional/extra:** Implementação em nuvem — API com autoscaling (`api/`, `Dockerfile`, `terraform/`) — https://d3u4nxzfcqh14d.cloudfront.net/
- [x] Vídeo de demonstração (até 15 min): https://youtu.be/B7IVvdhWeQM
