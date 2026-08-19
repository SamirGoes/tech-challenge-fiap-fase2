# Tech Challenge — Fase 2: Otimização via Algoritmo Genético + LLM

Projeto 1 do Tech Challenge Fase 2 (pós-graduação IADT/FIAP): otimização de modelos de diagnóstico de câncer de mama usando **Algoritmos Genéticos**, com integração de uma **LLM (Claude)** para interpretar os resultados em linguagem natural.

Este repositório dá continuidade ao trabalho da [Fase 1](https://github.com/Rafael-Evangelista/tech-challenge-fiap), que treinou 3 modelos (Regressão Linear, Regressão Logística, Random Forest) para classificar tumores de mama como malignos ou benignos, a partir do dataset Wisconsin Breast Cancer.

## O que este projeto faz

1. **Otimiza os hiperparâmetros dos 3 modelos da Fase 1** usando um Algoritmo Genético implementado do zero (`src/ga_utils.py`).
2. **Roda 3 experimentos** com configurações diferentes do GA (população/mutação/gerações), um por algoritmo.
3. **Compara** cada modelo original com sua versão otimizada, no mesmo holdout de teste.
4. **Integra a API da Anthropic (Claude)** para traduzir predições individuais em explicações de linguagem natural voltadas a profissionais de saúde (`src/llm_utils.py`).

## Estrutura

```
tech-challenge-fiap-fase2/
├── data/
│   └── data.csv                           # dataset Wisconsin Breast Cancer (569 amostras)
├── src/
│   ├── ga_utils.py                        # Algoritmo Genético: criar/mutar/fitness por algoritmo, crossover, seleção
│   └── llm_utils.py                       # Integração com a API Anthropic (Claude)
├── notebooks/
│   └── 03_ga_llm_breast_cancer.ipynb      # Orquestra os 3 experimentos, comparação e demonstração da LLM
├── tests/                                 # Testes automatizados (pytest)
├── experiments/results/                   # Saída dos experimentos: histórico de fitness, melhores hiperparâmetros, comparação final
├── docs/
│   ├── ARCHITECTURE.md                    # Diagramas de arquitetura (Mermaid), incluindo a nuvem
│   ├── GUIA_CONCEITOS.md                  # Glossário/FAQ dos conceitos de GA e prompt engineering
│   └── GA_EXPLICADO.md                    # Leitura linha a linha do código do GA, com exemplos
├── api/                                   # [opcional] API do modelo otimizado (FastAPI + Lambda)
│   ├── main.py
│   └── model/                             # modelo/scaler treinados, gerados por scripts/treinar_modelo_final.py
├── scripts/
│   └── treinar_modelo_final.py            # [opcional] treina e persiste o melhor modelo do GA
├── terraform/                             # [opcional] infraestrutura como código (AWS)
├── Dockerfile                             # [opcional] imagem da API para deploy em nuvem
├── RELATORIO_TECNICO.md                   # Relatório técnico da Fase 2
└── requirements.txt
```

## Como rodar

```bash
python -m pip install -r requirements.txt

# Necessário só para a seção de LLM do notebook — sem isso, os 3 experimentos
# do GA rodam normalmente, só não chama a API de verdade.
export ANTHROPIC_API_KEY="sua-chave-aqui"

jupyter notebook notebooks/03_ga_llm_breast_cancer.ipynb
```

### Testes automatizados

```bash
python -m pytest tests/ -v
```

## Resultados

Baseline (Fase 1) vs. otimizado pelo GA, no mesmo holdout de teste (20%, `random_state=42`):

| Modelo | Versão | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Regressão Linear | Original | 0.9649 | 1.0000 | 0.9048 | 0.9500 |
| Regressão Linear | **Otimizado (GA)** | 0.9649 | 0.9318 | **0.9762** | 0.9535 |
| Regressão Logística | Original | 0.9737 | 0.9756 | 0.9524 | 0.9639 |
| Regressão Logística | **Otimizado (GA)** | **0.9912** | **1.0000** | **0.9762** | **0.9880** |
| Random Forest | Original | 0.9737 | 1.0000 | 0.9286 | 0.9630 |
| Random Forest | Otimizado (GA) | 0.9649 | 1.0000 | 0.9048 | 0.9500 |

Detalhes de cada experimento (hiperparâmetros encontrados, curvas de convergência, discussão sobre o Random Forest não ter melhorado no holdout) estão no [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md).

## Implementação em nuvem (opcional)

O melhor modelo otimizado (Regressão Logística) é exposto como uma API (FastAPI) rodando em AWS Lambda com autoscaling automático, atrás de uma Function URL — arquitetura completa em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#implementação-em-nuvem-opcional-item-de-pontuação-extra), passos de deploy em [`terraform/README.md`](terraform/README.md).

```bash
# 1. Treinar e persistir o modelo (gera api/model/*.joblib)
python scripts/treinar_modelo_final.py

# 2. Testar a API localmente, sem AWS
export ANTHROPIC_API_KEY="sua-chave-aqui"
uvicorn api.main:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"features": {...}}'
```

## Entregáveis

- [x] Repositório Git com código, testes e documentação
- [x] Algoritmo Genético implementado (`src/ga_utils.py`) — 3 experimentos com configurações diferentes
- [x] Integração com LLM (`src/llm_utils.py`)
- [x] Testes automatizados (`tests/`)
- [x] Relatório técnico (`RELATORIO_TECNICO.md`)
- [x] Notebook de demonstração (`notebooks/03_ga_llm_breast_cancer.ipynb`)
- [x] **Opcional/extra:** Implementação em nuvem — API com autoscaling (`api/`, `Dockerfile`, `terraform/`)
- [ ] Vídeo de demonstração (até 15 min, YouTube/Vimeo)
