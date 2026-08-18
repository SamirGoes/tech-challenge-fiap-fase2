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
│   ├── ARCHITECTURE.md                    # Diagramas de arquitetura (Mermaid)
│   ├── GUIA_CONCEITOS.md                  # Glossário/FAQ dos conceitos de GA e prompt engineering
│   └── GA_EXPLICADO.md                    # Leitura linha a linha do código do GA, com exemplos
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

## Entregáveis

- [x] Repositório Git com código, testes e documentação
- [x] Algoritmo Genético implementado (`src/ga_utils.py`) — 3 experimentos com configurações diferentes
- [x] Integração com LLM (`src/llm_utils.py`)
- [x] Testes automatizados (`tests/`)
- [x] Relatório técnico (`RELATORIO_TECNICO.md`)
- [x] Notebook de demonstração (`notebooks/03_ga_llm_breast_cancer.ipynb`)
- [ ] Vídeo de demonstração (até 15 min, YouTube/Vimeo)
