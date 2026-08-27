"""Leitura do CSV de pacientes para o endpoint de lote.

Aceita o mesmo formato do data.csv: colunas das 30 features,
e opcionalmente id e diagnosis (M/B).
"""
import csv
from io import StringIO


def _limpar_nome(nome):
    return (nome or "").strip().strip('"')


def interpretar_diagnosis(valor):
    if not valor:
        return None
    texto = valor.strip().upper()
    if texto in {"M", "MALIGNANT", "MALIGNO", "1"}:
        return "Maligno"
    if texto in {"B", "BENIGN", "BENIGNO", "0"}:
        return "Benigno"
    return None


def ler_pacientes_csv(texto, feature_names):
    """Devolve (pacientes, erros). Cada paciente tem linha, id, diagnosis e features."""
    leitor = csv.DictReader(StringIO(texto.lstrip("\ufeff")))
    if not leitor.fieldnames:
        return [], [{"linha": 1, "erro": "CSV sem cabeçalho"}]

    leitor.fieldnames = [_limpar_nome(nome) for nome in leitor.fieldnames]
    colunas = [nome for nome in leitor.fieldnames if nome]
    faltando_no_arquivo = [nome for nome in feature_names if nome not in colunas]
    if faltando_no_arquivo:
        return [], [
            {
                "linha": 1,
                "erro": f"Colunas faltando no CSV: {faltando_no_arquivo}",
            }
        ]

    pacientes = []
    erros = []
    for numero, linha in enumerate(leitor, start=2):
        linha = {_limpar_nome(chave): (valor or "").strip() for chave, valor in linha.items() if _limpar_nome(chave)}
        if not any(linha.values()):
            continue
        try:
            features = {nome: float(linha[nome]) for nome in feature_names}
        except (KeyError, ValueError) as erro:
            erros.append({"linha": numero, "erro": f"Valor inválido: {erro}"})
            continue
        pacientes.append(
            {
                "linha": numero,
                "id": linha.get("id") or None,
                "diagnosis": interpretar_diagnosis(linha.get("diagnosis")),
                "features": features,
            }
        )
    return pacientes, erros
