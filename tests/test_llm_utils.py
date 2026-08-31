from unittest.mock import MagicMock, patch

import numpy as np
import llm_utils


def test_obter_features_importantes_ordena_por_importancia():
    class ModeloFake:
        feature_importances_ = np.array([0.1, 0.5, 0.2, 0.05])

    resultado = llm_utils.obter_features_importantes(
        ModeloFake(), ["a", "b", "c", "d"], [10, 20, 30, 40], top_n=2
    )
    assert [f["feature"] for f in resultado] == ["b", "c"]


def test_montar_prompt_contem_informacoes_principais():
    features = [{"feature": "area_worst", "valor": 1850.0, "importancia": 0.18}]
    prompt = llm_utils.montar_prompt(
        predicao="Maligno",
        chance_doenca=0.92,
        features_importantes=features,
        accuracy=0.978,
        recall=0.95,
        f1=0.97,
    )
    assert "MALIGNO" in prompt
    assert "92%" in prompt
    assert "área (extremo)" in prompt
    assert "97.8%" in prompt
    assert "Algoritmo Genético" in prompt


def test_gerar_explicacao_chama_api_mockada(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    bloco_texto = MagicMock(type="text", text="Explicação gerada.")
    resposta_falsa = MagicMock(content=[bloco_texto])

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = resposta_falsa
        mock_anthropic_cls.return_value = mock_client

        resultado = llm_utils.gerar_explicacao("prompt de teste")

    assert resultado == "Explicação gerada."
    mock_client.messages.create.assert_called_once()


def test_gerar_explicacao_sem_api_key_levanta_erro(monkeypatch):
    monkeypatch.setattr(llm_utils, "carregar_chave_do_env", lambda *a, **k: None)
    try:
        llm_utils.gerar_explicacao("prompt de teste", api_key=None)
        assert False, "deveria ter levantado RuntimeError"
    except RuntimeError:
        pass
