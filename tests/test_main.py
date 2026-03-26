import base64
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.mistral_analyser import MistralAnalyser


# --- MistralAnalyser ---

def test_mistral_raises_when_file_not_found():
    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_analyser.Mistral"):
            analyser = MistralAnalyser()
    with pytest.raises(FileNotFoundError):
        analyser.runner("/nonexistent/path/doc.jpg")


def test_mistral_raises_when_api_key_missing():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(KeyError):
            MistralAnalyser()


def test_mistral_returns_analysis_result(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_response = MagicMock()
    mock_response.document_annotation = json.dumps({"document_type": "id card", "id_doc": True})

    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_analyser.Mistral", return_value=mock_client):
            result = MistralAnalyser().runner(str(img))

    assert result == AnalysisResult(id_doc=True, document_type="id card")


def test_mistral_encodes_file_as_base64(tmp_path):
    content = b"sample-image-bytes"
    expected_b64 = base64.b64encode(content).decode("utf-8")
    img = tmp_path / "doc.jpg"
    img.write_bytes(content)

    mock_response = MagicMock()
    mock_response.document_annotation = json.dumps({"document_type": "other", "id_doc": False})
    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_analyser.Mistral", return_value=mock_client):
            MistralAnalyser().runner(str(img))

    call_kwargs = mock_client.ocr.process.call_args.kwargs
    assert expected_b64 in call_kwargs["document"]["image_url"]


# --- AzureAnalyser ---

def _make_azure_analyser():
    mock_client = MagicMock()
    mock_client.get_analyzer.return_value = MagicMock()
    env = {
        "CONTENTUNDERSTANDING_ENDPOINT": "https://fake.endpoint/",
        "CONTENTUNDERSTANDING_KEY": "fake-key",
    }
    with patch.dict("os.environ", env):
        with patch("docs_analyser.azure_analyser.ContentUnderstandingClient", return_value=mock_client):
            analyser = AzureAnalyser()
    return analyser, mock_client


def test_azure_raises_when_env_missing():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(KeyError):
            AzureAnalyser()


def test_azure_returns_analysis_result(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    analyser, mock_client = _make_azure_analyser()

    mock_fields = {
        "id_doc": {"valueBoolean": True},
        "document_type": {"valueString": "passport"},
    }
    mock_result = MagicMock()
    mock_result.contents[0].fields = mock_fields
    mock_client.begin_analyze_binary.return_value.result.return_value = mock_result

    result = analyser.runner(str(img))

    assert result == AnalysisResult(id_doc=True, document_type="passport")


def test_azure_creates_analyzer_if_missing(tmp_path):
    from azure.core.exceptions import ResourceNotFoundError

    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_client = MagicMock()
    mock_client.get_analyzer.side_effect = ResourceNotFoundError()
    mock_fields = {
        "id_doc": {"valueBoolean": False},
        "document_type": {"valueString": "other"},
    }
    mock_result = MagicMock()
    mock_result.contents[0].fields = mock_fields
    mock_client.begin_analyze_binary.return_value.result.return_value = mock_result

    env = {
        "CONTENTUNDERSTANDING_ENDPOINT": "https://fake.endpoint/",
        "CONTENTUNDERSTANDING_KEY": "fake-key",
    }
    with patch.dict("os.environ", env):
        with patch("docs_analyser.azure_analyser.ContentUnderstandingClient", return_value=mock_client):
            analyser = AzureAnalyser()

    mock_client.begin_create_analyzer.assert_called_once()
    result = analyser.runner(str(img))
    assert result == AnalysisResult(id_doc=False, document_type="other")