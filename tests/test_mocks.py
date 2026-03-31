import base64
import csv
import json
from pathlib import Path
from pprint import pprint
from unittest.mock import MagicMock, patch

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
    mock_response.document_annotation = json.dumps({"is_doc_id": True, "id_doc_type": "id card", "doc_type": "french national id card"})

    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_analyser.Mistral", return_value=mock_client):
            result = MistralAnalyser().runner(str(img))

    assert result == AnalysisResult(is_doc_id=True, id_doc_type="id card", doc_type="french national id card")


def test_mistral_encodes_file_as_base64(tmp_path):
    content = b"sample-image-bytes"
    expected_b64 = base64.b64encode(content).decode("utf-8")
    img = tmp_path / "doc.jpg"
    img.write_bytes(content)

    mock_response = MagicMock()
    mock_response.document_annotation = json.dumps({"is_doc_id": False, "id_doc_type": "not_identity_doc", "doc_type": "invoice"})
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
        "is_doc_id": {"valueBoolean": True},
        "id_doc_type": {"valueString": "passport"},
        "doc_type": {"valueString": "french passport"},
    }
    mock_result = MagicMock()
    mock_result.contents[0].fields = mock_fields
    mock_client.begin_analyze_binary.return_value.result.return_value = mock_result

    result = analyser.runner(str(img))

    assert result == AnalysisResult(is_doc_id=True, id_doc_type="passport", doc_type="french passport")


# --- MistralVisionAnalyser ---

def test_mistral_vision_returns_analysis_result(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_message = MagicMock()
    mock_message.content = json.dumps({"is_doc_id": True, "id_doc_type": "passport", "doc_type": "french passport"})
    mock_response = MagicMock()
    mock_response.choices[0].message = mock_message

    mock_client = MagicMock()
    mock_client.chat.complete.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_vision_analyser.Mistral", return_value=mock_client):
            from docs_analyser.mistral_vision_analyser import MistralVisionAnalyser
            result = MistralVisionAnalyser().runner(str(img))

    assert result == AnalysisResult(is_doc_id=True, id_doc_type="passport", doc_type="french passport")
    call_kwargs = mock_client.chat.complete.call_args.kwargs
    assert call_kwargs["model"] == "pixtral-12b-2409"


# --- AzureVisionAnalyser ---

def _make_azure_vision_analyser():
    mock_client = MagicMock()
    env = {
        "AZURE_OPENAI_KEY": "fake-key",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com/",
    }
    with patch.dict("os.environ", env):
        with patch("docs_analyser.azure_vision_analyser.AzureOpenAI", return_value=mock_client):
            from docs_analyser.azure_vision_analyser import AzureVisionAnalyser
            analyser = AzureVisionAnalyser()
    return analyser, mock_client


def test_azure_vision_returns_analysis_result(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    analyser, mock_client = _make_azure_vision_analyser()

    mock_message = MagicMock()
    mock_message.content = json.dumps({"is_doc_id": False, "id_doc_type": "not_identity_doc", "doc_type": "utility bill"})
    mock_response = MagicMock()
    mock_response.choices[0].message = mock_message
    mock_client.chat.completions.create.return_value = mock_response

    result = analyser.runner(str(img))

    assert result == AnalysisResult(is_doc_id=False, id_doc_type="not_identity_doc", doc_type="utility bill")
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4.1"


def test_azure_creates_analyzer_if_missing(tmp_path):
    from azure.core.exceptions import ResourceNotFoundError

    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_client = MagicMock()
    mock_client.get_analyzer.side_effect = ResourceNotFoundError()
    mock_fields = {
        "is_doc_id": {"valueBoolean": False},
        "id_doc_type": {"valueString": "not_identity_doc"},
        "doc_type": {"valueString": "invoice"},
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
    assert result == AnalysisResult(is_doc_id=False, id_doc_type="not_identity_doc", doc_type="invoice")


