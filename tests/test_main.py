import base64
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.mistral_analyser import MistralAnalyser
from pprint import pprint

# --- DATASET_DIR file listing ---

def test_dataset_dir_lists_jpg_and_png_files():
    from main import DATASET_DIR
    files = sorted(DATASET_DIR.rglob("*.jpg")) + sorted(DATASET_DIR.rglob("*.png"))
    pprint(f"Found {len(files)} files in {DATASET_DIR}")
    for f in files:
        pprint(f"{f}")
    assert len(files) > 0, f"No images found in {DATASET_DIR}"
    assert all(f.suffix in (".jpg", ".png") for f in files)
    assert all(f.is_file() for f in files)


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
    mock_response.document_annotation = json.dumps({"id_doc": True, "document_id_type": "id card", "document_type": "french national id card"})

    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_analyser.Mistral", return_value=mock_client):
            result = MistralAnalyser().runner(str(img))

    assert result == AnalysisResult(id_doc=True, document_id_type="id card", document_type="french national id card")


def test_mistral_encodes_file_as_base64(tmp_path):
    content = b"sample-image-bytes"
    expected_b64 = base64.b64encode(content).decode("utf-8")
    img = tmp_path / "doc.jpg"
    img.write_bytes(content)

    mock_response = MagicMock()
    mock_response.document_annotation = json.dumps({"id_doc": False, "document_id_type": "not_identity_doc", "document_type": "invoice"})
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
        "document_id_type": {"valueString": "passport"},
        "document_type": {"valueString": "french passport"},
    }
    mock_result = MagicMock()
    mock_result.contents[0].fields = mock_fields
    mock_client.begin_analyze_binary.return_value.result.return_value = mock_result

    result = analyser.runner(str(img))

    assert result == AnalysisResult(id_doc=True, document_id_type="passport", document_type="french passport")


# --- MistralVisionAnalyser ---

def test_mistral_vision_returns_analysis_result(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_message = MagicMock()
    mock_message.content = json.dumps({"id_doc": True, "document_id_type": "passport", "document_type": "french passport"})
    mock_response = MagicMock()
    mock_response.choices[0].message = mock_message

    mock_client = MagicMock()
    mock_client.chat.complete.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("docs_analyser.mistral_vision_analyser.Mistral", return_value=mock_client):
            from docs_analyser.mistral_vision_analyser import MistralVisionAnalyser
            result = MistralVisionAnalyser().runner(str(img))

    assert result == AnalysisResult(id_doc=True, document_id_type="passport", document_type="french passport")
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
    mock_message.content = json.dumps({"id_doc": False, "document_id_type": "not_identity_doc", "document_type": "utility bill"})
    mock_response = MagicMock()
    mock_response.choices[0].message = mock_message
    mock_client.chat.completions.create.return_value = mock_response

    result = analyser.runner(str(img))

    assert result == AnalysisResult(id_doc=False, document_id_type="not_identity_doc", document_type="utility bill")
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4.1"


def test_azure_creates_analyzer_if_missing(tmp_path):
    from azure.core.exceptions import ResourceNotFoundError

    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_client = MagicMock()
    mock_client.get_analyzer.side_effect = ResourceNotFoundError()
    mock_fields = {
        "id_doc": {"valueBoolean": False},
        "document_id_type": {"valueString": "not_identity_doc"},
        "document_type": {"valueString": "invoice"},
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
    assert result == AnalysisResult(id_doc=False, document_id_type="not_identity_doc", document_type="invoice")