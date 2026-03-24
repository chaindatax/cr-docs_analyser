import base64
from unittest.mock import MagicMock, mock_open, patch

import pytest

from main import mistral_ocr


def test_raises_when_no_filepath():
    with pytest.raises(ValueError, match="file argument is required"):
        mistral_ocr()


def test_raises_when_api_key_missing(tmp_path):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(KeyError):
            mistral_ocr(str(img))


def test_returns_ocr_response(tmp_path, monkeypatch):
    img = tmp_path / "doc.jpg"
    img.write_bytes(b"fake-image-data")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "pages": [{"index": 0, "markdown": "", "images": [], "dimensions": None}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1, "doc_size_bytes": None},
        "document_annotation": '{"document_type": "id card", "id_doc": true}',
    }

    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("main.Mistral", return_value=mock_client):
            with patch("main.Path") as mock_path:
                mock_path.return_value.__truediv__ = MagicMock(return_value=img)
                result = mistral_ocr(str(img))

    assert result["document_annotation"] == '{"document_type": "id card", "id_doc": true}'
    mock_client.ocr.process.assert_called_once()


def test_encodes_file_as_base64(tmp_path):
    content = b"sample-image-bytes"
    expected_b64 = base64.b64encode(content).decode("utf-8")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {}

    mock_client = MagicMock()
    mock_client.ocr.process.return_value = mock_response

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("main.Mistral", return_value=mock_client):
            with patch("builtins.open", mock_open(read_data=content)):
                mistral_ocr("doc.jpg")

    call_kwargs = mock_client.ocr.process.call_args.kwargs
    assert expected_b64 in call_kwargs["document"]["image_url"]
