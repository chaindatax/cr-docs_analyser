import base64
import json
import os

from mistralai.client import Mistral
from mistralai.client.models import JSONSchema, ResponseFormat

from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser, read_source_bytes


class MistralAnalyser(Analyser):
    """Document analyser backed by the Mistral OCR API.

    Encodes the image as base64 and submits it to ``mistral-ocr-latest``
    with a structured JSON schema to extract ``id_doc`` and ``document_type``.

    Requires the ``MISTRAL_API_KEY`` environment variable.
    """

    def __init__(self):
        """Initialise the Mistral client using ``MISTRAL_API_KEY``."""
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def runner(self, source: str) -> AnalysisResult:
        """Analyse a document image using Mistral OCR.

        Accepts either a local file path or an HTTPS URL (e.g. a blob SAS URL).
        PDFs are submitted as ``document_url``; images as ``image_url``.

        Args:
            source: Local path or HTTPS URL of the document to analyse.

        Returns:
            An :class:`AnalysisResult` populated from the model's JSON output.
        """
        raw = read_source_bytes(source)
        bare = source.split("?")[0].lower()
        if bare.endswith(".pdf"):
            b64 = base64.b64encode(raw).decode("utf-8")
            document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"}
        else:
            b64 = base64.b64encode(raw).decode("utf-8")
            document = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}

        response = self._client.ocr.process(
            document=document,
            model="mistral-ocr-latest",
            include_image_base64=False,
            document_annotation_format=ResponseFormat(
                type="json_schema",
                json_schema=JSONSchema(
                    name="response_schema",
                    schema_definition={
                        "type": "object",
                        "properties": FIELD_DEFINITIONS,
                    },
                    strict=True,
                ),
            ),
        )

        data = json.loads(response.document_annotation)
        return AnalysisResult(
            is_doc_id=data["is_doc_id"],
            id_doc_type=data["id_doc_type"],
            doc_type=data["doc_type"],
        )