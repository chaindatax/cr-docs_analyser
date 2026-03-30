import base64
import json
import os

from mistralai.client import Mistral
from mistralai.client.models import JSONSchema, ResponseFormat

from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser


class MistralAnalyser(Analyser):
    """Document analyser backed by the Mistral OCR API.

    Encodes the image as base64 and submits it to ``mistral-ocr-latest``
    with a structured JSON schema to extract ``id_doc`` and ``document_type``.

    Requires the ``MISTRAL_API_KEY`` environment variable.
    """

    def __init__(self):
        """Initialise the Mistral client using ``MISTRAL_API_KEY``."""
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def runner(self, file_path: str) -> AnalysisResult:
        """Analyse a document image using Mistral OCR.

        Reads the image, encodes it as base64, and calls the OCR API with
        a JSON schema that extracts ``id_doc`` and ``document_type``.

        Args:
            file_path: Path to the image file to analyse.

        Returns:
            An :class:`AnalysisResult` populated from the model's JSON output.
        """
        with open(file_path, "rb") as f:
            base64_file = base64.b64encode(f.read()).decode("utf-8")

        response = self._client.ocr.process(
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_file}",
            },
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
            id_doc=data["id_doc"],
            document_id_type=data["document_id_type"],
            document_type=data["document_type"],
        )