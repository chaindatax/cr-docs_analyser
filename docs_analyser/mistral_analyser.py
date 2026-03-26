import base64
import json
import os

from mistralai.client import Mistral
from mistralai.client.models import JSONSchema, ResponseFormat

from docs_analyser.base import AnalysisResult, Analyser


class MistralAnalyser(Analyser):
    def __init__(self):
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def runner(self, file_path: str) -> AnalysisResult:
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
                        "properties": {
                            "document_type": {
                                "type": "string",
                                "description": "Type of document: 'id card', 'passport', or 'other'",
                            },
                            "id_doc": {
                                "type": "boolean",
                                "description": "Is this document related to the identification of a person?",
                            },
                        },
                    },
                    strict=True,
                ),
            ),
        )

        data = json.loads(response.document_annotation)
        return AnalysisResult(
            id_doc=data["id_doc"],
            document_type=data["document_type"],
        )