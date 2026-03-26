import base64
import os

from mistralai.client import Mistral
from mistralai.client.models import JSONSchema, ResponseFormat


def mistral_ocr(file_path: str = None) -> dict:
    if file_path is None:
        raise ValueError("file argument is required")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    with open(file_path, "rb") as f:
        base64_file = base64.b64encode(f.read()).decode("utf-8")

    ocr_response = client.ocr.process(
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_file}"
        },
        model="mistral-ocr-latest",
        include_image_base64=True,
        document_annotation_format=ResponseFormat(
            type="json_schema",
            json_schema=JSONSchema(
                name="response_schema",
                schema_definition={
                    "properties": {
                        "document_type": {
                            "description": "describe the kind of document, one of this list: id card, passport, other",
                            "type": "string"
                        },
                        "id_doc": {
                            "description": "is this file a document related to identification of a person ?",
                            "type": "boolean"
                        }
                    },
                    "type": "object"
                },
                strict=True,
            ),
        )
    )

    return ocr_response.document_annotation
