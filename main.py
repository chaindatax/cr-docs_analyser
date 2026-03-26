import base64
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from pydantic import BaseModel


def mistral_ocr(file_path: str=None) -> BaseModel:
    from mistralai.client import Mistral
    from mistralai.client.models import JSONSchema, ResponseFormat

    if file_path is None:
        raise ValueError("file argument is required")

    api_key = os.environ["MISTRAL_API_KEY"]

    client = Mistral(api_key=api_key)

    def encode_file(file_path):
        with open(file_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')

    file_path = Path(__file__).parent / file_path
    base64_file = encode_file(file_path)

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

    return ocr_response.model_dump()

IDENTITY_ANALYZER_ID = "identityDocClassifier"
IDENTITY_ANALYZER_DEFINITION = {
    "description": "Classifies identity documents",
    "scenario": "documentIntelligence",
    "baseAnalyzerId": "prebuilt-document",
    "models": {"completion": "gpt-4.1", "embedding": "text-embedding-3-large"},
    "fieldSchema": {
        "fields": {
            "id_doc": {
                "type": "boolean",
                "description": "Is this document related to the identification of a person?",
            },
            "document_type": {
                "type": "string",
                "description": "Type of identity document, one of: 'id card', 'passport', 'other'",
            },
        }
    },
}


def azure_document_understanding(file_path: str = None) -> dict:
    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential

    if file_path is None:
        raise ValueError("file argument is required")

    endpoint = os.environ["CONTENTUNDERSTANDING_ENDPOINT"]
    key = os.getenv("CONTENTUNDERSTANDING_KEY")
    credential = AzureKeyCredential(key) if key else DefaultAzureCredential()

    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    client.update_defaults(model_deployments={
        "completion": "gpt-4.1",
        "embedding": "text-embedding-3-large",
        "gpt-4.1": "gpt-4.1",
        "text-embedding-3-large": "text-embedding-3-large",
    })

    try:
        client.get_analyzer(IDENTITY_ANALYZER_ID)
    except ResourceNotFoundError:
        print(f"Creating analyzer '{IDENTITY_ANALYZER_ID}'...")
        client.begin_create_analyzer(
            IDENTITY_ANALYZER_ID, IDENTITY_ANALYZER_DEFINITION, allow_replace=True
        ).result()

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    print(f"Analyzing {file_path}...")
    result = client.begin_analyze_binary(
        analyzer_id=IDENTITY_ANALYZER_ID,
        binary_input=file_bytes,
    ).result()

    fields = result.contents[0].fields
    return {
        "id_doc": fields.get("id_doc", {}).get("valueBoolean"),
        "document_type": fields.get("document_type", {}).get("valueString"),
    }

if __name__ == "__main__":
    load_dotenv()

    # pprint(main("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    pprint(azure_document_understanding("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    # pprint(main("dataset/passports/vias-pages-3783188661.jpg"))
    # pprint(main("dataset/passports/Ecp5OYUWAAAdVec-2854078686.jpg"))
    # pprint(mistral_ocr("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))
    # pprint(azure_document_understanding("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))

