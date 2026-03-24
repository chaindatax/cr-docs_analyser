import base64
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.models import  JSONSchema, ResponseFormat
from pydantic import BaseModel


def mistral_ocr(filepath: str=None) -> BaseModel:
    if filepath is None:
        raise ValueError("file argument is required")

    api_key = os.environ["MISTRAL_API_KEY"]

    client = Mistral(api_key=api_key)

    def encode_file(file_path):
        with open(file_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')

    file_path = Path(__file__).parent / filepath
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

if __name__ == "__main__":
    load_dotenv()

    # pprint(main("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    # pprint(main("dataset/passports/vias-pages-3783188661.jpg"))
    # pprint(main("dataset/passports/Ecp5OYUWAAAdVec-2854078686.jpg"))
    pprint(mistral_ocr("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))

