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

def azure_document_understanding(file_path: str=None) -> BaseModel | dict:
    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.ai.contentunderstanding.models import (
        AnalysisResult,
        DocumentContent,
    )
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential

    if file_path is None:
        raise ValueError("file argument is required")

    endpoint = os.environ["CONTENTUNDERSTANDING_ENDPOINT"]
    key = os.getenv("CONTENTUNDERSTANDING_KEY")
    credential = AzureKeyCredential(key) if key else DefaultAzureCredential()

    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    # [START analyze_document_from_binary]

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    print(f"Analyzing {file_path} with prebuilt-read...")
    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-read",
        binary_input=file_bytes,
    )
    result: AnalysisResult = poller.result()
    # [END analyze_document_from_binary]

    # [START extract_markdown]
    print("\nMarkdown Content:")
    print("=" * 50)

    # A PDF file has only one content element even if it contains multiple pages
    content = result.contents[0]
    print(content.markdown)

    print("=" * 50)

    return content.fields
    # [END extract_markdown]

    # [START access_document_properties]
    # Check if this is document content to access document-specific properties
    if isinstance(content, DocumentContent):
        print(f"\nDocument type: {content.mime_type or '(unknown)'}")
        print(f"Start page: {content.start_page_number}")
        print(f"End page: {content.end_page_number}")

        # Check for pages
        if content.pages and len(content.pages) > 0:
            print(f"\nNumber of pages: {len(content.pages)}")
            for page in content.pages:
                unit = content.unit or "units"
                print(f"  Page {page.page_number}: {page.width} x {page.height} {unit}")

        # Check for tables
        if content.tables and len(content.tables) > 0:
            print(f"\nNumber of tables: {len(content.tables)}")
            table_counter = 1
            for table in content.tables:
                print(
                    f"  Table {table_counter}: {table.row_count} rows x {table.column_count} columns"
                )
                table_counter += 1
    # [END access_document_properties]


    return None

if __name__ == "__main__":
    load_dotenv()

    # pprint(main("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    # pprint(main("dataset/passports/vias-pages-3783188661.jpg"))
    # pprint(main("dataset/passports/Ecp5OYUWAAAdVec-2854078686.jpg"))
    # pprint(mistral_ocr("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))
    pprint(azure_document_understanding("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))

