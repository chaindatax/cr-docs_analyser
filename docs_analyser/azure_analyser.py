import os

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

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