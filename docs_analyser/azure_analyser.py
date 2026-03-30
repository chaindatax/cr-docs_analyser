import os

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

from docs_analyser.base import AnalysisResult, Analyser, FIELD_DEFINITIONS

IDENTITY_ANALYZER_ID = "identityDocClassifier"
IDENTITY_ANALYZER_DEFINITION = {
    "description": "Classifies identity documents",
    "scenario": "documentIntelligence",
    "baseAnalyzerId": "prebuilt-document",
    "models": {"completion": "gpt-4.1", "embedding": "text-embedding-3-large"},
    "fieldSchema": {
        "fields": FIELD_DEFINITIONS
    },
}


class AzureAnalyser(Analyser):
    """Document analyser backed by Azure Content Understanding.

    Uses a custom analyzer (``identityDocClassifier``) built on top of
    ``prebuilt-document`` with ``gpt-4.1`` for field extraction and
    ``text-embedding-3-large`` for semantic chunking.

    The analyzer is created automatically on first instantiation and reused
    on subsequent calls.

    Requires the ``CONTENTUNDERSTANDING_ENDPOINT`` environment variable.
    ``CONTENTUNDERSTANDING_KEY`` is optional; if absent,
    :class:`~azure.identity.DefaultAzureCredential` is used instead.
    """

    def __init__(self):
        """Initialise the Azure client and ensure the custom analyzer exists."""
        endpoint = os.environ["CONTENTUNDERSTANDING_ENDPOINT"]
        key = os.getenv("CONTENTUNDERSTANDING_KEY")
        credential = AzureKeyCredential(key) if key else DefaultAzureCredential()
        self._client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)
        self._client.update_defaults(model_deployments={
            "completion": "gpt-4.1",
            "embedding": "text-embedding-3-large",
            "gpt-4.1": "gpt-4.1",
            "text-embedding-3-large": "text-embedding-3-large",
        })
        self._ensure_analyzer()

    def _ensure_analyzer(self):
        """Create the custom analyzer on the Azure resource if it does not exist."""
        try:
            self._client.get_analyzer(IDENTITY_ANALYZER_ID)
        except ResourceNotFoundError:
            print(f"Creating analyzer '{IDENTITY_ANALYZER_ID}'...")
            self._client.begin_create_analyzer(
                IDENTITY_ANALYZER_ID, IDENTITY_ANALYZER_DEFINITION, allow_replace=True
            ).result()

    def runner(self, file_path: str) -> AnalysisResult:
        """Analyse a document image using Azure Content Understanding.

        Submits the image to the ``identityDocClassifier`` analyzer and maps
        the returned fields to an :class:`AnalysisResult`.

        Args:
            file_path: Path to the image file to analyse.

        Returns:
            An :class:`AnalysisResult` populated from the analyzer's field extraction.
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        result = self._client.begin_analyze_binary(
            analyzer_id=IDENTITY_ANALYZER_ID,
            binary_input=file_bytes,
        ).result()

        fields = result.contents[0].fields
        return AnalysisResult(
            id_doc=fields.get("id_doc", {}).get("valueBoolean"),
            document_type=fields.get("document_type", {}).get("valueString"),
        )