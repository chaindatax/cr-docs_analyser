"""Document analyser backed by Azure OpenAI GPT-4.1 vision.

Unlike :mod:`azure_analyser` which uses the Content Understanding API, this
analyser sends the image directly to a ``gpt-4.1`` deployment via the Azure
OpenAI chat completion API with a structured JSON prompt.

Requires the ``AZURE_OPENAI_KEY`` and ``AZURE_OPENAI_ENDPOINT`` environment
variables, and a ``gpt-4.1`` model deployment in the target Azure AI Foundry
resource.
"""

import base64
import json
import os

from openai import AzureOpenAI

from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser, is_url


class AzureVisionAnalyser(Analyser):
    """Document analyser backed by Azure OpenAI GPT-4.1 vision.

    For local files the image is base64-encoded and sent as a data-URI.
    For HTTPS URLs (e.g. blob SAS URLs) the URL is passed directly so the
    model can fetch it, avoiding an in-memory download.

    Requires ``AZURE_OPENAI_KEY`` and ``AZURE_OPENAI_ENDPOINT`` environment
    variables.
    """

    def __init__(self):
        """Initialise the Azure OpenAI client.

        Uses ``AZURE_OPENAI_KEY`` and ``AZURE_OPENAI_ENDPOINT`` from the
        environment with API version ``2025-01-01-preview``.
        """
        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2025-01-01-preview",
        )

    def runner(self, source: str) -> AnalysisResult:
        """Analyse a document image using GPT-4.1 vision.

        Accepts either a local file path or an HTTPS URL (e.g. a blob SAS URL).
        Local files are base64-encoded; remote URLs are passed directly to the
        model.

        Args:
            source: Local path or HTTPS URL of the document to analyse.

        Returns:
            An :class:`~docs_analyser.base.AnalysisResult` populated from the
            model's JSON response.
        """
        if is_url(source):
            image_url = source
        else:
            with open(source, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            image_url = f"data:image/jpeg;base64,{b64}"

        response = self._client.chat.completions.create(
            model="gpt-4.1",  # nom du déploiement dans Azure AI Foundry
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Classify this document visually. "
                            "Reply ONLY with valid JSON: "
                            f"{json.dumps(FIELD_DEFINITIONS)}"
                        ),
                    },
                ],
            }],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        return AnalysisResult(
            is_doc_id=data["is_doc_id"],
            id_doc_type=data["id_doc_type"],
            doc_type=data["doc_type"],
        )
