"""Document analyser backed by the Mistral vision model (Pixtral).

Unlike :mod:`mistral_analyser` which uses the OCR API, this analyser sends
the image directly to ``pixtral-12b-2409`` via the chat completion API with a
structured JSON prompt.  This makes it usable for image formats that the OCR
endpoint may not support.

Requires the ``MISTRAL_API_KEY`` environment variable.
"""

import base64
import json
import os

from mistralai.client import Mistral

from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser, read_source_bytes


class MistralVisionAnalyser(Analyser):
    """Document analyser backed by the Mistral Pixtral vision model.

    Encodes the image as a base64 data-URI and submits it to
    ``pixtral-12b-2409`` via the chat completion API, requesting a JSON
    response that matches :data:`~docs_analyser.base.FIELD_DEFINITIONS`.

    Requires the ``MISTRAL_API_KEY`` environment variable.
    """

    def __init__(self):
        """Initialise the Mistral client using ``MISTRAL_API_KEY``."""
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def runner(self, source: str) -> AnalysisResult:
        """Analyse a document image using the Pixtral vision model.

        Accepts either a local file path or an HTTPS URL (e.g. a blob SAS URL).
        The file is downloaded in-memory and base64-encoded before being sent.

        Args:
            source: Local path or HTTPS URL of the document to analyse.

        Returns:
            An :class:`~docs_analyser.base.AnalysisResult` populated from the
            model's JSON response.

        Note:
            When ``doc_type`` is returned as a non-string value by the model
            (e.g. a JSON object), it is serialised back to a JSON string to
            keep :class:`~docs_analyser.base.AnalysisResult` consistent.
        """
        b64 = base64.b64encode(read_source_bytes(source)).decode()
        image_url_content = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}

        response = self._client.chat.complete(
            model="pixtral-12b-2409",
            messages=[{
                "role": "user",
                "content": [
                    image_url_content,
                    {
                        "type": "text",
                        "text": (
                            "Look at this image and classify the document. "
                            "Reply ONLY with valid JSON matching this schema: "
                            f"{json.dumps(FIELD_DEFINITIONS)}"
                        ),
                    },
                ],
            }],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        doc_type = data["doc_type"]
        return AnalysisResult(
            is_doc_id=data["is_doc_id"],
            id_doc_type=data["id_doc_type"],
            doc_type=json.dumps(doc_type) if not isinstance(doc_type, str) else doc_type,
        )
