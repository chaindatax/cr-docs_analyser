import base64, json, os
from mistralai.client import Mistral
from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser


class MistralVisionAnalyser(Analyser):
    def __init__(self):
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def runner(self, file_path: str) -> AnalysisResult:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = self._client.chat.complete(
            model="pixtral-12b-2409",
            # ou "mistral-small-latest" (aussi vision)
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    },
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
        return AnalysisResult(id_doc=data["id_doc"], document_type=data["document_type"])