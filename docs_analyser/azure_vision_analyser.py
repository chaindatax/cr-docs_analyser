import base64, json, os
from openai import AzureOpenAI
from docs_analyser.base import FIELD_DEFINITIONS, AnalysisResult, Analyser


class AzureVisionAnalyser(Analyser):
    def __init__(self):
        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2025-01-01-preview",
        )

    def runner(self, file_path: str) -> AnalysisResult:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = self._client.chat.completions.create(
            model="gpt-4.1",  # nom du déploiement dans Azure AI Foundry
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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
