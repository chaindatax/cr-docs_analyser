# Docs Analyser

Batch analysis of identity documents (ID cards, passports) using four independent AI backends — Mistral OCR, Mistral Vision (Pixtral), Azure Content Understanding, and Azure OpenAI Vision — with result comparison.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file at the project root:

```
MISTRAL_API_KEY=your_mistral_key

CONTENTUNDERSTANDING_ENDPOINT=https://your-resource.services.ai.azure.com/
CONTENTUNDERSTANDING_KEY=your_azure_key

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key
```

### Azure prerequisites

Both Azure analysers require models deployed in your Azure AI Foundry resource:

- `gpt-4.1` — field extraction (Content Understanding) and vision classification (Azure OpenAI)
- `text-embedding-3-large` — semantic document chunking (Content Understanding only)

## Usage

```bash
uv run main.py        # analyse all files in dataset/ and write results.csv
uv run pytest tests/  # run unit tests
uv run pytest tests/test_dataset.py -v  # run integration tests against real APIs
```

Place images (`.jpg` or `.png`) in `dataset/` before running. The output `results.csv` is written to the project root.

## Architecture

```
docs_analyser/
├── base.py                    # Analyser ABC + AnalysisResult dataclass
├── mistral_analyser.py        # MistralAnalyser       — Mistral OCR API
├── mistral_vision_analyser.py # MistralVisionAnalyser — Pixtral vision model
├── azure_analyser.py          # AzureAnalyser         — Azure Content Understanding
└── azure_vision_analyser.py   # AzureVisionAnalyser   — Azure OpenAI vision model
main.py                        # async batch runner, writes results.csv
tests/
├── test_main.py               # unit tests (mocked)
└── test_dataset.py            # integration tests (real API calls, skipped if keys absent)
```

### Base classes (`base.py`)

`Analyser` is an abstract base class with a single method:

```python
def runner(self, file_path: str) -> AnalysisResult
```

`AnalysisResult` is a dataclass with three fields:

| Field | Type | Description |
|---|---|---|
| `id_doc` | `bool` | Whether the document is an identity document |
| `document_id_type` | `str` | `"id card"`, `"passport"`, `"proof_of_residency"`, or `"not_identity_doc"` |
| `document_type` | `str` | Free-form document type as described by the model |

### MistralAnalyser

Encodes the image as base64 and calls the `mistral-ocr-latest` model with a structured JSON schema to extract all three fields from the document text content.

### MistralVisionAnalyser

Sends the image to `pixtral-12b-2409` via the chat completion API with a prompt requesting JSON output. Classifies the document visually without relying on OCR text extraction — works even when text is blurry or partially obscured.

### AzureAnalyser

Uses Azure Content Understanding with a custom analyzer (`identityDocClassifier`) built on top of `prebuilt-document`. The analyzer is created automatically on first run and reused on subsequent calls. Uses `gpt-4.1` for field extraction and `text-embedding-3-large` for document chunking.

### AzureVisionAnalyser

Sends the image to `gpt-4.1` (vision-capable) via Azure OpenAI chat completions with a prompt requesting JSON output. Like the Mistral vision analyser, classifies the document visually without OCR.

### Batch runner (`main.py`)

Processes all images in `dataset/` in batches of 10 using `asyncio`:

- Each batch runs files concurrently via `asyncio.gather`
- Within each file, all four analyser calls run in parallel via `asyncio.gather` + `asyncio.to_thread`
- Results are written to `results.csv` preserving the original file order

### Output CSV

| Column | Description |
|---|---|
| `file_path` | Relative path to the image |
| `filename` | Image filename |
| `mistral_id_doc` | `id_doc` from Mistral OCR |
| `mistral_document_id_type` | `document_id_type` from Mistral OCR |
| `mistral_document_type` | `document_type` from Mistral OCR |
| `azure_id_doc` | `id_doc` from Azure Content Understanding |
| `azure_document_id_type` | `document_id_type` from Azure Content Understanding |
| `azure_document_type` | `document_type` from Azure Content Understanding |
| `mistral_vision_id_doc` | `id_doc` from Mistral Vision |
| `mistral_vision_document_id_type` | `document_id_type` from Mistral Vision |
| `mistral_vision_document_type` | `document_type` from Mistral Vision |
| `azure_vision_id_doc` | `id_doc` from Azure OpenAI Vision |
| `azure_vision_document_id_type` | `document_id_type` from Azure OpenAI Vision |
| `azure_vision_document_type` | `document_type` from Azure OpenAI Vision |
| `aligned` | `True` if all four analysers agree on all three fields |