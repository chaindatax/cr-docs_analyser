# Docs Analyser

Batch analysis of identity documents (ID cards, passports) using two independent AI backends — Mistral OCR and Azure Content Understanding — with result comparison.

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
```

### Azure prerequisites

The Azure analyser requires the following models deployed in your Azure AI Foundry resource:

- `gpt-4.1` — used for field extraction
- `text-embedding-3-large` — used for semantic document chunking

## Usage

```bash
uv run main.py        # analyse all files in dataset/ and write results.csv
uv run pytest tests/  # run tests
```

Place images (`.jpg` or `.png`) in `dataset/` before running. The output `results.csv` is written to the project root.

## Architecture

```
docs_analyser/
├── base.py              # Analyser (abstract base class) + AnalysisResult (dataclass)
├── mistral_analyser.py  # MistralAnalyser — uses Mistral OCR API
└── azure_analyser.py    # AzureAnalyser  — uses Azure Content Understanding
main.py                  # batch runner
tests/
└── test_main.py
```

### Base classes (`base.py`)

`Analyser` is an abstract base class with a single method:

```python
def runner(self, file_path: str) -> AnalysisResult
```

`AnalysisResult` is a dataclass with two fields:

| Field | Type | Values |
|---|---|---|
| `id_doc` | `bool` | Whether the document is an identity document |
| `document_type` | `str` | `"id card"`, `"passport"`, or `"other"` |

### MistralAnalyser

Encodes the image as base64 and calls the `mistral-ocr-latest` model with a structured JSON schema. The schema instructs the model to extract `id_doc` and `document_type` directly from the image content.

### AzureAnalyser

Uses Azure Content Understanding with a custom analyzer (`identityDocClassifier`) built on top of `prebuilt-document`. The analyzer is created automatically on first run and reused on subsequent calls. It uses `gpt-4.1` for field extraction and `text-embedding-3-large` for document chunking.

### Batch runner (`main.py`)

Processes all images in `dataset/` in batches of 10 using `asyncio`:

- Each batch runs files concurrently via `asyncio.gather`
- Within each file, Mistral and Azure calls run in parallel via `asyncio.gather` + `asyncio.to_thread` (which wraps the synchronous clients)
- Results are written to `results.csv` preserving the original file order

### Output CSV

| Column | Description |
|---|---|
| `file_path` | Relative path to the image |
| `filename` | Image filename |
| `mistral_id_doc` | `id_doc` result from Mistral |
| `mistral_doctype` | `document_type` result from Mistral |
| `azure_id_doc` | `id_doc` result from Azure |
| `azure_doctype` | `document_type` result from Azure |
| `aligned` | `True` if both analysers agree on all fields |