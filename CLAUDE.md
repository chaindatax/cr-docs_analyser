# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batch analyser for identity documents (ID cards, passports). Runs images through two independent AI backends (Mistral OCR and Azure Content Understanding), extracts `id_doc` (bool) and `document_type` (id card / passport / other), and writes a comparison CSV.

## Environment Setup

Requires a `.env` file with:
```
MISTRAL_API_KEY=your_key_here
CONTENTUNDERSTANDING_ENDPOINT=https://your-resource.services.ai.azure.com/
CONTENTUNDERSTANDING_KEY=your_azure_key
```

Azure also requires `gpt-4.1` and `text-embedding-3-large` deployed in the Azure AI Foundry resource.

## Commands

Uses `uv` for package management (Python 3.12):

```bash
uv sync               # Install dependencies
uv run main.py        # Run batch analysis → results.csv
uv run pytest tests/  # Run tests
```

## Architecture

```
docs_analyser/
├── base.py              # Analyser ABC + AnalysisResult dataclass
├── mistral_analyser.py  # MistralAnalyser(Analyser)
└── azure_analyser.py    # AzureAnalyser(Analyser)
main.py                  # async batch runner, writes results.csv
tests/test_main.py
```

- `Analyser` is an abstract base class; both analysers implement `runner(file_path) -> AnalysisResult`
- `main.py` uses `asyncio.gather` for concurrent batch processing and `asyncio.to_thread` to wrap the synchronous API clients
- The Azure custom analyzer (`identityDocClassifier`) is created on first run and reused on subsequent calls

## Dependencies

- `mistralai>=2.1.3` — Mistral OCR API
- `azure-ai-contentunderstanding>=1.0.1` — Azure Content Understanding
- `azure-identity>=1.25.3` — Azure credential handling
- `python-dotenv>=1.2.2` — loads `.env`