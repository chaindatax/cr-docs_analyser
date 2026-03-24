# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A document analyser that uses the Mistral AI OCR API to process identity documents (ID cards, passports) and extract structured information from images.

## Environment Setup

Requires a `.env` file with:
```
MISTRAL_API_KEY=your_key_here
```

## Commands

Uses `uv` for package management (Python 3.12):

```bash
uv sync          # Install dependencies
uv run main.py   # Run the analyser
```

## Architecture

Single-script project (`main.py`). The entry point loads env vars via `python-dotenv`, then:

1. Encodes a local image file as base64
2. Calls `mistralai` OCR API (`mistral-ocr-latest` model) with a JSON schema for structured output
3. The schema extracts `document_type` (id card / passport / other) and `id_doc` (boolean)

The `file_path` variable in `main.py` must be updated to point to the actual document to analyse before running.

## Dependencies

- `mistralai>=2.1.3` — Mistral AI client (OCR + structured JSON output)
- `python-dotenv>=1.2.2` — loads `.env` for API key
