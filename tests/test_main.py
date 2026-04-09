"""Integration tests that run each analyser against a real file from the dataset.

Tests are skipped automatically when the required environment variables are not
set, so they can coexist with CI pipelines that lack API credentials.

``SAMPLE_FILES`` is a parametrised fixture that picks the first sorted file from
each leaf subdirectory of ``dataset/``, giving at least one representative
sample per document category.
"""

import csv
import os
from pathlib import Path
from pprint import pprint

import pytest
from dotenv import load_dotenv

from docs_analyser.base import AnalysisResult
from tests.test_dataset import first_file_per_leaf_subdir

load_dotenv()

DATASET_ROOT = Path(__file__).parent.parent / "dataset"

EXTENSIONS = (".jpg", ".png", ".pdf", ".webp")


# --- DATASET_DIR file listing ---

LABELS_CSV = Path(__file__).parent.parent / "dataset_labels.csv"



def _collect_dataset_files() -> list:
    """Return all supported image/PDF files found under ``DATASET_DIR``."""
    from main import DATASET_DIR
    files = []
    for ext in EXTENSIONS:
        files += sorted(DATASET_DIR.rglob(f"*{ext}"))
    return files


def test_dataset_dir_lists_jpg_and_png_files():
    files = _collect_dataset_files()
    pprint(f"Found {len(files)} files")
    for f in files:
        pprint(str(f))
    assert len(files) > 0
    assert all(f.suffix in EXTENSIONS for f in files)
    assert all(f.is_file() for f in files)



@pytest.mark.skip
def test_generate_dataset_labels_csv():
    files = _collect_dataset_files()
    with LABELS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "filename", "is_doc_id", "id_doc_type", "doc_type"])
        writer.writeheader()
        writer.writerows({"file_path": str(p.parent), "filename": p.name, "is_doc_id": "", "id_doc_type": "", "doc_type": ""} for p in files)
    assert LABELS_CSV.exists()
    assert LABELS_CSV.stat().st_size > 0



# First file (sorted) from each leaf subdirectory
SAMPLE_FILES = [
    pytest.param(path, id=path.parent.name)
    for path in first_file_per_leaf_subdir(DATASET_ROOT)
]

needs_mistral = pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)
needs_azure_cu = pytest.mark.skipif(
    not (os.getenv("CONTENTUNDERSTANDING_ENDPOINT") and os.getenv("CONTENTUNDERSTANDING_KEY")),
    reason="CONTENTUNDERSTANDING_ENDPOINT / CONTENTUNDERSTANDING_KEY not set",
)
needs_azure_oai = pytest.mark.skipif(
    not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_KEY")),
    reason="AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY not set",
)


@needs_mistral
@pytest.mark.parametrize("file_path", SAMPLE_FILES)
def test_mistral_analyser_on_dataset(file_path):
    from docs_analyser.mistral_analyser import MistralAnalyser
    result = MistralAnalyser().runner(str(file_path))
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.is_doc_id, bool)
    assert isinstance(result.doc_type, str)


@needs_mistral
@pytest.mark.parametrize("file_path", SAMPLE_FILES)
def test_mistral_vision_analyser_on_dataset(file_path):
    pprint(f"testing mistral vision analyser on dataset with file: {file_path}")
    from docs_analyser.mistral_vision_analyser import MistralVisionAnalyser
    result = MistralVisionAnalyser().runner(str(file_path))
    pprint(f"result: {result}")
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.is_doc_id, bool)
    assert isinstance(result.doc_type, str)


@needs_azure_cu
@pytest.mark.parametrize("file_path", SAMPLE_FILES)
def test_azure_analyser_on_dataset(file_path):
    from docs_analyser.azure_analyser import AzureAnalyser
    result = AzureAnalyser().runner(str(file_path))
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.is_doc_id, bool)
    assert isinstance(result.doc_type, str)


@needs_azure_oai
@pytest.mark.parametrize("file_path", SAMPLE_FILES)
def test_azure_vision_analyser_on_dataset(file_path):
    pprint(f"testing mistral vision analyser on dataset with file: {file_path}")
    from docs_analyser.azure_vision_analyser import AzureVisionAnalyser
    result = AzureVisionAnalyser().runner(str(file_path))
    pprint(f"result: {result}")
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.is_doc_id, bool)
    assert isinstance(result.doc_type, str)