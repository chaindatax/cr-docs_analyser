"""Integration tests: run all 4 analysers on the first file of each dataset subdirectory.

Skipped automatically when API credentials are not set in the environment.
Run with: uv run pytest tests/test_dataset.py -v
"""
import os
from pathlib import Path
from  pprint  import pprint
import pytest
from dotenv import load_dotenv
load_dotenv()

from docs_analyser.base import AnalysisResult

DATASET_ROOT = Path(__file__).parent.parent / "dataset"



# --- first_file_per_leaf_subdir ---

def test_returns_one_file_per_leaf_subdir(tmp_path):
    # tree: root/a/f1.jpg, root/a/f2.jpg, root/b/sub/f3.jpg, root/b/sub/f1.jpg
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f2.jpg").write_bytes(b"")
    (tmp_path / "a" / "f1.jpg").write_bytes(b"")
    (tmp_path / "b" / "sub").mkdir(parents=True)
    (tmp_path / "b" / "sub" / "f3.jpg").write_bytes(b"")
    (tmp_path / "b" / "sub" / "f1.jpg").write_bytes(b"")

    result = first_file_per_leaf_subdir(tmp_path)

    # b/ is not a leaf (has sub/), so only a/ and b/sub/ are returned
    assert len(result) == 2
    # each result is the first sorted file of its leaf dir
    assert result[0] == tmp_path / "a" / "f1.jpg"
    assert result[1] == tmp_path / "b" / "sub" / "f1.jpg"


def test_excludes_non_leaf_dirs(tmp_path):
    (tmp_path / "parent" / "child").mkdir(parents=True)
    (tmp_path / "parent" / "child" / "doc.jpg").write_bytes(b"")
    (tmp_path / "parent" / "sibling").mkdir()
    (tmp_path / "parent" / "sibling" / "doc.jpg").write_bytes(b"")

    result = first_file_per_leaf_subdir(tmp_path)

    dirs = {p.parent.name for p in result}
    assert "parent" not in dirs
    assert dirs == {"child", "sibling"}


def test_empty_root_returns_empty_list(tmp_path):
    assert first_file_per_leaf_subdir(tmp_path) == []



def first_file_per_leaf_subdir(root: Path) -> list[Path]:
    """Return the first sorted file from each leaf subdirectory under *root*."""
    return [
        sorted(subdir.iterdir())[0]
        for subdir in sorted(root.rglob("*"))
        if subdir.is_dir() and not any(child.is_dir() for child in subdir.iterdir())
    ]


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