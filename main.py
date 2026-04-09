import asyncio
import csv
import os
from pathlib import Path

from dotenv import load_dotenv

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.azure_vision_analyser import AzureVisionAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.blob_source import BlobSource
from docs_analyser.mistral_analyser import MistralAnalyser
from docs_analyser.mistral_vision_analyser import MistralVisionAnalyser

DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("results.csv")
BATCH_SIZE = 10
FIELDNAMES = [
    "file_path",
    "filename",
    "mistral_is_doc_id",
    "mistral_id_doc_type",
    "mistral_doc_type",
    "azure_is_doc_id",
    "azure_id_doc_type",
    "azure_doc_type",
    "mistral_vision_is_doc_id",
    "mistral_vision_id_doc_type",
    "mistral_vision_doc_type",
    "azure_vision_is_doc_id",
    "azure_vision_id_doc_type",
    "azure_vision_doc_type",
    "aligned",
]


async def analyse_file(
    source: str,
    file_path_label: str,
    filename: str,
    mistral: MistralAnalyser,
    azure: AzureAnalyser,
    mistral_vision: MistralVisionAnalyser,
    azure_vision: AzureVisionAnalyser,
) -> dict:
    """Run all four analysers on a single file concurrently.

    Args:
        source: Local path or HTTPS URL of the document.
        file_path_label: Value written to the ``file_path`` CSV column.
        filename: Value written to the ``filename`` CSV column.
        mistral: Initialised :class:`MistralAnalyser` instance.
        azure: Initialised :class:`AzureAnalyser` instance.
        mistral_vision: Initialised :class:`MistralVisionAnalyser` instance.
        azure_vision: Initialised :class:`AzureVisionAnalyser` instance.

    Returns:
        A dict with keys matching ``FIELDNAMES``, including ``aligned``.
    """
    async def run(analyser, name) -> AnalysisResult | None:
        try:
            return await asyncio.to_thread(analyser.runner, source)
        except Exception as e:
            print(f"  [{name} error] {filename}: {e}")
            return None

    mistral_result, azure_result, mistral_vision_result, azure_vision_result = await asyncio.gather(
        run(mistral, "mistral"),
        run(azure, "azure"),
        run(mistral_vision, "mistral_vision"),
        run(azure_vision, "azure_vision"),
    )

    results = [mistral_result, azure_result, mistral_vision_result, azure_vision_result]
    aligned = (
        all(r is not None for r in results)
        and len({r.is_doc_id for r in results}) == 1
        and len({r.id_doc_type for r in results}) == 1
        and len({r.doc_type for r in results}) == 1
    )
    print(f"  done: {filename} — aligned={aligned}")

    def f(r, attr): return getattr(r, attr) if r else ""

    return {
        "file_path": file_path_label,
        "filename": filename,
        "mistral_is_doc_id": f(mistral_result, "is_doc_id"),
        "mistral_id_doc_type": f(mistral_result, "id_doc_type"),
        "mistral_doc_type": f(mistral_result, "doc_type"),
        "azure_is_doc_id": f(azure_result, "is_doc_id"),
        "azure_id_doc_type": f(azure_result, "id_doc_type"),
        "azure_doc_type": f(azure_result, "doc_type"),
        "mistral_vision_is_doc_id": f(mistral_vision_result, "is_doc_id"),
        "mistral_vision_id_doc_type": f(mistral_vision_result, "id_doc_type"),
        "mistral_vision_doc_type": f(mistral_vision_result, "doc_type"),
        "azure_vision_is_doc_id": f(azure_vision_result, "is_doc_id"),
        "azure_vision_id_doc_type": f(azure_vision_result, "id_doc_type"),
        "azure_vision_doc_type": f(azure_vision_result, "doc_type"),
        "aligned": aligned,
    }


def _local_files(dataset_dir: Path) -> list[tuple[str, str, str]]:
    """Return ``(source, file_path_label, filename)`` for every supported local file."""
    paths = (
        sorted(dataset_dir.rglob("*.jpg"))
        + sorted(dataset_dir.rglob("*.jpeg"))
        + sorted(dataset_dir.rglob("*.png"))
        + sorted(dataset_dir.rglob("*.pdf"))
    )
    return [(str(p), str(p.parent), p.name) for p in paths]


async def analyse_all(output_csv: Path, batch_size: int = BATCH_SIZE):
    """Analyse all files and write results to a CSV file.

    Source is determined by the ``BLOB_SAS_URL`` environment variable:
    - If set: blobs are listed from the Azure container and analysed via their
      SAS URLs — no local download required.
    - If absent: files are read from the local ``dataset/`` directory.

    Args:
        output_csv: Destination path for the CSV output.
        batch_size: Number of files to process concurrently per batch.
    """
    sas_url = os.getenv("BLOB_SAS_URL")
    if sas_url:
        print("Source: Azure Blob Storage")
        files = [(url, fp, name) for fp, name, url in BlobSource(sas_url).list_files()]
    else:
        print(f"Source: local {DATASET_DIR}/")
        files = _local_files(DATASET_DIR)

    print(f"Found {len(files)} files, processing in batches of {batch_size}...\n")

    mistral = MistralAnalyser()
    azure = AzureAnalyser()
    mistral_vision = MistralVisionAnalyser()
    azure_vision = AzureVisionAnalyser()
    rows = []

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start: batch_start + batch_size]
        print(f"Batch {batch_start // batch_size + 1}: {[name for _, _, name in batch]}")
        batch_rows = await asyncio.gather(
            *[
                analyse_file(src, fp, name, mistral, azure, mistral_vision, azure_vision)
                for src, fp, name in batch
            ]
        )
        rows.extend(batch_rows)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)

    misaligned = [r for r in rows if not r["aligned"]]
    print(f"\nResults written to {output_csv}")
    print(f"Aligned: {len(rows) - len(misaligned)}/{len(rows)}")
    if misaligned:
        print("Misaligned files:")
        for r in misaligned:
            print(
                f"  {r['filename']}: "
                f"mistral=({r['mistral_is_doc_id']}, {r['mistral_id_doc_type']}, {r['mistral_doc_type']}) "
                f"azure=({r['azure_is_doc_id']}, {r['azure_id_doc_type']}, {r['azure_doc_type']}) "
                f"mistral_vision=({r['mistral_vision_is_doc_id']}, {r['mistral_vision_id_doc_type']}, {r['mistral_vision_doc_type']}) "
                f"azure_vision=({r['azure_vision_is_doc_id']}, {r['azure_vision_id_doc_type']}, {r['azure_vision_doc_type']})"
            )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(analyse_all(OUTPUT_CSV))