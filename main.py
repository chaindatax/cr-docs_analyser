import asyncio
import csv
from pathlib import Path

from dotenv import load_dotenv

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.azure_vision_analyser import AzureVisionAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.mistral_analyser import MistralAnalyser
from docs_analyser.mistral_vision_analyser import MistralVisionAnalyser

DATASET_DIR = Path("dataset/misc")
OUTPUT_CSV = Path("results.csv")
BATCH_SIZE = 10
FIELDNAMES = [
    "file_path",
    "filename",
    "mistral_id_doc",
    "mistral_document_id_type",
    "mistral_document_type",
    "azure_id_doc",
    "azure_document_id_type",
    "azure_document_type",
    "mistral_vision_id_doc",
    "mistral_vision_document_id_type",
    "mistral_vision_document_type",
    "azure_vision_id_doc",
    "azure_vision_document_id_type",
    "azure_vision_document_type",
    "aligned",
]


async def analyse_file(
    file_path: Path,
    mistral: MistralAnalyser,
    azure: AzureAnalyser,
    mistral_vision: MistralVisionAnalyser,
    azure_vision: AzureVisionAnalyser,
) -> dict:
    """Run all four analysers on a single file concurrently.

    All four calls are launched in parallel via :func:`asyncio.gather`.
    Errors from any analyser are caught and logged; the corresponding fields
    are left empty in the output row.

    Args:
        file_path: Path to the image file to analyse.
        mistral: Initialised :class:`MistralAnalyser` instance.
        azure: Initialised :class:`AzureAnalyser` instance.
        mistral_vision: Initialised :class:`MistralVisionAnalyser` instance.
        azure_vision: Initialised :class:`AzureVisionAnalyser` instance.

    Returns:
        A dict with keys matching ``FIELDNAMES``, including ``aligned``.
    """
    async def run(analyser, name) -> AnalysisResult | None:
        try:
            return await asyncio.to_thread(analyser.runner, str(file_path))
        except Exception as e:
            print(f"  [{name} error] {file_path.name}: {e}")
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
        and len({r.id_doc for r in results}) == 1
        and len({r.document_id_type for r in results}) == 1
        and len({r.document_type for r in results}) == 1
    )
    print(f"  done: {file_path.name} — aligned={aligned}")

    def f(r, attr): return getattr(r, attr) if r else ""

    return {
        "file_path": str(file_path),
        "filename": file_path.name,
        "mistral_id_doc": f(mistral_result, "id_doc"),
        "mistral_document_id_type": f(mistral_result, "document_id_type"),
        "mistral_document_type": f(mistral_result, "document_type"),
        "azure_id_doc": f(azure_result, "id_doc"),
        "azure_document_id_type": f(azure_result, "document_id_type"),
        "azure_document_type": f(azure_result, "document_type"),
        "mistral_vision_id_doc": f(mistral_vision_result, "id_doc"),
        "mistral_vision_document_id_type": f(mistral_vision_result, "document_id_type"),
        "mistral_vision_document_type": f(mistral_vision_result, "document_type"),
        "azure_vision_id_doc": f(azure_vision_result, "id_doc"),
        "azure_vision_document_id_type": f(azure_vision_result, "document_id_type"),
        "azure_vision_document_type": f(azure_vision_result, "document_type"),
        "aligned": aligned,
    }


async def analyse_all(dataset_dir: Path, output_csv: Path, batch_size: int = BATCH_SIZE):
    """Analyse all images in a directory and write results to a CSV file.

    Scans ``dataset_dir`` for ``.jpg`` and ``.png`` files, processes them in
    batches of ``batch_size`` using both :class:`MistralAnalyser` and
    :class:`AzureAnalyser` concurrently, and writes a comparison CSV to
    ``output_csv``.

    Prints a summary of aligned vs misaligned results on completion.

    Args:
        dataset_dir: Directory containing images to analyse.
        output_csv: Destination path for the CSV output.
        batch_size: Number of files to process concurrently per batch.
    """
    files = sorted(dataset_dir.rglob("*.jpg")) + sorted(dataset_dir.rglob("*.png"))
    print(f"Found {len(files)} files, processing in batches of {batch_size}...\n")

    mistral = MistralAnalyser()
    azure = AzureAnalyser()
    mistral_vision = MistralVisionAnalyser()
    azure_vision = AzureVisionAnalyser()
    rows = []

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start: batch_start + batch_size]
        print(f"Batch {batch_start // batch_size + 1}: {[f.name for f in batch]}")
        batch_rows = await asyncio.gather(
            *[analyse_file(f, mistral, azure, mistral_vision, azure_vision) for f in batch]
        )
        rows.extend(batch_rows)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
                f"mistral=({r['mistral_id_doc']}, {r['mistral_document_id_type']}, {r['mistral_document_type']}) "
                f"azure=({r['azure_id_doc']}, {r['azure_document_id_type']}, {r['azure_document_type']}) "
                f"mistral_vision=({r['mistral_vision_id_doc']}, {r['mistral_vision_document_id_type']}, {r['mistral_vision_document_type']}) "
                f"azure_vision=({r['azure_vision_id_doc']}, {r['azure_vision_document_id_type']}, {r['azure_vision_document_type']})"
            )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(analyse_all(DATASET_DIR, OUTPUT_CSV))