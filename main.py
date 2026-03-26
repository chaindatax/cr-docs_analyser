import asyncio
import csv
from pathlib import Path

from dotenv import load_dotenv

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.mistral_analyser import MistralAnalyser

DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("results.csv")
BATCH_SIZE = 10
FIELDNAMES = [
    "file_path",
    "filename",
    "mistral_id_doc",
    "mistral_doctype",
    "azure_id_doc",
    "azure_doctype",
    "aligned",
]


async def analyse_file(
    file_path: Path,
    mistral: MistralAnalyser,
    azure: AzureAnalyser,
) -> dict:
    async def run(analyser, name) -> AnalysisResult | None:
        try:
            return await asyncio.to_thread(analyser.runner, str(file_path))
        except Exception as e:
            print(f"  [{name} error] {file_path.name}: {e}")
            return None

    mistral_result, azure_result = await asyncio.gather(
        run(mistral, "mistral"),
        run(azure, "azure"),
    )

    aligned = (
        mistral_result is not None
        and azure_result is not None
        and mistral_result.id_doc == azure_result.id_doc
        and mistral_result.document_type == azure_result.document_type
    )
    print(f"  done: {file_path.name} — aligned={aligned}")

    return {
        "file_path": str(file_path),
        "filename": file_path.name,
        "mistral_id_doc": mistral_result.id_doc if mistral_result else "",
        "mistral_doctype": mistral_result.document_type if mistral_result else "",
        "azure_id_doc": azure_result.id_doc if azure_result else "",
        "azure_doctype": azure_result.document_type if azure_result else "",
        "aligned": aligned,
    }


async def analyse_all(dataset_dir: Path, output_csv: Path, batch_size: int = BATCH_SIZE):
    files = sorted(dataset_dir.rglob("*.jpg")) + sorted(dataset_dir.rglob("*.png"))
    print(f"Found {len(files)} files, processing in batches of {batch_size}...\n")

    mistral = MistralAnalyser()
    azure = AzureAnalyser()
    rows = []

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start: batch_start + batch_size]
        print(f"Batch {batch_start // batch_size + 1}: {[f.name for f in batch]}")
        batch_rows = await asyncio.gather(
            *[analyse_file(f, mistral, azure) for f in batch]
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
            print(f"  {r['filename']}: mistral=({r['mistral_id_doc']}, {r['mistral_doctype']}) azure=({r['azure_id_doc']}, {r['azure_doctype']})")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(analyse_all(DATASET_DIR, OUTPUT_CSV))