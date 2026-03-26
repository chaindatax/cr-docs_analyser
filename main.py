import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def analyse_file(
    file_path: Path,
    mistral: MistralAnalyser,
    azure: AzureAnalyser,
) -> dict:
    mistral_result: AnalysisResult | None = None
    azure_result: AnalysisResult | None = None

    try:
        mistral_result = mistral.runner(str(file_path))
    except Exception as e:
        print(f"  [mistral error] {file_path.name}: {e}")

    try:
        azure_result = azure.runner(str(file_path))
    except Exception as e:
        print(f"  [azure error] {file_path.name}: {e}")

    aligned = (
        mistral_result is not None
        and azure_result is not None
        and mistral_result.id_doc == azure_result.id_doc
        and mistral_result.document_type == azure_result.document_type
    )

    return {
        "file_path": str(file_path),
        "filename": file_path.name,
        "mistral_id_doc": mistral_result.id_doc if mistral_result else "",
        "mistral_doctype": mistral_result.document_type if mistral_result else "",
        "azure_id_doc": azure_result.id_doc if azure_result else "",
        "azure_doctype": azure_result.document_type if azure_result else "",
        "aligned": aligned,
    }


def analyse_all(dataset_dir: Path, output_csv: Path, batch_size: int = BATCH_SIZE):
    files = sorted(dataset_dir.rglob("*.jpg")) + sorted(dataset_dir.rglob("*.png"))
    print(f"Found {len(files)} files, processing in batches of {batch_size}...\n")

    mistral = MistralAnalyser()
    azure = AzureAnalyser()

    rows: list[dict] = [None] * len(files)

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start: batch_start + batch_size]
        print(f"Batch {batch_start // batch_size + 1}: {[f.name for f in batch]}")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_index = {
                executor.submit(analyse_file, file_path, mistral, azure): batch_start + i
                for i, file_path in enumerate(batch)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                rows[index] = future.result()
                print(f"  done: {rows[index]['filename']} — aligned={rows[index]['aligned']}")

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
    analyse_all(DATASET_DIR, OUTPUT_CSV)