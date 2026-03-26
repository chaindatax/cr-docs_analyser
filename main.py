import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.base import AnalysisResult
from docs_analyser.mistral_analyser import MistralAnalyser

DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("results.csv")
FIELDNAMES = [
    "file_path",
    "filename",
    "mistral_id_doc",
    "mistral_doctype",
    "azure_id_doc",
    "azure_doctype",
]


def analyse_all(dataset_dir: Path, output_csv: Path):
    files = sorted(dataset_dir.rglob("*.jpg")) + sorted(dataset_dir.rglob("*.png"))

    mistral = MistralAnalyser()
    azure = AzureAnalyser()

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for file_path in files:
            print(f"Processing {file_path}...", end=" ", flush=True)

            mistral_result: AnalysisResult | None = None
            azure_result: AnalysisResult | None = None

            try:
                mistral_result = mistral.runner(str(file_path))
            except Exception as e:
                print(f"\n  [mistral error] {e}", end=" ")

            try:
                azure_result = azure.runner(str(file_path))
            except Exception as e:
                print(f"\n  [azure error] {e}", end=" ")

            writer.writerow({
                "file_path": str(file_path),
                "filename": file_path.name,
                "mistral_id_doc": mistral_result.id_doc if mistral_result else "",
                "mistral_doctype": mistral_result.document_type if mistral_result else "",
                "azure_id_doc": azure_result.id_doc if azure_result else "",
                "azure_doctype": azure_result.document_type if azure_result else "",
            })
            print("done")

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    load_dotenv()
    analyse_all(DATASET_DIR, OUTPUT_CSV)