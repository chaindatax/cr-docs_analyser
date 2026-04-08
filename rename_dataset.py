"""Rename dataset files to the format:
<IdContact>_<IdCourier>_<CodeTypologie>_<NomFichierXelians>.<extension>
Also updates dataset_labels.csv and results.csv with the new filenames.
"""

import csv
import os
import uuid
import random
import string

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
LABELS_CSV = os.path.join(os.path.dirname(__file__), "dataset_labels.csv")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

# Fixed CodeTypologie per subdirectory
TYPOLOGIE_MAP = {
    "id_cards":  "C031",
    "passports": "C032",
    "jdd":       "C033",
    "false_doc": "C034",
    "false_id":  "C035",
}

# UUID pattern prefix length: uuid(36) + _ + RE(12) + _ + C0xx(4) + _ = 55 chars
_PREFIX_LEN = 36 + 1 + 12 + 1 + 4 + 1


def random_uuid() -> str:
    return str(uuid.uuid4())


def random_courier_id() -> str:
    digits = "".join(random.choices(string.digits, k=10))
    return f"RE{digits}"


def _original_name(new_name: str) -> str:
    """Extract the original filename from a already-renamed file."""
    return new_name[_PREFIX_LEN:]


def _build_reverse_map() -> dict[tuple[str, str], str]:
    """Build mapping (subdir, original_filename) -> new_filename from current filesystem."""
    mapping: dict[tuple[str, str], str] = {}
    for subdir in TYPOLOGIE_MAP:
        folder = os.path.join(DATASET_DIR, subdir)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.startswith("."):
                continue
            orig = _original_name(name)
            if orig:
                mapping[(subdir, orig)] = name
    return mapping


def rename_files(dry_run: bool = False) -> dict[tuple[str, str], str]:
    """Rename files and return (subdir, old_name) -> new_name mapping."""
    rename_map: dict[tuple[str, str], str] = {}

    for subdir, code_typo in TYPOLOGIE_MAP.items():
        folder = os.path.join(DATASET_DIR, subdir)
        if not os.path.isdir(folder):
            print(f"[SKIP] {folder} not found")
            continue

        for filename in sorted(os.listdir(folder)):
            if filename.startswith("."):
                continue

            old_path = os.path.join(folder, filename)
            if not os.path.isfile(old_path):
                continue

            id_contact = random_uuid()
            id_courier = random_courier_id()
            new_name = f"{id_contact}_{id_courier}_{code_typo}_{filename}"
            new_path = os.path.join(folder, new_name)

            print(f"{'[DRY] ' if dry_run else ''}  {filename}")
            print(f"    → {new_name}")

            rename_map[(subdir, filename)] = new_name
            if not dry_run:
                os.rename(old_path, new_path)

    return rename_map


def _update_csv_file(
    csv_path: str,
    rename_map: dict[tuple[str, str], str],
    delimiter: str,
    dry_run: bool,
) -> None:
    """Update filenames in a CSV file using the rename map."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames

    updated = 0
    for row in rows:
        subdir = row["file_path"].split("/")[-1]
        old_name = row["filename"]
        key = (subdir, old_name)
        if key in rename_map:
            row["filename"] = rename_map[key]
            updated += 1
        else:
            print(f"[WARN] no rename found for {csv_path}: {subdir}/{old_name}")

    label = os.path.basename(csv_path)
    print(f"{'[DRY] ' if dry_run else ''}{label}: {updated}/{len(rows)} rows updated")

    if not dry_run:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)


def update_csvs(rename_map: dict[tuple[str, str], str], dry_run: bool = False) -> None:
    _update_csv_file(LABELS_CSV, rename_map, delimiter=",", dry_run=dry_run)
    _update_csv_file(RESULTS_CSV, rename_map, delimiter=";", dry_run=dry_run)


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv

    if dry:
        print("=== DRY RUN (pass no args to actually rename) ===\n")

    rename_map = rename_files(dry_run=dry)

    print()
    update_csvs(rename_map, dry_run=dry)
    print("\nDone.")