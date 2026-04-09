"""Pre-processing step: convert PDF files to JPG images (one per page).

Run this script **before** ``main.py`` whenever the dataset contains PDFs.

Source is determined by ``BLOB_SAS_URL`` (same logic as ``main.py``):

- **Local** (default): scans ``dataset/`` recursively for ``.pdf`` files,
  writes one JPG per page alongside the PDF, then removes the PDF.
- **Azure Blob Storage**: downloads each PDF blob in-memory, converts it,
  uploads one JPG blob per page to the same virtual folder, then deletes
  the original PDF blob.

Page naming convention::

    <original_stem>_p001.jpg
    <original_stem>_p002.jpg
    ...

For a single-page PDF the ``_p001`` suffix is still added for consistency.

Usage::

    uv run preprocess_pdfs.py               # local dataset/
    BLOB_SAS_URL="https://..." uv run preprocess_pdfs.py

Options (env vars):
    BLOB_SAS_URL        Container-level SAS URL (Read + List + Write + Delete).
    PDF_DPI             Render resolution in DPI (default: 150).
    PDF_KEEP_ORIGINALS  Set to "true" to keep the original PDF after conversion
                        (default: false — PDFs are deleted after conversion).
"""

import io
import os
from pathlib import Path

import fitz  # pymupdf
from azure.storage.blob import ContainerClient, ContentSettings
from dotenv import load_dotenv

DATASET_DIR = Path("dataset")
DPI = int(os.getenv("PDF_DPI", "150"))
KEEP_ORIGINALS = os.getenv("PDF_KEEP_ORIGINALS", "false").strip().lower() in ("true", "1")


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def pdf_bytes_to_jpg_pages(pdf_bytes: bytes, dpi: int = DPI) -> list[bytes]:
    """Convert PDF bytes to a list of JPEG byte buffers, one per page.

    Args:
        pdf_bytes: Raw PDF content.
        dpi: Render resolution (higher = better quality, larger files).

    Returns:
        List of JPEG byte buffers in page order.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    zoom = dpi / 72  # fitz native resolution is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        pages.append(pix.tobytes("jpeg"))
    doc.close()
    return pages


def page_name(stem: str, page_index: int) -> str:
    """Return the JPG filename for a given page (1-based)."""
    return f"{stem}_p{page_index + 1:03d}.jpg"


# ---------------------------------------------------------------------------
# Local processing
# ---------------------------------------------------------------------------

def process_local(dataset_dir: Path, keep_originals: bool = KEEP_ORIGINALS) -> None:
    pdf_files = sorted(dataset_dir.rglob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in {dataset_dir}/\n")

    for pdf_path in pdf_files:
        print(f"  {pdf_path.relative_to(dataset_dir)}")
        pdf_bytes = pdf_path.read_bytes()
        pages = pdf_bytes_to_jpg_pages(pdf_bytes)

        for i, jpg_bytes in enumerate(pages):
            out_path = pdf_path.with_name(page_name(pdf_path.stem, i))
            out_path.write_bytes(jpg_bytes)
            print(f"    → {out_path.name}")

        if not keep_originals:
            pdf_path.unlink()
            print(f"    ✗ removed {pdf_path.name}")

    print(f"\nDone. Converted {len(pdf_files)} PDF(s).")


# ---------------------------------------------------------------------------
# Azure Blob Storage processing
# ---------------------------------------------------------------------------

def process_blob(sas_url: str, keep_originals: bool = KEEP_ORIGINALS) -> None:
    client = ContainerClient.from_container_url(sas_url)

    pdf_blobs = [b for b in client.list_blobs() if b.name.lower().endswith(".pdf")]
    if not pdf_blobs:
        print("No PDF blobs found.")
        return

    print(f"Found {len(pdf_blobs)} PDF blob(s)\n")

    for blob_props in pdf_blobs:
        blob_name = blob_props.name
        print(f"  {blob_name}")

        # Download PDF in-memory
        stream = client.get_blob_client(blob_name).download_blob()
        pdf_bytes = stream.readall()

        pages = pdf_bytes_to_jpg_pages(pdf_bytes)
        stem = blob_name[: -len(".pdf")]  # preserve virtual folder prefix

        for i, jpg_bytes in enumerate(pages):
            jpg_blob_name = page_name(stem, i)
            client.get_blob_client(jpg_blob_name).upload_blob(
                io.BytesIO(jpg_bytes),
                overwrite=True,
                content_settings=_jpg_content_settings(),
            )
            print(f"    → {jpg_blob_name}")

        if not keep_originals:
            client.get_blob_client(blob_name).delete_blob()
            print(f"    ✗ deleted {blob_name}")

    print(f"\nDone. Converted {len(pdf_blobs)} PDF blob(s).")


def _jpg_content_settings():
    return ContentSettings(content_type="image/jpeg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()
    sas_url = os.getenv("BLOB_SAS_URL")
    if sas_url:
        print(f"Source: Azure Blob Storage  |  DPI={DPI}  |  keep_originals={KEEP_ORIGINALS}\n")
        process_blob(sas_url)
    else:
        print(f"Source: local {DATASET_DIR}/  |  DPI={DPI}  |  keep_originals={KEEP_ORIGINALS}\n")
        process_local(DATASET_DIR)
