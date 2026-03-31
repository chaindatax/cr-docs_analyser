"""Shared helpers for dataset traversal used across test modules."""
from pathlib import Path

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".pdf")


def first_file_per_leaf_subdir(root: Path) -> list[Path]:
    """Return the first sorted non-hidden image file from each leaf subdirectory under *root*."""
    results = []
    for subdir in sorted(root.rglob("*")):
        if not subdir.is_dir():
            continue
        if any(child.is_dir() for child in subdir.iterdir()):
            continue
        candidates = sorted(
            f for f in subdir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix in EXTENSIONS
        )
        if candidates:
            results.append(candidates[0])
    return results
