from abc import ABC, abstractmethod
from dataclasses import dataclass


def is_url(source: str) -> bool:
    """Return ``True`` if *source* is an HTTPS/HTTP URL rather than a local path."""
    return source.startswith("https://") or source.startswith("http://")


def read_source_bytes(source: str) -> bytes:
    """Return the raw bytes of *source*, whether it is a local path or a URL.

    Used by analysers that do not support direct URL fetching (e.g. Mistral),
    so the content is downloaded in-memory without touching disk.
    """
    if is_url(source):
        import urllib.request
        with urllib.request.urlopen(source) as resp:
            return resp.read()
    with open(source, "rb") as f:
        return f.read()

FIELD_DEFINITIONS: dict[str, dict] = {
    "is_doc_id": {
        "type": "boolean",
        "description": "Is this document related to the identification of a person?",
    },
    "id_doc_type": {
        "type": "string",
        "description": "Type of identity document: 'id card', 'passport', 'proof_of_residency', or 'not_identity_doc'",
    },
    "doc_type": {
        "type": "string",
        "description": "Type of document, can be any kind of type",
    },
}
"""Shared field definitions used by all analysers to extract ``is_doc_id``, ``id_doc_type`` and ``doc_type``."""


@dataclass
class AnalysisResult:
    """Result of a document analysis.

    Attributes:
        is_doc_id: Whether the document is an identity document.
        id_doc_type: Identity document type — ``"id card"``, ``"passport"``, ``"proof_of_residency"``, or ``"not_identity_doc"``.
        doc_type: Free-form document type as described by the model.
    """

    is_doc_id: bool
    id_doc_type: str
    doc_type: str


class Analyser(ABC):
    """Abstract base class for document analysers.

    Subclasses must implement :meth:`runner` to analyse a document image
    and return an :class:`AnalysisResult`.
    """

    @abstractmethod
    def runner(self, file_path: str) -> AnalysisResult:
        """Analyse a document image and return a structured result.

        Args:
            file_path: Path to the image file to analyse.

        Returns:
            An :class:`AnalysisResult` with ``is_doc_id``, ``id_doc_type`` and ``doc_type``.
        """