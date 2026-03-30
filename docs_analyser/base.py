from abc import ABC, abstractmethod
from dataclasses import dataclass

FIELD_DEFINITIONS: dict[str, dict] = {
    "id_doc": {
        "type": "boolean",
        "description": "Is this document related to the identification of a person?",
    },
    "document_type": {
        "type": "string",
        "description": "Type of identity document: 'id card', 'passport', or 'other'",
    },
}
"""Shared field definitions used by all analysers to extract ``id_doc`` and ``document_type``."""


@dataclass
class AnalysisResult:
    """Result of a document analysis.

    Attributes:
        id_doc: Whether the document is an identity document.
        document_type: Type of document — ``"id card"``, ``"passport"``, or ``"other"``.
    """

    id_doc: bool
    document_type: str


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
            An :class:`AnalysisResult` with ``id_doc`` and ``document_type``.
        """