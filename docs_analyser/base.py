from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    id_doc: bool
    document_type: str


class Analyser(ABC):
    @abstractmethod
    def runner(self, file_path: str) -> AnalysisResult:
        pass