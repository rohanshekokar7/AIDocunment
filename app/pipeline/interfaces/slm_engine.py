from abc import ABC, abstractmethod
from app.pipeline.models import DocumentContext

class SLMEngine(ABC):
    @abstractmethod
    def classify(self, document_context: DocumentContext) -> dict:
        pass
