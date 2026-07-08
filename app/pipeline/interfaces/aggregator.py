from abc import ABC, abstractmethod
from app.pipeline.models import DocumentContext

class FeatureAggregator(ABC):
    @abstractmethod
    def aggregate(self, document_context: DocumentContext) -> DocumentContext:
        pass
