from abc import ABC, abstractmethod
from app.pipeline.models import DocumentContext

class VisionEngine(ABC):
    @abstractmethod
    def detect_writing_type(self, document_context: DocumentContext) -> str:
        """
        Takes the document context, analyzes its images, and determines
        if the document writing type is 'Printed', 'Handwritten', or 'Mixed'.
        """
        pass
