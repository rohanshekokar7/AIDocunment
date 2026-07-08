from abc import ABC, abstractmethod
from typing import Any

class DocumentValidator(ABC):
    @abstractmethod
    def validate(self, file_path: str) -> bool:
        pass
