from abc import ABC, abstractmethod
import numpy as np

class ImageFilter(ABC):
    """
    Abstract Base Class for an image processing filter.
    Follows the Open/Closed Principle and Liskov Substitution Principle.
    """
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a specific transformation to the input image.
        
        Args:
            image (np.ndarray): Input image in BGR or Grayscale format.
            
        Returns:
            np.ndarray: The transformed image.
        """
        pass
