from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """Base class for person/background detectors."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the detector."""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect person in the image and return a binary mask.
        
        Args:
            image: RGB image as numpy array with shape (H, W, 3)
            
        Returns:
            Binary mask as numpy array with shape (H, W) where:
            - 1 represents person
            - 0 represents background
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.
        
        Args:
            image: RGB image as numpy array with shape (H, W, 3)
            
        Returns:
            Preprocessed image
        """
        pass 