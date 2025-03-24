import mediapipe as mp
import numpy as np
import cv2
import logging
from ..base import BaseDetector

logger = logging.getLogger(__name__)

class MediaPipeDetector(BaseDetector):
    """MediaPipe-based person segmentation detector."""
    
    def __init__(self):
        """Initialize the MediaPipe detector."""
        logger.info("Initializing MediaPipe detector")
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0 for general use, 1 for landscape
        )
        logger.info("MediaPipe detector initialized successfully")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.
        
        Args:
            image: RGB image as numpy array with shape (H, W, 3)
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert BGR to RGB
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return processed
        except Exception as e:
            logger.error("Error in preprocessing: %s", str(e))
            raise
    
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
        try:
            # Preprocess image
            rgb_image = self.preprocess(image)
            
            # Get segmentation results
            results = self.selfie_segmentation.process(rgb_image)
            
            # Get the segmentation mask
            mask = results.segmentation_mask
            
            # Convert to binary mask (threshold at 0.5)
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            return binary_mask
        except Exception as e:
            logger.error("Error in detection: %s", str(e))
            raise 