import torch
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
import sys
import logging

# Add the root directory to the path so we can import the training module
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.model import UNet
from ..base import BaseDetector

logger = logging.getLogger(__name__)

class TorchDetector(BaseDetector):
    def __init__(self):
        """Initialize the PyTorch-based detector."""
        logger.info("Initializing PyTorch detector")
        # Set up device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model_path = 'checkpoints/best_model.pth'
        self.model = UNet(n_classes=2, size='small', in_channels=3)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Set up transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Target size for model input
        self.target_size = 256
        logger.info("PyTorch detector initialized successfully")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.
        
        Args:
            image: BGR image as numpy array with shape (H, W, 3)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size while maintaining aspect ratio
            h, w = image_rgb.shape[:2]
            
            # Calculate new dimensions
            if h > w:
                new_h = self.target_size
                new_w = int(w * (self.target_size / h))
            else:
                new_w = self.target_size
                new_h = int(h * (self.target_size / w))
                
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            
            # Create square image with padding
            square_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            y_offset = (self.target_size - new_h) // 2
            x_offset = (self.target_size - new_w) // 2
            square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized
            
            # Apply transforms
            img_tensor = self.transform(square_img)
            return img_tensor, (x_offset, y_offset, new_w, new_h)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect person in the image and return a binary mask.
        
        Args:
            image: BGR image as numpy array with shape (H, W, 3)
            
        Returns:
            Binary mask as numpy array with shape (H, W) where:
            - 1 represents person
            - 0 represents background
        """
        try:
            with torch.no_grad():
                # Preprocess image
                img_tensor, crop_info = self.preprocess(image)
                img_batch = img_tensor.unsqueeze(0).to(self.device)
                
                # Run inference
                outputs = self.model(img_batch)
                mask_prob = torch.sigmoid(outputs[:, 1])  # Take foreground probability
                mask = (mask_prob > 0.5).float()
                
                # Convert mask to numpy
                mask_np = mask[0].cpu().numpy()
                
                # Postprocess mask
                x_offset, y_offset, new_w, new_h = crop_info
                h, w = image.shape[:2]
                
                # Extract the valid portion of the mask
                valid_mask = mask_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
                
                # Resize back to original frame size
                display_mask = cv2.resize(valid_mask, (w, h))
                
                return display_mask
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            raise 