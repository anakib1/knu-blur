import torch
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
import sys

# Add the root directory to the path so we can import the training module
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.model import UNet
from app.predictors.base import BaseDetector

class TorchDetector(BaseDetector):
    def __init__(self, model_path: str = 'checkpoints/best_model.pth'):
        """Initialize the PyTorch-based detector.
        
        Args:
            model_path: Path to the trained model checkpoint
        """
        # Set up device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = UNet(n_classes=2, size='small', in_channels=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set up transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Target size for model input
        self.target_size = 256
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size while maintaining aspect ratio
        h, w = frame_rgb.shape[:2]
        
        # Calculate new dimensions
        if h > w:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
        else:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
            
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Create square image with padding
        square_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
        
        # Apply transforms
        img_tensor = self.transform(square_img)
        return img_tensor, (x_offset, y_offset, new_w, new_h)
    
    def postprocess_mask(self, mask, frame_shape, crop_info):
        """Convert model output to display mask."""
        x_offset, y_offset, new_w, new_h = crop_info
        h, w = frame_shape[:2]
        
        # Extract the valid portion of the mask
        valid_mask = mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        # Resize back to original frame size
        display_mask = cv2.resize(valid_mask, (w, h))
        return display_mask
    
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Predict segmentation mask for the input frame.
        
        Args:
            frame: Input frame in BGR format (H, W, 3)
            
        Returns:
            mask: Binary segmentation mask (H, W)
        """
        with torch.no_grad():
            # Preprocess frame
            img_tensor, crop_info = self.preprocess_frame(frame)
            img_batch = img_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            outputs = self.model(img_batch)
            mask_prob = torch.sigmoid(outputs[:, 1])  # Take foreground probability
            mask = (mask_prob > 0.5).float()
            
            # Convert mask to numpy
            mask_np = mask[0].cpu().numpy()
            
            # Postprocess mask
            display_mask = self.postprocess_mask(mask_np, frame.shape, crop_info)
            
            return display_mask 