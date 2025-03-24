import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from detector.base import BaseDetector
from detector import MediaPipeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackgroundBlurApp:
    def __init__(self, detector: BaseDetector, blur_strength: int = 31):
        """
        Initialize the background blur application.
        
        Args:
            detector: Instance of a detector implementing BaseDetector
            blur_strength: Strength of the Gaussian blur (odd number)
        """
        self.detector = detector
        self.blur_strength = blur_strength
        self.cap: Optional[cv2.VideoCapture] = None
        self.mode = 'blur'  # 'blur' or 'color'
        self.color = (0, 0, 255)  # BGR format for red
        self.colors = {
            'r': (0, 0, 255),    # Red
            'g': (0, 255, 0),    # Green
            'b': (255, 0, 0),    # Blue
            'w': (255, 255, 255), # White
            'k': (0, 0, 0)       # Black
        }
        logger.info("BackgroundBlurApp initialized with blur_strength=%d", blur_strength)
        
    def start(self, camera_id: int = 0):
        """Start the real-time background blur application."""
        logger.info("Attempting to open camera with ID: %d", camera_id)
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera!")
            return
            
        logger.info("Camera opened successfully")
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        logger.info("Camera properties - Width: %d, Height: %d, FPS: %d", width, height, fps)
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame!")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info("Processing frame %d", frame_count)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display result and controls
            self.display_frame(processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received")
                break
            elif key == ord('m'):
                self.mode = 'color' if self.mode == 'blur' else 'blur'
                logger.info(f"Mode switched to: {self.mode}")
            elif key in [ord(k) for k in self.colors.keys()]:
                self.color = self.colors[chr(key)]
                logger.info(f"Color changed to: {chr(key)}")
            elif key == ord('+'):
                self.blur_strength = min(99, self.blur_strength + 2)
                logger.info(f"Blur strength increased to: {self.blur_strength}")
            elif key == ord('-'):
                self.blur_strength = max(3, self.blur_strength - 2)
                logger.info(f"Blur strength decreased to: {self.blur_strength}")
                
        self.cleanup()
        
    def display_frame(self, frame: np.ndarray):
        """Display the frame with controls information."""
        # Create a copy of the frame for displaying text
        display_frame = frame.copy()
        
        # Add controls information
        controls = [
            "Controls:",
            "m: Toggle mode (blur/color)",
            "r,g,b,w,k: Change color",
            "+,-: Adjust blur strength",
            "q: Quit"
        ]
        
        # Add mode and current settings
        status = [
            f"Mode: {self.mode}",
            f"Blur strength: {self.blur_strength}"
        ]
        
        # Add text to frame
        y = 30
        for text in controls + status:
            cv2.putText(display_frame, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y += 20
            
        cv2.imshow('Background Effect', display_frame)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with background effect.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Processed frame with background effect
        """
        try:
            # Get person mask
            mask = self.detector.detect(frame)
            
            if self.mode == 'blur':
                # Create strongly blurred background
                blurred = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
                blurred = cv2.GaussianBlur(blurred, (self.blur_strength + 10, self.blur_strength + 10), 0)
                background = blurred
            else:  # color mode
                # Create solid color background
                background = np.full_like(frame, self.color)
            
            # Combine original and background based on mask
            result = np.where(mask[:, :, None] == 1, frame, background)
            
            return result
        except Exception as e:
            logger.error("Error processing frame: %s", str(e))
            return frame
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
        cv2.destroyAllWindows()
        logger.info("Windows closed")

def main():
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--detector', type=str, default='mediapipe',
                          choices=['mediapipe', 'torch'],
                          help='Detector to use (mediapipe or torch)')
        args = parser.parse_args()
        
        if args.detector == 'mediapipe':
            logger.info("Initializing MediaPipe detector")
            from detector.models.mediapipe_detector import MediaPipeDetector
            detector = MediaPipeDetector()
        else:
            logger.info("Initializing PyTorch detector")
            from detector.models.torch_detector import TorchDetector
            detector = TorchDetector()
        
        logger.info("Creating BackgroundBlurApp")
        app = BackgroundBlurApp(detector, blur_strength=31)
        
        logger.info("Starting application")
        app.start()
    except Exception as e:
        logger.error("Application error: %s", str(e))

if __name__ == "__main__":
    main() 