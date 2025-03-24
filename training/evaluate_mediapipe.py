import os
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import logging
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .dataset import LaPaDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MediaPipeEvaluator:
    def __init__(self, image_size: tuple = (256, 256)):
        """
        Initialize the MediaPipe evaluator.
        
        Args:
            image_size: Size to resize images to before processing
        """
        self.image_size = image_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
    def process_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Process an image with MediaPipe and return a binary mask.
        
        Args:
            image: RGB image as torch tensor (C, H, W)
            
        Returns:
            Binary mask where 1 indicates face region
        """
        # Convert to numpy and ensure correct format
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Process image with MediaPipe (expects RGB)
        results = self.face_mesh.process(image_np)
        
        # Create empty mask
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if results.multi_face_landmarks:
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to points
            points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * self.image_size[1])  # width
                y = int(landmark.y * self.image_size[0])  # height
                points.append([x, y])
            
            # Create convex hull of face points
            points = np.array(points, dtype=np.int32)
            
            # Create mask using PIL for better compatibility
            mask_img = Image.new('L', (self.image_size[1], self.image_size[0]), 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask_img)
            hull = [tuple(point) for point in points[np.array(points).astype(np.float32).sum(axis=1).argsort()]]
            draw.polygon(hull, fill=1)
            mask = np.array(mask_img)
        
        return mask
    
    def calculate_metrics(self, dataset: LaPaDataset) -> dict:
        """
        Calculate accuracy and IoU metrics.
        
        Args:
            dataset: LaPa dataset instance
            
        Returns:
            Dictionary containing metrics
        """
        total_correct = 0
        total_pixels = 0
        class_iou = {i: {'intersection': 0, 'union': 0} for i in range(2)}  # Binary: face vs background
        
        for idx in tqdm(range(len(dataset)), desc='Evaluating'):
            # Get image and ground truth mask
            image, gt_mask = dataset[idx]
            
            # Get MediaPipe prediction
            pred_mask = self.process_image(image)
            
            # Convert ground truth to binary (face vs background)
            gt_binary = (gt_mask > 0).numpy()
            
            # Calculate accuracy
            total_correct += (pred_mask == gt_binary).sum()
            total_pixels += gt_binary.size
            
            # Calculate IoU
            intersection = (pred_mask & gt_binary).sum()
            union = (pred_mask | gt_binary).sum()
            
            class_iou[1]['intersection'] += intersection
            class_iou[1]['union'] += union
            
            # Background IoU
            bg_intersection = ((1 - pred_mask) & (1 - gt_binary)).sum()
            bg_union = ((1 - pred_mask) | (1 - gt_binary)).sum()
            
            class_iou[0]['intersection'] += bg_intersection
            class_iou[0]['union'] += bg_union
        
        # Calculate final metrics
        accuracy = total_correct / total_pixels
        ious = {
            cls: data['intersection'] / (data['union'] + 1e-6)
            for cls, data in class_iou.items()
        }
        mean_iou = sum(ious.values()) / len(ious)
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'class_ious': ious
        }
    
    def plot_roc_curves(self, dataset: LaPaDataset, save_path: str):
        """
        Generate and save ROC curves.
        
        Args:
            dataset: LaPa dataset instance
            save_path: Path to save the ROC curve plot
        """
        all_probs = []
        all_labels = []
        
        for idx in tqdm(range(len(dataset)), desc='Generating ROC curves'):
            # Get image and ground truth mask
            image, gt_mask = dataset[idx]
            
            # Get MediaPipe prediction
            pred_mask = self.process_image(image)
            
            # Convert ground truth to binary
            gt_binary = (gt_mask > 0).numpy()
            
            # Store predictions and labels
            all_probs.append(pred_mask.flatten())
            all_labels.append(gt_binary.flatten())
        
        # Concatenate all predictions and labels
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'Face Detection (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Face Detection')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
    
    def visualize_predictions(self, dataset: LaPaDataset, num_samples: int = 5, save_dir: str = 'evaluations'):
        """
        Visualize predictions and save ground truth masks.
        
        Args:
            dataset: LaPa dataset instance
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        vis_dir = save_dir / 'visualizations'
        pred_dir = save_dir / 'predictions'
        gt_dir = save_dir / 'ground_truth'
        vis_dir.mkdir(exist_ok=True)
        pred_dir.mkdir(exist_ok=True)
        gt_dir.mkdir(exist_ok=True)
        
        # Get random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in indices:
            # Get image and ground truth mask
            image, gt_mask = dataset[idx]
            
            # Convert to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = ((image_np * std + mean) * 255).astype(np.uint8)
            
            # Get MediaPipe prediction
            pred_mask = self.process_image(image)
            
            # Convert ground truth to binary
            gt_binary = (gt_mask > 0).numpy()
            
            # Save predictions and ground truth masks
            Image.fromarray(pred_mask * 255).save(pred_dir / f'pred_{idx:04d}.png')
            Image.fromarray(gt_binary.astype(np.uint8) * 255).save(gt_dir / f'gt_{idx:04d}.png')
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(131)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            
            # Ground truth
            plt.subplot(132)
            plt.imshow(gt_binary, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Prediction
            plt.subplot(133)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('MediaPipe Prediction')
            plt.axis('off')
            
            plt.savefig(vis_dir / f'comparison_{idx:04d}.png')
            plt.close()
            
            # Save original image
            Image.fromarray(image_np).save(vis_dir / f'original_{idx:04d}.png')
        
        logger.info(f"Saved {num_samples} samples to {save_dir}")
        logger.info(f"- Visualizations: {vis_dir}")
        logger.info(f"- Predictions: {pred_dir}")
        logger.info(f"- Ground Truth: {gt_dir}")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create dataset
    data_dir = 'data/LaPa'
    dataset = LaPaDataset(data_dir, split='val')
    
    # Create evaluator
    evaluator = MediaPipeEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(dataset)
    
    # Log metrics
    logger.info("MediaPipe Evaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
    logger.info(f"Background IoU: {metrics['class_ious'][0]:.4f}")
    logger.info(f"Face IoU: {metrics['class_ious'][1]:.4f}")
    
    # Generate ROC curves
    evaluator.plot_roc_curves(dataset, 'evaluations/mediapipe_roc.png')
    
    # Visualize predictions and save masks
    evaluator.visualize_predictions(dataset, num_samples=10)

if __name__ == "__main__":
    main() 