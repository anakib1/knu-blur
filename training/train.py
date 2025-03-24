import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import logging
from pathlib import Path
import gc

from .dataset import LaPaDataset
from .model import UNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def visualize_batch(images, masks, predictions, epoch, save_dir):
    """
    Visualize a batch of images, masks, and predictions.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        masks: Tensor of shape (B, H, W)
        predictions: Tensor of shape (B, 2, H, W)
        epoch: Current epoch number
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and denormalize images
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    # Get binary predictions
    pred_masks = (torch.sigmoid(predictions[:, 1]) > 0.5).float()  # Use class 1 (foreground) probability
    
    # Create visualization for each image in batch
    for i in range(min(4, len(images))):  # Show up to 4 images
        plt.figure(figsize=(15, 5))
        
        # Original image
        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = ((img * std + mean) * 255).astype(np.uint8)
        
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(132)
        plt.imshow(masks[i], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(133)
        plt.imshow(pred_masks[i], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.savefig(save_dir / f'epoch_{epoch:03d}_sample_{i:02d}.png')
        plt.close()

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        learning_rate: float = 0.001,
        num_epochs: int = 50
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Use Binary Cross Entropy with Logits for binary segmentation
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(self.train_loader, desc='Training')):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Take only the foreground class logits and ensure correct shape
            outputs = outputs[:, 1]  # Shape: [B, H, W]
            
            loss = self.criterion(outputs, masks.float())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Visualize first batch of each epoch
            if batch_idx == 0:
                # For visualization, we need to reshape back to [B, 2, H, W]
                vis_outputs = torch.stack([-outputs, outputs], dim=1)
                visualize_batch(images, masks, vis_outputs, len(self.train_losses), 'checkpoints/train_vis')
            
            # Clear GPU memory if needed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.val_loader, desc='Validating')):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # Take only the foreground class logits and ensure correct shape
                outputs = outputs[:, 1]  # Shape: [B, H, W]
                
                loss = self.criterion(outputs, masks.float())
                
                total_loss += loss.item()
                
                # Visualize first batch of each validation
                if batch_idx == 0:
                    # For visualization, we need to reshape back to [B, 2, H, W]
                    vis_outputs = torch.stack([-outputs, outputs], dim=1)
                    visualize_batch(images, masks, vis_outputs, len(self.train_losses), 'checkpoints/val_vis')
                
                # Clear GPU memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return total_loss / len(self.val_loader)
    
    def calculate_metrics(self, loader: DataLoader) -> dict:
        """Calculate accuracy and IoU metrics."""
        self.model.eval()
        total_correct = 0
        total_pixels = 0
        intersection = 0
        union = 0
        
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                outputs = outputs[:, 1]  # Take foreground logits
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                # Calculate accuracy
                total_correct += (predictions == masks).sum().item()
                total_pixels += masks.numel()
                
                # Calculate IoU using multiplication instead of bitwise operations
                intersection += (predictions * masks).sum().item()
                union += ((predictions + masks) > 0).float().sum().item()
                
                # Clear GPU memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate final metrics
        accuracy = total_correct / total_pixels
        iou = intersection / (union + 1e-6)
        
        return {
            'accuracy': accuracy,
            'iou': iou
        }
    
    def plot_roc_curves(self, loader: DataLoader, save_path: str):
        """Generate and save ROC curve."""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs[:, 1])  # Take foreground probabilities
                
                all_probs.append(probs.cpu().flatten().numpy())
                all_labels.append(masks.cpu().flatten().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
    
    def train(self, save_dir: str):
        """Train the model."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(self.train_loader)
            val_metrics = self.calculate_metrics(self.val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Train IoU: {train_metrics['iou']:.4f}")
            logger.info(f"Val IoU: {val_metrics['iou']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pth')
                logger.info("Saved best model")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(save_dir / 'loss_curves.png')
            plt.close()
            
            # Generate ROC curve
            self.plot_roc_curves(self.val_loader, save_dir / 'roc_curve.png')
            
            # Clear memory after each epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create datasets with caching and prefetching
    data_dir = 'data/LaPa'
    image_size = (256, 256)
    
    # Define transforms
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LaPaDataset(
        data_dir, 
        split='train',
        transform=transform,
        cache_size=1000,
        prefetch_size=100
    )
    
    val_dataset = LaPaDataset(
        data_dir,
        split='val',
        transform=transform,
        cache_size=500,
        prefetch_size=50
    )
    
    # Create data loaders
    batch_size = 32 if device.type == 'mps' else 16  # Increased batch size for smaller model
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Using dataset's own prefetching
        pin_memory=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Using dataset's own prefetching
        pin_memory=True,
        persistent_workers=False
    )
    
    # Create model with specified size
    model = UNet(
        n_classes=2,  # Binary segmentation: background and foreground
        size='small',  # Use small model by default
        in_channels=3
    )
    logger.info(f"Created model with {model.get_model_size():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=50
    )
    
    try:
        # Train model
        trainer.train(save_dir='checkpoints')
    finally:
        # Clean up
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main() 