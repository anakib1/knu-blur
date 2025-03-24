import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        num_classes: int,
        learning_rate: float = 0.001,
        num_epochs: int = 50
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on
            num_classes: Number of classes
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
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
        
        for images, masks in tqdm(self.train_loader, desc='Training'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Clear GPU memory if needed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Clear GPU memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
        return total_loss / len(self.val_loader)
    
    def calculate_metrics(self, loader: DataLoader) -> dict:
        """Calculate accuracy and IoU metrics."""
        self.model.eval()
        total_correct = 0
        total_pixels = 0
        class_iou = {i: {'intersection': 0, 'union': 0} for i in range(self.num_classes)}
        
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Calculate accuracy
                total_correct += (predictions == masks).sum().item()
                total_pixels += masks.numel()
                
                # Calculate IoU for each class
                for cls in range(self.num_classes):
                    pred_mask = (predictions == cls)
                    true_mask = (masks == cls)
                    
                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    
                    class_iou[cls]['intersection'] += intersection
                    class_iou[cls]['union'] += union
                
                # Clear GPU memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
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
    
    def plot_roc_curves(self, loader: DataLoader, save_path: str):
        """Generate and save ROC curves for each class."""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(masks.cpu().numpy())
                
                # Clear GPU memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(
                (all_labels == i).flatten(),
                all_probs[:, i].flatten()
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
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
            logger.info(f"Train Mean IoU: {train_metrics['mean_iou']:.4f}")
            logger.info(f"Val Mean IoU: {val_metrics['mean_iou']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_dir / 'best_model.pth')
                logger.info("Saved best model")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Clear memory after each epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
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
        
        # Generate ROC curves
        self.plot_roc_curves(self.val_loader, save_dir / 'roc_curves.png')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create datasets with caching and prefetching
    data_dir = 'data/LaPa'
    train_dataset = LaPaDataset(
        data_dir, 
        split='train', 
        cache_size=1000,
        prefetch_size=100  # Prefetch 100 samples in background
    )
    val_dataset = LaPaDataset(
        data_dir, 
        split='val', 
        cache_size=500,
        prefetch_size=50  # Prefetch 50 samples in background
    )
    
    # Create data loaders with optimized settings
    batch_size = 16 if device.type == 'mps' else 8
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 since we're using our own prefetching
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Keep at 0 since we're using our own prefetching
        pin_memory=True,
        persistent_workers=False
    )
    
    # Create model
    model = UNet(n_classes=train_dataset.num_classes)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=train_dataset.num_classes,
        learning_rate=0.001,
        num_epochs=50
    )
    
    try:
        # Train model
        trainer.train(save_dir='checkpoints')
    finally:
        # Ensure we clean up resources
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main() 