import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import logging
from typing import Optional, Tuple, Dict, List
import threading
from queue import Queue
import time

logger = logging.getLogger(__name__)

class LaPaDataset(Dataset):
    """Dataset class for LaPa face parsing dataset with caching and prefetching."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, cache_size: int = 1000, prefetch_size: int = 100):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the LaPa dataset
            split: One of ['train', 'val', 'test']
            transform: Optional transforms to apply to images
            cache_size: Number of samples to keep in memory cache
            prefetch_size: Number of samples to prefetch in background
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = (256, 256)
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        
        # Define transforms for images and masks
        self.image_transform = transform or T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        
        # Load image and label paths
        self.image_dir = self.root_dir / split / 'images'
        self.label_dir = self.root_dir / split / 'labels'
        
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')))[:100]
        self.label_files = sorted(list(self.label_dir.glob('*.png')))[:100]
        
        if len(self.image_files) != len(self.label_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) "
                           f"doesn't match number of labels ({len(self.label_files)})")
        
        # Initialize cache and prefetch queue
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_lock = threading.Lock()
        self.prefetch_queue = Queue(maxsize=prefetch_size)
        self.prefetch_thread = None
        self.stop_prefetch = False
        
        # For binary segmentation (background vs. foreground)
        self.num_classes = 2
        
        # Start prefetching thread
        self.start_prefetch_thread()
        
        # Preload some samples into cache
        logger.info(f"Preloading {min(cache_size, len(self))} samples into cache...")
        for i in range(min(cache_size, len(self))):
            self._load_and_cache_item(i)
    
    def start_prefetch_thread(self):
        """Start the background prefetching thread."""
        self.stop_prefetch = False
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.info("Started prefetching thread")
    
    def stop_prefetch_thread(self):
        """Stop the background prefetching thread."""
        self.stop_prefetch = True
        if self.prefetch_thread:
            self.prefetch_thread.join()
            logger.info("Stopped prefetching thread")
    
    def _prefetch_worker(self):
        """Background worker that prefetches data."""
        current_idx = 0
        while not self.stop_prefetch:
            try:
                # Try to add items to the prefetch queue
                while not self.prefetch_queue.full() and current_idx < len(self):
                    if current_idx not in self.cache:
                        try:
                            image, mask = self._load_and_cache_item(current_idx)
                            self.prefetch_queue.put((current_idx, image, mask), timeout=1)
                        except Exception as e:
                            logger.warning(f"Error prefetching item {current_idx}: {e}")
                    current_idx = (current_idx + 1) % len(self)  # Cycle through dataset
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                time.sleep(1)  # Sleep longer on error
    
    def _load_and_cache_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and transform a single item from disk."""
        # Load image using torchvision
        image = Image.open(self.image_files[idx]).convert('RGB')
        image = self.image_transform(image)
        
        # Load mask using torchvision
        mask = Image.open(self.label_files[idx])
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0)  # Remove channel dimension
        
        # Convert multi-class mask to binary (0: background, 1: foreground/face)
        mask = (mask > 0).long()  # Any non-zero class is considered foreground
        
        return image, mask
    
    def _get_cached_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from cache or load it if not cached."""
        with self.cache_lock:
            if idx in self.cache:
                return self.cache[idx]
            
            # Try to get from prefetch queue first
            try:
                while not self.prefetch_queue.empty():
                    prefetch_idx, image, mask = self.prefetch_queue.get_nowait()
                    if prefetch_idx not in self.cache:
                        if len(self.cache) >= self.cache_size:
                            oldest_idx = min(self.cache.keys())
                            del self.cache[oldest_idx]
                        self.cache[prefetch_idx] = (image, mask)
                        if prefetch_idx == idx:
                            return image, mask
            except Exception as e:
                logger.warning(f"Error getting from prefetch queue: {e}")
            
            # If still not in cache, load directly
            if idx not in self.cache:
                image, mask = self._load_and_cache_item(idx)
                if len(self.cache) >= self.cache_size:
                    oldest_idx = min(self.cache.keys())
                    del self.cache[oldest_idx]
                self.cache[idx] = (image, mask)
            
            return self.cache[idx]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image, mask) where:
                - image: Tensor of shape (3, H, W)
                - mask: Tensor of shape (H, W) with class indices
        """
        return self._get_cached_item(idx)
    
    def clear_cache(self):
        """Clear the cache and stop prefetching."""
        with self.cache_lock:
            self.cache.clear()
            self.stop_prefetch_thread()
            logger.info("Dataset cache cleared and prefetching stopped") 