import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Configurable UNet implementation.
    
    Args:
        n_classes: Number of output classes
        in_channels: Number of input channels (default: 3 for RGB)
        base_features: Number of features in the first layer (default: 32)
        depth: Number of downsampling/upsampling steps (default: 4)
        size: Predefined size configuration ('small', 'medium', 'large')
    """
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 3,
        base_features: int = None,
        depth: int = None,
        size: str = 'small'
    ):
        super().__init__()
        
        # Predefined configurations
        configs = {
            'small': {'base_features': 16, 'depth': 3},    # Lightweight
            'medium': {'base_features': 32, 'depth': 4},   # Standard
            'large': {'base_features': 64, 'depth': 5}     # Full size
        }
        
        # Use provided values or get from config
        config = configs[size]
        self.base_features = base_features or config['base_features']
        self.depth = depth or config['depth']
        
        # Input convolution
        self.input_conv = DoubleConv(in_channels, self.base_features)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        features = self.base_features
        for i in range(self.depth):
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(features, features * 2)
            ))
            features *= 2
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(self.depth):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2),
                DoubleConv(features, features // 2)  # features because of skip connection
            ))
            features //= 2
        
        # Output convolution
        self.output_conv = nn.Conv2d(self.base_features, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.input_conv(x)
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        encoder_outputs.append(x)
        
        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            encoder_outputs.append(x)
        
        # Decoder path with skip connections
        for i, dec in enumerate(self.decoder):
            x = dec[0](x)  # Upsample
            skip = encoder_outputs[-(i+2)]
            
            # Handle cases where feature maps have different sizes
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = dec[1](x)  # Double convolution
        
        # Final convolution
        return self.output_conv(x)
    
    def get_model_size(self):
        """Calculate and return the number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(n_classes: int, size: str = 'small', **kwargs):
    """
    Factory function to create a UNet model with predefined configurations.
    
    Args:
        n_classes: Number of output classes
        size: Model size ('small', 'medium', 'large')
        **kwargs: Additional arguments to pass to UNet
    
    Returns:
        Configured UNet model
    """
    model = UNet(n_classes=n_classes, size=size, **kwargs)
    print(f"Created {size} UNet with {model.get_model_size():,} parameters") 