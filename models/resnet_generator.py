"""
ResNet-based Generator for CycleGAN

This module implements the ResNet generator architecture as described in the CycleGAN paper:
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

Architecture Overview:
- Input: 256x256 RGB image (3 channels)
- Initial Convolution: 7x7 conv with reflection padding → 64 filters
- Downsampling: 2 layers (3x3 conv, stride=2) → 64→128→256 filters
- Residual Blocks: 9 blocks with 256 filters each
- Upsampling: 2 layers (transposed conv, stride=2) → 256→128→64 filters
- Output: 7x7 conv with reflection padding → 3 channels, Tanh activation

Key Design Choices:
- Reflection padding: Reduces border artifacts compared to zero padding
- Instance Normalization: Better for image generation than BatchNorm (especially with small batches)
- Residual blocks: Enable deeper networks and better gradient flow
- 9 residual blocks: Optimal for 256x256 images (6 blocks for 128x128)
"""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    Residual Block for ResNet Generator
    
    Each block consists of:
    1. Reflection padding (1 pixel)
    2. 3x3 Convolution (dim → dim)
    3. Instance Normalization
    4. ReLU activation
    5. Reflection padding (1 pixel)
    6. 3x3 Convolution (dim → dim)
    7. Instance Normalization
    8. Residual connection (input + output)
    
    The residual connection helps with gradient flow and allows the network
    to learn identity mappings when needed.
    """
    
    def __init__(self, dim):
        """
        Args:
            dim: Number of input/output channels (typically 256 in the middle layers)
        """
        super(ResNetBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            # First convolution
            nn.ReflectionPad2d(1),  # Pad 1 pixel on all sides (total padding = 2)
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor of shape (batch, dim, height, width)
            
        Returns:
            Output tensor of same shape as input
        """
        # Residual connection: output = input + transformed(input)
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based Generator for CycleGAN
    
    This generator transforms images from one domain to another (e.g., photo→Monet).
    The architecture uses:
    - Reflection padding to reduce border artifacts
    - Instance normalization for stable training
    - Residual blocks for deep feature learning
    - Transposed convolutions for upsampling
    
    Architecture flow:
    Input (256x256x3) 
    → Initial Conv (256x256x64)
    → Downsample 1 (128x128x128)
    → Downsample 2 (64x64x256)
    → 9x Residual Blocks (64x64x256)
    → Upsample 1 (128x128x128)
    → Upsample 2 (256x256x64)
    → Final Conv (256x256x3)
    """
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """
        Initialize ResNet Generator
        
        Args:
            input_nc: Number of input channels (default: 3 for RGB)
            output_nc: Number of output channels (default: 3 for RGB)
            ngf: Number of generator filters in first conv layer (default: 64)
            n_blocks: Number of residual blocks (default: 9 for 256x256 images)
        """
        assert n_blocks >= 0, "Number of residual blocks must be non-negative"
        super(ResNetGenerator, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        model = []
        
        # ========== Initial Convolutional Layer ==========
        # 7x7 conv with reflection padding to maintain spatial dimensions
        # Input: (batch, 3, 256, 256) → Output: (batch, 64, 256, 256)
        model += [
            nn.ReflectionPad2d(3),  # Pad 3 pixels → total 6 pixels (for 7x7 kernel)
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # ========== Downsampling Layers ==========
        # Reduce spatial dimensions while increasing feature channels
        # 256x256 → 128x128 → 64x64
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i  # 1, 2
            model += [
                nn.Conv2d(
                    ngf * mult,                    # Input channels: 64, 128
                    ngf * mult * 2,                # Output channels: 128, 256
                    kernel_size=3,
                    stride=2,                      # Halve spatial dimensions
                    padding=1,                     # Maintain kernel size
                    bias=False
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # ========== Residual Blocks ==========
        # Core feature transformation layers
        # Maintains 64x64 spatial size with 256 channels
        mult = 2 ** n_downsampling  # 4 → 256 channels
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]
        
        # ========== Upsampling Layers ==========
        # Increase spatial dimensions while decreasing feature channels
        # 64x64 → 128x128 → 256x256
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)  # 4, 2
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,                    # Input channels: 256, 128
                    int(ngf * mult / 2),           # Output channels: 128, 64
                    kernel_size=3,
                    stride=2,                      # Double spatial dimensions
                    padding=1,                     # Standard padding
                    output_padding=1,              # Ensure exact size match
                    bias=False
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # ========== Final Convolutional Layer ==========
        # Convert features back to RGB image
        # Output: (batch, 3, 256, 256) with values in [-1, 1]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Scale output to [-1, 1] range
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        """
        Forward pass through the generator
        
        Args:
            input: Input image tensor of shape (batch, 3, 256, 256)
                  Values should be in range [-1, 1] (normalized)
        
        Returns:
            Generated image tensor of shape (batch, 3, 256, 256)
            Values in range [-1, 1]
        """
        return self.model(input)

