"""
PatchGAN Discriminator for CycleGAN with Hinge Loss Support

This module implements the PatchGAN discriminator architecture as described in the CycleGAN paper.
PatchGAN classifies local image patches as real or fake, which is more effective than 
classifying the entire image for texture and style transfer tasks.

Architecture Overview (for 256x256 input):
- Input: 256x256 RGB image (3 channels)
- Layer 1: 4x4 conv, stride=2 → 128x128x64 (no normalization)
- Layer 2: 4x4 conv, stride=2 → 64x64x128 (with InstanceNorm)
- Layer 3: 4x4 conv, stride=2 → 32x32x256 (with InstanceNorm)
- Layer 4: 4x4 conv, stride=1 → 32x32x512 (with InstanceNorm)
- Output: 4x4 conv, stride=1 → 32x32x1 (patch predictions)

The output is a 32x32 patch where each value represents the authenticity of a 
corresponding 70x70 patch in the input image (receptive field = 70x70).

Key Design Choices:
- LeakyReLU (slope=0.2): Allows gradients to flow through negative values
- No normalization in first layer: Preserves original image statistics
- Instance Normalization: Better for image generation than BatchNorm
- Patch-based output: More efficient and better at capturing local textures
- Raw logits output: Compatible with Hinge loss (no sigmoid)
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for CycleGAN
    
    This discriminator outputs a patch of predictions rather than a single value.
    Each output value corresponds to a local patch in the input image, making it
    more effective at capturing fine-grained textures and details.
    
    The architecture uses:
    - 4x4 convolutions (standard for GANs)
    - Stride 2 for downsampling (except last two layers)
    - LeakyReLU activation (slope=0.2)
    - Instance Normalization (except first layer)
    - Raw logits output (for Hinge loss compatibility)
    """
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """
        Initialize PatchGAN Discriminator
        
        Args:
            input_nc: Number of input channels (default: 3 for RGB)
            ndf: Number of discriminator filters in first conv layer (default: 64)
            n_layers: Number of layers in the discriminator (default: 3)
                     For 256x256 images, typically use 3 layers (70x70 PatchGAN)
        """
        super(PatchGANDiscriminator, self).__init__()
        
        self.input_nc = input_nc
        self.ndf = ndf
        
        # Build the discriminator layers
        model = []
        
        # ========== First Layer ==========
        # No normalization in the first layer to preserve original image statistics
        # Input: (batch, 3, 256, 256) → Output: (batch, 64, 128, 128)
        model += [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # ========== Intermediate Layers ==========
        # Downsample while increasing feature channels
        # 128x128 → 64x64 → 32x32
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # n_layers=3: creates 2 intermediate layers
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8 (512 filters max)
            model += [
                nn.Conv2d(
                    ndf * nf_mult_prev,      # Input: 64, 128
                    ndf * nf_mult,           # Output: 128, 256
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # ========== Second-to-Last Layer ==========
        # Stride 1 to maintain spatial dimensions
        # Input: (batch, 256, 32, 32) → Output: (batch, 512, 32, 32)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)  # 8 → 512 filters
        model += [
            nn.Conv2d(
                ndf * nf_mult_prev,      # 256
                ndf * nf_mult,           # 512
                kernel_size=4,
                stride=1,                # No downsampling
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # ========== Final Layer ==========
        # Output patch predictions (one value per patch)
        # Input: (batch, 512, 32, 32) → Output: (batch, 1, 32, 32)
        model += [
            nn.Conv2d(
                ndf * nf_mult,           # 512
                1,                       # Single output channel
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            )
            # No activation - raw logits (for Hinge loss compatibility)
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        """
        Forward pass through the discriminator
        
        Args:
            input: Input image tensor of shape (batch, 3, 256, 256)
                  Values should be in range [-1, 1] (normalized)
        
        Returns:
            Patch predictions tensor of shape (batch, 1, 32, 32)
            Each value represents the authenticity of a 70x70 patch in the input
            Values are raw logits (not passed through sigmoid) - compatible with Hinge loss
        """
        return self.model(input)
