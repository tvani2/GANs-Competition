"""
U-Net-based Generator for CycleGAN

This module implements the U-Net generator architecture, which uses skip connections
to preserve fine-grained details during image translation.

Architecture Overview:
- Input: 256x256 RGB image (3 channels)
- Encoder: Downsampling path with skip connections
- Bottleneck: Feature transformation layers
- Decoder: Upsampling path with skip connections from encoder
- Output: 256x256 RGB image (3 channels), Tanh activation

Key Design Choices:
- Skip connections: Preserve fine details by connecting encoder to decoder
- Reflection padding: Reduces border artifacts
- Instance Normalization: Better for image generation than BatchNorm
- U-Net structure: Better at preserving spatial details than ResNet
"""

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """
    U-Net Block with optional skip connection
    
    Can be used in:
    - Encoder (downsampling): no skip connection
    - Decoder (upsampling): with skip connection from encoder
    - Bottleneck: no skip connection
    """
    
    def __init__(self, in_channels, out_channels, down=True, use_skip=False, use_dropout=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            down: If True, downsampling (encoder). If False, upsampling (decoder)
            use_skip: If True, use skip connection (for decoder blocks)
            use_dropout: If True, apply dropout (typically in decoder)
        """
        super(UNetBlock, self).__init__()
        
        self.use_skip = use_skip
        
        if down:
            # Encoder: Downsampling
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            # Decoder: Upsampling
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
            ]
            
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            
            layers.append(nn.ReLU(inplace=True))
            self.conv = nn.Sequential(*layers)
    
    def forward(self, x, skip=None):
        """
        Forward pass
        
        Args:
            x: Input tensor
            skip: Skip connection tensor from encoder (if use_skip=True)
            
        Returns:
            Output tensor
        """
        if self.use_skip and skip is not None:
            # Concatenate skip connection along channel dimension
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class UNetGenerator(nn.Module):
    """
    U-Net-based Generator for CycleGAN
    
    This generator uses skip connections to preserve fine details during translation.
    Better at preserving spatial information than ResNet generator.
    
    Architecture:
    Encoder (downsampling):
    - Input (256x256x3)
    - Conv 64 (128x128x64)
    - Conv 128 (64x64x128)
    - Conv 256 (32x32x256)
    - Conv 512 (16x16x512)
    - Conv 512 (8x8x512)
    - Conv 512 (4x4x512)
    - Conv 512 (2x2x512)
    - Conv 512 (1x1x512) - bottleneck
    
    Decoder (upsampling with skip connections):
    - Deconv 512 (2x2x512) + skip from encoder
    - Deconv 512 (4x4x512) + skip
    - Deconv 512 (8x8x512) + skip
    - Deconv 512 (16x16x512) + skip
    - Deconv 256 (32x32x256) + skip
    - Deconv 128 (64x64x128) + skip
    - Deconv 64 (128x128x64) + skip
    - Final conv (256x256x3)
    """
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        """
        Initialize U-Net Generator
        
        Args:
            input_nc: Number of input channels (default: 3 for RGB)
            output_nc: Number of output channels (default: 3 for RGB)
            ngf: Number of generator filters in first conv layer (default: 64)
        """
        super(UNetGenerator, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        # ========== Encoder (Downsampling) ==========
        # Each layer halves spatial dimensions and doubles channels
        self.enc1 = UNetBlock(input_nc, ngf, down=True)  # 256→128
        self.enc2 = UNetBlock(ngf, ngf * 2, down=True)  # 128→64
        self.enc3 = UNetBlock(ngf * 2, ngf * 4, down=True)  # 64→32
        self.enc4 = UNetBlock(ngf * 4, ngf * 8, down=True)  # 32→16
        self.enc5 = UNetBlock(ngf * 8, ngf * 8, down=True)  # 16→8
        self.enc6 = UNetBlock(ngf * 8, ngf * 8, down=True)  # 8→4
        self.enc7 = UNetBlock(ngf * 8, ngf * 8, down=True)  # 4→2
        self.enc8 = UNetBlock(ngf * 8, ngf * 8, down=True)  # 2→1 (bottleneck)
        
        # ========== Decoder (Upsampling with Skip Connections) ==========
        # Each layer doubles spatial dimensions and halves channels
        # Skip connections concatenate encoder features
        self.dec1 = UNetBlock(ngf * 8, ngf * 8, down=False, use_skip=True, use_dropout=True)  # 1→2
        self.dec2 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_skip=True, use_dropout=True)  # 2→4
        self.dec3 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_skip=True, use_dropout=True)  # 4→8
        self.dec4 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_skip=True)  # 8→16
        self.dec5 = UNetBlock(ngf * 8 * 2, ngf * 4, down=False, use_skip=True)  # 16→32
        self.dec6 = UNetBlock(ngf * 4 * 2, ngf * 2, down=False, use_skip=True)  # 32→64
        self.dec7 = UNetBlock(ngf * 2 * 2, ngf, down=False, use_skip=True)  # 64→128
        
        # ========== Final Output Layer ==========
        # Convert to RGB with Tanh activation
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Scale output to [-1, 1] range
        )
    
    def forward(self, input):
        """
        Forward pass through U-Net generator
        
        Args:
            input: Input image tensor of shape (batch, 3, 256, 256)
                  Values should be in range [-1, 1] (normalized)
        
        Returns:
            Generated image tensor of shape (batch, 3, 256, 256)
            Values in range [-1, 1]
        """
        # Encoder (downsampling) - save outputs for skip connections
        e1 = self.enc1(input)  # 128x128
        e2 = self.enc2(e1)  # 64x64
        e3 = self.enc3(e2)  # 32x32
        e4 = self.enc4(e3)  # 16x16
        e5 = self.enc5(e4)  # 8x8
        e6 = self.enc6(e5)  # 4x4
        e7 = self.enc7(e6)  # 2x2
        e8 = self.enc8(e7)  # 1x1 (bottleneck)
        
        # Decoder (upsampling with skip connections)
        d1 = self.dec1(e8, e7)  # 2x2
        d2 = self.dec2(d1, e6)  # 4x4
        d3 = self.dec3(d2, e5)  # 8x8
        d4 = self.dec4(d3, e4)  # 16x16
        d5 = self.dec5(d4, e3)  # 32x32
        d6 = self.dec6(d5, e2)  # 64x64
        d7 = self.dec7(d6, e1)  # 128x128
        
        # Final output
        output = self.final(d7)  # 256x256
        
        return output

