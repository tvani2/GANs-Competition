"""
Enhanced CycleGAN Loss Functions with LSGAN - Production Ready

Based on Epoch 40 analysis showing:
- Sharp edges (need softening)
- Too much photographic detail (need suppression)

Adds 2 carefully selected perceptual losses:
1. Edge Softness Loss (λ=0.1) - PRIMARY: Soften sharp edges
2. High-Frequency Suppression (λ=0.05) - SECONDARY: Remove photographic sharpness

These are the most impactful for your images. Ready for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.losses_cyclegan_lsgan import (
    lsgan_loss_discriminator,
    lsgan_loss_generator,
    cycle_consistency_loss,
    identity_loss
)


class EdgeSoftnessLoss(nn.Module):
    """
    Edge Softness Loss using Sobel Edge Detection
    
    Monet has soft edges (~0.34 Sobel strength) vs photos (sharper).
    This is the MOST IMPORTANT loss for your images.
    """
    
    def __init__(self, weight=1.0):
        super(EdgeSoftnessLoss, self).__init__()
        self.weight = weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_edge_strength(self, gray_image):
        """Compute Sobel edge strength"""
        # Ensure Sobel filters match input dtype
        sobel_x = self.sobel_x.to(dtype=gray_image.dtype)
        sobel_y = self.sobel_y.to(dtype=gray_image.dtype)
        
        gx = F.conv2d(gray_image, sobel_x, padding=1)
        gy = F.conv2d(gray_image, sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return magnitude.view(magnitude.shape[0], -1).mean(dim=1)
    
    def forward(self, generated, target):
        """
        Compare edge strength between generated and target images
        
        Args:
            generated: Generated Monet images (B, 3, H, W) in [-1, 1]
            target: Real Monet images (B, 3, H, W) in [-1, 1]
        """
        # Convert to grayscale (ensure dtype compatibility)
        dtype = generated.dtype
        gray_gen = (generated[:, 0:1] * 0.299 + generated[:, 1:2] * 0.587 + generated[:, 2:3] * 0.114).to(dtype)
        gray_gen = (gray_gen + 1.0) / 2.0  # Normalize to [0, 1]
        
        gray_target = (target[:, 0:1] * 0.299 + target[:, 1:2] * 0.587 + target[:, 2:3] * 0.114).to(dtype)
        gray_target = (gray_target + 1.0) / 2.0
        
        edge_gen = self.compute_edge_strength(gray_gen)
        edge_target = self.compute_edge_strength(gray_target)
        
        # L1 loss between edge strengths
        loss = F.l1_loss(edge_gen, edge_target)
        return self.weight * loss


class HighFreqSuppressionLoss(nn.Module):
    """
    High-Frequency Suppression Loss using FFT
    
    Monet suppresses fine details (high-freq PSD ~1.75) vs photos (higher).
    This removes photographic sharpness.
    """
    
    def __init__(self, weight=1.0):
        super(HighFreqSuppressionLoss, self).__init__()
        self.weight = weight
    
    def compute_high_freq_psd(self, gray_image):
        """Compute high-frequency PSD using FFT"""
        import numpy as np
        
        batch_size = gray_image.shape[0]
        psd_values = []
        
        for i in range(batch_size):
            # Convert to float32 for numpy operations, then back to original dtype
            img = gray_image[i, 0].float().cpu().numpy()
            
            # Compute 2D FFT
            fft = np.fft.fft2(img)
            fft_shifted = np.fft.fftshift(fft)
            
            # Power spectral density
            psd = np.abs(fft_shifted)**2
            psd_log = np.log10(psd + 1e-10)
            
            # Extract high-frequency region (outer region)
            h, w = psd_log.shape
            center_h, center_w = h // 2, w // 2
            
            # High frequency: outer region (not in center 1/8)
            high_mask = np.ones((h, w), dtype=bool)
            high_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = False
            high_freq = psd_log[high_mask].mean()
            
            psd_values.append(high_freq)
        
        # Return tensor with same dtype as input
        return torch.tensor(psd_values, dtype=gray_image.dtype, device=gray_image.device)
    
    def forward(self, generated, target):
        """
        Compare high-frequency content between generated and target
        
        Args:
            generated: Generated Monet images (B, 3, H, W) in [-1, 1]
            target: Real Monet images (B, 3, H, W) in [-1, 1]
        """
        # Convert to grayscale (ensure dtype compatibility)
        dtype = generated.dtype
        gray_gen = (generated[:, 0:1] * 0.299 + generated[:, 1:2] * 0.587 + generated[:, 2:3] * 0.114).to(dtype)
        gray_gen = (gray_gen + 1.0) / 2.0
        
        gray_target = (target[:, 0:1] * 0.299 + target[:, 1:2] * 0.587 + target[:, 2:3] * 0.114).to(dtype)
        gray_target = (gray_target + 1.0) / 2.0
        
        psd_gen = self.compute_high_freq_psd(gray_gen)
        psd_target = self.compute_high_freq_psd(gray_target)
        
        # L1 loss between high-freq PSDs
        loss = F.l1_loss(psd_gen, psd_target)
        return self.weight * loss


class EnhancedCycleGANLossesLSGAN:
    """
    Production-Ready Enhanced CycleGAN Losses with LSGAN
    
    Adds 2 perceptual losses based on your Epoch 39 analysis:
    - Edge Softness (λ=0.1): Soften sharp edges - MOST IMPORTANT
    - High-Freq Suppression (λ=0.05): Remove photographic detail
    
    Usage:
        losses = EnhancedCycleGANLossesLSGAN(
            lambda_cycle=10.0,
            lambda_identity=0.5,
            lambda_edge=0.1,
            lambda_freq=0.05
        )
    """
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5, 
                 lambda_edge=0.1, lambda_freq=0.05):
        """
        Initialize Enhanced CycleGAN Losses
        
        Args:
            lambda_cycle: Weight for cycle consistency loss (default: 10.0)
            lambda_identity: Weight for identity loss (default: 0.5)
            lambda_edge: Weight for edge softness loss (default: 0.1)
            lambda_freq: Weight for high-freq suppression (default: 0.05)
        """
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq
        
        # Initialize perceptual losses
        self.edge_softness_loss = EdgeSoftnessLoss(weight=1.0)
        self.high_freq_loss = HighFreqSuppressionLoss(weight=1.0)
        
        print(f"✅ Enhanced LSGAN losses initialized:")
        print(f"   - Cycle consistency: λ={lambda_cycle}")
        print(f"   - Identity: λ={lambda_identity}")
        print(f"   - Edge softness: λ={lambda_edge} (Target: Monet ~0.34)")
        print(f"   - High-freq suppression: λ={lambda_freq} (Target: Monet ~1.75)")
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Compute discriminator loss using LSGAN (unchanged)"""
        return lsgan_loss_discriminator(real_pred, fake_pred)
    
    def generator_adversarial_loss(self, fake_pred):
        """Compute generator adversarial loss using LSGAN (unchanged)"""
        return lsgan_loss_generator(fake_pred)
    
    def generator_loss(self, fake_A_pred, fake_B_pred,
                      real_A, cycle_A, real_B, cycle_B,
                      fake_A, fake_B,
                      identity_A=None, identity_B=None):
        """
        Compute total generator loss with perceptual enhancements
        
        Args:
            fake_A_pred: Discriminator predictions on fake A
            fake_B_pred: Discriminator predictions on fake B
            real_A: Real photos
            cycle_A: Reconstructed photos (A→B→A)
            real_B: Real Monet paintings
            cycle_B: Reconstructed Monet (B→A→B)
            fake_A: Generated photos (B→A)
            fake_B: Generated Monet (A→B) - for perceptual losses
            identity_A: Identity mapping for A (optional)
            identity_B: Identity mapping for B (optional)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Standard LSGAN adversarial losses
        adv_loss_A = self.generator_adversarial_loss(fake_B_pred)
        adv_loss_B = self.generator_adversarial_loss(fake_A_pred)
        adv_loss = adv_loss_A + adv_loss_B
        
        # Cycle consistency losses
        cycle_loss_A = cycle_consistency_loss(real_A, cycle_A, self.lambda_cycle)
        cycle_loss_B = cycle_consistency_loss(real_B, cycle_B, self.lambda_cycle)
        cycle_loss = cycle_loss_A + cycle_loss_B
        
        # Identity losses (optional)
        identity_loss_total = 0.0
        if identity_A is not None and identity_B is not None:
            identity_loss_A = identity_loss(real_A, identity_A, self.lambda_identity)
            identity_loss_B = identity_loss(real_B, identity_B, self.lambda_identity)
            identity_loss_total = identity_loss_A + identity_loss_B
        
        # Perceptual losses (NEW)
        edge_loss_value = 0.0
        freq_loss_value = 0.0
        
        try:
            # 1. Edge Softness Loss - Soften sharp edges
            if self.lambda_edge > 0:
                edge_loss_B = self.edge_softness_loss(fake_B, real_B)
                edge_loss_value = self.lambda_edge * edge_loss_B
            
            # 2. High-Frequency Suppression - Remove photographic detail
            if self.lambda_freq > 0:
                freq_loss_B = self.high_freq_loss(fake_B, real_B)
                freq_loss_value = self.lambda_freq * freq_loss_B
                
        except Exception as e:
            print(f"Warning: Perceptual loss computation failed: {e}")
        
        # Total loss
        total_loss = adv_loss + cycle_loss + identity_loss_total + edge_loss_value + freq_loss_value
        
        return {
            'adversarial': adv_loss,
            'cycle': cycle_loss,
            'identity': identity_loss_total,
            'edge': edge_loss_value,
            'freq': freq_loss_value,
            'total': total_loss
        }
        
