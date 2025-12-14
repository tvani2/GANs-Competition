"""
CycleGAN Loss Functions

This module implements all loss functions needed for CycleGAN training:
- Hinge Adversarial Loss (for discriminator and generator)
- Cycle Consistency Loss
- Identity Loss

All losses are implemented from scratch (no high-level wrappers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_loss_discriminator(real_pred, fake_pred):
    """
    Hinge Loss for Discriminator
    
    The discriminator wants to:
    - Maximize real_pred (make it > 1)
    - Minimize fake_pred (make it < -1)
    
    Loss = mean(ReLU(1 - real_pred)) + mean(ReLU(1 + fake_pred))
    
    Args:
        real_pred: Discriminator predictions on real images (batch, 1, H, W)
        fake_pred: Discriminator predictions on fake images (batch, 1, H, W)
        
    Returns:
        Scalar loss value
    """
    # Real images: want real_pred > 1, so loss = max(0, 1 - real_pred)
    real_loss = F.relu(1.0 - real_pred).mean()
    
    # Fake images: want fake_pred < -1, so loss = max(0, 1 + fake_pred)
    fake_loss = F.relu(1.0 + fake_pred).mean()
    
    return real_loss + fake_loss


def hinge_loss_generator(fake_pred):
    """
    Hinge Loss for Generator
    
    The generator wants to:
    - Maximize fake_pred (make it > 1, fool the discriminator)
    
    Loss = -mean(fake_pred)
    Or equivalently: mean(ReLU(1 - fake_pred)) - 1
    
    Args:
        fake_pred: Discriminator predictions on fake images (batch, 1, H, W)
        
    Returns:
        Scalar loss value
    """
    # Generator wants fake_pred to be as high as possible (fool discriminator)
    # Hinge loss: -mean(fake_pred)
    return -fake_pred.mean()


def cycle_consistency_loss(real, reconstructed, lambda_cycle=10.0):
    """
    Cycle Consistency Loss
    
    Ensures that translating A→B→A should recover the original A.
    Uses L1 loss (mean absolute error).
    
    Loss = lambda_cycle * L1(real, reconstructed)
    
    Args:
        real: Original image (batch, 3, H, W)
        reconstructed: Reconstructed image after cycle (batch, 3, H, W)
        lambda_cycle: Weight for cycle loss (default: 10.0)
        
    Returns:
        Scalar loss value
    """
    return lambda_cycle * F.l1_loss(real, reconstructed)


def identity_loss(real, identity, lambda_identity=0.5):
    """
    Identity Loss (Optional but commonly used)
    
    When given an image from the target domain, the generator should output
    the same image (identity mapping). This helps preserve color and tone.
    
    Loss = lambda_identity * L1(real, identity)
    
    Args:
        real: Real image from target domain (batch, 3, H, W)
        identity: Generator output when given target domain image (batch, 3, H, W)
        lambda_identity: Weight for identity loss (default: 0.5)
        
    Returns:
        Scalar loss value
    """
    return lambda_identity * F.l1_loss(real, identity)


class CycleGANLosses:
    """
    Convenience class to compute all CycleGAN losses
    
    Usage:
        losses = CycleGANLosses(lambda_cycle=10.0, lambda_identity=0.5)
        
        # Discriminator losses
        disc_A_loss = losses.discriminator_loss(real_A_pred, fake_A_pred)
        disc_B_loss = losses.discriminator_loss(real_B_pred, fake_B_pred)
        
        # Generator losses
        gen_loss = losses.generator_loss(
            fake_A_pred, fake_B_pred,
            real_A, cycle_A, real_B, cycle_B,
            identity_A, identity_B
        )
    """
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5):
        """
        Initialize CycleGAN Losses
        
        Args:
            lambda_cycle: Weight for cycle consistency loss (default: 10.0)
            lambda_identity: Weight for identity loss (default: 0.5)
        """
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Compute discriminator loss using Hinge loss
        
        Args:
            real_pred: Discriminator predictions on real images
            fake_pred: Discriminator predictions on fake images
            
        Returns:
            Scalar loss value
        """
        return hinge_loss_discriminator(real_pred, fake_pred)
    
    def generator_adversarial_loss(self, fake_pred):
        """
        Compute generator adversarial loss using Hinge loss
        
        Args:
            fake_pred: Discriminator predictions on fake images
            
        Returns:
            Scalar loss value
        """
        return hinge_loss_generator(fake_pred)
    
    def generator_loss(self, fake_A_pred, fake_B_pred,
                      real_A, cycle_A, real_B, cycle_B,
                      identity_A=None, identity_B=None):
        """
        Compute total generator loss
        
        Includes:
        - Adversarial losses (for both generators)
        - Cycle consistency losses (for both directions)
        - Identity losses (optional)
        
        Args:
            fake_A_pred: Discriminator predictions on fake A images
            fake_B_pred: Discriminator predictions on fake B images
            real_A: Real A images
            cycle_A: Reconstructed A images (A→B→A)
            real_B: Real B images
            cycle_B: Reconstructed B images (B→A→B)
            identity_A: Identity mapping for A (optional)
            identity_B: Identity mapping for B (optional)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Adversarial losses
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
        
        # Total loss
        total_loss = adv_loss + cycle_loss + identity_loss_total
        
        return {
            'adversarial': adv_loss,
            'cycle': cycle_loss,
            'identity': identity_loss_total,
            'total': total_loss
        }
