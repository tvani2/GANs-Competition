"""
CycleGAN Loss Functions with LSGAN (Least Squares GAN)

This module implements all loss functions needed for CycleGAN training using LSGAN:
- LSGAN Adversarial Loss (for discriminator and generator)
- Cycle Consistency Loss
- Identity Loss

LSGAN uses least squares loss instead of cross-entropy, which provides:
- Smoother gradients
- Better image quality
- Less vanishing gradient problems
- Used in the original CycleGAN paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lsgan_loss_discriminator(real_pred, fake_pred):
    """
    LSGAN Loss for Discriminator (Least Squares GAN)
    
    The discriminator wants to:
    - Make real_pred close to 1 (real images should score 1)
    - Make fake_pred close to 0 (fake images should score 0)
    
    Loss = 0.5 * mean((real_pred - 1)²) + 0.5 * mean(fake_pred²)
    
    Args:
        real_pred: Discriminator predictions on real images (batch, 1, H, W)
        fake_pred: Discriminator predictions on fake images (batch, 1, H, W)
        
    Returns:
        Scalar loss value
    """
    # Real images: want real_pred = 1, so loss = (real_pred - 1)²
    real_loss = torch.mean((real_pred - 1.0) ** 2)
    
    # Fake images: want fake_pred = 0, so loss = fake_pred²
    fake_loss = torch.mean(fake_pred ** 2)
    
    # Average of both losses
    return (real_loss + fake_loss) * 0.5


def lsgan_loss_generator(fake_pred):
    """
    LSGAN Loss for Generator (Least Squares GAN)
    
    The generator wants to:
    - Make fake_pred close to 1 (fool the discriminator)
    
    Loss = 0.5 * mean((fake_pred - 1)²)
    
    Args:
        fake_pred: Discriminator predictions on fake images (batch, 1, H, W)
        
    Returns:
        Scalar loss value
    """
    # Generator wants fake_pred = 1 (fool discriminator)
    # LSGAN loss: (fake_pred - 1)²
    return torch.mean((fake_pred - 1.0) ** 2) * 0.5


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


class CycleGANLossesLSGAN:
    """
    Convenience class to compute all CycleGAN losses with LSGAN
    
    Usage:
        losses = CycleGANLossesLSGAN(lambda_cycle=10.0, lambda_identity=0.5)
        
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
        Initialize CycleGAN Losses with LSGAN
        
        Args:
            lambda_cycle: Weight for cycle consistency loss (default: 10.0)
            lambda_identity: Weight for identity loss (default: 0.5)
        """
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Compute discriminator loss using LSGAN loss
        
        Args:
            real_pred: Discriminator predictions on real images
            fake_pred: Discriminator predictions on fake images
            
        Returns:
            Scalar loss value
        """
        return lsgan_loss_discriminator(real_pred, fake_pred)
    
    def generator_adversarial_loss(self, fake_pred):
        """
        Compute generator adversarial loss using LSGAN loss
        
        Args:
            fake_pred: Discriminator predictions on fake images
            
        Returns:
            Scalar loss value
        """
        return lsgan_loss_generator(fake_pred)
    
    def generator_loss(self, fake_A_pred, fake_B_pred,
                      real_A, cycle_A, real_B, cycle_B,
                      identity_A=None, identity_B=None):
        """
        Compute total generator loss
        
        Includes:
        - Adversarial losses (for both generators) using LSGAN
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
        # Adversarial losses (LSGAN)
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


if __name__ == "__main__":
    print("CycleGAN Loss Functions with LSGAN (Least Squares GAN)")
    print("=" * 60)
    print("\nLSGAN uses least squares loss instead of hinge loss:")
    print("  - Discriminator: 0.5 * [(D(real) - 1)² + D(fake)²]")
    print("  - Generator: 0.5 * (D(fake) - 1)²")
    print("\nAdvantages of LSGAN:")
    print("  - Smoother gradients (quadratic loss)")
    print("  - Better image quality")
    print("  - Less vanishing gradient problems")
    print("  - Used in original CycleGAN paper")
    print("\nUsage:")
    print("  from losses_cyclegan_lsgan import CycleGANLossesLSGAN")
    print("  losses = CycleGANLossesLSGAN(lambda_cycle=10.0, lambda_identity=0.5)")

