"""
Test script for CycleGAN Models (Generator and Discriminator)

Run this to verify both models work correctly:
    python test_generator.py
"""

import torch
from models.resnet_generator import ResNetGenerator
from models.discriminator import PatchGANDiscriminator

def test_generator():
    """Test the ResNet generator with dummy input"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create generator
    generator = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    generator = generator.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Generator Architecture:")
    print(f"{'='*60}")
    print(f"  - Input channels: 3 (RGB)")
    print(f"  - Output channels: 3 (RGB)")
    print(f"  - Base filters: 64")
    print(f"  - Residual blocks: 9")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create dummy input (batch_size=2, channels=3, height=256, width=256)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Forward pass
    generator.eval()
    with torch.no_grad():
        output = generator(dummy_input)
    
    print(f"\nTest Results:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"  - Expected range: [-1.0, 1.0] (Tanh activation)")
    
    # Verify output is in correct range
    assert output.shape == (batch_size, 3, 256, 256), "Output shape mismatch!"
    assert output.min() >= -1.0 and output.max() <= 1.0, "Output not in [-1, 1] range!"
    
    print("\n✓ Generator tests passed!")
    return generator, output


def test_discriminator():
    """Test the PatchGAN discriminator with dummy input"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create discriminator
    discriminator = PatchGANDiscriminator(input_nc=3, ndf=64, n_layers=3)
    discriminator = discriminator.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Discriminator Architecture:")
    print(f"{'='*60}")
    print(f"  - Input channels: 3 (RGB)")
    print(f"  - Base filters: 64")
    print(f"  - Number of layers: 3 (70x70 PatchGAN)")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create dummy input (batch_size=2, channels=3, height=256, width=256)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Forward pass
    discriminator.eval()
    with torch.no_grad():
        output = discriminator(dummy_input)
    
    print(f"\nTest Results:")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"  - Expected shape: (batch, 1, 32, 32) - patch predictions")
    print(f"  - Receptive field: 70x70 patches in input image")
    
    # Verify output shape
    assert output.shape == (batch_size, 1, 32, 32), f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, 1, 32, 32)"
    
    print("\n✓ Discriminator tests passed!")
    return discriminator, output


def test_cyclegan_pair():
    """Test generator-discriminator pair (typical CycleGAN setup)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"CycleGAN Pair Test:")
    print(f"{'='*60}")
    
    # Create models (for CycleGAN, you need two generators and two discriminators)
    generator_A2B = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
    discriminator_B = PatchGANDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    
    # Create dummy input
    batch_size = 2
    real_A = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Generator forward pass
    fake_B = generator_A2B(real_A)
    
    # Discriminator forward pass on fake image
    disc_fake_B = discriminator_B(fake_B)
    
    # Discriminator forward pass on real image (simulated)
    real_B = torch.randn(batch_size, 3, 256, 256).to(device)
    disc_real_B = discriminator_B(real_B)
    
    print(f"  - Real A shape: {real_A.shape}")
    print(f"  - Fake B shape: {fake_B.shape}")
    print(f"  - Discriminator output on fake B: {disc_fake_B.shape}")
    print(f"  - Discriminator output on real B: {disc_real_B.shape}")
    
    assert fake_B.shape == (batch_size, 3, 256, 256), "Generator output shape mismatch!"
    assert disc_fake_B.shape == (batch_size, 1, 32, 32), "Discriminator output shape mismatch!"
    assert disc_real_B.shape == (batch_size, 1, 32, 32), "Discriminator output shape mismatch!"
    
    print("\n✓ CycleGAN pair tests passed!")


if __name__ == "__main__":
    print("Testing CycleGAN Models...")
    print("="*60)
    
    # Test generator
    generator, gen_output = test_generator()
    
    # Test discriminator
    discriminator, disc_output = test_discriminator()
    
    # Test CycleGAN pair
    test_cyclegan_pair()
    
    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print(f"{'='*60}")

