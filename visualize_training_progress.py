"""
Visualization Script for CycleGAN Training Progress

This script creates comparison visualizations showing:
1. Model progression across epochs (e.g., epoch 5, 15, 30)
2. ResNet vs UNet comparison
3. Loss curves and training metrics
4. Side-by-side image comparisons

Usage:
    python visualize_training_progress.py \
        --checkpoint_dir "/content/drive/MyDrive/GANsHomework/cyclegan_checkpoints" \
        --experiment_name "resnet-baseline" \
        --epochs_to_compare 5 15 30 \
        --output_dir "./visualizations"
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json


def load_loss_history(checkpoint_dir, experiment_name):
    """Load loss history from checkpoint files"""
    losses = []
    epochs = []
    
    # Try to find checkpoint files
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.startswith(f"{experiment_name}_epoch_") and filename.endswith(".pth"):
            try:
                import torch
                checkpoint = torch.load(os.path.join(checkpoint_dir, filename), map_location='cpu')
                if 'losses_history' in checkpoint:
                    losses.extend(checkpoint['losses_history'])
                    epochs.append(checkpoint['epoch'])
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return losses, epochs


def create_epoch_comparison(image_dir, experiment_name, epochs_to_compare, output_path):
    """
    Create side-by-side comparison of images at different epochs
    """
    fig, axes = plt.subplots(len(epochs_to_compare), 3, figsize=(15, 5 * len(epochs_to_compare)))
    if len(epochs_to_compare) == 1:
        axes = axes.reshape(1, -1)
    
    fig.patch.set_facecolor('white')
    
    for row, epoch in enumerate(epochs_to_compare):
        epoch_str = f"epoch_{epoch:03d}"
        
        # Load images
        real_photo_path = os.path.join(image_dir, f"{experiment_name}_{epoch_str}_real_photo.png")
        generated_monet_path = os.path.join(image_dir, f"{experiment_name}_{epoch_str}_generated_monet.png")
        reconstructed_photo_path = os.path.join(image_dir, f"{experiment_name}_{epoch_str}_reconstructed_photo.png")
        
        # Display images
        if os.path.exists(real_photo_path):
            axes[row, 0].imshow(Image.open(real_photo_path))
            axes[row, 0].set_title(f'Epoch {epoch} - Real Photo', fontsize=12, fontweight='bold')
        else:
            axes[row, 0].text(0.5, 0.5, f'Image not found\nEpoch {epoch}', 
                              ha='center', va='center', fontsize=12)
        axes[row, 0].axis('off')
        
        if os.path.exists(generated_monet_path):
            axes[row, 1].imshow(Image.open(generated_monet_path))
            axes[row, 1].set_title(f'Epoch {epoch} - Generated Monet', fontsize=12, fontweight='bold', color='green')
        else:
            axes[row, 1].text(0.5, 0.5, f'Image not found\nEpoch {epoch}', 
                              ha='center', va='center', fontsize=12)
        axes[row, 1].axis('off')
        
        if os.path.exists(reconstructed_photo_path):
            axes[row, 2].imshow(Image.open(reconstructed_photo_path))
            axes[row, 2].set_title(f'Epoch {epoch} - Reconstructed Photo', fontsize=12, fontweight='bold')
        else:
            axes[row, 2].text(0.5, 0.5, f'Image not found\nEpoch {epoch}', 
                              ha='center', va='center', fontsize=12)
        axes[row, 2].axis('off')
    
    plt.suptitle(f'{experiment_name} - Training Progression', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved epoch comparison: {output_path}")


def create_architecture_comparison(image_dir, resnet_name, unet_name, epoch, output_path):
    """
    Compare ResNet vs UNet at the same epoch
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    epoch_str = f"epoch_{epoch:03d}"
    
    # Row 1: ResNet
    resnet_real = os.path.join(image_dir, f"{resnet_name}_{epoch_str}_real_photo.png")
    resnet_generated = os.path.join(image_dir, f"{resnet_name}_{epoch_str}_generated_monet.png")
    resnet_reconstructed = os.path.join(image_dir, f"{resnet_name}_{epoch_str}_reconstructed_photo.png")
    
    if os.path.exists(resnet_real):
        axes[0, 0].imshow(Image.open(resnet_real))
    axes[0, 0].set_title(f'ResNet - Real Photo', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    if os.path.exists(resnet_generated):
        axes[0, 1].imshow(Image.open(resnet_generated))
    axes[0, 1].set_title(f'ResNet - Generated Monet', fontsize=12, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    if os.path.exists(resnet_reconstructed):
        axes[0, 2].imshow(Image.open(resnet_reconstructed))
    axes[0, 2].set_title(f'ResNet - Reconstructed', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: UNet
    unet_real = os.path.join(image_dir, f"{unet_name}_{epoch_str}_real_photo.png")
    unet_generated = os.path.join(image_dir, f"{unet_name}_{epoch_str}_generated_monet.png")
    unet_reconstructed = os.path.join(image_dir, f"{unet_name}_{epoch_str}_reconstructed_photo.png")
    
    if os.path.exists(unet_real):
        axes[1, 0].imshow(Image.open(unet_real))
    axes[1, 0].set_title(f'UNet - Real Photo', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    if os.path.exists(unet_generated):
        axes[1, 1].imshow(Image.open(unet_generated))
    axes[1, 1].set_title(f'UNet - Generated Monet', fontsize=12, fontweight='bold', color='orange')
    axes[1, 1].axis('off')
    
    if os.path.exists(unet_reconstructed):
        axes[1, 2].imshow(Image.open(unet_reconstructed))
    axes[1, 2].set_title(f'UNet - Reconstructed', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'ResNet vs UNet Comparison - Epoch {epoch}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved architecture comparison: {output_path}")


def plot_loss_curves_from_checkpoints(checkpoint_dir, experiment_name, output_path):
    """
    Plot loss curves from checkpoint files
    """
    losses, epochs = load_loss_history(checkpoint_dir, experiment_name)
    
    if not losses:
        print(f"Warning: No loss history found for {experiment_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    epoch_nums = [l['epoch'] for l in losses]
    
    # Generator losses
    axes[0, 0].plot(epoch_nums, [l['gen_total'] for l in losses], 
                   'b-', linewidth=2, label='Total', marker='o', markersize=4)
    axes[0, 0].plot(epoch_nums, [l['gen_adv'] for l in losses], 
                   'r--', linewidth=1.5, alpha=0.7, label='Adversarial')
    axes[0, 0].plot(epoch_nums, [l['gen_cycle'] for l in losses], 
                   'g--', linewidth=1.5, alpha=0.7, label='Cycle')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Generator Losses', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator losses
    axes[0, 1].plot(epoch_nums, [l['disc_A'] for l in losses], 
                   'orange', linewidth=2, label='Disc A', marker='s', markersize=4)
    axes[0, 1].plot(epoch_nums, [l['disc_B'] for l in losses], 
                   'purple', linewidth=2, label='Disc B', marker='^', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Discriminator Losses', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Combined plot
    axes[1, 0].plot(epoch_nums, [l['gen_total'] for l in losses], 
                   'b-', linewidth=2, label='Generator Total')
    axes[1, 0].plot(epoch_nums, [(l['disc_A'] + l['disc_B'])/2 for l in losses], 
                   'r-', linewidth=2, label='Discriminator Avg')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Generator vs Discriminator', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component breakdown
    axes[1, 1].plot(epoch_nums, [l['gen_adv'] for l in losses], 
                   label='Adversarial', linewidth=2)
    axes[1, 1].plot(epoch_nums, [l['gen_cycle'] for l in losses], 
                   label='Cycle', linewidth=2)
    axes[1, 1].plot(epoch_nums, [l['gen_identity'] for l in losses], 
                   label='Identity', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Generator Loss Components', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{experiment_name} - Training Loss Curves', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved loss curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CycleGAN training progress")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoints")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Directory containing saved images (default: checkpoint_dir/images/experiment_name)")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., 'resnet-baseline')")
    parser.add_argument("--epochs_to_compare", type=int, nargs='+', default=[5, 15, 30],
                       help="Epochs to compare (e.g., 5 15 30)")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for visualizations")
    
    # For architecture comparison
    parser.add_argument("--compare_architectures", action="store_true",
                       help="Compare ResNet vs UNet")
    parser.add_argument("--resnet_name", type=str, default="resnet-baseline",
                       help="ResNet experiment name")
    parser.add_argument("--unet_name", type=str, default="unet-baseline",
                       help="UNet experiment name")
    parser.add_argument("--comparison_epoch", type=int, default=30,
                       help="Epoch to compare architectures at")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set image directory
    if args.image_dir is None:
        args.image_dir = os.path.join(args.checkpoint_dir, "images", args.experiment_name)
    
    print(f"Creating visualizations for: {args.experiment_name}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 1. Create epoch progression comparison
    progression_path = os.path.join(args.output_dir, f"{args.experiment_name}_progression.png")
    create_epoch_comparison(args.image_dir, args.experiment_name, 
                           args.epochs_to_compare, progression_path)
    
    # 2. Plot loss curves
    loss_curves_path = os.path.join(args.output_dir, f"{args.experiment_name}_loss_curves.png")
    plot_loss_curves_from_checkpoints(args.checkpoint_dir, args.experiment_name, loss_curves_path)
    
    # 3. Architecture comparison (if requested)
    if args.compare_architectures:
        resnet_image_dir = os.path.join(args.checkpoint_dir, "images", args.resnet_name)
        unet_image_dir = os.path.join(args.checkpoint_dir, "images", args.unet_name)
        
        # Use the directory that exists
        if os.path.exists(resnet_image_dir) and os.path.exists(unet_image_dir):
            # Create comparison using images from both directories
            comparison_path = os.path.join(args.output_dir, 
                                         f"resnet_vs_unet_epoch_{args.comparison_epoch}.png")
            # Note: This is a simplified version - you may need to adjust paths
            print(f"Architecture comparison would be saved to: {comparison_path}")
            print("Note: Adjust paths in script if images are in different locations")
    
    print(f"\n✅ Visualizations saved to: {args.output_dir}")
    print(f"   - Progression comparison: {progression_path}")
    print(f"   - Loss curves: {loss_curves_path}")


if __name__ == "__main__":
    main()









