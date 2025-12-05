"""
CycleGAN Training Script with WandB Integration

This script trains CycleGAN models with:
- WandB logging for metrics, images, and model artifacts
- Checkpoint saving to Google Drive
- Experiment management (changing one variable at a time)
- Resuming training from checkpoints
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import wandb
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display issues
import matplotlib.pyplot as plt
# Configure matplotlib to prevent too many open figures warning
plt.rcParams['figure.max_open_warning'] = 0
import signal
import sys
import time

# Import your models and losses
from models import ResNetGenerator, UNetGenerator, PatchGANDiscriminator
from losses.losses_cyclegan import CycleGANLosses


class ImageDataset(torch.utils.data.Dataset):
    """
    Simple dataset for loading images from two directories
    Modify this based on your actual dataset structure
    """
    def __init__(self, domain_A_path, domain_B_path, transform=None):
        self.domain_A_path = domain_A_path
        self.domain_B_path = domain_B_path
        
        # Get list of image files
        self.domain_A_images = sorted([f for f in os.listdir(domain_A_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.domain_B_images = sorted([f for f in os.listdir(domain_B_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
    def __len__(self):
        return max(len(self.domain_A_images), len(self.domain_B_images))
    
    def __getitem__(self, idx):
        # Cycle through images to handle different sizes
        img_A_path = os.path.join(self.domain_A_path, 
                                 self.domain_A_images[idx % len(self.domain_A_images)])
        img_B_path = os.path.join(self.domain_B_path, 
                                 self.domain_B_images[idx % len(self.domain_B_images)])
        
        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return img_A, img_B


def tensor_to_image(tensor):
    """Convert tensor to numpy image for visualization"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)
    # Convert to numpy and ensure correct dtype (float32 for WandB compatibility)
    image = tensor.cpu().detach().permute(1, 2, 0).numpy()
    # Ensure dtype is float32 (WandB requires byte, short, float32, or float64)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # Ensure values are in [0, 1] range
    image = np.clip(image, 0.0, 1.0)
    return image


def save_fixed_image_pairs(fixed_pairs_results, epoch, experiment_name, output_dir):
    """
    Save 5 fixed photo->painting pairs for consistent comparison across epochs
    
    Args:
        fixed_pairs_results: List of tuples (real_A, fake_B, rec_A) for 5 pairs
        epoch: Current epoch number
        experiment_name: Name of the experiment
        output_dir: Base directory to save images (will create epoch subdirectory)
    """
    epoch_str = f"epoch_{epoch+1:03d}"  # e.g., epoch_015, epoch_030
    
    # Create epoch-specific subdirectory
    epoch_output_dir = os.path.join(output_dir, epoch_str)
    os.makedirs(epoch_output_dir, exist_ok=True)
    
    try:
        from PIL import Image as PILImage
        
        # Convert float32 [0,1] to uint8 [0,255] for saving
        def to_uint8(img):
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Create a grid showing all 5 pairs: 5 rows x 3 columns (Photo, Generated Monet, Reconstructed Photo)
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        fig.patch.set_facecolor('white')
        
        for pair_idx, (real_A, fake_B, rec_A) in enumerate(fixed_pairs_results):
            # Convert tensors to images
            img_A = tensor_to_image(real_A[0])
            img_B = tensor_to_image(fake_B[0])
            img_rec_A = tensor_to_image(rec_A[0])
            
            # Save individual images for this pair
            pair_dir = os.path.join(epoch_output_dir, f"pair_{pair_idx+1}")
            os.makedirs(pair_dir, exist_ok=True)
            
            PILImage.fromarray(to_uint8(img_A)).save(
                os.path.join(pair_dir, f'{experiment_name}_{epoch_str}_real_photo.png')
            )
            PILImage.fromarray(to_uint8(img_B)).save(
                os.path.join(pair_dir, f'{experiment_name}_{epoch_str}_generated_monet.png')
            )
            PILImage.fromarray(to_uint8(img_rec_A)).save(
                os.path.join(pair_dir, f'{experiment_name}_{epoch_str}_reconstructed_photo.png')
            )
            
            # Display in grid
            axes[pair_idx, 0].imshow(img_A)
            axes[pair_idx, 0].set_title(f'Pair {pair_idx+1} - Real Photo', fontsize=10, fontweight='bold')
            axes[pair_idx, 0].axis('off')
            
            axes[pair_idx, 1].imshow(img_B)
            axes[pair_idx, 1].set_title(f'Pair {pair_idx+1} - Generated Monet', fontsize=10, fontweight='bold', color='green')
            axes[pair_idx, 1].axis('off')
            
            axes[pair_idx, 2].imshow(img_rec_A)
            axes[pair_idx, 2].set_title(f'Pair {pair_idx+1} - Reconstructed', fontsize=10, fontweight='bold')
            axes[pair_idx, 2].axis('off')
        
        plt.suptitle(f'{experiment_name} - Epoch {epoch+1} (5 Fixed Pairs)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        grid_path = os.path.join(epoch_output_dir, f'{experiment_name}_{epoch_str}_all_pairs.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"   Saved 5 fixed image pairs for epoch {epoch+1} to {epoch_output_dir}")
        
    except Exception as e:
        print(f"Warning: Failed to save fixed image pairs: {e}. Continuing...")


def log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, epoch, step, 
                   experiment_name):
    """
    Create and log before/after comparison grid to WandB with enhanced styling
    """
    fig = None
    fig2 = None
    fig3 = None
    
    try:
        # Configure matplotlib to prevent too many open figures warning
        plt.rcParams['figure.max_open_warning'] = 0
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        # Row 1: Photo → Monet → Photo
        img_A = tensor_to_image(real_A[0])
        img_B = tensor_to_image(fake_B[0])
        img_rec_A = tensor_to_image(rec_A[0])
        
        axes[0, 0].imshow(img_A)
        axes[0, 0].set_title('Real Photo', fontsize=14, fontweight='bold', pad=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_B)
        axes[0, 1].set_title('Generated Monet', fontsize=14, fontweight='bold', pad=10, color='green')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_rec_A)
        axes[0, 2].set_title('Reconstructed Photo', fontsize=14, fontweight='bold', pad=10)
        axes[0, 2].axis('off')
        
        # Row 2: Monet → Photo → Monet
        img_B_real = tensor_to_image(real_B[0])
        img_A_fake = tensor_to_image(fake_A[0])
        img_rec_B = tensor_to_image(rec_B[0])
        
        axes[1, 0].imshow(img_B_real)
        axes[1, 0].set_title('Real Monet', fontsize=14, fontweight='bold', pad=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_A_fake)
        axes[1, 1].set_title('Generated Photo', fontsize=14, fontweight='bold', pad=10, color='green')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img_rec_B)
        axes[1, 2].set_title('Reconstructed Monet', fontsize=14, fontweight='bold', pad=10)
        axes[1, 2].axis('off')
        
        plt.suptitle(f'{experiment_name} - Epoch {epoch+1}, Step {step}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Log to WandB
        wandb.log({f"images/comparison_grid": wandb.Image(fig)})
        plt.close(fig)
        fig = None
        
        # Also log individual image pairs for better tracking
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
        fig2.patch.set_facecolor('white')
        axes2[0].imshow(img_A)
        axes2[0].set_title('Input Photo', fontsize=12, fontweight='bold')
        axes2[0].axis('off')
        axes2[1].imshow(img_B)
        axes2[1].set_title('Generated Monet', fontsize=12, fontweight='bold', color='green')
        axes2[1].axis('off')
        plt.tight_layout()
        wandb.log({f"images/photo_to_monet": wandb.Image(fig2)})
        plt.close(fig2)
        fig2 = None
        
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
        fig3.patch.set_facecolor('white')
        axes3[0].imshow(img_B_real)
        axes3[0].set_title('Input Monet', fontsize=12, fontweight='bold')
        axes3[0].axis('off')
        axes3[1].imshow(img_A_fake)
        axes3[1].set_title('Generated Photo', fontsize=12, fontweight='bold', color='green')
        axes3[1].axis('off')
        plt.tight_layout()
        wandb.log({f"images/monet_to_photo": wandb.Image(fig3)})
        plt.close(fig3)
        fig3 = None
        
    except Exception as e:
        # Ensure all figures are closed even on error
        if fig is not None:
            plt.close(fig)
        if fig2 is not None:
            plt.close(fig2)
        if fig3 is not None:
            plt.close(fig3)
        raise e


def save_checkpoint(epoch, gen_A2B, gen_B2A, disc_A, disc_B, 
                   optimizer_G, optimizer_D, losses_history, checkpoint_dir, 
                   experiment_name, save_latest=False):
    """
    Save complete checkpoint to Google Drive and WandB
    
    Args:
        save_latest: If True, also save as 'latest.pth' for quick recovery
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 
                                   f'{experiment_name}_epoch_{epoch}.pth')
    
    checkpoint = {
        'epoch': epoch,
        'experiment_name': experiment_name,
        'gen_A2B_state_dict': gen_A2B.state_dict(),
        'gen_B2A_state_dict': gen_B2A.state_dict(),
        'disc_A_state_dict': disc_A.state_dict(),
        'disc_B_state_dict': disc_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'losses_history': losses_history,
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as 'latest.pth' for quick recovery
        if save_latest:
            latest_path = os.path.join(checkpoint_dir, f'{experiment_name}_latest.pth')
            torch.save(checkpoint, latest_path)
            print(f"Latest checkpoint saved: {latest_path}")
        
        # Also save to WandB as artifact (non-blocking, with error handling)
        try:
            artifact = wandb.Artifact(f'{experiment_name}-epoch-{epoch}', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to save to WandB: {e}")
            print("Checkpoint saved locally, continuing training...")
        
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, gen_A2B, gen_B2A, disc_A, disc_B, 
                    optimizer_G, optimizer_D, device):
    """
    Load checkpoint and resume training
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    gen_A2B.load_state_dict(checkpoint['gen_A2B_state_dict'])
    gen_B2A.load_state_dict(checkpoint['gen_B2A_state_dict'])
    disc_A.load_state_dict(checkpoint['disc_A_state_dict'])
    disc_B.load_state_dict(checkpoint['disc_B_state_dict'])
    
    # Load optimizer states
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # Return starting epoch and experiment name
    start_epoch = checkpoint['epoch'] + 1
    experiment_name = checkpoint.get('experiment_name', 'unknown')
    losses_history = checkpoint.get('losses_history', [])
    
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")
    
    return start_epoch, experiment_name, losses_history


def train_epoch(gen_A2B, gen_B2A, disc_A, disc_B, dataloader, 
                optimizer_G, optimizer_D, losses_fn, device, epoch, scaler=None):
    """
    Train for one epoch
    """
    gen_A2B.train()
    gen_B2A.train()
    disc_A.train()
    disc_B.train()
    
    epoch_losses = {
        'gen_total': 0.0,
        'gen_adv': 0.0,
        'gen_cycle': 0.0,
        'gen_identity': 0.0,
        'disc_A': 0.0,
        'disc_B': 0.0,
    }
    
    num_batches = len(dataloader)
    last_progress_time = time.time()
    
    for step, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        
        # ==================== Train Generators ====================
        optimizer_G.zero_grad()
        
        # Use mixed precision for forward pass (faster training)
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward pass
            fake_B = gen_A2B(real_A)
            fake_A = gen_B2A(real_B)
            
            rec_A = gen_B2A(fake_B)  # A → B → A
            rec_B = gen_A2B(fake_A)  # B → A → B
            
            # Identity mapping (optional, helps preserve color)
            identity_A = gen_B2A(real_A)
            identity_B = gen_A2B(real_B)
            
            # Discriminator predictions on fake images
            fake_B_pred = disc_B(fake_B)
            fake_A_pred = disc_A(fake_A)
            
            # Generator losses
            gen_losses = losses_fn.generator_loss(
                fake_A_pred, fake_B_pred,
                real_A, rec_A, real_B, rec_B,
                identity_A, identity_B
            )
            
            gen_total = gen_losses['total']
        
        # Backward pass with mixed precision support
        if use_amp:
            scaler.scale(gen_total).backward()
            scaler.step(optimizer_G)
            scaler.update()
        else:
            gen_total.backward()
            optimizer_G.step()
        
        # ==================== Train Discriminators ====================
        # Discriminator A
        optimizer_D.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            real_A_pred = disc_A(real_A)
            fake_A_pred_detached = disc_A(fake_A.detach())
            disc_A_loss = losses_fn.discriminator_loss(real_A_pred, fake_A_pred_detached)
        
        if use_amp:
            scaler.scale(disc_A_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            disc_A_loss.backward()
            optimizer_D.step()
        
        # Discriminator B
        optimizer_D.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            real_B_pred = disc_B(real_B)
            fake_B_pred_detached = disc_B(fake_B.detach())
            disc_B_loss = losses_fn.discriminator_loss(real_B_pred, fake_B_pred_detached)
        
        if use_amp:
            scaler.scale(disc_B_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            disc_B_loss.backward()
            optimizer_D.step()
        
        # Accumulate losses
        epoch_losses['gen_total'] += gen_total.item()
        epoch_losses['gen_adv'] += gen_losses['adversarial'].item()
        epoch_losses['gen_cycle'] += gen_losses['cycle'].item()
        epoch_losses['gen_identity'] += gen_losses['identity'].item()
        epoch_losses['disc_A'] += disc_A_loss.item()
        epoch_losses['disc_B'] += disc_B_loss.item()
        
        # Periodic progress output to prevent inactivity timeout (every 5 minutes)
        # This helps prevent Colab from disconnecting due to inactivity
        current_time = time.time()
        if current_time - last_progress_time > 300:  # 5 minutes
            elapsed = current_time - last_progress_time
            print(f"   Progress: Step {step+1}/{num_batches} (Epoch {epoch+1}) - "
                  f"Gen Loss: {gen_total.item():.4f}, "
                  f"Disc A: {disc_A_loss.item():.4f}, "
                  f"Disc B: {disc_B_loss.item():.4f} "
                  f"[{elapsed:.0f}s since last update]")
            last_progress_time = current_time
        
        # Log to WandB every N steps (reduced frequency for better performance)
        # Logging less frequently reduces overhead and speeds up training
        log_freq = 200  # Log every 200 steps instead of 100
        image_log_freq = 1000  # Log images every 1000 steps instead of 500
        
        if step % log_freq == 0:
            try:
                wandb.log({
                    "loss/gen_total": gen_total.item(),
                    "loss/gen_adv": gen_losses['adversarial'].item(),
                    "loss/gen_cycle": gen_losses['cycle'].item(),
                    "loss/gen_identity": gen_losses['identity'].item(),
                    "loss/disc_A": disc_A_loss.item(),
                    "loss/disc_B": disc_B_loss.item(),
                    "epoch": epoch,
                    "step": epoch * num_batches + step,
                })
            except Exception as e:
                print(f"Warning: WandB logging failed: {e}. Continuing training...")
            
            # Log image grid less frequently to avoid expensive matplotlib operations
            if step % image_log_freq == 0:
                try:
                    log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, 
                                 epoch, step, wandb.run.name if wandb.run else "training")
                except Exception as e:
                    print(f"Warning: Image logging failed: {e}. Continuing training...")
    
    # Average losses over epoch
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses, (real_A, fake_B, rec_A, real_B, fake_A, rec_B)


def plot_loss_curves(losses_history):
    """
    Create and log loss curves plot to WandB
    """
    fig = None
    try:
        epochs = [l['epoch'] for l in losses_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        # Generator losses
        axes[0, 0].plot(epochs, [l['gen_total'] for l in losses_history], 
                       'b-', linewidth=2, label='Total')
        axes[0, 0].plot(epochs, [l['gen_adv'] for l in losses_history], 
                       'r--', linewidth=1.5, alpha=0.7, label='Adversarial')
        axes[0, 0].plot(epochs, [l['gen_cycle'] for l in losses_history], 
                       'g--', linewidth=1.5, alpha=0.7, label='Cycle')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Generator Losses', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator losses
        axes[0, 1].plot(epochs, [l['disc_A'] for l in losses_history], 
                       'orange', linewidth=2, label='Disc A')
        axes[0, 1].plot(epochs, [l['disc_B'] for l in losses_history], 
                       'purple', linewidth=2, label='Disc B')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Discriminator Losses', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Combined plot
        axes[1, 0].plot(epochs, [l['gen_total'] for l in losses_history], 
                       'b-', linewidth=2, label='Generator Total')
        axes[1, 0].plot(epochs, [(l['disc_A'] + l['disc_B'])/2 for l in losses_history], 
                       'r-', linewidth=2, label='Discriminator Avg')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Generator vs Discriminator', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Component breakdown
        axes[1, 1].plot(epochs, [l['gen_adv'] for l in losses_history], 
                       label='Adversarial', linewidth=2)
        axes[1, 1].plot(epochs, [l['gen_cycle'] for l in losses_history], 
                       label='Cycle', linewidth=2)
        axes[1, 1].plot(epochs, [l['gen_identity'] for l in losses_history], 
                       label='Identity', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('Generator Loss Components', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"plots/loss_curves": wandb.Image(fig)})
        plt.close(fig)
        fig = None
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        raise e


# Global variables for signal handler
current_epoch = 0
current_gen_A2B = None
current_gen_B2A = None
current_disc_A = None
current_disc_B = None
current_optimizer_G = None
current_optimizer_D = None
current_losses_history = []
current_experiment_name = ""
current_checkpoint_dir = ""


def train(args):
    """
    Main training function
    """
    global current_epoch, current_gen_A2B, current_gen_B2A, current_disc_A, current_disc_B
    global current_optimizer_G, current_optimizer_D, current_losses_history
    global current_experiment_name, current_checkpoint_dir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== Initialize WandB ====================
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config={
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lambda_cycle": args.lambda_cycle,
            "lambda_identity": args.lambda_identity,
            "checkpoint_freq": args.checkpoint_freq,
            "resume_from": args.resume_from,
        },
        tags=[args.architecture, "cyclegan"],
        settings=wandb.Settings(_disable_stats=False),  # Enable system metrics
    )
    
    # Log system info
    if torch.cuda.is_available():
        wandb.config.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    
    # ==================== Setup Data ====================
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(args.data_A, args.data_B, transform=transform)
    # Optimize DataLoader for faster data loading
    # num_workers: Use 4-8 for better parallelization (adjust based on CPU cores)
    # pin_memory: Faster GPU transfer when using CUDA
    # persistent_workers: Keep workers alive between epochs to avoid recreation overhead
    num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers, or CPU count if less
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else 2  # Prefetch batches
    )
    
    # ==================== Initialize Models ====================
    if args.architecture == 'resnet':
        gen_A2B = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
        gen_B2A = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
    elif args.architecture == 'unet':
        gen_A2B = UNetGenerator(input_nc=3, output_nc=3, ngf=64).to(device)
        gen_B2A = UNetGenerator(input_nc=3, output_nc=3, ngf=64).to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    disc_A = PatchGANDiscriminator(input_nc=3, ndf=64).to(device)
    disc_B = PatchGANDiscriminator(input_nc=3, ndf=64).to(device)
    
    # ==================== Initialize Optimizers ====================
    optimizer_G = optim.Adam(
        list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    
    optimizer_D = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    
    # ==================== Initialize Loss Function ====================
    losses_fn = CycleGANLosses(
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
    )
    
    # ==================== Mixed Precision Training ====================
    # Use Automatic Mixed Precision (AMP) for faster training and lower memory usage
    # This can speed up training by 1.5-2x on modern GPUs
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed Precision Training (AMP) enabled - faster training!")
    
    # ==================== Load Checkpoint if Resuming ====================
    start_epoch = 0
    losses_history = []
    
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(
                f" Checkpoint file not found: {args.resume_from}\n"
                f"   Please check the path and try again.\n"
                f"   Example checkpoint path: /content/drive/MyDrive/GANsHomework/cyclegan_checkpoints/resnet-baseline_epoch_5.pth"
            )
        start_epoch, exp_name, losses_history = load_checkpoint(
            args.resume_from,
            gen_A2B, gen_B2A, disc_A, disc_B,
            optimizer_G, optimizer_D,
            device
        )
        # Update experiment name if resuming
        if exp_name != args.experiment_name:
            print(f"  Warning: Experiment name mismatch. Using checkpoint name: {exp_name}")
            args.experiment_name = exp_name
        print(f" Resuming from epoch {start_epoch}, will train until epoch {args.epochs}")
    
    # ==================== Setup Image Output Directory ====================
    # Save images for comparison across epochs (for your tutor's graphs!)
    fixed_image_pairs_original = None  # Store original images for consistent comparison
    # Set image save frequency (default to checkpoint_freq if not specified)
    image_save_freq = args.image_save_freq if args.image_save_freq is not None else args.checkpoint_freq
    
    if args.save_images:
        if args.image_output_dir is None:
            args.image_output_dir = os.path.join(args.checkpoint_dir, "images", args.experiment_name)
        os.makedirs(args.image_output_dir, exist_ok=True)
        print(f"   Images will be saved to: {args.image_output_dir}")
        print(f"   Images saved at epochs: every {image_save_freq} epochs")
        print(f"   Saving 5 fixed photo->painting pairs for consistent comparison")
        
        # Get 5 fixed image pairs at the start (same images used across all epochs)
        print("   Selecting 5 fixed image pairs...")
        fixed_image_pairs_original = []
        with torch.no_grad():
            gen_A2B.eval()
            gen_B2A.eval()
            for batch_idx, (real_A, real_B) in enumerate(dataloader):
                if batch_idx >= 5:
                    break
                # Store original images (will regenerate transformations at each checkpoint)
                fixed_image_pairs_original.append(real_A.to(device))
            gen_A2B.train()
            gen_B2A.train()
        print(f"   Selected 5 fixed image pairs for consistent comparison")
    
    # ==================== Training Loop ====================
    print(f"   Starting training: {args.experiment_name}")
    print(f"   Architecture: {args.architecture}")
    print(f"   Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Checkpoint frequency: Every {args.checkpoint_freq} epochs")
    print(f"   Backup checkpoint: Every epoch (saved as 'latest.pth')")
    
    # Initialize global variables for signal handler
    current_gen_A2B = gen_A2B
    current_gen_B2A = gen_B2A
    current_disc_A = disc_A
    current_disc_B = disc_B
    current_optimizer_G = optimizer_G
    current_optimizer_D = optimizer_D
    current_losses_history = losses_history
    current_experiment_name = args.experiment_name
    current_checkpoint_dir = args.checkpoint_dir
    
    # Signal handler for graceful shutdown (Ctrl+C or disconnection)
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interruption detected! Saving checkpoint before exit...")
        try:
            save_checkpoint(
                current_epoch, current_gen_A2B, current_gen_B2A, 
                current_disc_A, current_disc_B,
                current_optimizer_G, current_optimizer_D, 
                current_losses_history,
                current_checkpoint_dir, current_experiment_name,
                save_latest=True
            )
            print("✅ Emergency checkpoint saved! You can resume with --resume_from")
        except Exception as e:
            print(f"❌ Failed to save emergency checkpoint: {e}")
        sys.exit(0)
    
    # Register signal handlers (works on Unix/Linux/Colab)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            epoch_start_time = time.time()
            
            try:
                # Train one epoch
                epoch_losses, sample_images = train_epoch(
                    gen_A2B, gen_B2A, disc_A, disc_B,
                    dataloader, optimizer_G, optimizer_D,
                    losses_fn, device, epoch, scaler
                )
                
                # Store losses
                losses_history.append({
                    'epoch': epoch,
                    **epoch_losses
                })
                current_losses_history = losses_history
                
                # Log epoch summary to WandB (with error handling)
                try:
                    log_dict = {f"epoch_loss/{k}": v for k, v in epoch_losses.items()}
                    log_dict['epoch'] = epoch
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: WandB logging failed: {e}. Continuing...")
                
                # Log loss curves plot (with error handling)
                if len(losses_history) > 1:
                    try:
                        plot_loss_curves(losses_history)
                    except Exception as e:
                        print(f"Warning: Loss curve plotting failed: {e}. Continuing...")
                
                # Log final image grid for epoch (with error handling)
                try:
                    real_A, fake_B, rec_A, real_B, fake_A, rec_B = sample_images
                    log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, 
                                  epoch, len(dataloader), args.experiment_name)
                except Exception as e:
                    print(f"Warning: Image grid logging failed: {e}. Continuing...")
                
                # Save 5 fixed image pairs for comparison across epochs
                if args.save_images and fixed_image_pairs_original is not None:
                    # Use image_save_freq (which defaults to checkpoint_freq if not set)
                    if (epoch + 1) % image_save_freq == 0 or (epoch + 1) == args.epochs:
                        try:
                            # Regenerate transformations for the fixed pairs using current model
                            gen_A2B.eval()
                            gen_B2A.eval()
                            fixed_pairs_results = []
                            with torch.no_grad():
                                for real_A_fixed in fixed_image_pairs_original:
                                    fake_B_fixed = gen_A2B(real_A_fixed)
                                    rec_A_fixed = gen_B2A(fake_B_fixed)
                                    fixed_pairs_results.append((real_A_fixed, fake_B_fixed, rec_A_fixed))
                            gen_A2B.train()
                            gen_B2A.train()
                            
                            # Save the 5 fixed pairs
                            save_fixed_image_pairs(fixed_pairs_results, epoch, args.experiment_name, args.image_output_dir)
                        except Exception as e:
                            print(f"Warning: Failed to save fixed image pairs: {e}. Continuing...")
                
                # Save checkpoint every epoch as backup (quick recovery)
                try:
                    save_checkpoint(
                        epoch + 1, gen_A2B, gen_B2A, disc_A, disc_B,
                        optimizer_G, optimizer_D, losses_history,
                        args.checkpoint_dir, args.experiment_name,
                        save_latest=True  # Always save latest.pth
                    )
                except Exception as e:
                    print(f"ERROR: Failed to save backup checkpoint: {e}")
                    raise  # Re-raise if checkpoint save fails
                
                # Save numbered checkpoint at specified frequency
                if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
                    try:
                        save_checkpoint(
                            epoch + 1, gen_A2B, gen_B2A, disc_A, disc_B,
                            optimizer_G, optimizer_D, losses_history,
                            args.checkpoint_dir, args.experiment_name,
                            save_latest=False  # Don't duplicate latest save
                        )
                    except Exception as e:
                        print(f"ERROR: Failed to save numbered checkpoint: {e}")
                        # Don't raise - we already have latest.pth
                
                epoch_time = time.time() - epoch_start_time
                print(f"   Generator Loss: {epoch_losses['gen_total']:.4f}")
                print(f"   Discriminator A Loss: {epoch_losses['disc_A']:.4f}")
                print(f"   Discriminator B Loss: {epoch_losses['disc_B']:.4f}")
                print(f"   Epoch time: {epoch_time/60:.1f} minutes")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user!")
                raise  # Re-raise to trigger signal handler
            except Exception as e:
                print(f"\n\n❌ Error during epoch {epoch+1}: {e}")
                print("Attempting to save emergency checkpoint...")
                try:
                    save_checkpoint(
                        epoch + 1, gen_A2B, gen_B2A, disc_A, disc_B,
                        optimizer_G, optimizer_D, losses_history,
                        args.checkpoint_dir, args.experiment_name,
                        save_latest=True
                    )
                    print("✅ Emergency checkpoint saved! You can resume training.")
                except Exception as save_error:
                    print(f"❌ Failed to save emergency checkpoint: {save_error}")
                raise  # Re-raise the original error
        
        print("\n✅ Training completed!")
        
    except KeyboardInterrupt:
        # Signal handler will save checkpoint
        pass
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Attempting final emergency checkpoint save...")
        try:
            save_checkpoint(
                current_epoch + 1, gen_A2B, gen_B2A, disc_A, disc_B,
                optimizer_G, optimizer_D, losses_history,
                args.checkpoint_dir, args.experiment_name,
                save_latest=True
            )
            print(" Final checkpoint saved!")
        except:
            print(" Failed to save final checkpoint")
        raise
    finally:
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleGAN with WandB")
    
    # Data arguments
    parser.add_argument("--data_A", type=str, required=True,
                       help="Path to domain A images (e.g., photos)")
    parser.add_argument("--data_B", type=str, required=True,
                       help="Path to domain B images (e.g., Monet paintings)")
    
    # Model arguments
    parser.add_argument("--architecture", type=str, default="resnet",
                       choices=["resnet", "unet"],
                       help="Generator architecture (resnet or unet)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                       help="Learning rate")
    parser.add_argument("--lambda_cycle", type=float, default=10.0,
                       help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_identity", type=float, default=0.5,
                       help="Weight for identity loss")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use Automatic Mixed Precision (AMP) for faster training (default: True)")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/content/drive/MyDrive/cyclegan_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Image saving arguments (for comparison and visualization)
    parser.add_argument("--save_images", action="store_true", default=True,
                       help="Save generated images to disk for comparison across epochs (default: True)")
    parser.add_argument("--image_output_dir", type=str, default=None,
                       help="Base directory to save generated images (default: checkpoint_dir/images/experiment_name). Images are organized by epoch in subdirectories.")
    parser.add_argument("--image_save_freq", type=int, default=None,
                       help="Save images every N epochs (default: same as checkpoint_freq)")
    
    # WandB arguments
    parser.add_argument("--project_name", type=str, default="cyclegan-experiments",
                       help="WandB project name")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., 'resnet-baseline')")
    
    args = parser.parse_args()
    
    train(args)