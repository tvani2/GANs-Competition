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
import matplotlib.pyplot as plt

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
    # Convert to numpy
    image = tensor.cpu().detach().permute(1, 2, 0).numpy()
    return image


def log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, epoch, step, 
                   experiment_name):
    """
    Create and log before/after comparison grid to WandB with enhanced styling
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    # Row 1: Photo → Monet → Photo
    axes[0, 0].imshow(tensor_to_image(real_A[0]))
    axes[0, 0].set_title('Real Photo', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tensor_to_image(fake_B[0]))
    axes[0, 1].set_title('Generated Monet', fontsize=14, fontweight='bold', pad=10, color='green')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(tensor_to_image(rec_A[0]))
    axes[0, 2].set_title('Reconstructed Photo', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    # Row 2: Monet → Photo → Monet
    axes[1, 0].imshow(tensor_to_image(real_B[0]))
    axes[1, 0].set_title('Real Monet', fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(tensor_to_image(fake_A[0]))
    axes[1, 1].set_title('Generated Photo', fontsize=14, fontweight='bold', pad=10, color='green')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(tensor_to_image(rec_B[0]))
    axes[1, 2].set_title('Reconstructed Monet', fontsize=14, fontweight='bold', pad=10)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'{experiment_name} - Epoch {epoch+1}, Step {step}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Log to WandB
    wandb.log({f"images/comparison_grid": wandb.Image(fig)})
    plt.close()
    
    # Also log individual image pairs for better tracking
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    fig2.patch.set_facecolor('white')
    axes2[0].imshow(tensor_to_image(real_A[0]))
    axes2[0].set_title('Input Photo', fontsize=12, fontweight='bold')
    axes2[0].axis('off')
    axes2[1].imshow(tensor_to_image(fake_B[0]))
    axes2[1].set_title('Generated Monet', fontsize=12, fontweight='bold', color='green')
    axes2[1].axis('off')
    plt.tight_layout()
    wandb.log({f"images/photo_to_monet": wandb.Image(fig2)})
    plt.close(fig2)
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
    fig3.patch.set_facecolor('white')
    axes3[0].imshow(tensor_to_image(real_B[0]))
    axes3[0].set_title('Input Monet', fontsize=12, fontweight='bold')
    axes3[0].axis('off')
    axes3[1].imshow(tensor_to_image(fake_A[0]))
    axes3[1].set_title('Generated Photo', fontsize=12, fontweight='bold', color='green')
    axes3[1].axis('off')
    plt.tight_layout()
    wandb.log({f"images/monet_to_photo": wandb.Image(fig3)})
    plt.close(fig3)


def save_checkpoint(epoch, gen_A2B, gen_B2A, disc_A, disc_B, 
                   optimizer_G, optimizer_D, losses_history, checkpoint_dir, 
                   experiment_name):
    """
    Save complete checkpoint to Google Drive and WandB
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
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    # Also save to WandB as artifact
    artifact = wandb.Artifact(f'{experiment_name}-epoch-{epoch}', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
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
        
        # Log to WandB every N steps (reduced frequency for better performance)
        # Logging less frequently reduces overhead and speeds up training
        log_freq = 200  # Log every 200 steps instead of 100
        image_log_freq = 1000  # Log images every 1000 steps instead of 500
        
        if step % log_freq == 0:
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
            
            # Log image grid less frequently to avoid expensive matplotlib operations
            if step % image_log_freq == 0:
                log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, 
                             epoch, step, wandb.run.name)
    
    # Average losses over epoch
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses, (real_A, fake_B, rec_A, real_B, fake_A, rec_B)


def plot_loss_curves(losses_history):
    """
    Create and log loss curves plot to WandB
    """
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
    plt.close()


def train(args):
    """
    Main training function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
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
    
    # ==================== Training Loop ====================
    print(f"   Starting training: {args.experiment_name}")
    print(f"   Architecture: {args.architecture}")
    print(f"   Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"   Batch size: {args.batch_size}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        
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
        
        # Log epoch summary to WandB
        log_dict = {f"epoch_loss/{k}": v for k, v in epoch_losses.items()}
        log_dict['epoch'] = epoch
        wandb.log(log_dict)
        
        # Log loss curves plot
        if len(losses_history) > 1:
            plot_loss_curves(losses_history)
        
        # Log final image grid for epoch
        real_A, fake_B, rec_A, real_B, fake_A, rec_B = sample_images
        log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, 
                      epoch, len(dataloader), args.experiment_name)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(
                epoch + 1, gen_A2B, gen_B2A, disc_A, disc_B,
                optimizer_G, optimizer_D, losses_history,
                args.checkpoint_dir, args.experiment_name
            )
        
        print(f"   Generator Loss: {epoch_losses['gen_total']:.4f}")
        print(f"   Discriminator A Loss: {epoch_losses['disc_A']:.4f}")
        print(f"   Discriminator B Loss: {epoch_losses['disc_B']:.4f}")
    
    print("\n✅ Training completed!")
    wandb.finish()


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
    
    # WandB arguments
    parser.add_argument("--project_name", type=str, default="cyclegan-experiments",
                       help="WandB project name")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., 'resnet-baseline')")
    
    args = parser.parse_args()
    
    train(args)