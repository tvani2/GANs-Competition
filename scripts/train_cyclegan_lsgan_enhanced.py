"""
CycleGAN Training Script with Enhanced LSGAN Loss

This script uses enhanced LSGAN with:
- Edge Softness Loss (λ=0.1) - Soften sharp edges
- High-Frequency Suppression (λ=0.05) - Remove photographic detail

Based on Epoch 40 analysis.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
import signal
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import models and ENHANCED losses
from models import ResNetGenerator, PatchGANDiscriminator
from losses.losses_cyclegan_lsgan_enhanced import EnhancedCycleGANLossesLSGAN

# Import shared functions from original training script
from train_cyclegan import (
    ImageDataset,
    tensor_to_image,
    save_checkpoint,
    load_checkpoint,
    plot_loss_curves,
    calculate_ssim_batch,
    calculate_psnr_batch,
    calculate_l1_distance,
    calculate_color_histogram_distance,
    calculate_diversity_score,
    evaluate_model,
    log_evaluation_metrics,
    save_fixed_image_pairs,
    log_image_grid
)


def train_epoch_enhanced(gen_A2B, gen_B2A, disc_A, disc_B, dataloader, 
                        optimizer_G, optimizer_D, losses_fn, device, epoch, scaler=None):
    """
    Train for one epoch with ENHANCED LSGAN loss
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
        'gen_edge': 0.0,       # NEW
        'gen_freq': 0.0,       # NEW
        'disc_A': 0.0,
        'disc_B': 0.0,
    }
    
    num_batches = len(dataloader)
    last_progress_time = time.time()
    progress_update_freq = max(1, num_batches // 20)
    
    for step, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        
        # Progress bar
        if step % progress_update_freq == 0 or step == num_batches - 1:
            progress_pct = (step + 1) / num_batches * 100
            bar_length = 30
            filled_length = int(bar_length * (step + 1) / num_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r  Batch [{step+1:4d}/{num_batches:4d}] |{bar}| {progress_pct:5.1f}%", end='', flush=True)
        
        # ==================== Train Generators ====================
        optimizer_G.zero_grad()
        
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass
            fake_B = gen_A2B(real_A)
            fake_A = gen_B2A(real_B)
            
            rec_A = gen_B2A(fake_B)
            rec_B = gen_A2B(fake_A)
            
            identity_A = gen_B2A(real_A)
            identity_B = gen_A2B(real_B)
            
            # Discriminator predictions
            fake_B_pred = disc_B(fake_B)
            fake_A_pred = disc_A(fake_A)
            
            # Generator losses (ENHANCED with edge + freq)
            gen_losses = losses_fn.generator_loss(
                fake_A_pred, fake_B_pred,
                real_A, rec_A, real_B, rec_B,
                fake_A, fake_B,  # Pass fake images for perceptual losses
                identity_A, identity_B
            )
            
            gen_total = gen_losses['total']
        
        # Backward pass
        if use_amp:
            scaler.scale(gen_total).backward()
            scaler.step(optimizer_G)
            scaler.update()
        else:
            gen_total.backward()
            optimizer_G.step()
        
        # ==================== Train Discriminators ====================
        optimizer_D.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=use_amp):
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
        
        optimizer_D.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=use_amp):
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
        # Edge and freq losses might be 0.0 (float) or tensor
        epoch_losses['gen_edge'] += gen_losses['edge'].item() if isinstance(gen_losses['edge'], torch.Tensor) else gen_losses['edge']
        epoch_losses['gen_freq'] += gen_losses['freq'].item() if isinstance(gen_losses['freq'], torch.Tensor) else gen_losses['freq']
        epoch_losses['disc_A'] += disc_A_loss.item()
        epoch_losses['disc_B'] += disc_B_loss.item()
        
        # Detailed progress
        current_time = time.time()
        show_detailed_progress = (
            step % progress_update_freq == 0 or 
            current_time - last_progress_time > 120 or
            step == num_batches - 1
        )
        
        if show_detailed_progress and step > 0:
            edge_val = gen_losses['edge'].item() if isinstance(gen_losses['edge'], torch.Tensor) else gen_losses['edge']
            freq_val = gen_losses['freq'].item() if isinstance(gen_losses['freq'], torch.Tensor) else gen_losses['freq']
            print(f"\r  Batch [{step+1:4d}/{num_batches:4d}] - "
                  f"G_loss: {gen_total.item():.4f} "
                  f"(adv: {gen_losses['adversarial'].item():.4f}, "
                  f"cyc: {gen_losses['cycle'].item():.4f}, "
                  f"id: {gen_losses['identity'].item():.4f}, "
                  f"edge: {edge_val:.4f}, "
                  f"freq: {freq_val:.4f}) | "
                  f"D_loss: {(disc_A_loss.item() + disc_B_loss.item())/2:.4f}")
            last_progress_time = current_time
        
        # WandB logging
        log_freq = 200
        image_log_freq = 1000
        
        if step % log_freq == 0:
            try:
                edge_val = gen_losses['edge'].item() if isinstance(gen_losses['edge'], torch.Tensor) else gen_losses['edge']
                freq_val = gen_losses['freq'].item() if isinstance(gen_losses['freq'], torch.Tensor) else gen_losses['freq']
                wandb.log({
                    "loss/gen_total": gen_total.item(),
                    "loss/gen_adv": gen_losses['adversarial'].item(),
                    "loss/gen_cycle": gen_losses['cycle'].item(),
                    "loss/gen_identity": gen_losses['identity'].item(),
                    "loss/gen_edge": edge_val,
                    "loss/gen_freq": freq_val,
                    "loss/disc_A": disc_A_loss.item(),
                    "loss/disc_B": disc_B_loss.item(),
                    "epoch": epoch,
                    "step": epoch * num_batches + step,
                })
            except Exception as e:
                print(f"Warning: WandB logging failed: {e}")
            
            if step % image_log_freq == 0:
                try:
                    log_image_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, 
                                 epoch, step, wandb.run.name if wandb.run else "training")
                except Exception as e:
                    print(f"Warning: Image logging failed: {e}")
    
    print()
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses, (real_A, fake_B, rec_A, real_B, fake_A, rec_B)


# Use the train() function from original script but with enhanced training
def train_enhanced(args):
    """Main training function with ENHANCED LSGAN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"🎯 Loss type: ENHANCED LSGAN (Edge + Freq)")
    
    # WandB init
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config={
            "architecture": args.architecture,
            "loss_type": "Enhanced LSGAN",
            "lambda_edge": 0.1,
            "lambda_freq": 0.05,
            **vars(args)
        },
        tags=[args.architecture, "cyclegan", "lsgan", "enhanced"],
    )
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(args.data_A, args.data_B, transform=transform)
    num_workers = args.num_workers if args.num_workers is not None else (
        0 if not torch.cuda.is_available() or args.resume_from else min(8, os.cpu_count() or 4)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Initialize models
    gen_A2B = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
    gen_B2A = ResNetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
    disc_A = PatchGANDiscriminator(input_nc=3, ndf=64).to(device)
    disc_B = PatchGANDiscriminator(input_nc=3, ndf=64).to(device)
    
    # Initialize optimizers
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
    
    # Initialize ENHANCED loss function
    losses_fn = EnhancedCycleGANLossesLSGAN(
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_edge=0.1,   # Edge softness
        lambda_freq=0.05   # High-freq suppression
    )
    print(f"✅ Using ENHANCED LSGAN loss function")
    
    # Mixed precision
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed Precision Training (AMP) enabled")
    
    # Load checkpoint if resuming
    start_epoch = 0
    losses_history = []
    
    if args.resume_from:
        start_epoch, exp_name, losses_history = load_checkpoint(
            args.resume_from,
            gen_A2B, gen_B2A, disc_A, disc_B,
            optimizer_G, optimizer_D,
            device
        )
        print(f"✅ Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"🎨 Starting Enhanced LSGAN Training")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"Enhancements: Edge Softness + High-Freq Suppression")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n{'='*70}")
            print(f"🎨 Epoch {epoch+1}/{args.epochs} (Enhanced LSGAN)")
            print(f"{'='*70}")
            epoch_start_time = time.time()
            
            # Train one epoch
            epoch_losses, sample_images = train_epoch_enhanced(
                gen_A2B, gen_B2A, disc_A, disc_B,
                dataloader, optimizer_G, optimizer_D,
                losses_fn, device, epoch, scaler
            )
            
            # Store losses
            losses_history.append({'epoch': epoch, **epoch_losses})
            
            # Log to WandB
            try:
                log_dict = {f"epoch_loss/{k}": v for k, v in epoch_losses.items()}
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
            except Exception as e:
                print(f"Warning: WandB logging failed: {e}")
            
            # Save checkpoint
            try:
                save_checkpoint(
                    epoch, gen_A2B, gen_B2A, disc_A, disc_B,
                    optimizer_G, optimizer_D, losses_history,
                    args.checkpoint_dir, args.experiment_name,
                    save_latest=True
                )
            except Exception as e:
                print(f"ERROR: Failed to save checkpoint: {e}")
            
            if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
                try:
                    save_checkpoint(
                        epoch, gen_A2B, gen_B2A, disc_A, disc_B,
                        optimizer_G, optimizer_D, losses_history,
                        args.checkpoint_dir, args.experiment_name,
                        save_latest=False
                    )
                except Exception as e:
                    print(f"ERROR: Failed to save numbered checkpoint: {e}")
            
            epoch_time = time.time() - epoch_start_time
            mins, secs = divmod(int(epoch_time), 60)
            print(f"\n  ✅ Epoch {epoch+1}/{args.epochs} completed in {mins}m {secs}s")
            print(f"     Generator Loss: {epoch_losses['gen_total']:.4f} "
                  f"(adv: {epoch_losses['gen_adv']:.4f}, "
                  f"cyc: {epoch_losses['gen_cycle']:.4f}, "
                  f"id: {epoch_losses['gen_identity']:.4f}, "
                  f"edge: {epoch_losses['gen_edge']:.4f}, "
                  f"freq: {epoch_losses['gen_freq']:.4f})")
            print(f"     Discriminator Loss: A={epoch_losses['disc_A']:.4f}, "
                  f"B={epoch_losses['disc_B']:.4f}")
        
        print("\n✅ Training completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
    finally:
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleGAN with Enhanced LSGAN")
    
    # Data
    parser.add_argument("--data_A", type=str, required=True)
    parser.add_argument("--data_B", type=str, required=True)
    
    # Model
    parser.add_argument("--architecture", type=str, default="resnet")
    
    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_identity", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/content/drive/MyDrive/GANsHomework/cyclegan_checkpoints_enhanced")
    parser.add_argument("--checkpoint_freq", type=int, default=4)
    parser.add_argument("--resume_from", type=str, default=None)
    
    # Image saving
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--image_output_dir", type=str, default=None)
    parser.add_argument("--image_save_freq", type=int, default=10)
    
    # WandB
    parser.add_argument("--project_name", type=str, default="cyclegan-experiments")
    parser.add_argument("--experiment_name", type=str, required=True)
    
    args = parser.parse_args()
    
    train_enhanced(args)

