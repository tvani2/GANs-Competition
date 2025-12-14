"""
Colab Setup Script - Run this first in Colab!

This script sets up everything needed for training:
1. Mounts Google Drive
2. Installs dependencies
3. Clones repository (if needed)
4. Sets up directories
5. Logs in to WandB
"""

import os
import sys

def setup_colab():
    """Setup Colab environment for CycleGAN training"""
    
    print("=" * 60)
    print("🚀 Setting up Colab for CycleGAN Training")
    print("=" * 60)
    
    # Step 1: Mount Google Drive
    print("\n📁 Step 1: Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted!")
    except ImportError:
        print("⚠️  Not in Colab environment. Skipping drive mount.")
        print("   (This is OK if running locally)")
    except Exception as e:
        print(f"❌ Error mounting drive: {e}")
        return False
    
    # Step 2: Install dependencies
    print("\n📦 Step 2: Installing dependencies...")
    os.system("pip install -q wandb torch torchvision tqdm")
    print("✅ Dependencies installed!")
    
    # Step 3: Clone repository (if not already cloned)
    print("\n🔗 Step 3: Setting up repository...")
    repo_path = "/content/GANs-Competition"
    
    if not os.path.exists(repo_path):
        print("   Cloning repository from GitHub...")
        # TODO: Replace with your actual GitHub URL
        github_url = input("Enter your GitHub repository URL (or press Enter to skip): ").strip()
        if github_url:
            os.system(f"git clone {github_url} {repo_path}")
            print(f"✅ Repository cloned to {repo_path}")
        else:
            print("⚠️  Skipping repository clone. Make sure to clone manually!")
    else:
        print(f"✅ Repository already exists at {repo_path}")
    
    # Add to Python path
    if repo_path not in sys.path:
        sys.path.append(repo_path)
        print(f"✅ Added {repo_path} to Python path")
    
    # Step 4: Create directories on Google Drive
    print("\n📂 Step 4: Creating directories on Google Drive...")
    try:
        checkpoint_dir = "/content/drive/MyDrive/cyclegan_checkpoints"
        output_dir = "/content/drive/MyDrive/cyclegan_outputs"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✅ Checkpoint directory: {checkpoint_dir}")
        print(f"✅ Output directory: {output_dir}")
    except Exception as e:
        print(f"⚠️  Could not create directories: {e}")
        print("   (This is OK if not in Colab)")
    
    # Step 5: Login to WandB
    print("\n🔐 Step 5: Logging in to WandB...")
    try:
        import wandb
        wandb.login()
        print("✅ WandB login successful!")
    except Exception as e:
        print(f"⚠️  WandB login error: {e}")
        print("   You can login manually with: wandb.login()")
    
    # Step 6: Verify imports
    print("\n🧪 Step 6: Verifying imports...")
    try:
        from models import ResNetGenerator, UNetGenerator, PatchGANDiscriminator
        from losses.losses_cyclegan import CycleGANLosses
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure repository is cloned and path is correct!")
        return False
    
    # Step 7: Check GPU
    print("\n🎮 Step 7: Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU available. Training will be slow on CPU.")
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your dataset paths")
    print("2. Run training with: python train_cyclegan.py --data_A <path> --data_B <path> ...")
    print("3. Monitor on WandB: https://wandb.ai")
    
    return True


if __name__ == "__main__":
    setup_colab()









