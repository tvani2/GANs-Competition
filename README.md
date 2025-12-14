# GANs Competition - CycleGAN Implementation

Complete CycleGAN implementation with WandB integration for experiment tracking and Google Drive checkpoint management.

## 📁 Project Structure

```
GANs-Competition/
├── models/
│   ├── discriminator.py          # PatchGAN Discriminator
│   ├── resnet_generator.py       # ResNet-based Generator
│   └── unet_generator.py         # U-Net Generator
├── losses/
│   ├── losses_cyclegan.py        # CycleGAN loss functions (Hinge, Cycle, Identity)
│   └── enhanced_losses.py        # Additional enhanced losses
├── analysis_scripts/              # Data analysis and visualization scripts
├── train_cyclegan.py             # Main training script with WandB integration
├── colab_setup.py                # Colab environment setup script
├── requirements.txt              # Python dependencies
├── COLAB_SETUP_GUIDE.md          # Detailed Colab setup guide
├── QUICK_START.md                # Quick start guide
└── ANSWERS_TO_YOUR_QUESTIONS.md  # Direct answers to common questions
```

## 🚀 Quick Start

### 1. Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python train_cyclegan.py \
    --data_A "/path/to/photos" \
    --data_B "/path/to/monet" \
    --architecture resnet \
    --experiment_name "resnet-baseline" \
    --epochs 30
```

### 2. Colab Setup

See `QUICK_START.md` for the fastest setup, or `COLAB_SETUP_GUIDE.md` for detailed instructions.

**Quick Colab commands:**
```python
# In Colab notebook:
!pip install wandb
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/yourusername/GANs-Competition.git
import sys
sys.path.append('/content/GANs-Competition')
```

## 📚 Documentation

- **[ANSWERS_TO_YOUR_QUESTIONS.md](ANSWERS_TO_YOUR_QUESTIONS.md)** - Direct answers to all your questions about GitHub, WandB, checkpoints, etc.
- **[QUICK_START.md](QUICK_START.md)** - Fast setup guide for Colab
- **[COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)** - Comprehensive Colab setup guide

## 🔧 Training Features

✅ **WandB Integration**
- Automatic metric logging
- Image grid visualization
- Model checkpoint artifacts
- Experiment comparison

✅ **Checkpoint Management**
- Save to Google Drive (persistent)
- Save to WandB artifacts (versioning)
- Resume training from any checkpoint

✅ **Experiment Management**
- Change one variable at a time
- Compare ResNet vs U-Net architectures
- Track all hyperparameters

✅ **Visualization**
- Before/after image grids
- Loss curves (automatic)
- System metrics (GPU, memory)

## 📊 Running Experiments

### Experiment 1: ResNet Baseline
```bash
python train_cyclegan.py \
    --data_A "/path/to/photos" \
    --data_B "/path/to/monet" \
    --architecture resnet \
    --experiment_name "resnet-baseline" \
    --epochs 30 \
    --checkpoint_dir "/content/drive/MyDrive/checkpoints"
```

### Experiment 2: U-Net (Only architecture changed!)
```bash
python train_cyclegan.py \
    --data_A "/path/to/photos" \
    --data_B "/path/to/monet" \
    --architecture unet \
    --experiment_name "unet-baseline" \
    --epochs 30 \
    --checkpoint_dir "/content/drive/MyDrive/checkpoints"
```

### Resume Training
```bash
python train_cyclegan.py \
    --resume_from "/content/drive/MyDrive/checkpoints/resnet-baseline_epoch_10.pth" \
    --data_A "/path/to/photos" \
    --data_B "/path/to/monet" \
    --experiment_name "resnet-baseline" \
    ...
```

## 🎯 Key Features for Your Project Requirements

### ✅ Code (25%)
- Generator and Discriminator implemented from scratch
- Loss functions implemented manually
- Training loop written from scratch

### ✅ Experiments (25%)
- Change one variable at a time (architecture: ResNet vs U-Net)
- Visual evaluation (image grids)
- Metric evaluation (FID, MiFID - can be added)

### ✅ WandB Report (20%)
- All experiments logged to WandB
- Architecture descriptions
- Experiment comparisons
- Results and analysis

## 📝 Common Questions Answered

**Q: GitHub vs Direct Paste?**  
A: Use GitHub! Clone your repo in Colab. See `ANSWERS_TO_YOUR_QUESTIONS.md`

**Q: How to include WandB?**  
A: Already integrated! Just login with `wandb.login()`. See `QUICK_START.md`

**Q: Where to save checkpoints?**  
A: Both Google Drive (for persistence) and WandB (for versioning). See `COLAB_SETUP_GUIDE.md`

**Q: How to resume training?**  
A: Use `--resume_from` flag with checkpoint path. See training examples above.

**Q: How to log plots?**  
A: Automatic! WandB creates plots from logged metrics. Images logged with `wandb.Image()`. See `train_cyclegan.py`

**Q: What does Colab lose?**  
A: Everything in `/content/` directory. Save checkpoints to `/content/drive/MyDrive/`. See `ANSWERS_TO_YOUR_QUESTIONS.md`

## 🔗 WandB Dashboard

After training, visit your WandB dashboard:
```
https://wandb.ai/YOUR_USERNAME/cyclegan-experiments
```

You'll see:
- Loss curves
- Generated image grids
- Model checkpoints
- System metrics
- Experiment comparisons

## 📦 Requirements

See `requirements.txt` for full list. Main dependencies:
- torch >= 1.9.0
- torchvision >= 0.10.0
- wandb >= 0.12.0
- numpy, PIL, matplotlib, etc.

## 🎓 Next Steps

1. ✅ Upload code to GitHub
2. ✅ Set up Colab environment
3. ✅ Run baseline experiment (ResNet)
4. ✅ Run comparison experiment (U-Net)
5. ✅ Compare results on WandB
6. ✅ Create WandB report with analysis

## 📖 Detailed Guides

- **Getting Started**: Read `QUICK_START.md`
- **Colab Setup**: Read `COLAB_SETUP_GUIDE.md`
- **Questions**: Read `ANSWERS_TO_YOUR_QUESTIONS.md`
- **Training**: See `train_cyclegan.py` with full documentation

## 🤝 Contributing

This is a competition project. Follow the experiment guidelines:
- Change one variable at a time
- Document all changes
- Log everything to WandB
- Save checkpoints regularly

## 📄 License

Academic/Competition use only.
