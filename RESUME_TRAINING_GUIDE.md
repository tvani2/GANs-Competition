# How to Resume Training from Checkpoint 📁

## Quick Answer: YES! You can resume from any checkpoint!

The code automatically:
- ✅ Saves model weights
- ✅ Saves optimizer states
- ✅ Saves training history
- ✅ Continues from exact same point

---

## Step-by-Step: Training 30 → 100 Epochs

### Phase 1: Train for 30 Epochs

```python
# In Colab - Train initial 30 epochs
!cd /content/GANs-Competition && python train_cyclegan.py \
    --data_A "/content/drive/MyDrive/datasets/photos" \
    --data_B "/content/drive/MyDrive/datasets/monet" \
    --architecture resnet \
    --epochs 30 \
    --batch_size 1 \
    --checkpoint_dir "/content/drive/MyDrive/cyclegan_checkpoints" \
    --checkpoint_freq 5 \
    --experiment_name "resnet-baseline"
```

**What you'll see:**
- Training runs for epochs 1-30
- Checkpoints saved at: epoch 5, 10, 15, 20, 25, 30
- Final checkpoint: `resnet-baseline_epoch_30.pth`

**Check checkpoint exists:**
```python
import os
checkpoint_path = "/content/drive/MyDrive/cyclegan_checkpoints/resnet-baseline_epoch_30.pth"
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / 1e6
    print(f"✅ Checkpoint found! Size: {size_mb:.1f} MB")
else:
    print("❌ Checkpoint not found")
```

---

### Phase 2: Resume Training (30 → 100 Epochs)

```python
# Resume from epoch 30 and continue to epoch 100
!cd /content/GANs-Competition && python train_cyclegan.py \
    --data_A "/content/drive/MyDrive/datasets/photos" \
    --data_B "/content/drive/MyDrive/datasets/monet" \
    --architecture resnet \
    --epochs 100 \
    --batch_size 1 \
    --checkpoint_dir "/content/drive/MyDrive/cyclegan_checkpoints" \
    --checkpoint_freq 5 \
    --resume_from "/content/drive/MyDrive/cyclegan_checkpoints/resnet-baseline_epoch_30.pth" \
    --experiment_name "resnet-baseline"
```

**What you'll see:**
```
📂 Loading checkpoint: resnet-baseline_epoch_30.pth
✅ Checkpoint loaded. Resuming from epoch 31
🎯 Starting training: resnet-baseline
   Architecture: resnet
   Epochs: 100 (starting from 31)  ← Notice this!

📊 Epoch 31/100
   Generator Loss: ...
   ...
```

**Training continues:**
- Epochs 31-100 (70 more epochs)
- Uses exact same weights from epoch 30
- Optimizer states restored (learning rate schedule continues)
- Loss history preserved

---

## Important Notes

### ✅ Set `--epochs` to Final Target (100), NOT Remaining (70)

**WRONG:**
```bash
--epochs 70 --resume_from epoch_30.pth  # ❌ This would train epochs 31-100, then stop!
```

**RIGHT:**
```bash
--epochs 100 --resume_from epoch_30.pth  # ✅ Trains epochs 31-100 correctly
```

The code automatically calculates: `for epoch in range(start_epoch, args.epochs)`
- If checkpoint is at epoch 30, `start_epoch = 31`
- Loop runs: `range(31, 100)` → epochs 31-99 (but final checkpoint saves at 100)

---

## What Gets Restored?

When you resume, the checkpoint contains:

1. **Model Weights:**
   - `gen_A2B.state_dict()` - Generator A→B
   - `gen_B2A.state_dict()` - Generator B→A
   - `disc_A.state_dict()` - Discriminator A
   - `disc_B.state_dict()` - Discriminator B

2. **Optimizer States:**
   - `optimizer_G.state_dict()` - Generator optimizer
   - `optimizer_D.state_dict()` - Discriminator optimizer
   - (Includes momentum, learning rate schedules, etc.)

3. **Training History:**
   - `losses_history` - All previous epoch losses
   - Continuation in WandB (same run or new run)

4. **Metadata:**
   - Current epoch number
   - Experiment name

---

## Example: Complete Training Workflow

### Scenario: Colab disconnects after 25 epochs

**Step 1: Check available checkpoints**
```python
import os
checkpoint_dir = "/content/drive/MyDrive/cyclegan_checkpoints"
checkpoints = [f for f in os.listdir(checkpoint_dir) if 'resnet-baseline' in f and f.endswith('.pth')]
checkpoints.sort()
print("Available checkpoints:")
for cp in checkpoints:
    print(f"  - {cp}")
```

**Step 2: Resume from latest checkpoint**
```python
# Resume from epoch 25 (or 30 if available)
latest_checkpoint = "/content/drive/MyDrive/cyclegan_checkpoints/resnet-baseline_epoch_25.pth"

!cd /content/GANs-Competition && python train_cyclegan.py \
    --data_A "/content/drive/MyDrive/datasets/photos" \
    --data_B "/content/drive/MyDrive/datasets/monet" \
    --architecture resnet \
    --epochs 100 \
    --resume_from "{latest_checkpoint}" \
    --experiment_name "resnet-baseline"
```

---

## WandB Continuation

**Option 1: Continue same WandB run**
- Use same `--experiment_name`
- WandB will continue logging to same run
- Loss curves continue seamlessly

**Option 2: New WandB run (separate tracking)**
- Change `--experiment_name` (e.g., "resnet-baseline-continued")
- Creates new run, but uses same model weights

**Recommendation:** Keep same experiment name for continuous tracking!

---

## Troubleshooting

### Checkpoint not found?
```python
# List all checkpoints
import os
checkpoint_dir = "/content/drive/MyDrive/cyclegan_checkpoints"
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if f.endswith('.pth')]
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in sorted(checkpoints):
        print(f"  - {cp}")
else:
    print(f"Directory not found: {checkpoint_dir}")
```

### Wrong epoch count?
Remember: `--epochs 100` means "train until epoch 100 total"
- If resuming from epoch 30, it trains epochs 31-100
- Don't calculate remaining epochs manually!

### Checkpoint corrupted?
The code will show an error. Make sure:
- File exists
- File size > 0 MB
- Not corrupted during save

---

## Summary

✅ **YES, you can resume from checkpoints!**
✅ **Checkpoints save automatically every 5 epochs**
✅ **Use `--resume_from` flag to continue**
✅ **Set `--epochs` to final target (100), not remaining (70)**
✅ **All weights, optimizer states, and history are preserved**

**Your training is never lost if you save checkpoints to Google Drive!** 🎉









