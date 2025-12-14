# Progress Display & Image Saving Updates

## 🎯 Changes Made

### 1. **Enhanced Batch Progress Display** 

#### Real-time Progress Bar
Now shows a visual progress bar during each epoch:
```
  Batch [ 234/1000] |███████████████████░░░░░░░░░░| 23.4%
```

**Features:**
- Updates **20 times per epoch** (not too spammy, not too sparse)
- Clean progress bar with percentage
- Shows current batch number and total batches
- Non-blocking carriage return for smooth updates

#### Detailed Progress Updates
Every ~5-10% or every 2 minutes, shows full loss breakdown:
```
  Batch [ 234/1000] - G_loss: 2.3456 (adv: 0.8234, cyc: 1.3456, id: 0.1766) | D_loss: 0.4567 (A: 0.4321, B: 0.4813)
```

**Shows:**
- Generator total loss with breakdown:
  - `adv`: Adversarial loss
  - `cyc`: Cycle consistency loss  
  - `id`: Identity loss
- Discriminator losses for both domains A and B

### 2. **Better Epoch Display**

#### Epoch Header (More Visible)
```
======================================================================
🎨 Epoch 15/30
======================================================================
```

#### Epoch Summary (More Informative)
```
  ✅ Epoch 15/30 completed in 12m 34s
     Generator Loss: 2.3456 (adv: 0.8234, cyc: 1.3456, id: 0.1766)
     Discriminator Loss: A=0.4321, B=0.4813, avg=0.4567
     💾 Images saved for epoch 15
```

### 3. **More Frequent Image Saving**

#### New Default: Save Every Epoch
Changed from saving every 4-5 epochs to **saving EVERY epoch** by default.

**Benefits:**
- Track visual progress more closely
- Better for creating comparison grids
- Helps identify when the model starts producing good results
- Great for making progress videos/gifs

#### Configuration
```bash
# Default (every epoch)
--image_save_freq 1

# Or customize
--image_save_freq 2  # Every 2 epochs
--image_save_freq 5  # Every 5 epochs
```

### 4. **Visual Feedback for Image Saves**
Now shows when images are saved:
```
💾 Images saved for epoch 15
```

## 📊 Example Training Output

```
======================================================================
🎨 Epoch 15/30
======================================================================
  Batch [  50/1000] |████░░░░░░░░░░░░░░░░░░░░░░░░░░|  5.0%
  Batch [  50/1000] - G_loss: 2.8945 (adv: 0.9234, cyc: 1.7456, id: 0.2255) | D_loss: 0.5234 (A: 0.5123, B: 0.5345)
  Batch [ 100/1000] |████████░░░░░░░░░░░░░░░░░░░░░░| 10.0%
  Batch [ 100/1000] - G_loss: 2.7123 (adv: 0.8876, cyc: 1.6234, id: 0.2013) | D_loss: 0.4987 (A: 0.4932, B: 0.5042)
  Batch [ 150/1000] |████████████░░░░░░░░░░░░░░░░░░| 15.0%
  Batch [ 150/1000] - G_loss: 2.5678 (adv: 0.8523, cyc: 1.5234, id: 0.1921) | D_loss: 0.4756 (A: 0.4698, B: 0.4814)
  ...
  Batch [1000/1000] |██████████████████████████████| 100.0%

  ✅ Epoch 15/30 completed in 12m 34s
     Generator Loss: 2.3456 (adv: 0.8234, cyc: 1.3456, id: 0.1766)
     Discriminator Loss: A=0.4321, B=0.4813, avg=0.4567
     💾 Images saved for epoch 15
```

## 🎨 Progress Bar Legend

- `█` = Completed batches
- `░` = Remaining batches
- Shows: `[current/total] |progress bar| percentage%`

## ⚙️ Technical Details

### Progress Update Frequency
- **Progress bar**: Updates ~20 times per epoch (`num_batches // 20`)
- **Detailed stats**: Every ~5% OR every 2 minutes (whichever comes first)
- **Prevents timeouts**: Keeps Colab/cloud instances active
- **Low overhead**: Uses carriage return (`\r`) for efficient updates

### Loss Display
All losses shown to 4 decimal places for precision tracking:
- Generator total and components (adversarial, cycle, identity)
- Discriminator for both domains (A, B) plus average
- Consistent formatting for easy parsing/logging

### Image Saving
- **Default frequency**: Every epoch (customizable)
- **Fixed pairs**: Same 5 image pairs across all epochs for consistent comparison
- **Directory structure**: `checkpoint_dir/images/experiment_name/epoch_XX/`
- **Feedback**: Shows confirmation when images are saved

## 💡 Usage Tips

### For Quick Experiments
```bash
--image_save_freq 1  # Save every epoch (default now)
```

### For Long Training Runs
```bash
--image_save_freq 2  # Every 2 epochs to save disk space
```

### For Final Models Only
```bash
--image_save_freq 5  # Less frequent, like checkpoints
```

## 🚀 Benefits

1. **Better Monitoring**: See exactly where training is at any moment
2. **Faster Debugging**: Identify issues earlier with per-batch progress
3. **Visual Tracking**: More frequent images show learning progression
4. **Cloud-Friendly**: Regular updates prevent timeout disconnections
5. **Professional Output**: Clean, informative display format

## 📝 Notes

- Progress bar uses Unicode box-drawing characters (`█`, `░`)
- All updates use `\r` (carriage return) for smooth, non-scrolling updates
- Newline is printed after epoch completes to preserve output
- Compatible with Colab, Jupyter, and terminal environments
- Images are saved to disk AND logged to WandB



