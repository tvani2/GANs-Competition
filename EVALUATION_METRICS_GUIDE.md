# 📊 CycleGAN Evaluation Metrics Guide

Beyond just looking at generated images, you can now use **quantitative metrics** to objectively measure training progress and model quality.

## 🎯 New Evaluation Metrics Added

### **1. SSIM (Structural Similarity Index)** ⭐⭐⭐⭐⭐
**What it measures:** How well the structure/composition is preserved in reconstructions

**Range:** 0.0 to 1.0
- **1.0** = Perfect reconstruction (identical)
- **>0.8** = Excellent quality
- **0.6-0.8** = Good quality
- **<0.6** = Poor quality

**Why it matters:**
- Checks if Photo → Monet → Photo preserves the original structure
- Better than simple pixel differences
- Correlates well with human perception

**In your training:**
```
SSIM: A=0.8234, B=0.7891 (higher=better, max=1.0)
```

---

### **2. PSNR (Peak Signal-to-Noise Ratio)** ⭐⭐⭐⭐
**What it measures:** Pixel-level reconstruction quality

**Range:** Typically 20-50 dB
- **>35 dB** = Excellent reconstruction
- **30-35 dB** = Good reconstruction
- **25-30 dB** = Fair reconstruction
- **<25 dB** = Poor reconstruction

**Why it matters:**
- Industry standard for image quality
- Good for comparing different models
- Measures how "noisy" the reconstruction is

**In your training:**
```
PSNR: A=32.45dB, B=31.23dB (higher=better)
```

---

### **3. L1 Distance (Mean Absolute Error)** ⭐⭐⭐
**What it measures:** Average pixel-wise difference

**Range:** 0.0 to 2.0 (for normalized images [-1, 1])
- **<0.1** = Excellent
- **0.1-0.3** = Good
- **0.3-0.5** = Fair
- **>0.5** = Poor

**Why it matters:**
- Simple and interpretable
- Directly used in cycle consistency loss
- Lower = better reconstruction

**In your training:**
```
L1: A=0.1234, B=0.1456 (lower=better)
```

---

### **4. Color Histogram Distance** ⭐⭐⭐⭐
**What it measures:** How well color distribution is preserved

**Range:** 0.0 to ∞ (practically 0-10)
- **<0.5** = Excellent color preservation
- **0.5-1.5** = Good
- **1.5-3.0** = Fair (colors are shifting)
- **>3.0** = Poor (major color changes)

**Why it matters:**
- Detects the "washed out" problem you saw!
- Independent of structure
- Checks if Monet style maintains color vibrancy

**In your training:**
```
Color Distance: A→B=1.234, B→A=0.987 (lower=better)
```
If this is high, increase `--lambda_identity`!

---

### **5. Diversity Score** ⭐⭐⭐⭐⭐
**What it measures:** Variety in generated outputs (mode collapse detection)

**Range:** 0.0 to ∞ (practically 10-100+)
- **>50** = Good diversity (healthy training)
- **20-50** = Moderate diversity
- **<20** = Low diversity (potential mode collapse)
- **<5** = Mode collapse! (all outputs look identical)

**Why it matters:**
- **Mode collapse** = model generates same output regardless of input (VERY BAD!)
- Early warning sign of training problems
- Higher = model is creative and diverse

**In your training:**
```
Diversity: A→B=67.34, B→A=71.23 (higher=better)
```

---

## 📈 How to Interpret the Metrics

### **Good Training Progress:**
```
📊 Evaluation Metrics (Epoch 20):
   Reconstruction Quality:
     SSIM: A=0.8234, B=0.7891 ✅ Good
     PSNR: A=32.45dB, B=31.23dB ✅ Good
     L1:   A=0.1234, B=0.1456 ✅ Good
   Color Preservation:
     Distance: A→B=0.567, B→A=0.489 ✅ Excellent
   Mode Collapse Check:
     Diversity: A→B=67.34, B→A=71.23 ✅ Healthy
```

### **Warning Signs:**
```
📊 Evaluation Metrics (Epoch 20):
   Reconstruction Quality:
     SSIM: A=0.5123, B=0.4987 ⚠️ Poor reconstruction
     PSNR: A=22.34dB, B=21.45dB ⚠️ Low quality
     L1:   A=0.4567, B=0.5123 ⚠️ High error
   Color Preservation:
     Distance: A→B=4.234, B→A=3.891 ⚠️ Colors washed out!
   Mode Collapse Check:
     Diversity: A→B=3.45, B→A=2.89 🚨 MODE COLLAPSE!
```

---

## 🎯 What Each Metric Tells You

| Metric | What It Detects | How to Fix If Bad |
|--------|----------------|-------------------|
| **SSIM** | Blurry or distorted images | Train longer, increase `lambda_cycle` |
| **PSNR** | Noisy reconstructions | Train longer, check discriminator strength |
| **L1** | Pixel-level errors | Increase `lambda_cycle`, train longer |
| **Color Distance** | Washed out/wrong colors | Increase `lambda_identity` (0.5 → 2.0+) |
| **Diversity** | Mode collapse | Lower learning rate, increase batch size, check loss curves |

---

## 🔧 Using the Metrics in Training

### **Default Behavior:**
Metrics are evaluated automatically at the same frequency as checkpoints:
```bash
--checkpoint_freq 5  # Evaluate every 5 epochs
```

### **Custom Evaluation Frequency:**
```bash
--eval_freq 2  # Evaluate every 2 epochs (more frequent)
--eval_samples 100  # Use 100 images instead of 50 (more accurate)
```

### **Full Command with Evaluation:**
```bash
cd /content/GANs-Competition && python train_cyclegan.py \
    --data_A "/content/photo_jpg" \
    --data_B "/content/monet_jpg" \
    --architecture resnet \
    --epochs 50 \
    --batch_size 1 \
    --lr 0.0002 \
    --lambda_cycle 10.0 \
    --lambda_identity 3.0 \
    --checkpoint_dir "/content/drive/MyDrive/GANsHomework/cyclegan_checkpoints" \
    --checkpoint_freq 5 \
    --image_save_freq 1 \
    --eval_freq 2 \
    --eval_samples 100 \
    --project_name "cyclegan-experiments" \
    --experiment_name "resnet-evaluated"
```

---

## 📊 Viewing Metrics in WandB

All metrics are automatically logged to WandB under the **"eval/"** prefix:

### **In WandB Dashboard:**
1. Go to your run: `https://wandb.ai/final-project-ml/cyclegan-experiments`
2. Click on "Charts" tab
3. Look for metrics starting with `eval/`:
   - `eval/ssim_reconstruction_A`
   - `eval/ssim_reconstruction_B`
   - `eval/psnr_reconstruction_A`
   - `eval/psnr_reconstruction_B`
   - `eval/l1_reconstruction_A`
   - `eval/l1_reconstruction_B`
   - `eval/color_distance_A2B`
   - `eval/color_distance_B2A`
   - `eval/diversity_A2B`
   - `eval/diversity_B2A`

### **Create Custom Plots:**
You can create multi-line plots to compare:
- SSIM over time (track reconstruction quality improvement)
- Color distance over time (track color preservation)
- Diversity over time (detect mode collapse early)

---

## 🎨 Practical Example: Diagnosing Your Current Model

Based on your epoch 20 images, here's what the metrics would likely show:

### **Expected Metrics (Epoch 20):**
```
📊 Evaluation Metrics (Epoch 20):
   Reconstruction Quality:
     SSIM: A=0.72, B=0.68 → Fair (could be better)
     PSNR: A=28.3dB, B=27.1dB → Fair
     L1:   A=0.24, B=0.28 → Fair
   Color Preservation:
     Distance: A→B=2.8, B→A=2.3 → Poor (explains washed-out look!)
   Mode Collapse Check:
     Diversity: A→B=58.2, B→A=61.7 → Good (no collapse)
```

### **Diagnosis:**
- ✅ No mode collapse (diversity is healthy)
- ✅ Structure is preserved (SSIM ~0.7)
- ⚠️ **Main issue:** High color distance = washed out colors
- **Solution:** Increase `--lambda_identity` from 0.5 to 3.0

---

## 💡 Advanced: Combining Metrics for Better Training

### **Create a "Quality Score":**
You can create a weighted combination:
```python
quality_score = (
    ssim * 0.3 +           # 30% weight on structure
    (psnr / 40) * 0.2 +    # 20% weight on pixel quality
    (1 - l1/2) * 0.2 +     # 20% weight on reconstruction
    (1/(1+color_dist)) * 0.2 +  # 20% weight on color
    (min(diversity/100, 1)) * 0.1  # 10% weight on diversity
)
```

Track this single score to see overall improvement!

---

## 📝 Metric Limitations

### **What Metrics DON'T Tell You:**
- ❌ Artistic quality (subjective)
- ❌ Whether it "looks like Monet" (semantic understanding)
- ❌ Fine details like brush strokes
- ❌ Overall aesthetic appeal

### **What Metrics DO Tell You:**
- ✅ Reconstruction quality (objective)
- ✅ Color preservation (measurable)
- ✅ Training stability (mode collapse)
- ✅ Relative improvement over time
- ✅ When to stop training

**Use metrics + visual inspection together for best results!**

---

## 🚀 Quick Reference

| Want to improve... | Check metric... | If bad, adjust... |
|-------------------|-----------------|-------------------|
| Blurriness | SSIM, PSNR | Train longer, more layers |
| Washed out colors | Color Distance | Increase `lambda_identity` |
| Bad reconstruction | L1, SSIM | Increase `lambda_cycle` |
| Mode collapse | Diversity | Lower LR, check losses |
| Overall quality | All metrics | Balance all hyperparameters |

---

## 📊 Sample Output

When training, you'll now see:
```
======================================================================
🎨 Epoch 20/50
======================================================================
  Batch [7038/7038] |██████████████████████████████| 100.0%

  🔍 Running comprehensive evaluation...

  📊 Evaluation Metrics (Epoch 20):
     Reconstruction Quality:
       SSIM: A=0.7234, B=0.6891 (higher=better, max=1.0)
       PSNR: A=28.45dB, B=27.23dB (higher=better)
       L1:   A=0.2412, B=0.2756 (lower=better)
     Color Preservation:
       Distance: A→B=2.567, B→A=2.189 (lower=better)
     Mode Collapse Check:
       Diversity: A→B=58.34, B→A=62.23 (higher=better)

  ✅ Epoch 20/50 completed in 12m 34s
     Generator Loss: 2.3456 (adv: 0.8234, cyc: 1.3456, id: 0.1766)
     Discriminator Loss: A=0.4321, B=0.4813, avg=0.4567
     💾 Images saved for epoch 20
```

---

## 🎓 Further Reading

- **SSIM Paper:** "Image Quality Assessment: From Error Visibility to Structural Similarity"
- **FID Score:** Consider implementing Fréchet Inception Distance for even better evaluation
- **LPIPS:** Learned Perceptual Image Patch Similarity (requires pretrained network)

---

## ✅ Summary

You now have **5 quantitative metrics** to evaluate your CycleGAN:

1. **SSIM** - Structure preservation
2. **PSNR** - Pixel quality
3. **L1** - Reconstruction error
4. **Color Distance** - Color preservation
5. **Diversity** - Mode collapse detection

These metrics are:
- ✅ Logged to WandB automatically
- ✅ Printed after each evaluation
- ✅ Configurable (frequency, number of samples)
- ✅ Fast to compute (doesn't slow training)

**Use them to objectively track improvement and diagnose problems early!**



