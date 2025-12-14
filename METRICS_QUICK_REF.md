# Quick Guide: Evaluating Your CycleGAN Beyond Visual Inspection

## 🎯 5 New Quantitative Metrics Added

### **1. SSIM (Structural Similarity)** - Structure Quality
- **0.8-1.0**: Excellent ✅
- **0.6-0.8**: Good
- **<0.6**: Poor ⚠️

### **2. PSNR (Peak Signal-to-Noise Ratio)** - Pixel Quality
- **>35 dB**: Excellent ✅
- **30-35 dB**: Good
- **<25 dB**: Poor ⚠️

### **3. L1 Distance** - Reconstruction Error
- **<0.1**: Excellent ✅
- **0.1-0.3**: Good
- **>0.5**: Poor ⚠️

### **4. Color Distance** - Color Preservation
- **<0.5**: Excellent ✅
- **0.5-1.5**: Good
- **>3.0**: Poor (washed out!) ⚠️
- **This is why your images look pale!**

### **5. Diversity Score** - Mode Collapse Detection
- **>50**: Healthy ✅
- **20-50**: Moderate
- **<5**: Mode collapse! 🚨

---

## 📊 What You'll See During Training

```
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
```

---

## 🔧 How to Use

### **Default (evaluates every checkpoint_freq epochs):**
```bash
# Already enabled! No changes needed
```

### **More frequent evaluation:**
```bash
--eval_freq 2  # Evaluate every 2 epochs
```

### **More accurate (but slower):**
```bash
--eval_samples 100  # Use 100 samples instead of 50
```

---

## 🎯 Quick Diagnosis Table

| Symptom | Check Metric | Fix |
|---------|--------------|-----|
| Blurry images | SSIM < 0.6 | Train longer, increase `lambda_cycle` |
| Washed out colors | Color Distance > 3.0 | Increase `lambda_identity` to 3.0+ |
| Bad reconstruction | L1 > 0.5 | Increase `lambda_cycle` |
| All images look same | Diversity < 5 | Mode collapse! Lower LR |

---

## 📈 Your Current Issue (Epoch 20)

Based on your washed-out images, you likely have:
- ✅ SSIM: ~0.7 (decent structure)
- ⚠️ **Color Distance: ~2.5+ (too high!)**
- ✅ Diversity: ~60 (no mode collapse)

**Solution:** `--lambda_identity 3.0` (instead of 0.5)

---

## 💡 Pro Tip

Watch these in WandB:
- `eval/color_distance_A2B` - If increasing = colors getting worse
- `eval/diversity_A2B` - If decreasing = approaching mode collapse
- `eval/ssim_reconstruction_A` - If increasing = structure improving

---

## ✅ Installation Required

These metrics use scipy and scikit-image. Install with:
```bash
pip install scipy scikit-image
```

Already included in most Colab/Python environments!

---

See **EVALUATION_METRICS_GUIDE.md** for complete details!



