"""
Enhanced Loss Functions for CycleGAN

Based on comprehensive analysis showing critical differences between Monet and photos:
- Luminance: Monet is brighter (peak ~65-70) vs photos (darker, peak ~0-40)
- Saturation: Monet has balanced mid-range (0.1-0.8) vs photos (bimodal: ~0 and ~1.0)
- RMS Contrast: Monet is softer (peak ~5-6) vs photos (peak ~6-7, more extremes)
- Edge Softness: Monet has very soft edges (Sobel ~0.34) vs photos (sharper)
- High-Frequency: Monet suppresses fine details (high-freq PSD ~1.75) vs photos (higher)
- Gradient Smoothness: Monet has smooth transitions (gradient ~87) vs photos (sharper)
- Hard Edges: Monet has fewer hard edges (ratio ~19.7%) vs photos (more)

These losses explicitly guide the generator to learn these critical transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
import numpy as np


def rgb_to_lab_tensor(rgb_tensor):
    """
    Convert RGB tensor to Lab color space
    
    Args:
        rgb_tensor: Tensor of shape (B, 3, H, W) with values in [-1, 1]
        
    Returns:
        lab_tensor: Tensor of shape (B, 3, H, W) with L in [0, 100], a,b in [-128, 127]
    """
    # Convert from [-1, 1] to [0, 1]
    rgb_normalized = (rgb_tensor + 1.0) / 2.0
    
    # Convert to numpy for skimage
    batch_size = rgb_tensor.shape[0]
    lab_batch = []
    
    for i in range(batch_size):
        rgb_np = rgb_normalized[i].permute(1, 2, 0).cpu().detach().numpy()
        lab_np = color.rgb2lab(rgb_np)
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()
        lab_batch.append(lab_tensor)
    
    lab_batch = torch.stack(lab_batch)
    
    # Move to same device as input
    if rgb_tensor.is_cuda:
        lab_batch = lab_batch.cuda()
    
    return lab_batch


def rgb_to_hsv_tensor(rgb_tensor):
    """
    Convert RGB tensor to HSV color space
    
    Args:
        rgb_tensor: Tensor of shape (B, 3, H, W) with values in [-1, 1]
        
    Returns:
        hsv_tensor: Tensor of shape (B, 3, H, W) with H,S,V in [0, 1]
    """
    # Convert from [-1, 1] to [0, 1]
    rgb_normalized = (rgb_tensor + 1.0) / 2.0
    
    # Convert to numpy for skimage
    batch_size = rgb_tensor.shape[0]
    hsv_batch = []
    
    for i in range(batch_size):
        rgb_np = rgb_normalized[i].permute(1, 2, 0).cpu().detach().numpy()
        hsv_np = color.rgb2hsv(rgb_np)
        hsv_tensor = torch.from_numpy(hsv_np).permute(2, 0, 1).float()
        hsv_batch.append(hsv_tensor)
    
    hsv_batch = torch.stack(hsv_batch)
    
    # Move to same device as input
    if rgb_tensor.is_cuda:
        hsv_batch = hsv_batch.cuda()
    
    return hsv_batch


class LuminanceLoss(nn.Module):
    """
    Luminance loss in Lab color space
    
    Based on your analysis: Monet has higher luminance (peak ~65-70) 
    vs photos (darker, peak ~0-40). This loss explicitly guides the 
    generator to learn the correct brightness transformation.
    
    Usage:
        luminance_loss = LuminanceLoss(weight=1.0)
        loss = luminance_loss(generated_monet, target_monet_statistics)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1' or 'l2' or 'statistical' (compare mean/std)
        """
        super(LuminanceLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
        
    def forward(self, generated, target=None, target_mean=None, target_std=None):
        """
        Compute luminance loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean luminance (from your analysis: Monet ~65-70, Photos ~30-40)
            target_std: Target std luminance (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to Lab
        lab_gen = rgb_to_lab_tensor(generated)
        L_gen = lab_gen[:, 0:1, :, :]  # Extract L channel
        
        if target is not None:
            # Direct comparison with target image
            lab_target = rgb_to_lab_tensor(target)
            L_target = lab_target[:, 0:1, :, :]
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(L_gen, L_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(L_gen, L_target)
            else:
                # Statistical comparison
                gen_mean = L_gen.mean()
                gen_std = L_gen.std()
                target_mean = L_target.mean()
                target_std = L_target.std()
                loss = F.l1_loss(gen_mean, target_mean) + F.l1_loss(gen_std, target_std)
        
        elif target_mean is not None:
            # Compare to target statistics (from your analysis)
            gen_mean = L_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            else:
                # Statistical comparison
                gen_std = L_gen.std()
                mean_loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
                if target_std is not None:
                    std_loss = F.l1_loss(gen_std, torch.tensor(target_std).to(generated.device))
                    loss = mean_loss + std_loss
                else:
                    loss = mean_loss
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class SaturationLoss(nn.Module):
    """
    Saturation loss in HSV color space
    
    Based on your analysis: Monet has balanced mid-range saturation (0.1-0.8)
    vs photos (bimodal: very low ~0 and very high ~1.0). This loss guides
    the generator to learn the correct saturation distribution.
    
    Usage:
        saturation_loss = SaturationLoss(weight=1.0)
        loss = saturation_loss(generated_monet, target_monet_statistics)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1', 'l2', or 'distribution' (compare histogram)
        """
        super(SaturationLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
        
    def forward(self, generated, target=None, target_mean=None, target_std=None):
        """
        Compute saturation loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean saturation (from your analysis)
            target_std: Target std saturation (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to HSV
        hsv_gen = rgb_to_hsv_tensor(generated)
        S_gen = hsv_gen[:, 1:2, :, :]  # Extract S channel
        
        if target is not None:
            # Direct comparison with target image
            hsv_target = rgb_to_hsv_tensor(target)
            S_target = hsv_target[:, 1:2, :, :]
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(S_gen, S_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(S_gen, S_target)
            else:
                # Distribution comparison (histogram)
                # Compute histograms
                bins = 50
                gen_hist = torch.histc(S_gen.flatten(), bins=bins, min=0.0, max=1.0)
                target_hist = torch.histc(S_target.flatten(), bins=bins, min=0.0, max=1.0)
                # Normalize
                gen_hist = gen_hist / (gen_hist.sum() + 1e-8)
                target_hist = target_hist / (target_hist.sum() + 1e-8)
                # Compare
                loss = F.l1_loss(gen_hist, target_hist)
        
        elif target_mean is not None:
            # Compare to target statistics
            gen_mean = S_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            else:
                # Statistical comparison
                gen_std = S_gen.std()
                mean_loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
                if target_std is not None:
                    std_loss = F.l1_loss(gen_std, torch.tensor(target_std).to(generated.device))
                    loss = mean_loss + std_loss
                else:
                    loss = mean_loss
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class RMSContrastLoss(nn.Module):
    """
    RMS (Root Mean Square) Contrast Loss
    
    Based on your analysis: Monet has lower RMS contrast (peak ~5-6, softer edges)
    vs photos (peak ~6-7, sharper edges, more extremes up to 14).
    
    RMS contrast measures local variation in luminance - how sharp/smooth the image is.
    This is a CRITICAL difference for your GAN to learn.
    
    Usage:
        rms_loss = RMSContrastLoss(weight=1.0)
        loss = rms_loss(generated_monet, target_mean=6.0)
    """
    
    def __init__(self, weight=1.0, window_size=5, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            window_size: Size of local window for contrast computation (default: 5)
            loss_type: 'l1' or 'l2' or 'statistical' (compare mean/std)
        """
        super(RMSContrastLoss, self).__init__()
        self.weight = weight
        self.window_size = window_size
        self.loss_type = loss_type
        
        # Create averaging kernel for local mean computation
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        self.register_buffer('kernel', kernel)
    
    def compute_rms_contrast(self, lab_image):
        """
        Compute RMS contrast for a Lab image
        
        Formula: sqrt(mean((L - mean(L))^2))
        
        Args:
            lab_image: Lab color space tensor (B, 3, H, W)
            
        Returns:
            rms_contrast: Scalar RMS contrast value per image (B,)
        """
        L = lab_image[:, 0:1, :, :]  # Extract L channel
        
        # Compute local mean using convolution
        # Pad to maintain spatial dimensions
        padding = self.window_size // 2
        L_padded = F.pad(L, (padding, padding, padding, padding), mode='reflect')
        local_mean = F.conv2d(L_padded, self.kernel, padding=0)
        
        # Compute RMS contrast: sqrt(mean((L - mean(L))^2))
        squared_diff = (L - local_mean) ** 2
        rms_contrast = torch.sqrt(torch.mean(squared_diff.view(L.shape[0], -1), dim=1))
        
        return rms_contrast
    
    def forward(self, generated, target=None, target_mean=None, target_std=None):
        """
        Compute RMS contrast loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean RMS contrast (Monet ~6.0, Photos ~6.5)
            target_std: Target std RMS contrast (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to Lab
        lab_gen = rgb_to_lab_tensor(generated)
        rms_gen = self.compute_rms_contrast(lab_gen)
        
        if target is not None:
            # Direct comparison with target image
            lab_target = rgb_to_lab_tensor(target)
            rms_target = self.compute_rms_contrast(lab_target)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(rms_gen, rms_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(rms_gen, rms_target)
            else:
                # Statistical comparison
                gen_mean = rms_gen.mean()
                gen_std = rms_gen.std()
                target_mean = rms_target.mean()
                target_std = rms_target.std()
                loss = F.l1_loss(gen_mean, target_mean) + F.l1_loss(gen_std, target_std)
        
        elif target_mean is not None:
            # Compare to target statistics (from your analysis)
            gen_mean = rms_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            else:
                # Statistical comparison
                gen_std = rms_gen.std()
                mean_loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
                if target_std is not None:
                    std_loss = F.l1_loss(gen_std, torch.tensor(target_std).to(generated.device))
                    loss = mean_loss + std_loss
                else:
                    loss = mean_loss
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class EdgeSoftnessLoss(nn.Module):
    """
    Edge Softness Loss using Sobel Edge Detection
    
    Based on your analysis: Monet has very soft edges (Sobel edge strength ~0.34)
    vs photos (sharper edges, higher strength). This is CRITICAL for achieving
    the Monet soft-focus effect.
    
    Usage:
        edge_loss = EdgeSoftnessLoss(weight=0.5)
        loss = edge_loss(generated_monet, target_mean=0.34)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1' or 'l2' or 'statistical' (compare mean/std)
        """
        super(EdgeSoftnessLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_edge_strength(self, gray_image):
        """
        Compute Sobel edge strength
        
        Args:
            gray_image: Grayscale image tensor (B, 1, H, W) in [0, 1]
            
        Returns:
            edge_strength: Mean edge strength per image (B,)
        """
        # Apply Sobel filters
        gx = F.conv2d(gray_image, self.sobel_x, padding=1)
        gy = F.conv2d(gray_image, self.sobel_y, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        # Return mean edge strength per image
        return magnitude.view(magnitude.shape[0], -1).mean(dim=1)
    
    def forward(self, generated, target=None, target_mean=None, target_std=None):
        """
        Compute edge softness loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean edge strength (Monet ~0.34, Photos higher)
            target_std: Target std edge strength (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to grayscale
        gray_gen = 0.299 * generated[:, 0:1] + 0.587 * generated[:, 1:2] + 0.114 * generated[:, 2:3]
        gray_gen = (gray_gen + 1.0) / 2.0  # Normalize to [0, 1]
        
        edge_gen = self.compute_edge_strength(gray_gen)
        
        if target is not None:
            gray_target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
            gray_target = (gray_target + 1.0) / 2.0
            edge_target = self.compute_edge_strength(gray_target)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(edge_gen, edge_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(edge_gen, edge_target)
            else:
                gen_mean = edge_gen.mean()
                gen_std = edge_gen.std()
                target_mean = edge_target.mean()
                target_std = edge_target.std()
                loss = F.l1_loss(gen_mean, target_mean) + F.l1_loss(gen_std, target_std)
        
        elif target_mean is not None:
            gen_mean = edge_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            else:
                gen_std = edge_gen.std()
                mean_loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
                if target_std is not None:
                    std_loss = F.l1_loss(gen_std, torch.tensor(target_std).to(generated.device))
                    loss = mean_loss + std_loss
                else:
                    loss = mean_loss
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class HighFreqSuppressionLoss(nn.Module):
    """
    High-Frequency Suppression Loss using Power Spectral Density (PSD)
    
    Based on your analysis: Monet suppresses fine details (high-freq PSD ~1.75)
    vs photos (higher high-freq PSD). This creates the soft-focus effect and
    encourages broad strokes over fine details.
    
    Usage:
        freq_loss = HighFreqSuppressionLoss(weight=0.3)
        loss = freq_loss(generated_monet, target_mean=1.75)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1' or 'l2' or 'statistical' (compare mean/std)
        """
        super(HighFreqSuppressionLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
    
    def compute_high_freq_psd(self, gray_image):
        """
        Compute high-frequency PSD using FFT
        
        Args:
            gray_image: Grayscale image tensor (B, 1, H, W) in [0, 1]
            
        Returns:
            high_freq_psd: Mean high-frequency PSD per image (B,)
        """
        batch_size = gray_image.shape[0]
        psd_values = []
        
        for i in range(batch_size):
            img = gray_image[i, 0].cpu().numpy()
            
            # Compute 2D FFT
            fft = np.fft.fft2(img)
            fft_shifted = np.fft.fftshift(fft)
            
            # Power spectral density
            psd = np.abs(fft_shifted)**2
            psd_log = np.log10(psd + 1e-10)
            
            # Extract high-frequency region (outer region, away from center)
            h, w = psd_log.shape
            center_h, center_w = h // 2, w // 2
            
            # High frequency: outer region (not in center 1/8)
            high_mask = np.ones((h, w), dtype=bool)
            high_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = False
            high_freq = psd_log[high_mask].mean()
            
            psd_values.append(high_freq)
        
        return torch.tensor(psd_values, dtype=torch.float32).to(gray_image.device)
    
    def forward(self, generated, target=None, target_mean=None, target_std=None):
        """
        Compute high-frequency suppression loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean high-freq PSD (Monet ~1.75, Photos higher)
            target_std: Target std high-freq PSD (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to grayscale
        gray_gen = 0.299 * generated[:, 0:1] + 0.587 * generated[:, 1:2] + 0.114 * generated[:, 2:3]
        gray_gen = (gray_gen + 1.0) / 2.0  # Normalize to [0, 1]
        
        psd_gen = self.compute_high_freq_psd(gray_gen)
        
        if target is not None:
            gray_target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
            gray_target = (gray_target + 1.0) / 2.0
            psd_target = self.compute_high_freq_psd(gray_target)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(psd_gen, psd_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(psd_gen, psd_target)
            else:
                gen_mean = psd_gen.mean()
                gen_std = psd_gen.std()
                target_mean = psd_target.mean()
                target_std = psd_target.std()
                loss = F.l1_loss(gen_mean, target_mean) + F.l1_loss(gen_std, target_std)
        
        elif target_mean is not None:
            gen_mean = psd_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
            else:
                gen_std = psd_gen.std()
                mean_loss = F.l1_loss(gen_mean, torch.tensor(target_mean).to(generated.device))
                if target_std is not None:
                    std_loss = F.l1_loss(gen_std, torch.tensor(target_std).to(generated.device))
                    loss = mean_loss + std_loss
                else:
                    loss = mean_loss
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class GradientSmoothnessLoss(nn.Module):
    """
    Gradient Smoothness Loss
    
    Based on your analysis: Monet has smooth brightness transitions (gradient magnitude
    mean ~87, median ~63) vs photos (higher gradients, sharper transitions). This
    encourages smooth, blended color transitions.
    
    Usage:
        grad_loss = GradientSmoothnessLoss(weight=0.2)
        loss = grad_loss(generated_monet, target_mean=87.0, target_median=63.0)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1' or 'l2' or 'statistical' (compare mean/median)
        """
        super(GradientSmoothnessLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_gradient_magnitude(self, gray_image):
        """
        Compute gradient magnitude statistics
        
        Args:
            gray_image: Grayscale image tensor (B, 1, H, W) in [0, 1]
            
        Returns:
            mean_grad: Mean gradient magnitude per image (B,)
            median_grad: Median gradient magnitude per image (B,)
        """
        # Apply Sobel filters
        gx = F.conv2d(gray_image, self.sobel_x, padding=1)
        gy = F.conv2d(gray_image, self.sobel_y, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        # Compute statistics per image
        magnitude_flat = magnitude.view(magnitude.shape[0], -1)
        mean_grad = magnitude_flat.mean(dim=1)
        median_grad = magnitude_flat.median(dim=1)[0]
        
        return mean_grad, median_grad
    
    def forward(self, generated, target=None, target_mean=None, target_median=None):
        """
        Compute gradient smoothness loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_mean: Target mean gradient (Monet ~87, Photos higher)
            target_median: Target median gradient (Monet ~63, Photos higher)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to grayscale
        gray_gen = 0.299 * generated[:, 0:1] + 0.587 * generated[:, 1:2] + 0.114 * generated[:, 2:3]
        gray_gen = (gray_gen + 1.0) / 2.0  # Normalize to [0, 1]
        
        mean_grad_gen, median_grad_gen = self.compute_gradient_magnitude(gray_gen)
        
        if target is not None:
            gray_target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
            gray_target = (gray_target + 1.0) / 2.0
            mean_grad_target, median_grad_target = self.compute_gradient_magnitude(gray_target)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(mean_grad_gen, mean_grad_target) + F.l1_loss(median_grad_gen, median_grad_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(mean_grad_gen, mean_grad_target) + F.mse_loss(median_grad_gen, median_grad_target)
            else:
                loss = F.l1_loss(mean_grad_gen.mean(), mean_grad_target.mean()) + \
                       F.l1_loss(median_grad_gen.mean(), median_grad_target.mean())
        
        elif target_mean is not None:
            if self.loss_type == 'l1':
                mean_loss = F.l1_loss(mean_grad_gen.mean(), torch.tensor(target_mean).to(generated.device))
                if target_median is not None:
                    median_loss = F.l1_loss(median_grad_gen.mean(), torch.tensor(target_median).to(generated.device))
                    loss = mean_loss + median_loss
                else:
                    loss = mean_loss
            elif self.loss_type == 'l2':
                mean_loss = F.mse_loss(mean_grad_gen.mean(), torch.tensor(target_mean).to(generated.device))
                if target_median is not None:
                    median_loss = F.mse_loss(median_grad_gen.mean(), torch.tensor(target_median).to(generated.device))
                    loss = mean_loss + median_loss
                else:
                    loss = mean_loss
            else:
                loss = F.l1_loss(mean_grad_gen.mean(), torch.tensor(target_mean).to(generated.device))
                if target_median is not None:
                    loss += F.l1_loss(median_grad_gen.mean(), torch.tensor(target_median).to(generated.device))
        
        else:
            raise ValueError("Either 'target' or 'target_mean' must be provided")
        
        return self.weight * loss


class HardEdgeReductionLoss(nn.Module):
    """
    Hard Edge Reduction Loss using Canny Edge Detection
    
    Based on your analysis: Monet has fewer hard edges (ratio ~19.7%) vs photos
    (higher ratio). This encourages soft, blended boundaries rather than sharp edges.
    
    Usage:
        hard_edge_loss = HardEdgeReductionLoss(weight=0.2)
        loss = hard_edge_loss(generated_monet, target_ratio=0.197)
    """
    
    def __init__(self, weight=1.0, loss_type='l1'):
        """
        Args:
            weight: Weight for this loss component
            loss_type: 'l1' or 'l2' or 'statistical' (compare ratio)
        """
        super(HardEdgeReductionLoss, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
    
    def compute_hard_edge_ratio(self, gray_image):
        """
        Compute hard edge ratio using simplified Canny-like detection
        
        Uses gradient magnitude thresholding as approximation to Canny edges.
        
        Args:
            gray_image: Grayscale image tensor (B, 1, H, W) in [0, 1]
            
        Returns:
            edge_ratio: Ratio of hard edges per image (B,)
        """
        # Compute gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=gray_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=gray_image.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray_image, sobel_x, padding=1)
        gy = F.conv2d(gray_image, sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        # Threshold for hard edges (adaptive: use percentile)
        batch_size = magnitude.shape[0]
        edge_ratios = []
        
        for i in range(batch_size):
            mag = magnitude[i, 0]
            # Use 90th percentile as threshold (top 10% are hard edges)
            threshold = torch.quantile(mag, 0.9)
            hard_edges = (mag > threshold).float()
            edge_ratio = hard_edges.mean()
            edge_ratios.append(edge_ratio)
        
        return torch.stack(edge_ratios)
    
    def forward(self, generated, target=None, target_ratio=None):
        """
        Compute hard edge reduction loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional, for direct comparison)
            target_ratio: Target hard edge ratio (Monet ~0.197, Photos higher)
            
        Returns:
            loss: Scalar loss value
        """
        # Convert to grayscale
        gray_gen = 0.299 * generated[:, 0:1] + 0.587 * generated[:, 1:2] + 0.114 * generated[:, 2:3]
        gray_gen = (gray_gen + 1.0) / 2.0  # Normalize to [0, 1]
        
        ratio_gen = self.compute_hard_edge_ratio(gray_gen)
        
        if target is not None:
            gray_target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
            gray_target = (gray_target + 1.0) / 2.0
            ratio_target = self.compute_hard_edge_ratio(gray_target)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(ratio_gen, ratio_target)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(ratio_gen, ratio_target)
            else:
                loss = F.l1_loss(ratio_gen.mean(), ratio_target.mean())
        
        elif target_ratio is not None:
            gen_mean = ratio_gen.mean()
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(gen_mean, torch.tensor(target_ratio).to(generated.device))
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gen_mean, torch.tensor(target_ratio).to(generated.device))
            else:
                loss = F.l1_loss(gen_mean, torch.tensor(target_ratio).to(generated.device))
        
        else:
            raise ValueError("Either 'target' or 'target_ratio' must be provided")
        
        return self.weight * loss


class CombinedLuminanceSaturationLoss(nn.Module):
    """
    Combined loss that uses both luminance and saturation
    
    This is the recommended loss for your CycleGAN training, as it
    explicitly guides the generator to learn both critical transformations.
    """
    
    def __init__(self, luminance_weight=1.0, saturation_weight=1.0, 
                 luminance_loss_type='l1', saturation_loss_type='l1'):
        """
        Args:
            luminance_weight: Weight for luminance loss
            saturation_weight: Weight for saturation loss
            luminance_loss_type: 'l1', 'l2', or 'statistical'
            saturation_loss_type: 'l1', 'l2', or 'distribution'
        """
        super(CombinedLuminanceSaturationLoss, self).__init__()
        self.luminance_loss = LuminanceLoss(weight=luminance_weight, loss_type=luminance_loss_type)
        self.saturation_loss = SaturationLoss(weight=saturation_weight, loss_type=saturation_loss_type)
        
    def forward(self, generated, target=None, 
                target_lum_mean=None, target_lum_std=None,
                target_sat_mean=None, target_sat_std=None):
        """
        Compute combined luminance and saturation loss
        
        Args:
            generated: Generated image tensor (B, 3, H, W) in [-1, 1]
            target: Target image tensor (optional)
            target_lum_mean: Target mean luminance (Monet ~65-70, Photos ~30-40)
            target_lum_std: Target std luminance (optional)
            target_sat_mean: Target mean saturation (from your analysis)
            target_sat_std: Target std saturation (optional)
            
        Returns:
            total_loss: Combined loss
            lum_loss: Luminance loss component
            sat_loss: Saturation loss component
        """
        if target is not None:
            lum_loss = self.luminance_loss(generated, target=target)
            sat_loss = self.saturation_loss(generated, target=target)
        else:
            lum_loss = self.luminance_loss(generated, target_mean=target_lum_mean, target_std=target_lum_std)
            sat_loss = self.saturation_loss(generated, target_mean=target_sat_mean, target_std=target_sat_std)
        
        total_loss = lum_loss + sat_loss
        
        return total_loss, lum_loss, sat_loss


# Target statistics from your analysis
# These values should be updated based on your actual analysis results

MONET_TARGET_STATS = {
    'luminance_mean': 65.0,  # Monet peak around 65-70
    'luminance_std': 15.0,   # Approximate
    'saturation_mean': 0.3,  # Monet balanced mid-range (0.1-0.8)
    'saturation_std': 0.2,   # Approximate
    'rms_contrast_mean': 6.0,  # Monet peak around 5-6 (softer edges)
    'rms_contrast_std': 1.5,   # Approximate (more consistent)
    'edge_strength_mean': 0.34,  # Monet has very soft edges (Sobel)
    'edge_strength_std': 0.12,   # Approximate
    'high_freq_psd_mean': 1.75,  # Monet suppresses fine details
    'high_freq_psd_std': 0.2,    # Approximate
    'gradient_mean': 87.0,  # Monet has smooth transitions
    'gradient_median': 63.0,  # Monet median gradient
    'hard_edge_ratio': 0.197,  # Monet has ~19.7% hard edges
}

PHOTO_TARGET_STATS = {
    'luminance_mean': 35.0,  # Photos darker, peak around 0-40
    'luminance_std': 20.0,   # Approximate (wider distribution)
    'saturation_mean': 0.4,  # Photos bimodal (low ~0 and high ~1.0)
    'saturation_std': 0.35,  # Approximate (bimodal = higher std)
    'rms_contrast_mean': 6.5,  # Photos peak around 6-7 (sharper edges)
    'rms_contrast_std': 2.0,   # Approximate (more variety, more extremes)
    'edge_strength_mean': 0.5,  # Photos have sharper edges (higher than Monet)
    'edge_strength_std': 0.15,  # Approximate
    'high_freq_psd_mean': 2.0,  # Photos have more fine details (higher than Monet)
    'high_freq_psd_std': 0.3,    # Approximate
    'gradient_mean': 120.0,  # Photos have sharper transitions
    'gradient_median': 90.0,  # Photos median gradient
    'hard_edge_ratio': 0.25,  # Photos have more hard edges (~25%)
}


if __name__ == "__main__":
    print("Enhanced Loss Functions for CycleGAN")
    print("=" * 60)
    print("\nThese losses are designed based on comprehensive analysis showing:")
    print("  - Monet: Brighter (L ~65-70), balanced saturation (0.1-0.8)")
    print("  - Photos: Darker (L ~0-40), bimodal saturation (~0 and ~1.0)")
    print("  - Monet: Softer edges, suppressed fine details, smooth transitions")
    print("  - Photos: Sharper edges, more fine details, sharper transitions")
    print("\nAvailable Loss Functions:")
    print("  - LuminanceLoss: Brightness matching")
    print("  - SaturationLoss: Color saturation matching")
    print("  - RMSContrastLoss: Local contrast (softness)")
    print("  - EdgeSoftnessLoss: Edge softness (Sobel)")
    print("  - HighFreqSuppressionLoss: Fine detail suppression (PSD)")
    print("  - GradientSmoothnessLoss: Smooth transitions")
    print("  - HardEdgeReductionLoss: Reduce sharp edges (Canny)")
    print("\nUsage:")
    print("  from enhanced_losses import LuminanceLoss, EdgeSoftnessLoss, ...")
    print("  from enhanced_losses import MONET_TARGET_STATS, PHOTO_TARGET_STATS")

