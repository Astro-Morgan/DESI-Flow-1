"""Spectrum preprocessing shared by every model in this repository.

All models consume the same representation of a spectrum: an
inverse-variance-weighted, Gaussian-smoothed ("whitened") flux vector that is
standardized to zero mean / unit variance over its valid pixels. The pipeline,
given raw ``flux``, ``ivar`` (inverse variance), and a pixel ``mask``:

    1. Treat ``mask == 0`` as "good pixel"; everything else is ignored.
    2. Smooth with a normalized Gaussian kernel (default: 11 px, sigma = 11/6),
       weighting each pixel by sqrt(ivar) so noisy pixels contribute less:
           smoothed = conv(flux * sqrt(ivar) * good) / conv(sqrt(ivar) * good)
    3. Standardize using the mean/std of the *valid* smoothed pixels only.
    4. Zero out invalid pixels so masked regions carry no signal.

Two mathematically equivalent implementations are provided:

    * ``preprocess_for_model`` -- NumPy, one spectrum at a time. Used inside
      Dataset ``__getitem__`` (e.g. Plato's triplet dataset, where the
      whitened flux is also the input to the CCF ground-truth similarity).
    * ``GPUPreprocessor``      -- batched PyTorch ``nn.Module``. Used when raw
      (flux, ivar, mask) batches are shipped to the GPU and preprocessed
      there (e.g. Aristotle distillation, where CPU-side preprocessing would
      bottleneck the large batch sizes).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import gaussian


def gaussian_kernel(kernel_size: int = 11) -> np.ndarray:
    """Normalized 1D Gaussian kernel with the repo-wide sigma heuristic k/6."""
    kernel = gaussian(kernel_size, std=kernel_size / 6.0)
    return kernel / np.sum(kernel)


def preprocess_for_model(flux: np.ndarray,
                         ivar: np.ndarray,
                         mask: np.ndarray,
                         kernel_size: int = 11) -> np.ndarray:
    """Turn one raw spectrum into model-ready input.

    Args:
        flux: Raw flux, shape (L,).
        ivar: Inverse variance per pixel, shape (L,).
        mask: Pixel mask, shape (L,). Convention: 0 = good pixel.
        kernel_size: Width of the Gaussian smoothing kernel in pixels.

    Returns:
        Whitened, standardized spectrum of shape (1, L), float32 -- a single
        channel ready to be stacked into a (B, 1, L) batch for the ConvNet
        encoders.
    """
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    good = (mask == 0)

    kernel = gaussian_kernel(kernel_size)

    # IVAR-weighted moving average: conv the numerator and denominator
    # separately so masked / noisy pixels are down-weighted rather than
    # dragging the local mean toward zero.
    numerator = np.convolve(flux * np.sqrt(ivar) * good, kernel, mode='same')
    denominator = np.convolve(np.sqrt(ivar) * good, kernel, mode='same')

    whitened = np.zeros_like(numerator)
    valid = denominator > 1e-9  # pixels with any valid support in the window
    whitened[valid] = numerator[valid] / denominator[valid]

    # Standardize on valid pixels only, then re-zero the invalid regions.
    valid_pixels = whitened[valid]
    if valid_pixels.size >= 2:
        whitened = (whitened - valid_pixels.mean()) / (valid_pixels.std() + 1e-8)
        whitened = whitened * valid

    return whitened[None, :].astype(np.float32)  # (1, L)


class GPUPreprocessor(nn.Module):
    """Batched, on-GPU equivalent of :func:`preprocess_for_model`.

    Input:  (B, 3, L) raw batch with channels [flux, ivar, valid_flag],
            where valid_flag is 1.0 for good pixels (i.e. ``mask == 0``
            already converted to a float flag by the Dataset).
    Output: (B, 1, L) whitened, standardized spectra.

    The smoothing is expressed as a single ``conv1d`` over the batch, which is
    why this version exists: preprocessing large distillation batches on the
    CPU inside DataLoader workers was the training bottleneck.
    """

    def __init__(self, kernel_size: int = 11, device: str = 'cuda'):
        super().__init__()
        kernel = torch.from_numpy(gaussian_kernel(kernel_size)).float().to(device)
        # conv1d expects (out_channels, in_channels, K)
        self.register_buffer('kernel', kernel.view(1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, raw_batch: torch.Tensor) -> torch.Tensor:
        flux = raw_batch[:, 0:1, :]
        ivar = raw_batch[:, 1:2, :]
        good = raw_batch[:, 2:3, :]

        # Epsilon inside the sqrt keeps gradients finite where ivar == 0.
        sqrt_ivar = torch.sqrt(ivar + 1e-9)
        smooth_num = F.conv1d(flux * sqrt_ivar * good, self.kernel, padding=self.pad)
        smooth_den = F.conv1d(sqrt_ivar * good, self.kernel, padding=self.pad)

        valid = smooth_den > 1e-8
        whitened = torch.zeros_like(smooth_num)
        whitened[valid] = smooth_num[valid] / smooth_den[valid]

        # Per-spectrum mean/std over valid pixels (masked sum / count).
        counts = valid.sum(dim=2, keepdim=True).clamp(min=1.0)
        means = whitened.sum(dim=2, keepdim=True) / counts
        centered = (whitened - means) * valid
        stds = torch.sqrt((centered ** 2).sum(dim=2, keepdim=True) / counts + 1e-8)

        return centered / (stds + 1e-8)
