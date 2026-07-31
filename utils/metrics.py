"""评估指标"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def _as_2d(img: np.ndarray) -> np.ndarray:
    """将输入图像统一为 2D (H, W) 供 SSIM/PSNR 使用。

    约定：
    - (H, W) 直接返回
    - (H, W, 1) -> squeeze
    - (H, W, C>=2) -> 取中间通道（常见 2.5D 三通道输入的中心切片）
    """
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        mid = img.shape[2] // 2
        return img[:, :, mid]
    raise ValueError(f"Unsupported image shape for metrics: {img.shape}")


def get_ssim(img1, img2):
    """计算SSIM（默认仅评估中间通道）"""
    img1_2d = _as_2d(img1)
    img2_2d = _as_2d(img2)
    data_range = float(np.max([img1_2d.max(), img2_2d.max()]) - np.min([img1_2d.min(), img2_2d.min()]))
    if data_range == 0:
        data_range = 1.0
    return ssim(img1_2d, img2_2d, channel_axis=None, data_range=data_range)


def get_psnr(img1, img2):
    """计算PSNR（默认仅评估中间通道）"""
    img1_2d = _as_2d(img1)
    img2_2d = _as_2d(img2)
    data_range = float(np.max([img1_2d.max(), img2_2d.max()]) - np.min([img1_2d.min(), img2_2d.min()]))
    if data_range == 0:
        data_range = 1.0
    return psnr(img1_2d, img2_2d, data_range=data_range)


def dice_coefficient(mask1, mask2):
    """计算Dice系数"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    dice = 2 * np.sum(intersection) / (mask1_sum + mask2_sum)
    return dice
