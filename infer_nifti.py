#!/usr/bin/env python
"""RescueCTA inference for 3D NIfTI volumes using 2.5D slice stacks."""

import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm

import nibabel as nib

from models import build_generator
from utils import load_config, make_dirs


def select_generator_state(checkpoint, weights="ema"):
    """Select a generator state dict from a RescueCTA checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)}")

    if "G_state_dict" not in checkpoint:
        return checkpoint

    if weights == "ema":
        state_dict = checkpoint.get("G_ema_state_dict")
        if state_dict is None:
            raise KeyError(
                "EMA weights requested, but checkpoint has no "
                "'G_ema_state_dict'. Use --weights raw only when intended."
            )
        return state_dict
    if weights == "raw":
        return checkpoint["G_state_dict"]
    raise ValueError("weights must be either 'ema' or 'raw'")


def normalize_slice(image):
    """单切片归一化: HU -> [-1, 1]，与训练时保持一致

    对应 data/dataset.py 中的 _normalize 方法
    """
    image = image.copy().astype(np.float32)
    # 归一化到 [-1, 1]
    image = image + 1024
    image[image < 0] = 0
    image = image / 4095
    image = (image - 0.5) / 0.5
    return image.astype(np.float32)


def denormalize_slice(image):
    """反归一化: [-1, 1] -> HU值"""
    # 从 [-1, 1] 转换回 HU 值
    image = (image + 1) * 0.5  # -> [0, 1]
    image = image * 4095  # -> [0, 4095]
    image = image - 1024  # -> [-1024, 3071]
    return image


def resize_slice(image, target_size=(512, 512)):
    """调整切片大小"""
    if image.shape[:2] == target_size:
        return image
    return cv2.resize(image.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)


def restore_slice_size(image, original_shape):
    """恢复切片到原始大小"""
    if image.shape[:2] == original_shape:
        return image
    # cv2.resize 使用 (W, H)
    return cv2.resize(image.astype(np.float32), (original_shape[1], original_shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def find_brain_bbox(image, background_val=-1024, margin_ratio=0.05, margin_min=4):
    """基于背景值找到脑部区域的最小外接正方形，并外扩一定边界

    与 data/dataset.py 中的 _find_brain_bbox 保持一致
    """
    if image.ndim == 3:
        mask_source = image.mean(axis=2)
    else:
        mask_source = image

    mask = mask_source > (background_val + 1e-3)
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    height = y_max - y_min + 1
    width = x_max - x_min + 1
    side = max(height, width)

    # 额外外扩 margin
    margin = max(int(side * margin_ratio), margin_min)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, mask_source.shape[0] - 1)
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, mask_source.shape[1] - 1)

    # 重新计算为外扩后的正方形
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    side = max(height, width)

    # 扩展成正方形 bbox，保持中心
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    half = side // 2
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = y0 + side
    x1 = x0 + side

    # 防止越界
    if y1 > mask_source.shape[0]:
        overflow = y1 - mask_source.shape[0]
        y0 = max(y0 - overflow, 0)
        y1 = mask_source.shape[0]
    if x1 > mask_source.shape[1]:
        overflow = x1 - mask_source.shape[1]
        x0 = max(x0 - overflow, 0)
        x1 = mask_source.shape[1]

    return y0, y1, x0, x1


def compute_volume_bbox(data):
    """对3D volume逐层计算bbox，取并集作为全局bbox

    Args:
        data: (H, W, D) 3D数组，HU值

    Returns:
        (y0, y1, x0, x1) 或 None
    """
    H, W, D = data.shape
    global_y0, global_x0 = H, W
    global_y1, global_x1 = 0, 0
    found_any = False

    for i in range(D):
        bbox = find_brain_bbox(data[:, :, i])
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            global_y0 = min(global_y0, y0)
            global_y1 = max(global_y1, y1)
            global_x0 = min(global_x0, x0)
            global_x1 = max(global_x1, x1)
            found_any = True

    if not found_any:
        return None

    # 将并集结果重新扩展为正方形
    height = global_y1 - global_y0
    width = global_x1 - global_x0
    side = max(height, width)

    cy = (global_y0 + global_y1) // 2
    cx = (global_x0 + global_x1) // 2
    half = side // 2
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = y0 + side
    x1 = x0 + side

    if y1 > H:
        overflow = y1 - H
        y0 = max(y0 - overflow, 0)
        y1 = H
    if x1 > W:
        overflow = x1 - W
        x0 = max(x0 - overflow, 0)
        x1 = W

    return y0, y1, x0, x1


def inference_single_volume(
    nifti_path,
    generator,
    device,
    target_size=(512, 512),
    channel_strategy="mean",
):
    """推理单个3D NIfTI图像

    流程（与训练 data/dataset.py 的预处理对齐）：
    1. 加载3D图像，计算全局脑区 bbox
    2. 对每层裁剪 bbox 区域 → resize 512 → normalize
    3. 构建2.5D输入（3相邻层），边界使用复制
    4. 推理输出3通道，反归一化
    5. resize 回 bbox 裁剪尺寸，贴回原始全图
    6. 3通道平均融合作为最终输出

    Args:
        nifti_path: NIfTI文件路径
        generator: 生成器模型
        device: 计算设备
        target_size: 推理时的目标大小

    Returns:
        output_volume: 融合后的3D输出 (H, W, D)，HU值
        affine: 原始affine矩阵
        header: 修正后的header（与原始数据类型一致）
    """
    # 加载NIfTI文件
    nii = nib.load(nifti_path)
    # get_fdata() 会自动应用 scl_slope/scl_inter，返回的已经是真实 HU 值
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header.copy()

    # 记录原始数据类型，用于保存时对齐
    original_dtype = header.get_data_dtype()

    # 重置 scl_slope/scl_inter，因为反归一化后的数据已经是 HU 值，
    # 保存时不需要再做 slope/intercept 变换，否则再次 get_fdata() 会二次变换
    header.set_slope_inter(slope=1.0, inter=0.0)

    H, W, num_slices = data.shape

    # Step 0: 计算全局脑区 bbox（与训练时 _find_brain_bbox 逻辑一致）
    bbox = compute_volume_bbox(data)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        crop_side = max(y1 - y0, x1 - x0)
        scale_factor = target_size[0] / crop_side
        if scale_factor > 5:
            # 与训练一致：缩放因子过大时不裁剪
            bbox = None

    # Step 1: 预处理所有切片 (bbox crop + resize + normalize)
    preprocessed_slices = []
    for i in range(num_slices):
        slice_data = data[:, :, i].astype(np.float32)
        # 裁剪脑区
        if bbox is not None:
            slice_data = slice_data[y0:y1, x0:x1]
        # resize to target_size
        slice_resized = resize_slice(slice_data, target_size)
        # normalize
        slice_normalized = normalize_slice(slice_resized)
        preprocessed_slices.append(slice_normalized)

    crop_shape = (y1 - y0, x1 - x0) if bbox is not None else (H, W)

    # Step 2 & 3: 构建2.5D输入并推理
    output_ch0 = []
    output_ch1 = []
    output_ch2 = []

    for i in range(num_slices):
        # 构建3通道输入：[slice_{i-1}, slice_i, slice_{i+1}]
        idx_prev = max(0, i - 1)
        idx_curr = i
        idx_next = min(num_slices - 1, i + 1)

        input_3ch = np.stack([
            preprocessed_slices[idx_prev],
            preprocessed_slices[idx_curr],
            preprocessed_slices[idx_next]
        ], axis=0)  # (3, H, W)

        input_tensor = torch.from_numpy(input_3ch).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(input_tensor)  # (1, 3, H, W)

        output_np = output_tensor[0].cpu().numpy()  # (3, H, W)

        # 反归一化
        ch0 = denormalize_slice(output_np[0])
        ch1 = denormalize_slice(output_np[1])
        ch2 = denormalize_slice(output_np[2])

        # resize 回 bbox 裁剪尺寸
        ch0 = restore_slice_size(ch0, crop_shape)
        ch1 = restore_slice_size(ch1, crop_shape)
        ch2 = restore_slice_size(ch2, crop_shape)

        # 贴回原始全图，bbox 外区域填充 -1024（空气 HU）
        if bbox is not None:
            full0 = np.full((H, W), -1024.0, dtype=np.float32)
            full1 = np.full((H, W), -1024.0, dtype=np.float32)
            full2 = np.full((H, W), -1024.0, dtype=np.float32)
            full0[y0:y1, x0:x1] = ch0
            full1[y0:y1, x0:x1] = ch1
            full2[y0:y1, x0:x1] = ch2
            output_ch0.append(full0)
            output_ch1.append(full1)
            output_ch2.append(full2)
        else:
            output_ch0.append(ch0)
            output_ch1.append(ch1)
            output_ch2.append(ch2)

    # Step 4: 堆叠为3个3D图像
    volume_ch0 = np.stack(output_ch0, axis=2)  # (H, W, D)
    volume_ch1 = np.stack(output_ch1, axis=2)
    volume_ch2 = np.stack(output_ch2, axis=2)

    # Step 5: combine the three output channels.
    if channel_strategy == "mean":
        # Retains the behavior of the historical NIfTI inference script.
        output_volume = (volume_ch0 + volume_ch1 + volume_ch2) / 3.0
    elif channel_strategy == "center":
        output_volume = volume_ch1
    else:
        raise ValueError("channel_strategy must be either 'mean' or 'center'")

    # Step 6: 对齐到原始数据类型
    if np.issubdtype(original_dtype, np.integer):
        dtype_info = np.iinfo(original_dtype)
        output_volume = np.clip(output_volume, dtype_info.min, dtype_info.max)
        output_volume = np.round(output_volume).astype(original_dtype)
    else:
        output_volume = output_volume.astype(original_dtype)

    header.set_data_dtype(original_dtype)

    return output_volume, affine, header


def main():
    parser = argparse.ArgumentParser(description="RescueCTA NIfTI inference")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to input NIfTI files')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="Checkpoint weights to load (default: ema).",
    )
    parser.add_argument(
        "--channel_strategy",
        choices=("mean", "center"),
        default="mean",
        help=(
            "How to reduce the three output channels. 'mean' reproduces the "
            "historical NIfTI pipeline; 'center' keeps the central channel."
        ),
    )
    parser.add_argument('--target_size', type=int, default=512, help='Target size for inference (default: 512)')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    make_dirs(args.output_dir)

    # 加载模型配置（若提供）
    g_name = "rescuecta"
    image_channels = 3  # 2.5D模式固定使用3通道
    if args.config:
        cfg = load_config(args.config)
        g_name = cfg.get('g_name', g_name)

    # 加载模型
    print("Loading model...")
    generator = build_generator(g_name, device, in_channels=image_channels, out_channels=image_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = select_generator_state(checkpoint, args.weights)
    print(f"Using {args.weights.upper()} generator weights")

    # 处理可能的 'module.' 前缀（兼容 DDP 保存格式）
    if isinstance(state_dict, dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        generator.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(state_dict)}")

    generator.eval()
    print("Model loaded successfully")

    # 获取所有NIfTI文件
    nifti_files = sorted([
        f for f in os.listdir(args.data_root)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])

    if not nifti_files:
        print(f"No NIfTI files found in {args.data_root}")
        return

    print(f"Found {len(nifti_files)} NIfTI files")

    target_size = (args.target_size, args.target_size)

    # 循环处理所有文件
    for nifti_file in tqdm(nifti_files, desc="Processing 3D volumes"):
        nifti_path = os.path.join(args.data_root, nifti_file)

        # 推理
        output_volume, affine, header = inference_single_volume(
            nifti_path,
            generator,
            device,
            target_size,
            channel_strategy=args.channel_strategy,
        )

        # 保存为NIfTI（数据类型和 header 已在 inference_single_volume 中对齐）
        output_nii = nib.Nifti1Image(output_volume, affine, header)

        # 构建输出文件名
        output_filename = nifti_file
        if not output_filename.endswith('.gz'):
            output_filename = output_filename.replace('.nii', '.nii.gz')

        output_path = os.path.join(args.output_dir, output_filename)
        nib.save(output_nii, output_path)

    print(f"Inference completed. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
