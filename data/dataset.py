"""数据加载模块"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label, binary_fill_holes

try:
    import pydicom  # type: ignore
except ModuleNotFoundError:  # optional dependency (only needed for .dcm)
    pydicom = None


def connected_region(slice_array, threshold=-100):
    """连通域分析 - 去除床和边缘"""
    binary_slice = slice_array > threshold
    labeled_slice, _ = label(binary_slice)
    region_volumes = np.bincount(labeled_slice.ravel())
    region_volumes[0] = 0
    largest_region_label = region_volumes.argmax()
    largest_region = (labeled_slice == largest_region_label).astype(np.uint8)
    filled_largest_region = binary_fill_holes(largest_region)
    return filled_largest_region


def match_folders(*folders):
    """检查多个文件夹的关联性"""
    if len(folders) < 2:
        raise ValueError("至少需要两个文件夹进行比较")

    def folder_entries(folder):
        try:
            with os.scandir(folder) as it:
                return {entry.name for entry in it}
        except FileNotFoundError as e:
            raise ValueError(f"文件夹不存在: {folder}") from e

    base_folder = folders[0]
    base_files = folder_entries(base_folder)
    for folder in folders[1:]:
        current_files = folder_entries(folder)
        if base_files != current_files:
            diff = base_files.symmetric_difference(current_files)
            sample = sorted(list(diff))[:20]
            more = "" if len(diff) <= 20 else f" ... (+{len(diff) - 20} more)"
            raise ValueError(
                f"文件夹 {base_folder} 和 {folder} 的文件不匹配。"
                f" base={len(base_files)}, current={len(current_files)}, diff={len(diff)}. "
                f"示例: {sample}{more}"
            )


class SliceDataset(Dataset):
    """2D切片数据集"""

    def __init__(self, data_root, target_size=(512, 512),
                 source_dirname='source', cta_dirname='CTA',
                 cache_dir=None,
                 random_rotate: bool = False,
                 rotate_degrees: float = 0.0,
                 rotate_prob: float = 0.0):
        self.target_size = target_size
        self.random_rotate = random_rotate
        self.rotate_degrees = float(rotate_degrees)
        self.rotate_prob = float(rotate_prob)

        self.source_dir = os.path.join(data_root, source_dirname)
        self.cta_dir = os.path.join(data_root, cta_dirname)
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cta_cache_dir = os.path.join(self.cache_dir, 'cta')
            self.source_cache_dir = os.path.join(self.cache_dir, 'source')
            os.makedirs(self.cta_cache_dir, exist_ok=True)
            os.makedirs(self.source_cache_dir, exist_ok=True)

        # 验证文件夹一致性
        match_folders(self.source_dir, self.cta_dir)

        self.cta_names = sorted(os.listdir(self.cta_dir))

    def __len__(self):
        return len(self.cta_names)

    def _load_image(self, path):
        """加载图像文件"""
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.dcm'):
            if pydicom is None:
                raise ModuleNotFoundError(
                    "读取 .dcm 需要安装 pydicom：pip install pydicom"
                )
            return pydicom.dcmread(path).pixel_array
        else:
            raise ValueError("文件必须是 .npy 或 .dcm 格式")

    def _ensure_channel_last(self, image):
        """确保图像为(H, W, C)或(H, W)"""
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] != image.shape[-1]:
            image = np.transpose(image, (1, 2, 0))
        return image

    def _resize_image(self, image, interpolation=cv2.INTER_LINEAR):
        """调整图像大小，支持单通道或多通道"""
        if image.ndim == 2:
            resized = cv2.resize(image, self.target_size, interpolation=interpolation)
        elif image.ndim == 3:
            channels = []
            for c in range(image.shape[2]):
                channels.append(cv2.resize(image[:, :, c], self.target_size, interpolation=interpolation))
            resized = np.stack(channels, axis=2)
        else:
            raise ValueError("仅支持2D或3D图像")
        return resized.astype(image.dtype)

    def _find_brain_bbox(self, image, background_val=-1024, margin_ratio=0.05, margin_min=4):
        """基于背景值找到脑部区域的最小外接正方形，并外扩一定边界"""
        image = self._ensure_channel_last(image)
        if image.ndim == 3:
            mask_source = image.mean(axis=2)
        else:
            mask_source = image

        mask = mask_source > (background_val + 1e-3)
        if not mask.any():
            return None  # 找不到脑组织，返回None以跳过裁剪

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

    def _random_rotate_pair(self, cta_chw: np.ndarray, source_chw: np.ndarray):
        if not self.random_rotate or self.rotate_degrees <= 0 or self.rotate_prob <= 0:
            return cta_chw, source_chw
        if np.random.rand() > self.rotate_prob:
            return cta_chw, source_chw

        angle = float(np.random.uniform(-self.rotate_degrees, self.rotate_degrees))
        c, h, w = cta_chw.shape

        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_value = (-1.0,) * c if c > 1 else -1.0

        def rotate_chw(arr_chw):
            if arr_chw.shape[0] == 1:
                src = arr_chw[0]
                rotated = cv2.warpAffine(
                    src, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value,
                )
                return np.expand_dims(rotated, 0)

            src = np.transpose(arr_chw, (1, 2, 0))
            rotated = cv2.warpAffine(
                src, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            return np.transpose(rotated, (2, 0, 1))

        return rotate_chw(cta_chw), rotate_chw(source_chw)

    def _normalize(self, image):
        """图像归一化: HU -> [-1, 1]"""
        image = self._ensure_channel_last(image)
        image = image.copy()
        # 归一化到 [-1, 1]
        image = image + 1024
        image[image < 0] = 0
        image = image / 4095
        image = (image - 0.5) / 0.5

        return image.astype(np.float32)

    def __getitem__(self, index):
        name = self.cta_names[index]
        return_dic = {}

        cache_used = False
        cache_name = os.path.splitext(name)[0] + ".npy"
        if self.cache_dir:
            cta_cache_path = os.path.join(self.cta_cache_dir, cache_name)
            source_cache_path = os.path.join(self.source_cache_dir, cache_name)
            if os.path.exists(cta_cache_path) and os.path.exists(source_cache_path):
                cta_np = np.load(cta_cache_path)
                source_np = np.load(source_cache_path)
                cta_chw, source_chw = self._random_rotate_pair(cta_np, source_np)
                cta_tensor = torch.from_numpy(cta_chw)
                source_tensor = torch.from_numpy(source_chw)
                cache_used = True

        if not cache_used:
            # 加载图像并统一通道维到最后
            cta_image = self._ensure_channel_last(self._load_image(os.path.join(self.cta_dir, name)))
            source_image = self._ensure_channel_last(self._load_image(os.path.join(self.source_dir, name)))

            # 计算联合脑区bbox并裁剪（若缩放因子不超过阈值）
            bbox_cta = self._find_brain_bbox(cta_image)
            bbox_source = self._find_brain_bbox(source_image)
            bbox = None
            if bbox_cta and bbox_source:
                y0 = min(bbox_cta[0], bbox_source[0])
                y1 = max(bbox_cta[1], bbox_source[1])
                x0 = min(bbox_cta[2], bbox_source[2])
                x1 = max(bbox_cta[3], bbox_source[3])
                bbox = (y0, y1, x0, x1)
            else:
                bbox = bbox_cta or bbox_source
            if bbox:
                y0, y1, x0, x1 = bbox
                crop_side = max(y1 - y0, x1 - x0)
                scale_factor = self.target_size[0] / crop_side
                if scale_factor <= 5:
                    cta_image = cta_image[y0:y1, x0:x1, ...]
                    source_image = source_image[y0:y1, x0:x1, ...]

            # 调整大小
            cta_image = self._resize_image(cta_image, cv2.INTER_LINEAR)
            source_image = self._resize_image(source_image, cv2.INTER_LINEAR)

            # 归一化
            cta_image = self._normalize(cta_image)
            source_image = self._normalize(source_image)

            # 转换为 (C,H,W) tensor
            def to_tensor(arr):
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, 0)
                elif arr.ndim == 3:
                    if arr.shape[2] in (1, 3):
                        arr = np.transpose(arr, (2, 0, 1))
                    elif arr.shape[0] in (1, 3):
                        pass  # already channel-first
                    else:
                        raise ValueError("无法确定图像通道维度")
                else:
                    raise ValueError("仅支持2D或3D图像")
                return torch.from_numpy(arr)

            cta_tensor = to_tensor(cta_image)
            source_tensor = to_tensor(source_image)

            if self.cache_dir:
                cta_np = cta_tensor.numpy()
                source_np = source_tensor.numpy()
                np.save(cta_cache_path, cta_np)
                np.save(source_cache_path, source_np)
            else:
                cta_np = cta_tensor.numpy()
                source_np = source_tensor.numpy()

            cta_np, source_np = self._random_rotate_pair(cta_np, source_np)
            cta_tensor = torch.from_numpy(cta_np)
            source_tensor = torch.from_numpy(source_np)

        return_dic['cta'] = cta_tensor
        return_dic['source'] = source_tensor
        return_dic['name'] = name  # 添加样本名称，用于异常检测时记录

        return return_dic


def create_dataloader(data_root, batch_size=1, shuffle=False,
                      num_workers=0, distributed=False, return_sampler=False,
                      drop_last: bool = True,
                      **kwargs):
    """创建数据加载器

    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        shuffle: 是否打乱数据（非分布式模式时生效）
        num_workers: 数据加载线程数
        distributed: 是否使用分布式采样器（DDP 模式）
        return_sampler: 是否返回采样器（用于 DDP 模式下设置 epoch）
        **kwargs: 传递给 SliceDataset 的其他参数

    Returns:
        如果 return_sampler=True: (DataLoader, sampler)
        否则: DataLoader
    """
    dataset = SliceDataset(
        data_root,
        **kwargs
    )

    sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # 使用 sampler 时不能使用 shuffle

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        drop_last=bool(drop_last),
        pin_memory=True,
        sampler=sampler
    )

    if return_sampler:
        return loader, sampler
    return loader
