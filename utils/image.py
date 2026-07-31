"""图像处理工具"""

import numpy as np
from PIL import Image

try:
    import pydicom  # type: ignore
except ModuleNotFoundError:  # optional dependency (only needed for DICOM I/O)
    pydicom = None


def tensor2image(input_tensor, im_type=np.uint8, inference=False, window_center=None, window_width=None):
    """将Tensor转换为图像数组

    Args:
        input_tensor: 输入张量
        im_type: 输出图像类型
        inference: 是否为推理模式（使用窗宽窗位）
        window_center: 窗位（默认40，脑窗）
        window_width: 窗宽（默认400，脑窗）
    """
    input_num = input_tensor.cpu().detach().numpy().transpose((1, 2, 0))

    if inference:
        # 默认使用脑窗，适合脑CTA图像
        center = window_center if window_center is not None else 40
        width = window_width if window_width is not None else 400
        image_numpy = to_window_data(input_num, center, width)
    else:
        input_num = (input_num + 1) / 2 * 256
        image_numpy = np.clip(input_num, 0, 255)

    return image_numpy.astype(im_type)


def to_window_data(image, WC, WW):
    """应用窗宽窗位"""
    # 从 [-1, 1] 转换到 [0, 4095]
    image = (image + 1) * 0.5 * 4095
    # 转换到 HU 值 [-1024, 3071]
    image = image - 1024

    center = WC
    width = WW
    win_min = (2 * center - width) / 2.0 + 0.5
    win_max = (2 * center + width) / 2.0 + 0.5
    image[image < win_min] = win_min
    image[image > win_max] = win_max
    image = (image - win_min) / (win_max - win_min) * 255

    return image.astype(np.uint8)


def save_image(image_array, save_path):
    """保存图像"""
    image = Image.fromarray(image_array)
    image.save(save_path + '.jpg', format="JPEG")


def copy_numpy2dicom(arr, dcm_path):
    """将numpy数组复制到dicom"""
    if pydicom is None:
        raise ModuleNotFoundError("写入/读取 DICOM 需要安装 pydicom：pip install pydicom")
    ds = pydicom.dcmread(dcm_path, force=True)
    ds.PixelData = None
    arr = arr.astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.Rows, ds.Columns = arr.shape
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    return ds


def tensor2dicom(tensor, source_path, instance):
    """将Tensor转换为DICOM"""
    if pydicom is None:
        raise ModuleNotFoundError("写入/读取 DICOM 需要安装 pydicom：pip install pydicom")
    image = tensor.cpu().detach().float().numpy().transpose((1, 2, 0))
    image = (image + 1) * 0.5 * 4095
    image_numpy = image[:, :, 0]
    image_data = image_numpy.astype(np.uint16)
    target_dicom = copy_numpy2dicom(image_data, source_path)
    target_dicom.InstanceNumber = instance
    return target_dicom
